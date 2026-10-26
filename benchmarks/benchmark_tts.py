#!/usr/bin/env python3
"""Pocket TTS reproducible streaming benchmark. Stdlib only.

Runs a real native Rust Pocket TTS binary. Reads its unseekable RIFF stream,
times arriving PCM rather than header bytes, validates signal, collects process
RSS, and writes machine-readable trials + summary. No simulated output.
"""
from __future__ import annotations
import argparse
import array
import hashlib
import json
import math
import os
import platform
import re
import selectors
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SAMPLE_RATE = 24000
OUTPUT_PADDING_SEC = 0.2  # Kyutai Rust StreamingWavWriter::finalize
PROMPTS = {
    "short": "Hello. This is a voice latency test.",
    "medium": "This benchmark measures time to first audio, total synthesis latency, and streaming generation speed. The model is running on a local CPU.",
    "long": "A responsive speech engine should begin speaking before it has completed the entire sentence. We want to measure not only the first audible sample, but also the full generation time, output quality, latency variation, and memory requirements. These measurements help decide which inference engine and which model size should be deployed on different devices. We repeat the test with an original voice sample so the cost of preparing a voice can be separated from generation throughput.",
}
MODEL_VARIANTS = {
    "english": "english",
    "7m": "english_7m_2026-10",
    "18m-refl": "english_18m_refl_2026-10",
}
VOICES = {
    "preset": "alba",
    "reference": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
}
GEN_PATTERN = re.compile(r"Generated:\s*(\d+)\s*ms of audio in\s*(\d+)\s*ms")
ENC_PATTERN = re.compile(r"Encoding audio prompt took\s*(\d+)\s*ms")
PROMPT_PATTERN = re.compile(r"Prompting audio took\s*(\d+)\s*ms")
TEXT_PATTERN = re.compile(r"Prompting text took\s*(\d+)\s*ms")


def read_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass
    return None


def analyze_pcm(raw: bytes) -> dict:
    if len(raw) < 2:
        raise ValueError("no PCM data arrived")
    raw = raw[: len(raw) - len(raw) % 2]
    pcm = array.array("h")
    pcm.frombytes(raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    n = len(pcm)
    peak_int = max(abs(x) for x in pcm)
    rms = math.sqrt(sum(float(x) * float(x) for x in pcm) / n) / 32768.0
    # Audibility onset is a reproducible amplitude-gated approximation, not
    # a perceptual listening test or an ASR/quality metric.
    threshold = int(0.01 * 32768)
    first_audible_idx = next((i for i, x in enumerate(pcm) if abs(x) >= threshold), None)
    if first_audible_idx is None or rms < 1e-5:
        raise ValueError("output contains no detectable audible PCM")
    actual_duration = n / SAMPLE_RATE
    speech_duration = max(0.0, actual_duration - OUTPUT_PADDING_SEC)
    return {
        "sample_count": n,
        "sample_rate": SAMPLE_RATE,
        "output_duration_s": round(actual_duration, 5),
        "speech_duration_excluding_padding_s": round(speech_duration, 5),
        "peak": round(peak_int / 32768, 7),
        "rms": round(rms, 7),
        "onset_offset_ms": round(1000 * first_audible_idx / SAMPLE_RATE, 3),
        "first_audible_sample_index": first_audible_idx,
        "pcm_sha256": hashlib.sha256(raw).hexdigest(),
    }


def run_trial(binary: Path, model: str, voice: str, prompt: str, threads: int,
              quantize: bool, timeout_s: float, seed: int) -> dict:
    command = [
        str(binary), "generate", "--language", model, "--text", prompt,
        "--voice", voice, "--output-path", "-", "--max-tokens", "50",
    ]
    if quantize:
        command.append("--quantize")
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    env["POCKET_TTS_SEED"] = str(seed)

    start = time.monotonic()
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=stderr_file, env=env,
            bufsize=0,
        )
        assert process.stdout is not None
        stream = selectors.DefaultSelector()
        stream.register(process.stdout, selectors.EVENT_READ)
        header = bytearray()
        pcm = bytearray()
        first_byte_ms = None
        first_pcm_ms = None
        first_audible_chunk_ms = None
        first_pcm_chunk_len = None
        arrival_samples = []
        last_receive_ms = None
        chunk_intervals_ms = []
        rss_peak = 0
        eof = False
        try:
            while not eof:
                elapsed = time.monotonic() - start
                if elapsed > timeout_s:
                    process.kill()
                    raise TimeoutError(f"generation exceeded {timeout_s}s for {model} {voice}")
                rss = read_rss_kib(process.pid)
                if rss is not None:
                    rss_peak = max(rss_peak, rss)
                for _key, _mask in stream.select(timeout=0.025):
                    chunk = os.read(process.stdout.fileno(), 65536)
                    stamp_ms = (time.monotonic() - start) * 1000
                    if not chunk:
                        eof = True
                        break
                    if first_byte_ms is None:
                        first_byte_ms = stamp_ms
                    if last_receive_ms is not None:
                        chunk_intervals_ms.append(stamp_ms - last_receive_ms)
                    last_receive_ms = stamp_ms
                    arrival_samples.append(stamp_ms)
                    if len(header) < 44:
                        take = min(44 - len(header), len(chunk))
                        header.extend(chunk[:take])
                        chunk = chunk[take:]
                    if chunk:
                        if first_pcm_ms is None:
                            first_pcm_ms = stamp_ms
                            first_pcm_chunk_len = len(chunk) // 2
                        pcm.extend(chunk)
                        if first_audible_chunk_ms is None:
                            even = chunk[:len(chunk) - len(chunk)%2]
                            if even:
                                values = array.array("h")
                                values.frombytes(even)
                                if sys.byteorder != "little":
                                    values.byteswap()
                                if any(abs(x) >= 328 for x in values):
                                    first_audible_chunk_ms = stamp_ms
            code = process.wait(timeout=5)
        finally:
            stream.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        wall_ms = (time.monotonic() - start) * 1000
        stderr_file.seek(0)
        logs = stderr_file.read().decode("utf-8", errors="replace")
    if code != 0:
        # Do not expose any authentication environment variables or entire log.
        raise RuntimeError(f"inference exit {code}: {logs[-1600:]}")
    if len(header) != 44:
        raise ValueError(f"expected 44-byte RIFF header, got {len(header)} bytes")
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE" or header[36:40] != b"data":
        raise ValueError("stream is not 44-byte PCM RIFF WAV")
    _, channels, sr, _bitrate, _align, depth = struct.unpack_from("<HHIIHH", header, 20)
    if sr != SAMPLE_RATE or channels != 1 or depth != 16:
        raise ValueError(f"unexpected format: sample_rate={sr} channels={channels} bits={depth}")
    signal = analyze_pcm(bytes(pcm))
    gen = [(int(a), int(b)) for a, b in GEN_PATTERN.findall(logs)]
    generated_ms = sum(x for x, _ in gen)
    inference_ms = sum(y for _, y in gen)
    audio_seconds = signal["speech_duration_excluding_padding_s"]
    overall_seconds = wall_ms / 1000
    first_pcm = first_pcm_ms or 0
    return {
        "model": model, "voice": "reference" if voice.startswith("hf://") else "preset",
        "threads": threads, "quantized": quantize, "seed": seed,
        "input_chars": len(prompt), "input_words": len(prompt.split()),
        "wall_ms": round(wall_ms, 3),
        "first_header_byte_ms": round(first_byte_ms, 3) if first_byte_ms is not None else None,
        "first_pcm_ms": round(first_pcm, 3),
        "first_audible_chunk_ms": round(first_audible_chunk_ms, 3) if first_audible_chunk_ms is not None else None,
        "perceived_ttfa_estimate_ms": round(first_pcm + signal["onset_offset_ms"], 3),
        "first_pcm_chunk_samples": first_pcm_chunk_len,
        "stream_reads": len(arrival_samples),
        "inter_read_gap_p95_ms": round(percentile(chunk_intervals_ms, 95), 3),
        "peak_rss_mib": round(rss_peak / 1024, 3),
        "process_inclusive_rtf": round(overall_seconds / audio_seconds, 5),
        "process_inclusive_x_realtime": round(audio_seconds / overall_seconds, 4),
        "model_generation_ms": inference_ms if gen else None,
        "model_reported_audio_ms": generated_ms if gen else None,
        "model_only_rtf": round(inference_ms / generated_ms, 5) if generated_ms else None,
        "model_only_x_realtime": round(generated_ms / inference_ms, 5) if inference_ms else None,
        "reference_encode_ms": sum(int(x) for x in ENC_PATTERN.findall(logs)) or None,
        "reference_prompt_ms": sum(int(x) for x in PROMPT_PATTERN.findall(logs)) or None,
        "text_prompt_ms": sum(int(x) for x in TEXT_PATTERN.findall(logs)) or None,
        **signal,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    at = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] * (hi - at) + ordered[hi] * (at - lo)


def describe(values: list[float]) -> dict:
    return {k: round(v, 3) for k, v in {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": min(values), "max": max(values),
        "mean": statistics.mean(values),
    }.items()} if values else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", default="english,7m,18m-refl")
    parser.add_argument("--voices", default="preset,reference")
    parser.add_argument("--lengths", default="short,medium,long")
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--quantization", default="off,on")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-on-trial-error", action="store_true")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"native Rust inference executable missing: {args.binary}")
    models = [MODEL_VARIANTS[k] for k in args.models.split(",")]
    voices = [VOICES[k] for k in args.voices.split(",")]
    lengths = args.lengths.split(",")
    threads = [int(x) for x in args.threads.split(",")]
    quant = [v == "on" for v in args.quantization.split(",")]
    if args.repeats < 1 or any(t < 1 for t in threads):
        parser.error("repeats and CPU thread counts must be positive")
    result = {
        "schema_version": 1,
        "engine": "kyutai/pocket-tts v4.0.1 official native Rust XN",
        "backend_type": "new_process_per_request",
        "note": "TTFA includes process/model/voice load. Repeat processes benefit from cached files, not resident model state. Truly warm in-process model timings are not measured.",
        "metric_definitions": {
            "first_pcm_ms": "launch to first non-header PCM bytes arriving on stdout",
            "first_audible_chunk_ms": "launch to stdout read containing sample above 1% full-scale",
            "perceived_ttfa_estimate_ms": "first_pcm_ms + leading-audible-onset audio time; estimate, not sound-card latency",
            "process_inclusive_rtf": "whole-process wall time / output audio duration excluding fixed 200ms tail padding; lower is better",
            "model_only_rtf": "upstream reported generation_ms / generated_audio_ms; excludes loading/prompt encoding",
            "model_only_x_realtime": "inverse of model_only_rtf; higher is faster",
            "peak_rss_mib": "max /proc/PID VmRSS sampled at 25ms intervals; lower bound on peak",
            "inter_read_gap_p95_ms": "host pipe read arrival gaps; OS read boundaries not guaranteed to be model frame boundaries",
        },
        "platform": {
            "uname": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpu": Path("/proc/cpuinfo").read_text(errors="replace").split("model name\t: ")[-1].split("\n")[0] if Path("/proc/cpuinfo").exists() else "unknown",
            "git_sha": os.environ.get("GITHUB_SHA"),
            "model_seed": args.seed,
        },
        "trials": [], "errors": [],
    }
    configurations = [
        (m, v, n, t, q, repetition)
        for m in models for v in voices for n in lengths
        for t in threads for q in quant for repetition in range(args.repeats)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for idx, (model, voice, length, thread_count, q, repetition) in enumerate(configurations, 1):
        label = f"{idx}/{len(configurations)} {model} {length} {'clone' if voice.startswith('hf://') else 'preset'} t{thread_count} q{int(q)} rep{repetition}"
        print(f"BENCH START {label}", flush=True)
        try:
            trial = run_trial(args.binary, model, voice, PROMPTS[length], thread_count, q, args.timeout, args.seed)
            trial.update({"length": length, "repetition": repetition, "trial_index": idx, "process_cache_status": "first_run" if idx == 1 else "repeat_process"})
            result["trials"].append(trial)
            print(f"BENCH OK {label} TTFA={trial['first_audible_chunk_ms']}ms RTF={trial['process_inclusive_rtf']} model_RTF={trial['model_only_rtf']} RSS={trial['peak_rss_mib']}MiB", flush=True)
        except Exception as exc:
            error = {"label": label, "error": str(exc)}
            result["errors"].append(error)
            print(f"BENCH ERROR {label}: {str(exc)[:500]}", flush=True)
            if args.fail_on_trial_error:
                break
        # Save partial evidence even if later trial fails.
        args.output.write_text(json.dumps(result, indent=2))
    groups = {}
    for item in result["trials"]:
        key = (item["model"], item["voice"], item["threads"], item["quantized"])
        groups.setdefault(key, []).append(item)
    print("BENCH SUMMARY")
    summary_lines = ["# Pocket TTS native Rust streaming benchmark", "",
                     f"Successful runs: {len(result['trials'])} / {len(configurations)}; errors: {len(result['errors'])}",
                     "", "All timings are process-inclusive unless named model-only. Models are **not** kept resident between requests.",
                     "", "| Model | Voice | Threads | Quant | Runs | Audible TTFA p50/p95 (ms) | End-to-end RTF p50 | Model RTF p50 | Peak RSS MiB p50 |",
                     "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for (model, voice, t, q), rows in sorted(groups.items()):
        ttfas = describe([r["first_audible_chunk_ms"] for r in rows if r["first_audible_chunk_ms"] is not None])
        rtf = describe([r["process_inclusive_rtf"] for r in rows])
        native = describe([r["model_only_rtf"] for r in rows if r["model_only_rtf"] is not None])
        rss = describe([r["peak_rss_mib"] for r in rows])
        summary_lines.append(f"| {model} | {voice} | {t} | {'int8' if q else 'f32'} | {len(rows)} | {ttfas.get('p50',0):.0f}/{ttfas.get('p95',0):.0f} | {rtf.get('p50',0):.3f} | {native.get('p50',0):.3f} | {rss.get('p50',0):.1f} |")
    summary_lines += ["", "## Method and caveats",
                      "- Streaming audio captured as bytes from Kyutai's native Rust CLI stdout. The 44-byte WAV header is excluded from TTFA.",
                      "- Native CLI has a fixed 200 ms silence tail. RTF denominator excludes it. Model RTF comes from Kyutai's logged per-text synthesis.",
                      "- Repeated subprocess launches are **not warm resident model inference**. No direct Candle or CUDA benchmark was performed.",
                      "- Reference voice includes actual Mimi audio encoding, not only loading a saved voice embedding.",
                      "- First audible chunk uses a 1%-of-full-scale PCM threshold; perceived TTFA also reports leading-audio offset as an estimate.",
                      "- Speech quality/ASR WER and speaker similarity require independent measurements, not inferred from non-silent PCM.",
                      "", "See benchmark JSON for every trial, metrics, model and machine metadata."]
    summary = "\n".join(summary_lines) + "\n"
    args.output.with_suffix(".md").write_text(summary)
    print(summary, flush=True)
    if result["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
