#!/usr/bin/env python3
"""Native Rust Pocket TTS resident-model HTTP benchmark (stdlib only).

Launches Kyutai's official Rust 'serve' once per model/device configuration
and times streaming HTTP audio while model and voice-state caches remain warm.
Use beside benchmark_tts.py's fresh-process TTFA; never conflate them.
"""
from __future__ import annotations
import argparse
import http.client
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from benchmark_tts import (
    SAMPLE_RATE, PROMPTS, MODEL_VARIANTS, VOICES,
    analyze_pcm, describe, percentile, read_rss_kib
)

def choose_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch_server(binary: Path, model: str, threads: int, quantized: bool,
                  timeout: float) -> tuple[subprocess.Popen, tempfile._TemporaryFileWrapper | object, int, float]:
    port = choose_port()
    args = [
        str(binary), "serve", "--host", "127.0.0.1", "--port", str(port),
        "--language", model, "--default-voice", "alba",
    ]
    if quantized:
        args.append("--quantize")
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    env["POCKET_TTS_SEED"] = "42"
    logs = tempfile.TemporaryFile()
    started = time.monotonic()
    process = subprocess.Popen(args, stdout=logs, stderr=logs, env=env)
    try:
        while True:
            if process.poll() is not None:
                logs.seek(0)
                raise RuntimeError(
                    f"native Rust server exited {process.returncode}: "
                    + logs.read().decode("utf-8", errors="replace")[-1200:]
                )
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"server startup exceeded {timeout}s")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.3) as response:
                    if response.status == 200:
                        return process, logs, port, round((time.monotonic()-started)*1000, 3)
            except (OSError, TimeoutError):
                time.sleep(0.05)
    except Exception:
        process.kill()
        process.wait()
        logs.close()
        raise


def run_request(conn: http.client.HTTPConnection, process: subprocess.Popen,
                text: str, voice: str | None, timeout_s: float) -> dict:
    body = {"text": text}
    if voice:
        body["voice_url"] = voice
    data = urlencode(body).encode("utf-8")
    start = time.monotonic()
    conn.timeout = timeout_s
    conn.request("POST", "/tts", body=data, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "audio/wav",
    })
    response = conn.getresponse()
    headers_ms = (time.monotonic() - start) * 1000
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read(2048)!r}")
    if "audio/wav" not in (response.getheader("content-type") or ""):
        raise RuntimeError("server did not return WAV")
    header = bytearray()
    pcm = bytearray()
    first_pcm_ms = None
    first_audible_chunk_ms = None
    first_pcm_chunk_samples = None
    last_packet_ms = None
    gaps = []
    rss_peak = read_rss_kib(process.pid) or 0
    reads = 0
    while True:
        if time.monotonic() - start > timeout_s:
            raise TimeoutError("resident HTTP streaming timeout")
        chunk = response.read1(65536)
        when_ms = (time.monotonic() - start) * 1000
        if not chunk:
            break
        reads += 1
        rss_peak = max(rss_peak, read_rss_kib(process.pid) or 0)
        if last_packet_ms is not None:
            gaps.append(when_ms - last_packet_ms)
        last_packet_ms = when_ms
        if len(header) < 44:
            taken = min(44-len(header), len(chunk))
            header.extend(chunk[:taken])
            chunk = chunk[taken:]
        if chunk:
            if first_pcm_ms is None:
                first_pcm_ms = when_ms
                first_pcm_chunk_samples = len(chunk)//2
            pcm.extend(chunk)
            if first_audible_chunk_ms is None:
                # Valid PCM is 16-bit little-endian, mono, threshold = 1% FS.
                usable = chunk[:len(chunk) - len(chunk)%2]
                import array
                values = array.array("h")
                values.frombytes(usable)
                if sys.byteorder != "little":
                    values.byteswap()
                if any(abs(value) >= 328 for value in values):
                    first_audible_chunk_ms = when_ms
    wall_ms = (time.monotonic() - start) * 1000
    if len(header) != 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("resident server returned invalid WAV framing")
    signal = analyze_pcm(bytes(pcm))
    if signal["sample_rate"] != SAMPLE_RATE:
        raise ValueError("unexpected sample rate")
    seconds = signal["speech_duration_excluding_padding_s"]
    return {
        "headers_ms": round(headers_ms,3),
        "ttfa_first_pcm_ms": round(first_pcm_ms,3) if first_pcm_ms is not None else None,
        "ttfa_first_audible_chunk_ms": round(first_audible_chunk_ms,3) if first_audible_chunk_ms is not None else None,
        "ttfa_perceived_estimate_ms": round(first_pcm_ms + signal["onset_offset_ms"],3) if first_pcm_ms is not None else None,
        "wall_ms": round(wall_ms,3),
        "rtf": round(wall_ms/1000/seconds,5),
        "x_realtime": round(seconds/(wall_ms/1000),4),
        "peak_rss_mib": round(rss_peak/1024,3),
        "stream_reads": reads, "gap_p95_ms": round(percentile(gaps,95),3),
        "first_pcm_chunk_samples": first_pcm_chunk_samples,
        **signal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", default="english,7m,18m-refl")
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--quantization", default="off,on")
    parser.add_argument("--voices", default="preset,reference")
    parser.add_argument("--lengths", default="short,medium,long")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error("missing native Kyutai Rust executable")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    models = [MODEL_VARIANTS[v] for v in args.models.split(",")]
    threads = [int(v) for v in args.threads.split(",")]
    quantization = [v=="on" for v in args.quantization.split(",")]
    voices = args.voices.split(",")
    lengths = args.lengths.split(",")
    results = {
        "engine": "Kyutai Rust/XN, persistent HTTP streaming server",
        "mode": "resident_model",
        "schema_version": 1,
        "metrics": {
            "startup_ms": "launch of native Rust server until /health is ready; model and preset default voice loaded",
            "ttfa_first_pcm_ms": "HTTP request send to first arriving PCM payload, excludes 44-byte WAV header",
            "ttfa_first_audible_chunk_ms": "HTTP request send to first arriving chunk with amplitude >1% full scale",
            "rtf": "complete request wall time / output duration excluding Kyutai's fixed 200ms trailing silence",
            "reference_cache": "first voice_url request triggers encoder, later requests reuse the server LRU voice state; cached per server configuration",
            "peak_rss_mib": "sampled resident process VmRSS (observed lower bound), not OS file cache",
        },
        "platform": {
            "name": platform.platform(),
            "cpu_count": os.cpu_count(),
            "git_sha": os.environ.get("GITHUB_SHA"),
        },
        "servers": [], "trials": [], "errors": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected = len(models)*len(threads)*len(quantization)*len(voices)*len(lengths)*args.repeats
    counter = 0
    for model in models:
        for thread in threads:
            for quant in quantization:
                label = f"{model} threads={thread} quantized={quant}"
                print(f"RESIDENT START SERVER {label}", flush=True)
                try:
                    proc, logs, port, startup_ms = launch_server(args.binary, model, thread, quant, args.startup_timeout)
                except Exception as exc:
                    results["errors"].append({"server": label, "error": str(exc)})
                    print(f"RESIDENT SERVER ERROR {label}: {str(exc)[:300]}", flush=True)
                    args.output.write_text(json.dumps(results, indent=2))
                    continue
                results["servers"].append({
                    "model":model, "threads":thread, "quantized":quant,
                    "startup_ms":startup_ms, "base_rss_mib":round((read_rss_kib(proc.pid) or 0)/1024, 3)
                })
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=args.request_timeout)
                    first_reference = True
                    try:
                        for voice in voices:
                            for length in lengths:
                                for rep in range(args.repeats):
                                    counter += 1
                                    name = f"{counter}/{expected} {label} voice={voice} length={length} repeat={rep}"
                                    print(f"RESIDENT REQUEST {name}", flush=True)
                                    source = VOICES[voice] if voice=="reference" else None
                                    try:
                                        item = run_request(conn, proc, PROMPTS[length], source, args.request_timeout)
                                        item.update({
                                            "model":model, "voice":voice, "length":length,
                                            "threads":thread,"quantized":quant,"repetition":rep,
                                            "reference_voice_cache":"cold" if voice=="reference" and first_reference else ("warm" if voice=="reference" else "preloaded"),
                                            "server_startup_ms":startup_ms,
                                        })
                                        if voice=="reference":
                                            first_reference = False
                                        results["trials"].append(item)
                                        print(f"RESIDENT OK {name} TTFA={item['ttfa_first_audible_chunk_ms']}ms RTF={item['rtf']} RSS={item['peak_rss_mib']}MiB", flush=True)
                                    except Exception as exc:
                                        results["errors"].append({"trial":name,"error":str(exc)})
                                        print(f"RESIDENT ERROR {name}: {str(exc)[:400]}",flush=True)
                                        try:
                                            conn.close()
                                            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=args.request_timeout)
                                        except OSError:
                                            pass
                                    args.output.write_text(json.dumps(results,indent=2))
                    finally:
                        conn.close()
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    logs.close()
    groups = {}
    for item in results["trials"]:
        key=(item["model"],item["voice"],item["threads"],item["quantized"])
        groups.setdefault(key,[]).append(item)
    summary=[
        "# Resident-model Pocket TTS HTTP benchmark","",
        f"Successful trials: {len(results['trials'])}/{expected}; failures: {len(results['errors'])}",
        "", "These measurements **exclude model cold startup** from each request; server startup recorded separately.",
        "", "| Model | Voice | Threads | Quant | Count | TTFA p50/p95 (ms) | RTF p50 | End-to-end latency p50 (ms) | Peak RSS p50 (MiB) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (model,voice,t,q),rows in sorted(groups.items()):
        ttfa=describe([r["ttfa_first_audible_chunk_ms"] for r in rows if r["ttfa_first_audible_chunk_ms"] is not None])
        rtf=describe([r["rtf"] for r in rows])
        wall=describe([r["wall_ms"] for r in rows])
        rss=describe([r["peak_rss_mib"] for r in rows])
        summary.append(f"| {model} | {voice} | {t} | {'int8' if q else 'f32'} | {len(rows)} | {ttfa.get('p50',0):.1f}/{ttfa.get('p95',0):.1f} | {rtf.get('p50',0):.3f} | {wall.get('p50',0):.1f} | {rss.get('p50',0):.1f} |")
    startup=describe([x["startup_ms"] for x in results["servers"]])
    summary.extend(["", f"Server startup (all configurations): {startup}", "",
        "The first reference-voice HTTP request performs Mimi encoding; subsequent requests reuse cached voice state.",
        "Server handles generation serially under its built-in mutex. This matrix is single-client and does not measure concurrency capacity.",
        "Streaming includes HTTP transport and initial network-loopback overhead; sampled RSS is not process+kernel total memory.",
        "WAV 200ms final silence is excluded from RTF. First audible chunk is approximate and threshold-based.",
        "Raw trial JSON is authoritative for per-utterance statistics and failure diagnosis.",
    ])
    args.output.with_suffix(".md").write_text("\n".join(summary)+"\n")
    print("RESIDENT BENCH SUMMARY\n" + "\n".join(summary),flush=True)
    if results["errors"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
