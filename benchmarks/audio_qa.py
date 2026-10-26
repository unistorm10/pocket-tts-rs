#!/usr/bin/env python3
"""Acoustic and optional ASR verification of *real* Pocket TTS WAV output.

Stdlib for objective waveform gates, optional faster-whisper for intelligibility.
Never counts non-silent audio as proof of readable or cloned speech. JSON only,
no PCM or voice recordings in artifacts. Use an external speaker verifier for
voice identity; RMS, pitch and WER cannot establish speaker similarity.
"""
import argparse
import array
import json
import math
import re
import statistics
import sys
import wave
from pathlib import Path

def words(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", s.lower())

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(curr[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = curr
    return prev[-1]

def pcm_analysis(path):
    with wave.open(str(path), "rb") as f:
        channels, sr, width, n = (
            f.getnchannels(), f.getframerate(), f.getsampwidth(), f.getnframes()
        )
        if channels != 1 or sr != 24000 or width != 2:
            raise ValueError(f"expected mono 24kHz signed 16-bit PCM, got {channels}ch/{sr}Hz/{width*8}bit")
        if n < 2400:
            raise ValueError(f"too few samples: {n}")
        data = f.readframes(n)
    if len(data) != n * 2:
        raise ValueError("truncated WAV data")
    pcm = array.array("h")
    pcm.frombytes(data)
    if sys.byteorder != "little":
        pcm.byteswap()
    peak = max(abs(x) for x in pcm)/32768
    dc = statistics.mean(pcm)/32768
    rms = math.sqrt(sum(float(x)**2 for x in pcm)/n)/32768
    clipped = sum(abs(x) >= 32760 for x in pcm)/n
    # Approximate speech activity from 20ms frame RMS. This is not VAD and
    # cannot tell speech from music or noise.
    frame_n = 480
    rms_frames = []
    for i in range(0, n-frame_n+1, frame_n):
        rms_frames.append(math.sqrt(sum(float(x)**2 for x in pcm[i:i+frame_n])/frame_n)/32768)
    active = sum(x >= 0.008 for x in rms_frames)/len(rms_frames) if rms_frames else 0.0
    results = {
        "file": path.name, "sample_rate": sr, "channels": channels,
        "bit_depth": width*8, "samples": n, "duration_s": round(n/sr, 4),
        "peak": round(peak, 6), "rms": round(rms, 6),
        "dc_offset": round(dc, 6), "clipped_fraction": round(clipped, 6),
        "20ms_active_frame_fraction": round(active, 4),
    }
    checks = {
        "audio_signal": rms >= 0.001 and peak >= 0.01,
        "not_severely_clipped": clipped < 0.02,
        "low_dc_offset": abs(dc) < 0.05,
        "not_mostly_silent": active >= 0.15,
    }
    return results, checks

def asr_analysis(path, expected, model_name, threads):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed; never substitute audio-RMS for ASR") from exc
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
    segments, info = model.transcribe(
        str(path), language="en", beam_size=5, vad_filter=False,
        condition_on_previous_text=False, temperature=0,
    )
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    ref_words, got_words = words(expected), words(transcript)
    if not ref_words:
        raise ValueError("reference text has no words")
    ref_chars, got_chars = list("".join(ref_words)), list("".join(got_words))
    return {
        "engine": "faster-whisper", "model": model_name,
        "expected_text": expected, "transcript": transcript,
        "wer": round(edit_distance(ref_words, got_words)/len(ref_words), 5),
        "cer": round(edit_distance(ref_chars, got_chars)/max(1, len(ref_chars)), 5),
        "asr_language": info.language,
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-text")
    p.add_argument("--asr-model", help="require and run faster-whisper (e.g. small.en)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-wer", type=float, help="only valid together with --asr-model")
    args=p.parse_args()
    if args.max_wer is not None and (not args.asr_model or not args.expected_text):
        p.error("--max-wer requires --asr-model and --expected-text")
    report={"schema_version":1, "acoustics":None, "checks":{}, "asr":None,
            "speaker_similarity":"not_measured: requires a calibrated independent speaker-embedding model and consented reference",
            "naturalness":"not_measured: requires listening tests or separately calibrated quality predictor"}
    try:
        report["acoustics"], checks=pcm_analysis(args.wav)
        report["checks"].update(checks)
        if args.asr_model:
            if not args.expected_text:
                p.error("--asr-model requires --expected-text")
            report["asr"]=asr_analysis(args.wav,args.expected_text,args.asr_model,args.threads)
            if args.max_wer is not None:
                report["checks"]["wer_at_most_threshold"]=report["asr"]["wer"] <= args.max_wer
        elif args.expected_text:
            report["asr"]="not_measured: supply --asr-model to evaluate intelligibility"
    except Exception as err:
        report["error"]=str(err)
        report["checks"]["validation_completed"]=False
    report["passed"]=bool(report["checks"]) and all(report["checks"].values()) and "error" not in report
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:
        raise SystemExit(1)

if __name__=="__main__":
    main()
