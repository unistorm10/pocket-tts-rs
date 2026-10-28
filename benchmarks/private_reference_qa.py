#!/usr/bin/env python3
"""Independently evaluate five private reference-voice clones.

The private source WAVs stay on the runner. Export only aggregate measurements,
never raw audio, transcripts, decoded utterances or voice embeddings. Numbered
references have UNKNOWN relationships; do not treat them as impostors or
same-speaker positives without identity annotations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from speechbrain.inference.speaker import SpeakerRecognition

from audio_qa import edit_distance, pcm_analysis, words
from speaker_qa import speaker_embedding, speaker_score

PROMPTS = {
    1: "The museum opens early on Saturday so we can arrive before the crowds.",
    2: "Please put the blue notebook beside the old wooden table.",
}
PUBLIC_COMPARISONS = {
    "alba": "alba-mackenna/casual.wav",
    "bill": "voice-zero/bill_boerst.wav",
}


def transcribe(model: WhisperModel, wav: Path) -> str:
    chunks, _ = model.transcribe(
        str(wav), language="en", beam_size=5, vad_filter=False,
        condition_on_previous_text=False, temperature=0,
    )
    return " ".join(chunk.text.strip() for chunk in chunks).strip()


def evaluate(speech_dir: Path, output_dir: Path, variant: str, max_wer: float) -> dict:
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
    speaker = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / ".cache" / "pocket_tts_rs_speaker_ecapa"),
        run_opts={"device": "cpu"},
    )
    # Two known, publicly hosted comparison speakers. The private corpus has
    # no validated speaker IDs; do not assume any pair of its clips differ.
    anchors = {
        key: speaker_embedding(
            speaker, Path(hf_hub_download(repo_id="kyutai/tts-voices", filename=filename))
        )
        for key, filename in PUBLIC_COMPARISONS.items()
    }
    records = []
    failures = []
    # Encode each source once. Rank the clone against all five sources only
    # diagnostically: multiple recordings may belong to the same person.
    reference_embeddings = {}
    for index in range(1, 6):
        label = f"reference_{index}"
        reference = speech_dir / f"{label}.wav"
        transcript = speech_dir / f"{label}.txt"
        if not reference.is_file() or not transcript.is_file():
            raise ValueError(f"missing WAV/transcript for {label}")
        if not transcript.read_text(encoding="utf-8").strip():
            raise ValueError(f"empty reference transcript for {label}")
        reference_embeddings[label] = speaker_embedding(speaker, reference)
    for index in range(1, 6):
        label = f"reference_{index}"
        ref_embedding = reference_embeddings[label]
        for prompt_id, phrase in PROMPTS.items():
            wav_name = f"private-clone-{variant}-{label}-p{prompt_id}.wav"
            wav = output_dir / wav_name
            if not wav.is_file():
                raise ValueError(f"missing generated WAV for {label} prompt {prompt_id}")
            acoustic, checks = pcm_analysis(wav)
            decoded = transcribe(whisper, wav)
            expected = words(phrase)
            observed = words(decoded)
            wer = edit_distance(expected, observed) / len(expected)
            clone_embedding = speaker_embedding(speaker, wav)
            own_score = speaker_score(clone_embedding, ref_embedding)
            other_reference_scores = {
                name: speaker_score(clone_embedding, embedding)
                for name, embedding in reference_embeddings.items() if name != label
            }
            conditioning_rank = 1 + sum(
                score > own_score for score in other_reference_scores.values()
            )
            background_scores = {
                name: speaker_score(clone_embedding, embedding)
                for name, embedding in anchors.items()
            }
            # Uncalibrated diagnostic ranking, not a claim of identity.
            reference_is_closest = own_score > max(background_scores.values())
            passed = all(checks.values()) and math.isfinite(wer) and wer <= max_wer
            if not passed:
                failures.append(f"{label}:p{prompt_id}")
            records.append({
                "reference_id": label,
                "prompt_id": prompt_id,
                "wer": round(wer, 5),
                "speaker_score_to_conditioning": round(own_score, 6),
                "speaker_scores_to_public_comparisons": {
                    k: round(v, 6) for k, v in background_scores.items()
                },
                "conditioning_ranked_closest": reference_is_closest,
                "conditioning_rank_among_five": conditioning_rank,
                "nonconditioning_reference_scores": {
                    k: round(v, 6) for k, v in other_reference_scores.items()
                },
                "cross_reference_identities_unverified": True,
                "speaker_fidelity_calibrated": False,
                "acoustics": {
                    "duration_s": acoustic["duration_s"],
                    "rms": acoustic["rms"],
                    "peak": acoustic["peak"],
                    "clipped_fraction": acoustic["clipped_fraction"],
                },
                "acoustic_checks": checks,
                "asr_and_acoustic_gate": passed,
            })
    return {
        "schema_version": 1,
        "variant": variant,
        "reference_repo": "unistorm10/reference",
        "reference_ids": [f"reference_{i}" for i in range(1, 6)],
        "clips_per_reference": len(PROMPTS),
        "speaker_identities_known": False,
        "speaker_similarity_calibrated": False,
        "naturalness_assessed": False,
        "private_audio_or_transcripts_exported": False,
        "failed_asr_or_acoustics": failures,
        "passed_asr_and_acoustics": not failures,
        "results": records,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speech-dir", required=True, type=Path)
    p.add_argument("--generated-dir", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=["probe-english", "probe-7m", "probe-18m"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-wer", type=float, default=0.35)
    args = p.parse_args()
    if not 0 <= args.max_wer <= 1:
        p.error("max WER must be between 0 and 1")
    try:
        report = evaluate(args.speech_dir, args.generated_dir, args.variant, args.max_wer)
    except Exception as exc:
        # Avoid leaking private filesystem paths or transcript text in failure
        # JSON; detailed tracebacks belong only on the private runner.
        report = {"schema_version": 1, "variant": args.variant,
                  "evaluation_completed": False, "error_type": type(exc).__name__,
                  "private_audio_or_transcripts_exported": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PRIVATE_REFERENCE_QA_FAILED error_type=" + type(exc).__name__, flush=True)
        raise SystemExit(1) from None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PRIVATE_REFERENCE_QA variant={} samples={} failed_asr_or_acoustics={} "
        "identity_diagnostic_only=true".format(
            args.variant, len(report["results"]), len(report["failed_asr_or_acoustics"])
        ),
        flush=True,
    )
    if not report["passed_asr_and_acoustics"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
