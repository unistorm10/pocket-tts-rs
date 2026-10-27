#!/usr/bin/env python3
"""Independent speaker-identity evaluation for real Pocket TTS cloned audio.

Uses Apache-2.0 SpeechBrain ECAPA-TDNN speaker embeddings, not the TTS
generator itself. Compares generated audio to (a) the exact conditioning
recording, (b) a different recording of the same actor, and (c) another
speaker. No arbitrary cosine cut-off is silently equated with naturalness,
speaker identity, or production-ready cloning.
The example voice files are from Kyutai's public tts-voices collection.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from huggingface_hub import hf_hub_download
from speechbrain.inference.speaker import SpeakerRecognition

VOICES_REPO = "kyutai/tts-voices"
REFERENCE = "alba-mackenna/casual.wav"
POSITIVE = "alba-mackenna/announcer.wav"
NEGATIVE = "voice-zero/bill_boerst.wav"

def speaker_score(model: SpeakerRecognition, a: Path, b: Path) -> float:
    score, _ = model.verify_files(str(a), str(b))
    value = float(score.detach().cpu().flatten()[0])
    if not math.isfinite(value):
        raise ValueError("speaker embedding similarity is non-finite")
    return value

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None,
                        help="Optional locally recorded reference with consent")
    parser.add_argument("--positive-control", type=Path, default=None)
    parser.add_argument("--negative-control", type=Path, default=None)
    parser.add_argument("--require-control-ranking", action="store_true",
                        help="Fail when the independent same-speaker control is not above the impostor")
    args = parser.parse_args()
    if not args.generated.is_file():
        parser.error("the generated WAV file must exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "model_type": "independent pretrained ECAPA-TDNN speaker embedding",
        "generated_file": args.generated.name,
        "controls": {
            "reference": REFERENCE,
            "same_speaker_other_recording": POSITIVE,
            "different_speaker": NEGATIVE,
        },
        "similarity": None,
        "diagnostic_only": True,
        "note": "Uncalibrated cosine similarity does not by itself certify voice cloning or perceptual naturalness.",
    }
    try:
        def sample(local: Path | None, name: str) -> Path:
            return local if local is not None else Path(
                hf_hub_download(repo_id=VOICES_REPO, filename=name)
            )
        ref = sample(args.reference, REFERENCE)
        pos = sample(args.positive_control, POSITIVE)
        neg = sample(args.negative_control, NEGATIVE)
        classifier = SpeakerRecognition.from_hparams(
            source=report["model"],
            savedir=str(Path.home()/".cache"/"pocket_tts_rs_speaker_ecapa"),
            run_opts={"device": "cpu"},
        )
        comparisons = {
            "generated_vs_reference": speaker_score(classifier, args.generated, ref),
            "generated_vs_other_same_speaker": speaker_score(classifier, args.generated, pos),
            "reference_vs_other_same_speaker": speaker_score(classifier, ref, pos),
            "reference_vs_different_speaker": speaker_score(classifier, ref, neg),
            "generated_vs_different_speaker": speaker_score(classifier, args.generated, neg),
        }
        comparisons["generated_reference_margin_over_impostor"] = (
            comparisons["generated_vs_reference"] -
            comparisons["reference_vs_different_speaker"]
        )
        comparisons["generated_margin_over_impostor"] = (
            comparisons["generated_vs_reference"] -
            comparisons["generated_vs_different_speaker"]
        )
        comparisons["generated_positive_margin_over_impostor"] = (
            comparisons["generated_vs_other_same_speaker"] -
            comparisons["generated_vs_different_speaker"]
        )
        # Diagnostic sanity flags are not calibrated production thresholds.
        report["control_ranking"] = {
            "reference_positive_above_negative": (
                comparisons["reference_vs_other_same_speaker"] >
                comparisons["reference_vs_different_speaker"]
            ),
            "generated_reference_above_negative": (
                comparisons["generated_vs_reference"] >
                comparisons["generated_vs_different_speaker"]
            ),
            "generated_other_same_above_negative": (
                comparisons["generated_vs_other_same_speaker"] >
                comparisons["generated_vs_different_speaker"]
            ),
        }
        report["similarity"] = {k: round(v, 6) for k, v in comparisons.items()}
        report["completed"] = True
        if args.require_control_ranking and not all(report["control_ranking"].values()):
            report["control_rankings_passed"] = False
        else:
            report["control_rankings_passed"] = all(report["control_ranking"].values())
    except Exception as exc:
        report["completed"] = False
        report["error"] = str(exc)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["completed"] or (args.require_control_ranking and not report.get("control_rankings_passed", False)):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
