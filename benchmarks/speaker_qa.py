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
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from speechbrain.inference.speaker import SpeakerRecognition

VOICES_REPO = "kyutai/tts-voices"
REFERENCE = "alba-mackenna/casual.wav"
POSITIVE = "alba-mackenna/announcer.wav"
NEGATIVE = "voice-zero/bill_boerst.wav"

SPEAKER_SAMPLE_RATE = 16_000


def load_waveform(path: Path) -> torch.Tensor:
    """Read audio independently of torchaudio backends, resampling to 16 kHz.

    SpeechBrain 1.0.3's verify_files() calls torchaudio.load(), which is not
    usable on all minimal Linux runners. Its verify_batch()/encode_batch()
    accept waveforms directly, so preserve the identical ECAPA model and
    cosine metric while using libsndfile for decoding.
    """
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate <= 0 or waveform.ndim != 2 or waveform.shape[1] < 1:
        raise ValueError(f"invalid waveform format: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"non-finite audio samples: {path}")
    mono = waveform.mean(axis=1, dtype=np.float32)
    if rate != SPEAKER_SAMPLE_RATE:
        factor = math.gcd(rate, SPEAKER_SAMPLE_RATE)
        mono = resample_poly(
            mono, SPEAKER_SAMPLE_RATE // factor, rate // factor
        ).astype(np.float32, copy=False)
    if mono.size < SPEAKER_SAMPLE_RATE:
        raise ValueError(f"speaker sample must contain at least 1 s: {path}")
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    if not math.isfinite(rms) or rms < 1e-6:
        raise ValueError(f"speaker sample is silent: {path}")
    return torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)


def speaker_embedding(model: SpeakerRecognition, path: Path) -> torch.Tensor:
    """Compute ECAPA once per file, using the model's own 16-kHz audio contract."""
    waveform = load_waveform(path)
    with torch.inference_mode():
        embedding = model.encode_batch(waveform, normalize=False).flatten()
    if not torch.isfinite(embedding).all():
        raise ValueError(f"non-finite speaker embedding: {path}")
    if embedding.numel() == 0:
        raise ValueError(f"empty speaker embedding: {path}")
    return embedding


def speaker_score(a: torch.Tensor, b: torch.Tensor) -> float:
    # Identical embedding/cosine semantics to SpeakerRecognition.verify_batch;
    # no internal torchaudio file-loading backend is required.
    value = float(torch.nn.functional.cosine_similarity(
        a, b, dim=0, eps=1e-6
    ).item())
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
        "audio_reader": "soundfile+scipy (16 kHz mono, no torchaudio backend)",
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
        embeddings = {
            "generated": speaker_embedding(classifier, args.generated),
            "reference": speaker_embedding(classifier, ref),
            "positive_control": speaker_embedding(classifier, pos),
            "negative_control": speaker_embedding(classifier, neg),
        }
        generated = embeddings["generated"]
        reference = embeddings["reference"]
        positive = embeddings["positive_control"]
        negative = embeddings["negative_control"]
        comparisons = {
            "generated_vs_reference": speaker_score(generated, reference),
            "generated_vs_other_same_speaker": speaker_score(generated, positive),
            "reference_vs_other_same_speaker": speaker_score(reference, positive),
            "reference_vs_different_speaker": speaker_score(reference, negative),
            "generated_vs_different_speaker": speaker_score(generated, negative),
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
