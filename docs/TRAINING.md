# Pocket TTS training, standalone

## Supported today

The standalone Rust package `pocket-tts-training` is an explicit launcher for **Kyutai's actual training scripts at git tag v4.0.1**. It handles checkout/recipe validation, paired-data manifest preflight and process execution. No Phonation or SOMA dependency is required.

**Neural training presently runs in upstream PyTorch/CUDA, not in Candle**. There is not yet a native Rust autograd/trainer implementation, nor a validated fine-tuned model from this repository. Candle remains the inference target.

### Prerequisites

- Linux and an NVIDIA CUDA GPU (officially supported upstream).
- `uv`, Python 3.10+, and the official training dependencies installed with `uv sync`.
- A local checkout of `https://github.com/kyutai-labs/pocket-tts` at **tag v4.0.1**. Never execute an unpinned upstream branch in an automated training deployment.
- Prepared, rights-cleared audio/transcript pairs **with word-level alignments**. For voice identity training or cloning, collect consent and respect licenses.
- A training config pointing to both training and validation JSONL manifests. `finetune_language` requires a tokenizer trained for the target language.

```bash
git clone --branch v4.0.1 --depth 1 https://github.com/kyutai-labs/pocket-tts.git /opt/kyutai-pocket-tts
cd /opt/kyutai-pocket-tts
uv sync

# From standalone pocket-tts-rs:
cargo run -p pocket-tts-training -- --source /opt/kyutai-pocket-tts --recipe finetune

# Starts actual training only when --run is explicit and manifests exist:
cargo run -p pocket-tts-training -- --source /opt/kyutai-pocket-tts --recipe finetune --run
```

Training recipes: `scratch` (teacher), `finetune` (new voices/domain in same language), `finetune-language` (new language/new tokenizer), `depth-distill` (teacher to 6-layer student), and `drifting`. The launcher does **not** synthesize manifests, align audio, silently install PyTorch, or download a dataset.

On ~16 GB VRAM, Kyutai documents `batch_size: 16`, `grad_accum_steps: 4` as an achievable configuration for its training recipe. This does not mean the default `batch_size: 64` will fit. Edit a local copy of the upstream recipe and verify your CUDA driver, actual memory consumption and data pipeline before running.

Kyutai exports `model.safetensors` for inference and `checkpoint_*.pt` for resuming training. Do not mistake a PyTorch optimizer/training checkpoint for a Candle inference file. Future work includes a native Rust data/augmentation API, metadata/experiment tracking, LoRA/adapters when verified against the architecture, dataset/consent provenance, checkpoint conversion and inference parity tests.

Reference: https://github.com/kyutai-labs/pocket-tts/blob/v4.0.1/training/README.md
