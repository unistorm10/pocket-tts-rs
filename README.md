# pocket-tts-rs

Standalone Rust-native port of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), targeting Candle inference and a reusable library, CLI and optional HTTP server.

**Status:** Phase 0 foundation. Current model inference is not implemented. CLI/server are placeholders and do not synthesize speech.

The engine is completely independent of SOMA and Phonation. Phonation may integrate it as one of multiple backends via its own interface, but this repository must remain usable without Phonation.

## Workspace

- `pocket-tts-core`: configuration and download/cache utilities; current upstream config fixtures
- `pocket-tts-flow`: planned FlowLM implementation
- `pocket-tts-mimi`: planned Mimi codec
- `pocket-tts-model`: planned native public synthesis library
- `pocket-tts-cli`: planned standalone executable
- `pocket-tts-server`: planned optional Axum server
- `xtask`: official-checkpoint fetching, safetensors inspection and Candle weight probes
- `pocket-tts-training`: standalone training recipe validation and explicit upstream PyTorch/CUDA launcher

## Official checkpoint policy

The primary model source is [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), with **voice cloning required**. Access to the full checkpoint is gated by Kyutai's model terms and may require an authorized Hugging Face login/token. The library must not silently fall back to the reduced encoder-disabled checkpoint.

The `kyutai/pocket-tts-without-voice-cloning` checkpoints are deliberately reduced and permitted **only** for public CI and non-cloning comparisons. `xtask prefetch english` targets the full gated checkpoint; `--without-voice-cloning` must be explicit.

## Standalone training

The `pocket-tts-training` crate provides a training launcher for the **pinned official PyTorch/CUDA trainer**, independent of Phonation. See [Training](docs/TRAINING.md). End-to-end training and native Candle backpropagation are not yet implemented in this repository.

## Phase 0 tests

```bash
cargo test -p pocket-tts-core
cargo test --workspace
```

These commands validate the source when executed on a Rust-equipped machine, but passing them **does not** demonstrate speech inference.

See [standalone architecture](docs/ARCHITECTURE.md) and the [implementation plan](.github/prompts/plan-implementation-pocket-tts.md).

## Licenses

Source code: MIT (see `LICENSE`). Upstream code and downloaded model weights may have separate attribution and redistribution requirements. Verify the license of each model checkpoint before redistribution.
