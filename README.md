# pocket-tts-rs

Standalone Rust-native port of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), targeting Candle inference and a reusable library, CLI and optional HTTP server.

**Status:** **Real voice-cloned 24-kHz speech was generated and verified** through the standalone CLI using Kyutai's pinned native Rust/XN inference engine. The in-process Candle inference backend and production HTTP service are **not yet implemented**. This is an actual, working reference-backend milestone, not completed end-to-end platform parity.

The engine is completely independent of SOMA and Phonation. Phonation may integrate it as one of multiple backends via its own interface, but this repository must remain usable without Phonation.

## Reuse-first implementation strategy

Existing Rust engines have already implemented much of Pocket TTS. Before adding more new FlowLM/Mimi/Candle code, compare Kyutai's official v4.0.1 Rust/XN runtime and the substantial community Candle port. The [Rust implementation audit](docs/RUST_IMPLEMENTATION_AUDIT.md) records the candidate repositories, licenses, version gaps and required real-weight acceptance tests. It supersedes any older port-every-module-from-Python assumptions in the implementation plan.

## Performance benchmarks

The combined [288-trial report](docs/RESULTS_2026-10-26.md) includes a **144-trial warm resident-model HTTP benchmark** ([data](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40380071569/artifacts/12677703174)) and the full [144-trial fresh-process benchmark](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40377493188) passed against all three gated English models. The [raw metrics artifact](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40377493188/artifacts/12677901806) includes streaming first-audio timing, end-to-end/model-only RTF, peak RSS, FP32/INT8 and voice-preparation costs. At 4 CPU threads FP32 with preset voice, median first audio was **376 ms (English), 139 ms (18M REFL), 104 ms (7M)**. Fresh reference audio required **1.6–1.9 seconds** before first audible PCM. A second resident-model HTTP benchmark is implemented separately to isolate warm model/voice-cache behavior. See [benchmark methods](docs/BENCHMARKING.md).

## Verified native Rust voice-cloning baseline

On October 26, 2026, the self-hosted Linux runner compiled Kyutai's existing
`v4.0.1` native Rust engine, used a published Kyutai reference-audio WAV
with the full (encoder-enabled) `kyutai/pocket-tts` English checkpoint,
and generated **3.16 s of real 24-kHz mono speech (75,840 samples)**.
Our Rust CLI validated signal amplitude (peak 0.670105, RMS 0.118723);
an independent WAV parser confirmed the result. Kyutai reported generation
at **3.66× real-time**, excluding some setup overhead.
[CI evidence](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40368083804).
The model files and generated audio are not committed to Git.

This is a **process-backed reference implementation**, not yet a native
in-process reusable Candle inference backend and not a final streaming library.
The expensive gated reference-synthesis CI is manually triggered; ordinary CI
remains automatic.

See [reference inference usage](docs/REFERENCE_INFERENCE.md) and
[reuse-first implementation stages](docs/REUSE_FIRST_PLAN.md).

## Workspace

- `pocket-tts-core`: configuration and download/cache utilities; current upstream config fixtures
- `pocket-tts-flow`: planned FlowLM implementation
- `pocket-tts-mimi`: planned Mimi codec
- `pocket-tts-model`: backend-neutral Rust synthesis API, real WAV inspection and operational official Rust CLI adapter; Candle adapter still to be implemented
- `pocket-tts-cli`: operational `generate`/ `inspect` through a pinned existing native Rust reference executable; requires that binary
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
