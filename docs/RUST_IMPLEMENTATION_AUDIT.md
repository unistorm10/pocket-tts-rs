# Rust Pocket TTS implementation reuse audit

**Date:** 2026-10-26
**Scope:** Kyutai Pocket TTS v4.0.1, full official voice-cloning checkpoints, standalone Rust integration with a preferred Candle inference backend and optional Rust-native alternatives.

## Decision: reuse first; no new full neural port until baseline comparison

We should **not implement a complete transformer, Mimi neural codec, streaming attention, voice-conditioning pipeline or text-processing stack from scratch** until existing implementations have been evaluated with the exact three gated Kyutai checkpoints already downloaded on the Linux runner.

The current Candle core has only weight-loading/streaming-state/RoPE primitives. These are **not full inference**. Treat them as experiments/tests; do not multiply competing production implementations without measured reasons.

## Investigated codebases

| Codebase | Verified code/behavior | Backend | Source license | Verdict |
| --- | --- | --- | --- | --- |
| [Kyutai official `rust/`](https://github.com/kyutai-labs/pocket-tts/tree/v4.0.1/rust) | Complete first-party Rust CLI integration: v4.0.1 config parsing (including compact GQA/asymmetric decoder), tokenizer, native text processing, real streaming generation, voice-state import/export, reference-audio encoding/cloning. It uses `ptts` from `gabrieldemarmiesse/xn-ptts` at commit `7efa26973f81bba42cf2ace8da2c1eb9b605d54d`. `rust/Cargo.toml` has `publish = false` and a binary target, rather than a published reusable library. | XN/ptts CPU in this wrapper | MIT | **Primary correctness oracle, integration blueprint, first upstream binary to benchmark.** Can reuse module logic with attribution; do not force binary code into the library API unchanged. |
| [Kyutai-pinned ptts fork](https://github.com/gabrieldemarmiesse/xn-ptts/tree/pocket-tts-small-models/ptts) | Reusable `ptts` Rust neural runtime extended for Pocket TTS small models; explicit first-party dependency and pinned commit above. | XN (library) | MIT OR Apache-2.0 in package | **Prefer this over writing a new XN engine.** Use as baseline and possible optional `xn` feature/backend. Verify API and checkpoint mapping first. |
| [Gradium `xn-ptts`](https://github.com/gradium-ai/xn-ptts) | `ptts` library with streaming/session APIs, mobile/server/FFI and CPU, Metal, CUDA, Vulkan/WebGPU feature surfaces; Gradium Phonon model packages and format are a separate product. | XN | MIT OR Apache-2.0 | Strong reusable primitives and optimization work. **Do not assume** mainline Phonon checkpoint format or APIs load Kyutai's October 2026 safetensors without the Kyutai fork/adapter. |
| [Community Candle port, now `stepupgaming/pocket-tts`](https://github.com/stepupgaming/pocket-tts) (previously `babybirdprd/pocket-tts`) | A substantial Rust/Candle implementation, with `TTSModel`, FlowLM, Mimi, SEANet, transformer/streaming state, voice cloning via audio or exported state, CLI/server, WASM and PyO3. Current examined workspace version 0.6.2 uses Candle 0.9.1. | Candle | Cargo metadata MIT OR Apache-2.0, root LICENSE MIT | **Primary source for the desired Candle inference path. Reuse/port its modules and tests rather than rewriting layers.** **Not drop-in for October v4.0.1 compact variants:** inspected `config.rs` lacks `attn_dim`, `num_kv_heads`, `rank`, `z_dim`, `decoder_seanet` and `decoder_transformer`. Confirm quality and full checkpoint parity on runner before vendoring/adapting. |
| [Archived Candle attempt](https://github.com/jamesfebin/pocket-tts-candle) | Archived January 2026; README explicitly recommends successor community Candle port. | Candle | Check source notices before copying | **Do not choose as new foundation.** |
| [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) | Rust binding/runtime for ONNX exports, including Pocket TTS support on embedded targets documented by Kyutai. | ONNX Runtime/native FFI | Check runtime/build licenses separately | Alternative interoperability path if needed, **not** a Candle or native Kyutai safetensors substitute. |

## Why this changes the work

1. **Reference first:** Compile Kyutai's `rust/` target from v4.0.1 with its pinned `ptts` fork, on our self-hosted Linux runner. Reuse the **full** gated English model, and generate an actual WAV from a permitted predefined voice and a consented reference WAV. Verify non-silent PCM, sample rate, duration and audible/parity benchmarks. Repeat compact 7M and 18M REFL when supported. Save checksums and metrics, never gated weight bytes in Git/artifacts.
2. **Candle second, by module:** Try the Candle `TTSModel` implementation against the full English checkpoint. Establish exact missing-v4 config/weight-layout delta. Reuse its existing SEANet, Mimi, attention/cache, FlowLM and streaming implementations, under applicable source license and notices. Keep only adaptations/patches needed for Kyutai v4.0.1 and our public API. Do not rewrite primitives already validated upstream.
3. **Backend-neutral library:** `pocket-tts-model` must expose a standalone synthesis + streaming + reference-voice API. The default backend should follow the project's Candle preference **once it passes end-to-end tests**; an optional `ptts`/XN backend may provide an earlier working CPU baseline if benchmarked and acceptable. Never depend on Phonation or SOMA, and never expose Candle or XN tensors in the host-facing API.
4. **Measure:** Same pinned checkpoint, tokenizer, voice prompt, text, sampling config and comparable seed where applicable. Report model load time, first audio chunk, real-time factor, memory, WER, speaker similarity, and chunk continuity. Bench CPU first; verify CUDA/Metal/ARM features independently rather than assuming they work.
5. **Training:** Kyutai's official training remains PyTorch/CUDA. The existing `pocket-tts-training` launcher wraps the official pinned trainer. **No reliable complete Rust trainer is identified among these ports.** Do not reinvent training until a real training run, checkpoint export/import and evaluation work.

## Reuse and licensing rules

- Code licensing and gated Kyutai model conditions are **separate**. Keep model bytes in local HF caches and comply with the accepted license terms.
- Source code from Kyutai is MIT; the `ptts` library is MIT OR Apache-2.0; the community Candle repo's Cargo metadata says MIT OR Apache-2.0 while its root LICENSE contains MIT terms. Retain original copyright/license notices on reused source and document provenance and changes.
- Pin upstream Git revisions and review diffs when upgrading. Direct git dependencies may be used within the standalone workspace, but never pull in Phonation or SOMA.
- Do not claim a runtime is compatible because it compiles, downloads a checkpoint or can multiply a single weight. End-to-end PCM and reference-voice cloning must execute before any production-readiness claim.

## Current verification status

Our Rust scaffolding/config parsing, `pocket-tts-training` dry-run, and actual safetensors/Candle tensor probes have passed CI. Three full English-family models with active encoder tensors are cached on the self-hosted runner (see [CHECKPOINTS.md](CHECKPOINTS.md)).

**Not yet benchmarked or smoke-tested in our CI:** official Kyutai Rust CLI synthesizing WAV, community Candle TTSModel against v4.0.1, optional XN embeddings, or any training optimizer step. These are the next gates, not assumed achievements.

Source review: direct `Cargo.toml`, `rust/src/model.rs`, `rust/src/config.rs`, `rust/src/weights.rs` at Kyutai v4.0.1; `ptts/Cargo.toml` on both XN repositories; `crates/pocket-tts/src/tts_model.rs`, `models/*`, `config.rs`, `voice_state.rs`, workspace `Cargo.toml` and LICENSE in the moved community Candle repo; official Kyutai README's alternative-implementations listing.
