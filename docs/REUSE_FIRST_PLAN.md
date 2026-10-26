# Reuse-first execution plan: Pocket TTS v4.0.1

This plan supersedes the prior port-every-layer-from-Python phases in
`.github/prompts/plan-implementation-pocket-tts.md`.

## Required architecture
- Pocket TTS remains a **standalone** Rust library, CLI and optional server.
- No dependency on SOMA or Phonation. The latter integrates via its own adapter.
- Voice cloning and real training are requirements.
- **Candle preferred** for embedded inference; official Kyutai XN Rust is the reference engine and an optional runtime choice.
- No fake audio, mocked model execution, or claims from checkpoint probes.
- Never commit weights, tokens or reference audio. Use pinned full model checkpoints already cached.

## Milestone A: real voice-cloned audio baseline (validated 2026-10-26)
1. Checkout Kyutai v4.0.1 at pinned commit `a922b912dfa366975294afb568d815b305acde5d`.
2. Build official `rust/` Rust inference binary on our self-hosted Linux runner.
3. Exercise a **reference-audio** voice-cloning path with Kyutai's published example WAV.
4. Validate actual WAV with sample rate 24 kHz, non-silence, positive duration, plus RMS/peak.
5. Expose a minimal standalone `pocket-tts-model::SynthesisEngine` interface and `pocket-tts-cli generate` through `OfficialRustCli` (explicit reference backend, process based).
6. Verify Cargo tests, Clippy, lockfile, CI; keep the proof link.
**Gate passed:** [Run #40368083804](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40368083804) produced real 24 kHz cloned speech with an external reference-audio sample and verified non-silent PCM. This process-backed bridge is *not* the final streaming/embedded engine.

## Milestone A2: persistent Rust/XN bridge (validated 2026-10-26)
- `pocket_tts_model::ResidentRustServer` now starts the official pinned Rust/XN HTTP server on a localhost-only ephemeral port, keeps the model resident, and kills/reaps its child on drop.
- `stream_wav` supplies bounded 64 KiB chunks via a Rust callback; the host receives first-PCM and end-to-end latency. `synthesize_wav` writes to a WAV file.
- `finalize_streamed_wav` repairs Kyutai's intentionally oversized HTTP WAV header to the actual PCM length; regression test catches the previous EOF failure.
- `pocket-tts generate-resident --repeat 2` exercises persistent model and URL voice-state caching, including real reference-audio voice cloning.
- [7/7 real WAV acceptance](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40392318011) covers all three full voice-cloning models (two requests each) and a local WAV multipart upload, verified independently at 24 kHz with nonzero PCM.
- Limitation: **local file uploads are re-encoded** by the official server every request; only preset and voice URLs are reused in its existing LRU cache. Persistent local voice states and an in-process Candle backend remain to implement.
- Both expensive gated inference workflows are now manual-only and normal CI read-only.

## Milestone B: reuse the existing Candle implementation
1. Pin a reviewed `stepupgaming/pocket-tts` commit and save license notices.
2. Adapt its neural modules (`SEANet`, `Mimi`, `FlowLM`, attention/KV, text) into our backend under a `candle` feature. Prefer reusing upstream code to retyping it.
3. Handle Kyutai October 2026 config deltas: `attn_dim`, `num_kv_heads`, `rank`, `z_dim`, `decoder_seanet`/`decoder_transformer`, EOS and flow variants.
4. No fallback to zeroed `mimi.encoder`; encoder activation is checked and full model hashes pinned.
5. Compare fixed text/voice/seed against official Rust reference. Measure output audio, real-time factor, first chunk, RSS and voice embedding parity.
6. Implement native `SynthesisEngine` and a proper bounded streaming interface. Retain optional XN backend if performance warrants; do not rebuild a separate XN neural stack.

## Milestone C: library/CLI/service parity
- Public audio/reference-voice/import-export types without Candle/XN tensor leakage.
- Reuse already-loaded models and voices; avoid process-per-request for production.
- Context/cancellation, backpressure, async streaming and multiple concurrent sessions.
- Optional Axum local HTTP endpoint with capability discovery, upload handling and health.
- CPU first, then CUDA/Metal/ARM64 via separate hardware runner matrix.

## Milestone D: training
- Existing `pocket-tts-training` runs **official pinned PyTorch/CUDA** recipes: scratch, fine-tune, language and distillation.
- Prepare rights-cleared aligned audio/transcript datasets; verify data and GPU.
- Run actual training smoke step on a CUDA runner with small real dataset; checkpoints export to inference-compatible `model.safetensors`, validated with both Rust backends.
- LoRA/native Rust training only after measuring upstream trainer and real fine-tune parity. Do not pretend the existing launcher is Candle autograd.

## Cross-cutting acceptance
- Pin dependency revisions, weights and toolchains.
- Attach source-attribution/license notices when vendoring.
- CI: rustfmt, Clippy, unit tests; gated/manual real-audio tests.
- For any claimed production feature require a *real* output/benchmark and explicit artifact proof.
- Do not merge draft PR until required capabilities are demonstrated.
