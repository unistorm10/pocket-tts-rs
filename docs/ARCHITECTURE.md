# Architecture: standalone Pocket TTS engine

## Dependency rule

**pocket-tts-rs never depends on Phonation or any SOMA project.**
The independently versioned Rust library exports Pocket TTS domain types and synthesis operations. The optional CLI and HTTP server depend on that library. Phonation depends on the published Pocket TTS library and provides any adapter or implementation for its own `TtsBackend` contract.

```text
                   Phonation (host/orchestrator)
                       |
             phonation::TtsBackend wrapper
                       |
                pocket-tts-model   <---- CLI / optional Axum service
                       |
            +----------+----------+
            |                     |
       pocket-tts-flow       pocket-tts-mimi
            |                     |
            +----------+----------+
                       |
                 pocket-tts-core
                       |
                Candle + safetensors
```

This allows direct embedded use without HTTP, and standalone CLI/server use without Phonation.

## Ownership of responsibilities

- **pocket-tts-core**: strict model configurations, verified weight resolution, tensor modules, streaming primitives.
- **pocket-tts-flow**: tokenizer/conditioning, transformer, flow sampler and EOS semantics.
- **pocket-tts-mimi**: codec, encoder/decoder, stream-local state.
- **pocket-tts-model**: ergonomic public library API, loading lifecycle, voice-state import/export, streaming session API and clear errors.
- **pocket-tts-cli**: standalone executable backed by the same library.
- **pocket-tts-server**: optional HTTP interface backed by the same library.
- **pocket-tts-training**: separate local training workflow, using the pinned upstream PyTorch/CUDA trainer today; no runtime dependency on this crate.
- **Phonation**: backend discovery/selection, voice registry integration, request/response mapping, scheduling and orchestration.

## Public API design target (NOT YET IMPLEMENTED)

A caller should be able to load a verified model, select a preset/reference voice and synthesize PCM without importing Candle tensors. Streamed synthesis should emit ordered sample chunks with a sample rate and explicit termination/error signaling. Inference sessions must isolate state so that concurrent callers do not accidentally share KV caches or decoder history.

Capability discovery should indicate language(s), voice cloning, live streaming, supported devices and models, rather than claiming universal support. Native audio output should include its true sample rate, not a forced Phonation rate.

## Invariants / acceptance gates

1. Compile independently on supported targets and run all native unit tests.
2. Parse current upstream configurations including compact decoder differences and use the correct tokenizer per model.
3. Load real, revision-pinned weights and verify integrity; operate without network once cached.
4. Demonstrate correct, non-silent, real-weight audio with golden PyTorch comparison, plus explicit WER/audio benchmarks where meaningful.
5. Streaming and one-shot generation agree on content/timing within documented tolerances.
6. For full voice-cloning checkpoints require active encoder tensors; do not silently substitute the encoder-disabled public variant. Voice cloning must pass real reference-audio tests.
7. Keep inference independent of optional training backends. Check exported `model.safetensors` before deploying a trained model.
8. Keep model licenses and weight redistribution restrictions distinct from this repository's MIT source license.
9. Confirm a shared compatible Candle version against Phonation when embedded integration is implemented. Do not depend on Phonation to compile/test this repository.

## Current status (2026-10-26)

The workspace is Phase 0 scaffolding. The v4.0.1 configuration schema and regression fixtures are being added first. Neural inference, streaming, CLI commands and server endpoints are not implemented; do not describe the project as functional TTS until real synthesis tests pass.

See `.github/prompts/plan-implementation-pocket-tts.md` for implementation phases.
