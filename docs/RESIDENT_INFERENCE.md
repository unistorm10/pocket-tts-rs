# Standalone resident-model inference

This repository implements a real, **persistent Rust/XN inference backend** in `pocket-tts-model::ResidentRustServer`. It launches the existing official Kyutai v4.0.1 Rust executable using the `serve` command, binds to an ephemeral **127.0.0.1** port, and keeps the model loaded for multiple requests. The backend is independent of Phonation/SOMA and does not use Python.

**Validated**: [7/7 genuine cloned-voice WAVs](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40392318011) on the self-hosted Linux runner, with two requests to each of the three full models plus a local audio upload. [Read-only Rust CI](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40392317888) also passed.

## Run with CLI

Build Kyutai's official Rust binary from pinned `v4.0.1` code (see `REFERENCE_INFERENCE.md`). Accept the full model access terms and set `HF_TOKEN` in the **process environment**, never in a source file.

```bash
cargo run --release --locked -p pocket-tts-cli -- generate-resident \
  --reference-bin /path/to/kyutai-pocket-tts/rust/target/release/pocket-tts \
  --language english_7m_2026-10 \
  --voice hf://kyutai/tts-voices/alba-mackenna/casual.wav \
  --text "This is a real streaming voice cloning request." \
  --threads 4 --repeat 2 --output /tmp/pocket-test.wav
```

Outputs `/tmp/pocket-test-1.wav` and `/tmp/pocket-test-2.wav`, with model startup, first PCM and total request latency printed. A cached URL-based reference voice should have markedly lower latency on the second request.

To clone a local WAV, use `--voice /path/to/consented-voice.wav`. The adapter sends a **multipart** upload, which invokes Kyutai's full Mimi encoder. This path was verified with a real WAV and actual synthesized audio in the 7-WAV acceptance run.

## Use as a standalone library

```rust
use pocket_tts_model::{
    ResidentOptions, ResidentRustServer, SynthesisEngine, SynthesisRequest, Voice
};
use std::path::Path;

fn example() -> anyhow::Result<()> {
    let server = ResidentRustServer::start(
        Path::new("/path/to/native/pocket-tts"),
        ResidentOptions {
            language: "english_7m_2026-10".into(),
            ..ResidentOptions::default()
        },
    )?;
    let request = SynthesisRequest {
        text: "Hello from native Rust.".into(),
        language: "english_7m_2026-10".into(),
        voice: Voice::Preset("alba".into()),
        quantize: false,
    };
    let audio = server.synthesize_wav(&request, Path::new("out.wav"))?;
    assert_eq!(audio.sample_rate, 24_000);
    Ok(())
}
```

`ResidentRustServer::stream_wav(&request, |chunk| ...)` streams the actual WAV bytes through a caller-controlled callback without buffering the entire output. Its `StreamTiming` reports header arrival, first PCM and complete request duration.

**Important:** Kyutai's unseekable HTTP output emits a provisional WAV length before it knows how much speech will be produced. Call `pocket_tts_model::resident::finalize_streamed_wav(path)` after writing a complete streamed WAV to disk; `synthesize_wav` and `generate-resident` already perform this automatically. The function verifies and updates the 44-byte RIFF/data-size fields without changing PCM bytes.

## Voice caching and security boundaries

- Built-in voice presets load a saved speaker state; the official server also caches `voice_url` reference conditioning in a small LRU.
- A **local reference WAV** is multipart uploaded on every request and therefore **re-encoded** each time. A persisted local voice-state registry is a subsequent feature and should not be confused with cached URL voices.
- The official server serializes generation with a mutex. This interface supports multi-request use, but not parallel inference on one model instance. Shared-model concurrency and backpressure still need an in-process scheduler.
- The service binds to loopback with no public network binding; loopback is not an authentication boundary against other processes on the same computer.
- Drop kills only the child it started. We do not log the Hugging Face token, commit gated checkpoints, or retain generated WAVs in repository artifacts.
- The numerical engine here is **XN**, not Candle. The Candle neural port must reuse existing vetted modules and pass waveform/parity benchmarks against the pinned reference engine.

See `docs/RESULTS_2026-10-26.md` for the 288-trial TTFA/RTF benchmark. The model **7M, four CPU threads FP32, cached reference voice** had approximately 20 ms median *first audible-chunk* TTFA in its 5-sample warm cohort. Do not confuse that with first PCM or first uncached clone.
