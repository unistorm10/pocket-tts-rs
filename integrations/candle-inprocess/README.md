# Direct Candle integration candidate

This independent Rust workspace is the first **in-process** adapter for the
audited community Candle implementation. It implements the backend-neutral
`SynthesisEngine` and `StreamingSynthesisEngine` traits from
`pocket-tts-model` without spawning a process or using an HTTP service.

It is **not** included in the root workspace until build, real-model inference,
speaker identity and audio parity gates pass. The validated XN backend remains
untouched.

## Build and real checkpoint probe

From repository root:

```bash
bash scripts/prepare-candle-v4.sh integrations/candle-inprocess/vendor/pocket-tts
cargo test --manifest-path integrations/candle-inprocess/Cargo.toml
cargo run --release --manifest-path integrations/candle-inprocess/Cargo.toml --bin probe -- \
  probe-english /tmp/inprocess-candle.wav
```

The preparation script clones the **exact pinned audited commit** of
`stepupgaming/pocket-tts`, adds the three official full-model configuration
fixtures, and applies all nine v4 compatibility patches. The source remains
outside Git. This integration needs the gated full checkpoints and the
authorized `HF_TOKEN`. Encoder-disabled checkpoints are never selected.

**Implementation details:** One Candle `TTSModel` is resident across requests,
stable Hub conditioning is cached in-process, locally supplied recordings are
re-encoded when used, streaming yields real Mimi-decoded 16-bit PCM chunks,
and the file-producing API finalizes provisional WAV lengths. The backend
rejects unsupported language, model variants and unverified quantization.

**Limitations:** CPU and checkpoint validation first; no GPU parity or
production concurrency certification. WAV framing reports actual first PCM,
not merely a header. This is still an **experimental source-prepared**
integration rather than the final vendored standalone Candle engine.

Do not treat structural unit tests as speech quality acceptance. Real model
checks must pass independent ASR and speaker verification. Upstream Candle
source licensing is retained in the prepared source, not stripped.
