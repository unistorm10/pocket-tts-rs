# Real inference baseline

The existing official Kyutai v4.0.1 Rust CLI already implements FlowLM, Mimi,
streaming output and reference-audio voice cloning using ptts/XN. Instead of
reimplementing these layers, `pocket-tts-model` offers a backend-neutral
`SynthesisEngine` trait and the first `OfficialRustCli` adapter.

This adapter is intentionally **process-based**: it requires a locally built
copy of Kyutai's official Rust binary; it does not require Python, Phonation or
SOMA. It generates a WAV, checks its signal and reports duration and metrics.
It is a functioning reference baseline *only after* the gated integration test
has produced real audio successfully. The native in-process Candle backend and
native streaming library surface are not implemented yet.

Build and run:

```bash
git clone --depth 1 --branch v4.0.1 https://github.com/kyutai-labs/pocket-tts.git /tmp/kyutai-pocket-tts
cargo build --release --manifest-path /tmp/kyutai-pocket-tts/rust/Cargo.toml
export HF_TOKEN=YOUR_READ_TOKEN
cargo run --release -p pocket-tts-cli -- generate \
  --reference-bin /tmp/kyutai-pocket-tts/rust/target/release/pocket-tts \
  --text "Testing native Rust text to speech." \
  --voice /path/to/consented-reference.wav --output output.wav
```

Do not store `HF_TOKEN` literally in a script or the repository. The CI job
uses the repository secret. The model bytes remain in the runner cache; audio
is written only to `RUNNER_TEMP` for test validation.

Voice types:
- `Preset`: a Kyutai built-in voice name; may use a precomputed voice state.
- `ReferenceAudio`: a local file that exercises the real voice encoder.
- `State`: a saved `.safetensors` voice-state file.
- `HubAudio`: an authorized `hf://` or `https://` audio reference.

The official native CLI runs on CPU. Future Candle/XN feature support is
separate; no GPU capability is asserted for this adapter.

See `docs/REUSE_FIRST_PLAN.md` for next deliverables.
