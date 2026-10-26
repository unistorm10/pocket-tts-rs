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

## Actual runner acceptance result (October 26, 2026)

[Workflow run #40368083804](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40368083804) succeeded on the user-owned self-hosted Linux runner:
- Kyutai v4.0.1 native Rust release executable compiled from pinned source (after providing checksum-verified, user-space CMake 3.30.9).
- Reference-audio cloning was exercised with Kyutai's published Alba sample, **not** a pretrained voice-state shortcut.
- The full gated English model was used with an active encoder. Generated 75,840 mono PCM samples at 24,000 Hz, 3.16 seconds, peak 0.670105, RMS 0.118723.
- Kyutai logged 2,960 milliseconds of generated speech in 808 ms inference, a 3.66x real-time factor. The final WAV also includes boundary/padding audio, hence its 3.16 s duration.
- Independent Python `wave` validation confirmed mono, 24 kHz, duration and nonzero PCM.
- The model, reference audio and synthesized WAV were not checked into Git or uploaded as build artifacts.

The GitHub workflow is manual-only. To trigger it, open **Actions → Pocket TTS official Rust inference baseline → Run workflow** on branch `feat/standalone-v4-foundation`. GitHub repository secret `HF_TOKEN` is required for authorized full-model access.
