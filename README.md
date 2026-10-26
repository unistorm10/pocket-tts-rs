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
- `xtask`: planned golden-reference fixture and developer tooling

## Phase 0 tests

```bash
cargo test -p pocket-tts-core
cargo test --workspace
```

These commands validate the source when executed on a Rust-equipped machine, but passing them **does not** demonstrate speech inference.

See [standalone architecture](docs/ARCHITECTURE.md) and the [implementation plan](.github/prompts/plan-implementation-pocket-tts.md).

## Licenses

Source code: MIT (see `LICENSE`). Upstream code and downloaded model weights may have separate attribution and redistribution requirements. Verify the license of each model checkpoint before redistribution.
