# Porting notes

This project is a **Rust-native port of [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts)**.
It is vendored from the mature Rust/Candle implementation
[`babybirdprd/pocket-tts`](https://github.com/babybirdprd/pocket-tts) and
reorganised into a six-crate workspace.

There is **no Python in this repository**. The upstream Python implementation was
used only as a behavioural reference by the original Rust port; every artefact
here is Rust, WASM or web assets.

## Lineage

| Layer | Source |
|---|---|
| Model / algorithm | [`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts) (Python/PyTorch, reference) |
| Rust implementation | [`babybirdprd/pocket-tts`](https://github.com/babybirdprd/pocket-tts) (v0.6.2, Rust/Candle) |
| This repository | The above, restructured into 6 crates |

## Crate layout

The upstream Rust port ships as two crates (`pocket-tts`, `pocket-tts-cli`).
This repository splits the same code along its natural dependency seams:

```text
pocket-tts-core     primitives + shared NN blocks
   |                voice_state, config, audio, weights, quantize,
   |                modules/{attention,conv,mlp,rope,sdpa}, transformer
   |
   +--> pocket-tts-mimi     seanet, mimi          (neural audio codec)
   |
   +--> pocket-tts-flow     flow_lm, conditioners (FlowLM + text conditioning)
   |
   +--> pocket-tts-model    tts_model, pause, voice, wasm
   |                        (re-exports the core modules it needs)
   |
   +--> pocket-tts-server   routes, handlers, state, config, embedded web UI
   |
   +--> pocket-tts-cli      generate / serve commands -> single `pocket-tts` binary
```

### Why the split lands where it does

- `transformer` (streaming transformer + `ProjectedTransformer`) is used by
  **both** the Mimi codec and the FlowLM, so it lives in `pocket-tts-core`.
  Putting it in either model crate would create a cycle.
- `voice` (voice resolution: predefined names, local files, `hf://`, base64) is
  needed by the CLI *and* the HTTP server, so it lives in `pocket-tts-model`.
- `ServeArgs` / `UiMode` / the endpoint banner live in `pocket-tts-server`, not
  the CLI, so the server can be embedded as a library without pulling in `clap`
  argument parsing from the CLI.
- `pocket-tts-model` re-exports the `pocket-tts-core` modules that callers of
  `TTSModel` need, so ordinary consumers can depend on one crate.

## Removed from upstream

| Item | Why |
|---|---|
| `python-reference/` (36 files) | Python reference tree; not needed at runtime |
| `crates/pocket-tts-bindings/` | PyO3 bindings — this project is Python-free |
| `scripts/benchmark*.{sh,ps1}` | Benchmarks that compared against Python |
| `.github/workflows/pypi-publish.yml` | PyPI publishing |
| `find_config_path()` Python fallback | Loaded config from `python-reference/` |

`find_config_path()` now resolves config from `crates/pocket-tts-core/config/`
first, then the workspace root, then the current directory.

## Build notes

- **aarch64 build fix**: `candle-core`'s `gemm-f16` dependency emits `fullfp16`
  NEON assembly that some aarch64 rustc targets do not enable by default,
  causing a build failure (`instruction requires: fullfp16`). `.cargo/config.toml`
  enables it for `aarch64-unknown-linux-gnu`:
  ```toml
  [target.aarch64-unknown-linux-gnu]
  rustflags = ["-C", "target-feature=+fp16"]
  ```
- **Web UI**: `crates/pocket-tts-server/web/dist/` holds the **prebuilt** Vite
  bundle, so no Node/npm/bun toolchain is required to build the Rust code. To
  regenerate it you need one of those toolchains:
  ```bash
  cd crates/pocket-tts-server/web
  npm install && npm run build     # or: bun install && bun run build
  ```
  Build without the UI at all via `--no-default-features`.
- **WASM**: see `scripts/build-wasm.sh`. The wasm target requires the
  `wasm32-unknown-unknown` toolchain and `wasm-bindgen-cli`; the `getrandom`
  backend flags are already set in `.cargo/config.toml`.

## Test fixtures

`assets/` holds golden reference data (`ref.wav`, `ref_output.wav` and
`*.safetensors` tensors). Tests locate it relative to the workspace root, so
run them from the workspace (or with `--workspace`).

Some tests download model weights from Hugging Face and therefore need
`HF_TOKEN`; those are `#[ignore]`d by default.

## Licences

Dual-licensed `MIT OR Apache-2.0`, matching the upstream Rust port. See
`LICENSE`. The `.safetensors` fixtures under `assets/` are reference tensors
generated from the Kyutai model for numerical-parity testing.
