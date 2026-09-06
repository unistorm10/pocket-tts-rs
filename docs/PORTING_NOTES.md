# Porting notes: Python -> Rust file mapping

Updated at the end of every phase per the plan's anti-drift rule #5.

| Phase | Python source | Rust file |
|---|---|---|
| 0 | `pocket_tts/utils/config.py` | `crates/pocket-tts-core/src/config.rs` |
| 0 | `pocket_tts/utils/utils.py::download_if_necessary`, `make_cache_directory` | `crates/pocket-tts-core/src/download.rs` |

## Build notes

- **aarch64 (e.g. Apple Silicon under emulation, some ARM servers) build fix**:
  `candle-core`'s `gemm-f16` dependency emits `fullfp16` NEON assembly that
  isn't enabled by the default target on some aarch64 rustc targets, causing
  a build failure ("instruction requires: fullfp16"). Workaround, needed on
  the box this was verified on:
  ```
  RUSTFLAGS="-C target-feature=+fp16" cargo build ...
  RUSTFLAGS="-C target-feature=+fp16" cargo test ...
  ```
  Phase 0 was built and tested with this flag; `cargo test -p pocket-tts-core`
  passes (`config_parses_english_yaml`, `config_rejects_unknown_fields`).
  A follow-up phase should consider adding a `.cargo/config.toml` with this
  rustflag scoped to `[target.aarch64-unknown-linux-gnu]` so agents don't
  have to rediscover it.

