# Porting notes: Python -> Rust file mapping

Updated at the end of every phase per the plan's anti-drift rule #5.

| Phase | Python source | Rust file |
|---|---|---|
| 0 | `pocket_tts/utils/config.py` | `crates/pocket-tts-core/src/config.rs` |
| 0 | `pocket_tts/utils/utils.py::download_if_necessary`, `make_cache_directory` | `crates/pocket-tts-core/src/download.rs` |
