//! `pocket-tts-core`: shared primitives for the pocket-tts Rust port.
//!
//! Phase 0 provides configuration parsing (mirrors
//! `pocket_tts/utils/config.py`) and resource download/caching (mirrors
//! `pocket_tts/utils/utils.py::download_if_necessary`). Later phases add
//! streaming state, RoPE, streaming convolutions, and attention.

pub mod config;
pub mod download;
