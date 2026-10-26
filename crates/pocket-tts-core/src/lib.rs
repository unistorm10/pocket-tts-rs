//! `pocket-tts-core`: shared primitives for the pocket-tts Rust port.
//!
//! This crate holds everything that the model crates share:
//!
//! - [`voice_state`] — [`voice_state::ModelState`], the streaming KV/cache state
//!   passed through every layer.
//! - [`config`] — YAML model configuration, mirroring the upstream config schema.
//! - [`audio`] — WAV read/write, resampling and PCM conversion helpers.
//! - [`weights`] — `hf://`, `http(s)://` and local-path weight resolution.
//! - [`quantize`] — int8 quantisation support.
//! - [`modules`] — reusable NN blocks (`attention`, `conv`, `mlp`, `rope`, `sdpa`).
//! - [`transformer`] — the streaming transformer stack used by both the Mimi
//!   codec and the FlowLM language model.
//!
//! `transformer` lives here rather than in a model crate because it is shared by
//! both [`pocket-tts-mimi`] and [`pocket-tts-flow`]; keeping it in `core` is what
//! lets those two crates stay independent of each other.

pub mod audio;
pub mod config;
pub mod modules;
pub mod quantize;
pub mod transformer;
pub mod voice_state;
pub mod weights;

pub use quantize::{QuantizeConfig, QuantizedTensor};
pub use voice_state::ModelState;
