//! `pocket-tts-model`: high-level TTS orchestration.
//!
//! Ties the pieces together:
//!
//! - [`tts_model`] — [`tts_model::TTSModel`], the main entry point: loads weights,
//!   chunks text, drives FlowLM + Mimi and yields streaming audio frames.
//! - [`pause`] — parsing of explicit `[pause:Xms]` markers and natural pauses.
//! - [`voice`] — voice resolution (predefined voices, local files, `hf://` URLs,
//!   base64 audio) shared by the CLI and the HTTP server.
//! - [`wasm`] — browser bindings, compiled only for `wasm32`.

pub mod pause;
pub mod tts_model;

// Voice resolution (predefined names, local files, `hf://`, base64) is a
// native-only utility: it reads files and calls the file-based voice-state
// loaders, which do not exist on wasm32. Used by the CLI and the HTTP server.
#[cfg(not(target_arch = "wasm32"))]
pub mod voice;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use pause::{ParsedText, PauseMarker, parse_text_with_pauses};
pub use tts_model::TTSModel;

// The model crate is the public facade over the whole stack, so it re-exports
// the `pocket-tts-core` modules that callers of `TTSModel` need (audio helpers,
// config, quantisation, voice state, weight resolution). This means downstream
// code can depend on `pocket-tts-model` alone for ordinary use.
pub use pocket_tts_core::{
    ModelState, audio, config, quantize, quantize::QuantizedTensor, voice_state, weights,
};

// QuantizeConfig is re-exported separately because it shares a name-space with
// the module of the same name.
pub use pocket_tts_core::quantize::QuantizeConfig;
