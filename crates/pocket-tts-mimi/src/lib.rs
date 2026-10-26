//! `pocket-tts-mimi`: the Mimi neural audio codec.
//!
//! Two halves:
//!
//! - [`seanet`] — the SEANet convolutional encoder/decoder (with streaming
//!   convolution primitives).
//! - [`mimi`] — the codec proper: SEANet + quantizer + projected transformer,
//!   exposing `encode_to_latent` / `decode_from_latent`.

pub mod mimi;
pub mod seanet;

pub use mimi::MimiModel;
pub use seanet::{SEANetDecoder, SEANetEncoder};
