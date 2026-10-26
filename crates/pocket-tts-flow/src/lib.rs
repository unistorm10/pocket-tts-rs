//! `pocket-tts-flow`: the FlowLM transformer and its text conditioning.
//!
//! - [`flow_lm`] — the FlowLM model plus the LSD (latent self-distillation)
//!   decode loop that turns sampled latents into Mimi latents.
//! - [`conditioners`] — text conditioning; currently the lookup-table
//!   conditioner ([`conditioners::text::LUTConditioner`]) that tokenises text
//!   and embeds it.

pub mod conditioners;
pub mod flow_lm;

pub use conditioners::text::LUTConditioner;
pub use flow_lm::FlowLMModel;
