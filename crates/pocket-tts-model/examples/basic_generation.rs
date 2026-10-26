//! Minimal end-to-end example: load the model, clone a voice from a WAV file
//! and synthesise speech.
//!
//! Run with:
//! ```text
//! cargo run --release -p pocket-tts-model --example basic_generation
//! ```
//!
//! Weights are downloaded from Hugging Face on first run. Set `HF_TOKEN` if the
//! variant you use requires authentication.
//!
//! Without a `--voice` argument the example uses the predefined `alba` voice,
//! so it works without a local WAV file.

use anyhow::{Context, Result};
use pocket_tts_model::TTSModel;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let text = args
        .next()
        .unwrap_or_else(|| "Hello world! I am pocket-tts.".to_string());
    // An optional path to a WAV file used for voice cloning.
    let voice = args.next();

    println!("Loading model...");
    let model = TTSModel::load("b6369a24").context("failed to load model")?;

    let voice_state = match voice.as_deref() {
        Some(path) => {
            println!("Cloning voice from {path}...");
            model
                .get_voice_state(path)
                .with_context(|| format!("failed to derive voice state from {path}"))?
        }
        None => {
            println!("Using predefined voice 'alba'...");
            pocket_tts_model::voice::resolve_voice(&model, Some("alba"))?
        }
    };

    println!("Generating audio...");
    let audio = model.generate(&text, &voice_state)?;

    let out = "example_output.wav";
    pocket_tts_model::audio::write_wav(out, &audio, model.sample_rate as u32)?;

    println!("Wrote {out} ({} Hz)", model.sample_rate);
    Ok(())
}
