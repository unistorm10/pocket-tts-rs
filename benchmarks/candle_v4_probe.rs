//! Real checkpoint compatibility probe for pinned stepupgaming/pocket-tts.
//! Executed as an example inside the upstream Candle crate, without rewriting
//! or mocking its FlowLM, Mimi, or reference-voice encoder.
use anyhow::{Context, Result};
use pocket_tts::TTSModel;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let variant = args.next().context("missing model variant")?;
    let output = PathBuf::from(args.next().context("missing output WAV path")?);
    let start = std::time::Instant::now();
    eprintln!("CANDLE_ATTEMPT {variant}");
    // The community port's older defaults (temperature=.7, EOS=-4)
    // are not the released v4 compact-model generation defaults.
    let (temperature, eos_threshold) = if variant == "probe-7m" || variant == "probe-18m" {
        (0.3_f32, -1.0_f32)
    } else {
        (0.3_f32, -4.0_f32)
    };
    eprintln!("CANDLE_GENERATION_SETTINGS variant={variant} temp={temperature} eos={eos_threshold}");
    let model = TTSModel::load_with_params(&variant, temperature, 1, eos_threshold)
        .with_context(|| format!("CANDLE_WEIGHT_LOAD_FAILURE {variant}"))?;
    eprintln!(
        "CANDLE_WEIGHTS_LOADED {variant} elapsed_ms={:.1}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let path = pocket_tts::weights::download_if_necessary(
        "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    )?;
    let voice = model.get_voice_state(&path).context("CANDLE_VOICE_ENCODER_FAILURE")?;
    let audio = model
        .generate("Testing real Candle synthesis with a cloned voice.", &voice)
        .context("CANDLE_SYNTHESIS_FAILURE")?;
    pocket_tts::audio::write_wav(&output, &audio, model.sample_rate as u32)?;
    eprintln!("CANDLE_REAL_AUDIO {} elapsed_ms={:.1}", output.display(), start.elapsed().as_secs_f64()*1000.0);
    Ok(())
}
