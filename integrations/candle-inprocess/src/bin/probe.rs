//! Direct in-process Candle probe with the same voice and text as reference QA.
//! This loads full gated checkpoints; no mock, PCM fixture or reduced model.
use anyhow::{Context, Result};
use pocket_tts_candle_inprocess::CandleInProcess;
use pocket_tts_model::{SynthesisEngine, SynthesisRequest, Voice};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let variant = args.next().context("variant: probe-english, probe-7m, probe-18m")?;
    let output = PathBuf::from(args.next().context("missing WAV output path")?);
    let text = args
        .next()
        .unwrap_or_else(|| "Testing real Candle synthesis with a cloned voice.".to_owned());
    let started = std::time::Instant::now();
    let engine = CandleInProcess::load(&variant)?;
    let model_load = started.elapsed();
    let request = SynthesisRequest {
        text,
        language: "english".to_owned(),
        voice: Voice::HubAudio("hf://kyutai/tts-voices/alba-mackenna/casual.wav".to_owned()),
        quantize: false,
    };
    let audio = engine.synthesize_wav(&request, &output)?;
    eprintln!(
        "INPROCESS_CANDLE_OK variant={} model_load_ms={:.1} total_ms={:.1} audio_s={:.3} rms={:.5}",
        variant,
        model_load.as_secs_f64() * 1000.0,
        started.elapsed().as_secs_f64() * 1000.0,
        audio.duration_seconds(),
        audio.rms
    );
    Ok(())
}
