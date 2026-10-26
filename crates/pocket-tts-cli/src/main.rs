//! Standalone front-end with explicit reference and persistent Rust backends.
//! Candle inference is a separate compatibility effort; this code does not
//! pretend that the verified XN backend is implemented in Candle.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use pocket_tts_model::{
    inspect_wav, OfficialRustCli, ResidentOptions, ResidentRustServer, SynthesisEngine,
    SynthesisRequest, Voice,
};
use pocket_tts_model::resident::finalize_streamed_wav;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "pocket-tts", about = "Standalone Rust Pocket TTS")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate and validate audio with the official native Rust executable.
    Generate {
        #[arg(long, env = "POCKET_TTS_REFERENCE_BIN")]
        reference_bin: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long, default_value = "english")]
        language: String,
        #[arg(long, default_value = "alba")]
        voice: String,
        #[arg(long, default_value = "tts_output.wav")]
        output: PathBuf,
        #[arg(long)]
        quantize: bool,
    },
    /// Reuse the loaded Rust engine and voice state across multiple requests.
    GenerateResident {
        #[arg(long, env = "POCKET_TTS_REFERENCE_BIN")]
        reference_bin: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long, default_value = "english")]
        language: String,
        #[arg(long, default_value = "alba")]
        voice: String,
        #[arg(long, default_value = "tts_output.wav")]
        output: PathBuf,
        #[arg(long, default_value_t = 2)]
        repeat: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        #[arg(long)]
        quantize: bool,
    },
    /// Verify actual PCM in a locally generated 24-kHz WAV.
    Inspect { path: PathBuf },
}

fn parse_voice(voice: String) -> Voice {
    if voice.starts_with("hf://") || voice.starts_with("https://") {
        return Voice::HubAudio(voice);
    }
    let path = PathBuf::from(&voice);
    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            Voice::State(path)
        } else {
            Voice::ReferenceAudio(path)
        }
    } else if ["wav", "mp3", "flac", "safetensors"]
        .contains(&path.extension().and_then(|e| e.to_str()).unwrap_or(""))
    {
        Voice::ReferenceAudio(path)
    } else {
        Voice::Preset(voice)
    }
}

fn request(text: String, language: String, voice: String, quantize: bool) -> SynthesisRequest {
    SynthesisRequest {
        text,
        language,
        voice: parse_voice(voice),
        quantize,
    }
}

fn repeated_output(path: &Path, repetition: usize, repeats: usize) -> PathBuf {
    if repeats == 1 {
        return path.to_path_buf();
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tts_output");
    path.with_file_name(format!("{stem}-{}.wav", repetition + 1))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { path } => {
            let audio = inspect_wav(&path)?;
            println!(
                "file={} rate={} samples={} duration={:.3}s peak={:.6} rms={:.6}",
                audio.path.display(),
                audio.sample_rate,
                audio.samples,
                audio.duration_seconds(),
                audio.peak,
                audio.rms
            );
        }
        Commands::Generate {
            reference_bin,
            text,
            language,
            voice,
            output,
            quantize,
        } => {
            let engine = OfficialRustCli::new(reference_bin)?;
            let audio = engine
                .synthesize_wav(&request(text, language, voice, quantize), &output)
                .context("native reference generation failed")?;
            println!(
                "generated={} sample_rate={} samples={} duration_s={:.3} peak={:.6} rms={:.6}",
                audio.path.display(),
                audio.sample_rate,
                audio.samples,
                audio.duration_seconds(),
                audio.peak,
                audio.rms
            );
        }
        Commands::GenerateResident {
            reference_bin,
            text,
            language,
            voice,
            output,
            repeat,
            threads,
            quantize,
        } => {
            anyhow::ensure!(repeat > 0 && repeat <= 100, "repeat must be 1..=100");
            let req = request(text, language.clone(), voice, quantize);
            req.validate()?;
            let startup = Instant::now();
            let engine = ResidentRustServer::start(
                reference_bin,
                ResidentOptions {
                    language,
                    quantize,
                    threads,
                    ..ResidentOptions::default()
                },
            )?;
            println!(
                "resident_model_startup_ms={:.1}",
                startup.elapsed().as_secs_f64() * 1000.0
            );
            for rep in 0..repeat {
                let file_path = repeated_output(&output, rep, repeat);
                let mut file = File::create(&file_path)
                    .with_context(|| format!("creating {}", file_path.display()))?;
                let stats = match engine.stream_wav(&req, |bytes| {
                    file.write_all(bytes).context("writing streaming audio")
                }) {
                    Ok(stats) => stats,
                    Err(err) => {
                        drop(file);
                        let _ = std::fs::remove_file(&file_path);
                        return Err(err);
                    }
                };
                file.flush()?;
                drop(file);
                finalize_streamed_wav(&file_path)?;
                let audio = inspect_wav(&file_path)?;
                println!(
                    "resident_request={} file={} ttfa_first_pcm_ms={:.1} total_ms={:.1} rate={} samples={} peak={:.6} rms={:.6}",
                    rep+1, file_path.display(),
                    stats.first_pcm.as_secs_f64()*1000.0,
                    stats.total.as_secs_f64()*1000.0,
                    audio.sample_rate, audio.samples, audio.peak, audio.rms
                );
            }
        }
    }
    Ok(())
}
