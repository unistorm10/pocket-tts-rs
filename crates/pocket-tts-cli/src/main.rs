//! Standalone front-end with explicit backend selection.
//! Only the verified official-Rust executable backend is active at this
//! milestone. Candle inference is not mislabeled as implemented.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use pocket_tts_model::{OfficialRustCli, SynthesisEngine, SynthesisRequest, Voice};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "pocket-tts", about = "Standalone Rust Pocket TTS")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate and validate audio with Kyutai's native Rust reference engine.
    Generate {
        /// Path to the official Kyutai v4.0.1 Rust pocket-tts executable.
        #[arg(long, env = "POCKET_TTS_REFERENCE_BIN")]
        reference_bin: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long, default_value = "english")]
        language: String,
        /// Preset name, local audio/.safetensors file or hf:// audio source.
        #[arg(long, default_value = "alba")]
        voice: String,
        #[arg(long, default_value = "tts_output.wav")]
        output: PathBuf,
        /// Inference quantization in the official XN engine.
        #[arg(long)]
        quantize: bool,
    },
    /// Inspect an existing generated mono 24-kHz WAV.
    Inspect { path: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Inspect { path } => pocket_tts_model::inspect_wav(&path)?,
        Commands::Generate {
            reference_bin,
            text,
            language,
            voice,
            output,
            quantize,
        } => {
            let source = if voice.starts_with("hf://") || voice.starts_with("https://") {
                Voice::HubAudio(voice)
            } else {
                let path = PathBuf::from(&voice);
                if path.is_file() {
                    if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                        Voice::State(path)
                    } else {
                        Voice::ReferenceAudio(path)
                    }
                } else {
                    Voice::Preset(voice)
                }
            };
            let engine = OfficialRustCli::new(reference_bin)?;
            let request = SynthesisRequest {
                text,
                language,
                voice: source,
                quantize,
            };
            engine
                .synthesize_wav(&request, &output)
                .context("real Pocket TTS reference generation failed")?
        }
    };
    println!(
        "generated={} sample_rate={} samples={} duration_s={:.3} peak={:.6} rms={:.6}",
        result.path.display(),
        result.sample_rate,
        result.samples,
        result.duration_seconds(),
        result.peak,
        result.rms
    );
    Ok(())
}
