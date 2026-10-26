//! Standalone public Pocket TTS model surface.
//!
//! The first executable backend delegates to Kyutai's already-implemented Rust
//! inference CLI, compiled from a pinned upstream commit. This keeps a real
//! synthesis path available while an in-process Candle backend is adapted.
//! There is no dependency on Phonation, SOMA, Python or a network service.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Source of voice conditioning. Arbitrary local reference audio is forwarded
/// to the official encoder, not reduced to a preset embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Voice {
    Preset(String),
    ReferenceAudio(PathBuf),
    State(PathBuf),
    HubAudio(String),
}

impl Voice {
    pub fn as_argument(&self) -> Result<String> {
        match self {
            Self::Preset(name) => {
                anyhow::ensure!(!name.trim().is_empty(), "preset voice must not be empty");
                Ok(name.clone())
            }
            Self::ReferenceAudio(path) | Self::State(path) => {
                anyhow::ensure!(path.is_file(), "voice file is missing: {}", path.display());
                Ok(path.display().to_string())
            }
            Self::HubAudio(source) => {
                anyhow::ensure!(
                    source.starts_with("hf://") || source.starts_with("https://"),
                    "remote reference voice must be an hf:// or https:// URL"
                );
                Ok(source.clone())
            }
        }
    }
}

/// Arguments which do not depend on the underlying numerical engine.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub language: String,
    pub voice: Voice,
    pub quantize: bool,
}

impl SynthesisRequest {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.text.trim().is_empty(), "synthesis text is empty");
        anyhow::ensure!(
            !self.language.trim().is_empty(),
            "language must not be empty"
        );
        self.voice.as_argument()?;
        Ok(())
    }
}

/// Real WAV output measurements, read from the generated file.
#[derive(Debug, Clone)]
pub struct GeneratedAudio {
    pub path: PathBuf,
    pub sample_rate: u32,
    pub samples: u64,
    pub peak: f64,
    pub rms: f64,
}

impl GeneratedAudio {
    pub fn duration_seconds(&self) -> f64 {
        self.samples as f64 / f64::from(self.sample_rate)
    }
}

/// Backend-neutral interface. A native Candle backend must implement this
/// independently and should add native streaming through a companion trait.
pub trait SynthesisEngine {
    fn synthesize_wav(&self, request: &SynthesisRequest, output: &Path) -> Result<GeneratedAudio>;
}

/// Temporary native-Rust reference execution backend: Kyutai's v4.0.1 Rust
/// binary (XN/ptts). Unlike a Python/ONNX bridge, this is the upstream Rust
/// engine, but it launches a process per request. This is NOT the final
/// zero-reload embedded API or a streaming transport.
#[derive(Debug, Clone)]
pub struct OfficialRustCli {
    binary: PathBuf,
}

impl OfficialRustCli {
    pub fn new(binary: impl AsRef<Path>) -> Result<Self> {
        let binary = binary.as_ref();
        anyhow::ensure!(
            binary.is_file(),
            "Kyutai Rust binary not found: {}",
            binary.display()
        );
        Ok(Self {
            binary: binary
                .canonicalize()
                .context("resolving official Rust binary")?,
        })
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }
}

impl SynthesisEngine for OfficialRustCli {
    fn synthesize_wav(&self, request: &SynthesisRequest, output: &Path) -> Result<GeneratedAudio> {
        request.validate()?;
        anyhow::ensure!(
            output.extension().and_then(|e| e.to_str()) == Some("wav"),
            "output file must end in .wav"
        );
        let voice = request.voice.as_argument()?;
        if let Some(parent) = output.parent() {
            anyhow::ensure!(
                parent.is_dir(),
                "output directory missing: {}",
                parent.display()
            );
        }
        // Do not accidentally validate an old WAV if the upstream CLI fails.
        if output.exists() {
            std::fs::remove_file(output)
                .with_context(|| format!("removing stale output {}", output.display()))?;
        }
        // All parameters are passed as exact argv arguments, never via a shell.
        let mut command = Command::new(&self.binary);
        command.args([
            "generate",
            "--language",
            &request.language,
            "--text",
            &request.text,
            "--voice",
            &voice,
            "--output-path",
        ]);
        command.arg(output);
        if request.quantize {
            command.arg("--quantize");
        }
        let status = command
            .status()
            .context("executing official Rust Pocket TTS binary")?;
        if !status.success() {
            bail!("official Rust TTS generation failed ({status}); no audio result");
        }
        inspect_wav(output)
    }
}

/// Inspect actual PCM, rejecting empty/silent/wrong-format outputs.
/// The official model generates 24-kHz, mono speech; do not silently resample.
pub fn inspect_wav(path: &Path) -> Result<GeneratedAudio> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("opening generated WAV {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 24_000 && spec.channels == 1,
        "unexpected model audio format: {} Hz / {} channels",
        spec.sample_rate,
        spec.channels
    );
    anyhow::ensure!(spec.bits_per_sample > 0, "WAV bit depth is missing");
    let mut n = 0_u64;
    let mut peak = 0.0_f64;
    let mut energy = 0.0_f64;
    let mut accumulate = |value: f64| -> Result<()> {
        anyhow::ensure!(value.is_finite(), "non-finite generated audio");
        peak = peak.max(value.abs());
        energy += value * value;
        n += 1;
        Ok(())
    };
    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut reader = hound::WavReader::open(path)?;
            for sample in reader.samples::<f32>() {
                accumulate(f64::from(sample?))?;
            }
        }
        hound::SampleFormat::Int => {
            anyhow::ensure!(
                spec.bits_per_sample <= 32,
                "unsupported PCM bit depth {}",
                spec.bits_per_sample
            );
            let divisor = (1_u64 << (spec.bits_per_sample - 1)) as f64;
            let mut reader = hound::WavReader::open(path)?;
            for sample in reader.samples::<i32>() {
                accumulate(f64::from(sample?) / divisor)?;
            }
        }
    }
    anyhow::ensure!(
        n >= u64::from(spec.sample_rate / 10),
        "generated audio is too short"
    );
    let rms = (energy / n as f64).sqrt();
    anyhow::ensure!(peak > 1e-5 && rms > 1e-6, "generated audio is silent");
    Ok(GeneratedAudio {
        path: path.to_path_buf(),
        sample_rate: spec.sample_rate,
        samples: n,
        peak,
        rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_requests_fail_before_invoking_backend() {
        let r = SynthesisRequest {
            text: " ".into(),
            language: "english".into(),
            voice: Voice::Preset("alba".into()),
            quantize: false,
        };
        assert!(r.validate().is_err());
        assert!(Voice::ReferenceAudio(PathBuf::from("/does/not/exist.wav"))
            .as_argument()
            .is_err());
        assert!(Voice::HubAudio("ftp://untrusted".into())
            .as_argument()
            .is_err());
    }

    #[test]
    fn real_wav_inspection_detects_signal_and_silence() {
        let output = std::env::temp_dir().join(format!(
            "pocket-tts-rs-wav-{}-{}.wav",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::create(&output, spec).unwrap();
            for i in 0..4800 {
                let v: i16 = if i % 2 == 0 { 500 } else { -500 };
                writer.write_sample(v).unwrap();
            }
            writer.finalize().unwrap();
        }
        let result = inspect_wav(&output).unwrap();
        assert_eq!(result.sample_rate, 24_000);
        assert_eq!(result.samples, 4800);
        assert!((result.duration_seconds() - 0.2).abs() < 1e-5);
        assert!(result.rms > 0.0);
        std::fs::remove_file(output).unwrap();
    }
}
