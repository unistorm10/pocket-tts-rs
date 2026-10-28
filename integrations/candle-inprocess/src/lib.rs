//! Direct, persistent Rust/Candle Pocket TTS inference.
//!
//! The adapted community Candle source is prepared at a pinned upstream SHA
//! under vendor/pocket-tts by scripts/prepare-candle-v4.sh. It is not a Python,
//! ONNX, XN, process or HTTP inference bridge. This crate is deliberately an
//! isolated integration workspace until real model and parity CI passes.

use anyhow::{Context, Result, bail, ensure};
use pocket_tts::{ModelState, TTSModel, audio, weights};
use pocket_tts_model::{
    GeneratedAudio, StreamingSynthesisEngine, SynthesisEngine, SynthesisRequest, Voice,
    resident::{StreamTiming, finalize_streamed_wav},
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

/// A model loaded once and reused across generations, with reference-voice
/// caching only for stable URL-based conditioning. Local files are re-encoded
/// each time so updating a recorded voice never returns a stale clone.
pub struct CandleInProcess {
    model: TTSModel,
    variant: String,
    stable_voice_cache: Mutex<HashMap<String, ModelState>>,
}

impl CandleInProcess {
    /// Official gated voice-cloning checkpoints only. Never load reduced
    /// encoder-disabled configurations or pretend quantization is implemented.
    pub fn load(variant: &str) -> Result<Self> {
        let eos_threshold = match variant {
            "probe-english" => -4.0,
            "probe-7m" | "probe-18m" => -1.0,
            _ => bail!("unsupported full-checkpoint Candle variant: {variant}"),
        };
        let model = TTSModel::load_with_params(variant, 0.3, 1, eos_threshold)
            .with_context(|| format!("loading official full Candle {variant} checkpoint"))?;
        ensure!(model.sample_rate == 24_000, "unsupported model sample rate");
        Ok(Self {
            model,
            variant: variant.to_string(),
            stable_voice_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn variant(&self) -> &str {
        &self.variant
    }

    fn voice_state(&self, voice: &Voice) -> Result<ModelState> {
        let url = match voice {
            Voice::Preset(name) if name == "alba" => {
                Some("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
            }
            Voice::Preset(_) => bail!(
                "Candle currently supports only the verified alba preset; use explicit reference audio for other speakers"
            ),
            Voice::HubAudio(url) => Some(url.as_str()),
            Voice::ReferenceAudio(path) => {
                return self
                    .model
                    .get_voice_state(path)
                    .with_context(|| format!("encoding local speaker WAV {}", path.display()));
            }
            Voice::State(path) => {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("reading voice state {}", path.display()))?;
                return self
                    .model
                    .get_voice_state_from_prompt_bytes(&bytes)
                    .context("loading a full voice-conditioning audio_prompt tensor");
            }
        };
        let key = url.context("reference voice URL missing")?;
        if let Some(cached) = self
            .stable_voice_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("voice cache mutex poisoned"))?
            .get(key)
            .cloned()
        {
            return Ok(cached);
        }
        let path = weights::download_if_necessary(key)
            .with_context(|| format!("resolving voice reference {key}"))?;
        let state = self
            .model
            .get_voice_state(&path)
            .context("encoding reference audio through full Mimi encoder")?;
        self.stable_voice_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("voice cache mutex poisoned"))?
            .insert(key.to_owned(), state.clone());
        Ok(state)
    }

    fn validate(&self, request: &SynthesisRequest) -> Result<()> {
        request.validate()?;
        ensure!(
            request.language == "english",
            "Candle checkpoint is English-only"
        );
        ensure!(
            !request.quantize,
            "Candle quantization is not yet validated; refusing requested quantization"
        );
        Ok(())
    }
}

/// Provisional standard 24 kHz mono 16-bit RIFF header, finalized when
/// persisted. No misleading header-only first-audio latency is reported.
fn provisional_wav_header() -> [u8; 44] {
    let mut out = [0u8; 44];
    out[0..4].copy_from_slice(b"RIFF");
    out[4..8].copy_from_slice(&2_000_000_036_u32.to_le_bytes());
    out[8..12].copy_from_slice(b"WAVE");
    out[12..16].copy_from_slice(b"fmt ");
    out[16..20].copy_from_slice(&16_u32.to_le_bytes());
    out[20..22].copy_from_slice(&1_u16.to_le_bytes());
    out[22..24].copy_from_slice(&1_u16.to_le_bytes());
    out[24..28].copy_from_slice(&24_000_u32.to_le_bytes());
    out[28..32].copy_from_slice(&48_000_u32.to_le_bytes());
    out[32..34].copy_from_slice(&2_u16.to_le_bytes());
    out[34..36].copy_from_slice(&16_u16.to_le_bytes());
    out[36..40].copy_from_slice(b"data");
    out[40..44].copy_from_slice(&2_000_000_000_u32.to_le_bytes());
    out
}

impl StreamingSynthesisEngine for CandleInProcess {
    fn stream_wav(
        &self,
        request: &SynthesisRequest,
        on_bytes: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<StreamTiming> {
        self.validate(request)?;
        let start = Instant::now();
        let voice = self.voice_state(&request.voice)?;
        let header = provisional_wav_header();
        on_bytes(&header)?;
        let headers = start.elapsed();
        let mut first_pcm = None;
        let mut received_bytes = header.len() as u64;
        for next in self.model.generate_stream(&request.text, &voice) {
            let tensor = next.context("Candle streaming inference failed")?;
            let (batch, channels, _) = tensor.dims3()?;
            ensure!(
                batch == 1 && channels == 1,
                "Candle returned non-mono audio"
            );
            let pcm = audio::pcm_i16_le_bytes(&tensor.squeeze(0)?)?;
            if pcm.is_empty() {
                continue;
            }
            ensure!(pcm.len() % 2 == 0, "partial PCM sample");
            let new_total = received_bytes
                .checked_add(pcm.len() as u64)
                .context("streamed WAV size overflow")?;
            ensure!(
                new_total <= u64::from(u32::MAX),
                "WAV exceeds RIFF size limit"
            );
            if first_pcm.is_none() {
                first_pcm = Some(start.elapsed());
            }
            on_bytes(&pcm)?;
            received_bytes = new_total;
        }
        let first_pcm = first_pcm.context("Candle emitted no PCM audio")?;
        Ok(StreamTiming {
            headers,
            first_pcm,
            total: start.elapsed(),
            received_bytes,
        })
    }
}

impl SynthesisEngine for CandleInProcess {
    fn synthesize_wav(&self, request: &SynthesisRequest, output: &Path) -> Result<GeneratedAudio> {
        self.validate(request)?;
        ensure!(
            output.extension().and_then(|v| v.to_str()) == Some("wav"),
            "output must be .wav"
        );
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.as_os_str().is_empty() || parent.is_dir(),
            "output directory does not exist: {}",
            parent.display()
        );
        let mut writer = File::create(output).context("creating Candle WAV")?;
        let result = self.stream_wav(request, &mut |bytes| {
            writer.write_all(bytes).context("writing Candle PCM stream")
        });
        if let Err(error) = result {
            drop(writer);
            let _ = std::fs::remove_file(output);
            return Err(error);
        }
        if let Err(error) = writer.flush() {
            drop(writer);
            let _ = std::fs::remove_file(output);
            return Err(error.into());
        }
        drop(writer);
        let output_result =
            finalize_streamed_wav(output).and_then(|()| pocket_tts_model::inspect_wav(output));
        if output_result.is_err() {
            let _ = std::fs::remove_file(output);
        }
        output_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_wav_header_has_expected_format() {
        let h = provisional_wav_header();
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32::from_le_bytes(h[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u16::from_le_bytes(h[34..36].try_into().unwrap()), 16);
    }

    #[test]
    fn trait_supports_dynamic_dispatch() {
        fn requires_engine<T: StreamingSynthesisEngine + SynthesisEngine>() {}
        requires_engine::<CandleInProcess>();
        // This is a structural test, not a claim to load gated weights.
    }
}
