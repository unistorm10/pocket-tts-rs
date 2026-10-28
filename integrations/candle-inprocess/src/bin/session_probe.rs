//! Real two-request streaming acceptance for a single resident Candle model.
//! Full gated weights, same cached voice reference, actual incremental PCM.
//! No mock data, process restart, or post-hoc re-chunking.
use anyhow::{Context, Result, ensure};
use pocket_tts_candle_inprocess::CandleInProcess;
use pocket_tts_model::{
    StreamingSynthesisEngine, SynthesisRequest, Voice, inspect_wav,
    resident::finalize_streamed_wav,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SENTENCE: &str = "Testing persistent Candle streaming with a cloned voice.";

fn streamed_request(
    engine: &dyn StreamingSynthesisEngine,
    request: &SynthesisRequest,
    output: &Path,
) -> Result<(f64, f64, f64, f64, u64, usize)> {
    let mut file = File::create(output).context("creating streamed acceptance WAV")?;
    let mut chunks = 0usize;
    let timing = engine.stream_wav(request, &mut |bytes| {
        if chunks == 0 {
            ensure!(bytes.len() == 44 && &bytes[..4] == b"RIFF", "invalid initial WAV header");
        } else {
            ensure!(!bytes.is_empty() && bytes.len() % 2 == 0, "invalid streamed PCM chunk");
        }
        chunks += 1;
        file.write_all(bytes).context("writing streamed audio chunk")
    });
    let timing = match timing {
        Ok(result) => result,
        Err(err) => {
            drop(file);
            let _ = std::fs::remove_file(output);
            return Err(err);
        }
    };
    file.flush()?;
    drop(file);
    finalize_streamed_wav(output)?;
    let audio = inspect_wav(output)?;
    ensure!(chunks > 1, "model did not emit a PCM chunk after the header");
    ensure!(audio.samples > 24_000, "streamed model emitted less than 1 second");
    ensure!(timing.first_pcm >= timing.headers, "first PCM preceded WAV header");
    ensure!(timing.total >= timing.first_pcm, "total synthesis shorter than first PCM");
    ensure!(
        timing.received_bytes == std::fs::metadata(output)?.len(),
        "stream timing byte count differs from written output"
    );
    let audio_secs = audio.duration_seconds();
    ensure!(audio_secs > 0.0, "invalid streamed audio duration");
    Ok((
        timing.headers.as_secs_f64() * 1000.0,
        timing.first_pcm.as_secs_f64() * 1000.0,
        timing.total.as_secs_f64() * 1000.0,
        timing.total.as_secs_f64() / audio_secs,
        timing.received_bytes,
        chunks,
    ))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let variant = args.next().context("usage: session_probe VARIANT OUTPUT_DIR")?;
    let output_dir = PathBuf::from(args.next().context("missing output directory")?);
    ensure!(args.next().is_none(), "unexpected extra arguments");
    ensure!(output_dir.is_dir(), "output directory does not exist");
    let load_start = Instant::now();
    let engine = CandleInProcess::load(&variant)?;
    let model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let request = SynthesisRequest {
        text: SENTENCE.to_owned(),
        language: "english".to_owned(),
        voice: Voice::HubAudio(
            "hf://kyutai/tts-voices/alba-mackenna/casual.wav".to_owned(),
        ),
        quantize: false,
    };
    let mut reports = Vec::new();
    for label in ["cold", "warm"] {
        let output = output_dir.join(format!("session-{}-{}.wav", variant, label));
        let (headers_ms, first_pcm_ms, total_ms, rtf, bytes, chunks) =
            streamed_request(&engine, &request, &output)
                .with_context(|| format!("failed {label} request for {variant}"))?;
        eprintln!(
            "SESSION_CANDLE_OK variant={variant} request={label} first_pcm_ms={first_pcm_ms:.1} total_ms={total_ms:.1} rtf={rtf:.4} chunks={chunks} bytes={bytes}"
        );
        reports.push(serde_json::json!({
            "request": label,
            "wav": output.file_name().context("WAV filename missing")?.to_string_lossy(),
            "headers_ms": headers_ms,
            "first_pcm_ms": first_pcm_ms,
            "total_ms": total_ms,
            "rtf": rtf,
            "received_bytes": bytes,
            "callback_chunks": chunks,
        }));
    }
    let metrics = serde_json::json!({
        "schema_version": 1,
        "model_variant": variant,
        "speaker_reference": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
        "engine": "resident in-process Rust/Candle",
        "model_load_ms": model_load_ms,
        "requests": reports,
        "note": "Uncalibrated single-run measurements. No latency/performance acceptance threshold.",
    });
    let output = output_dir.join(format!("session-{}.metrics.json", variant));
    std::fs::write(output, serde_json::to_vec_pretty(&metrics)?)?;
    Ok(())
}
