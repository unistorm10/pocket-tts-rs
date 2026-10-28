//! Run cloned-voice acceptance against five private, consented WAV references.
//! This uses the original WAVs, never Qwen-specific ICL embeddings. No voice
//! samples or transcripts are embedded in this source or written to JSON.
use anyhow::{Context, Result, ensure};
use pocket_tts_candle_inprocess::CandleInProcess;
use pocket_tts_model::{SynthesisEngine, SynthesisRequest, Voice};
use std::path::PathBuf;
use std::time::Instant;

const PROMPTS: [&str; 2] = [
    "The museum opens early on Saturday so we can arrive before the crowds.",
    "Please put the blue notebook beside the old wooden table.",
];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let variant = args
        .next()
        .context("usage: reference_probe VARIANT SPEECH_DIR OUTPUT_DIR")?;
    let speech_dir = PathBuf::from(args.next().context("missing private speech directory")?);
    let output_dir = PathBuf::from(args.next().context("missing output directory")?);
    ensure!(args.next().is_none(), "unexpected extra arguments");
    ensure!(
        speech_dir.is_dir() && output_dir.is_dir(),
        "input and output directories must exist"
    );
    // All five known examples are required. No implicit speaker identity is
    // assigned from the numbering, and we do not log private transcripts.
    for id in 1..=5 {
        let stem = format!("reference_{id}");
        let reference = speech_dir.join(format!("{stem}.wav"));
        let transcript = speech_dir.join(format!("{stem}.txt"));
        ensure!(reference.is_file(), "reference audio missing for {stem}");
        ensure!(transcript.is_file(), "reference transcript missing for {stem}");
        ensure!(
            !std::fs::read_to_string(&transcript)?.trim().is_empty(),
            "empty reference transcript for {stem}"
        );
        ensure!(
            std::fs::metadata(&reference)?.len() > 44,
            "empty reference audio for {stem}"
        );
    }
    let started = Instant::now();
    let engine = CandleInProcess::load(&variant)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut results = Vec::new();
    for id in 1..=5 {
        let stem = format!("reference_{id}");
        let voice = speech_dir.join(format!("{stem}.wav"));
        for (prompt_index, prompt) in PROMPTS.iter().enumerate() {
            let prompt_id = prompt_index + 1;
            let filename = format!("private-clone-{variant}-{stem}-p{prompt_id}.wav");
            let output = output_dir.join(&filename);
            let request = SynthesisRequest {
                text: (*prompt).to_owned(),
                language: "english".to_owned(),
                voice: Voice::ReferenceAudio(voice.clone()),
                quantize: false,
            };
            let t = Instant::now();
            let audio = engine.synthesize_wav(&request, &output).with_context(|| {
                format!("in-process cloning failed for {stem} prompt {prompt_id}")
            })?;
            let elapsed = t.elapsed().as_secs_f64();
            let audio_seconds = audio.duration_seconds();
            ensure!(
                audio_seconds > 0.25,
                "generated speech too short for {stem}"
            );
            ensure!(audio.rms > 0.001, "generated speech silent for {stem}");
            let rtf = elapsed / audio_seconds;
            eprintln!(
                "PRIVATE_REFERENCE_CLONE_OK variant={variant} reference={stem} prompt={prompt_id} total_ms={:.1} rtf={rtf:.4}",
                elapsed * 1000.0,
            );
            results.push(serde_json::json!({
                "reference_id": stem,
                "prompt_id": prompt_id,
                "wav_file": filename,
                "duration_s": audio_seconds,
                "rms": audio.rms,
                "total_ms": elapsed * 1000.0,
                "rtf": rtf
            }));
        }
    }
    let report = serde_json::json!({
        "schema_version": 1,
        "model_variant": variant,
        "model_load_ms": load_ms,
        "references": 5,
        "prompts_per_reference": PROMPTS.len(),
        "conditioning": "original local WAV through full Mimi encoder",
        "speaker_identities": "unknown; numbering does not imply different speakers",
        "private_transcripts_exported": false,
        "results": results,
    });
    let report_path = output_dir.join(format!("private-clone-{variant}.metrics.json"));
    std::fs::write(report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
