# Comprehensive streaming TTS benchmark

The `benchmark_tts.py` harness runs **real** voice-conditioned generation via Kyutai's official pinned Rust/XN executable. It requires **no external Python libraries**, model mocks, or Phonation integration.

## Metrics and exact definitions

- **TTFA, time to first audio**: request/process launch to first arriving audible PCM read from stdout. WAV header bytes never count as audio. The first PCM arrival time and estimated time to first audible sample (including leading silence) are separate fields.
- **RTF** (also written TRF in some discussions): synthesis time divided by the duration of generated audio, lower is better. The inverse is real-time speed multiplier, higher is better. Both a **process-inclusive** (startup, loading, reference encode and speech) and **model-only** (upstream per-text generation log) measurement are emitted.
- **TTFB**: launch to first stdout WAV header byte. Do not use as TTFA.
- **Generation latency**: process start to exit. Includes loading models on every subprocess request.
- **Peak RSS**: sampled process physical memory usage on Linux. Excludes page cache and can miss peaks between samples.
- **Preparation**: upstream-reported Mimi reference audio encode, prompt-audio conditioning and prompt-text prep times.
- **Throughput**: produced audio seconds per end-to-end wall second, as well as model-reported real-time speed.
- **Voice types**: saved predefined Alba voice state versus Kyutai's *reference audio* requiring real encoder work.
- **Matrix**: English 109M stored-parameter checkpoint, 7M and 18M REFL variants; short/medium/long prompts; one and four threads; float and int8; repeated trials; seed 42.
- **Dispersion**: mean, min/max, p50, p90 and p95 helpers; JSON holds per-trial values.

## Important interpretation limits

A new process is launched for each trial. Repeated runs benefit from warm Hugging Face files and OS caches but **not resident model reuse**. True in-process warmed model TTFA requires a model-retaining Rust session/server; that is an explicit next milestone, not interchangeable with these values. Synthetic-output latency, the voice-free path, GPU/CUDA, and streaming backpressure under concurrency are likewise outside this first suite.

Kyutai's Rust `StreamingWavWriter::finalize` adds 200 milliseconds of trailing silence; the process-inclusive RTF denominator deducts that fixed output padding. Model-only RTF uses native log measurements, which may differ from a listener's perceived time. First audible onset is threshold-based, not a substitute for a playback hardware test.

## Reproduce

```bash
export HF_TOKEN=...  # authorized token, not stored in source
python3 benchmarks/benchmark_tts.py \
    --binary /path/to/official/rust/pocket-tts \
    --output benchmark-results.json \
    --models english,7m,18m-refl \
    --voices preset,reference \
    --lengths short,medium,long \
    --threads 1,4 \
    --quantization off,on \
    --repeats 3
```

This produces `benchmark-results.json` and `benchmark-results.md` with all measurements, and retains **no audio files or model weights** in the benchmark outputs. Hosted CI is manual after initial baseline, with artifacts uploaded from the job for review.

The next comparative suite must benchmark reused Candle against the same prompts/checkpoints, then add **resident-session TTFA** and simultaneous-stream capacity. Audio quality comparison, WER, speaker embeddings and human listening tests should be added rather than treating output amplitude as quality.
