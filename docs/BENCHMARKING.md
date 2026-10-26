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

## First completed full fresh-process matrix

[Workflow run #40377493188](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40377493188) completed **144/144 real-PCM trials without errors**, on the self-hosted `elitedesk-mini` Linux runner. [Download JSON and Markdown metrics](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40377493188/artifacts/12677901806).

At 4 CPU threads with FP32, the p50 streaming audible TTFA, process-inclusive RTF and observed peak RSS were:

| Variant | Conditioning | TTFA (ms) | RTF | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| english | preset | 376 | 0.296 | 624.6 |
| english | reference audio | 1903 | 0.479 | 897.5 |
| english_18m_refl_2026-10 | preset | 139 | 0.093 | 239.1 |
| english_18m_refl_2026-10 | reference audio | 1604 | 0.296 | 613.7 |
| english_7m_2026-10 | preset | 104 | 0.083 | 203.5 |
| english_7m_2026-10 | reference audio | 1593 | 0.278 | 572.9 |

The process benchmark includes model startup and voice preparation per request. Many INT8 configurations had lower memory requirements but **worse** latency/RTF than FP32. See per-trial data for 1-thread and INT8 results.

### Warm resident comparison

`benchmarks/benchmark_resident.py` drives the official Rust `serve` endpoint. It measures:
- server launch to `/health` ready (model + preset voice loaded);
- actual HTTP TTFA for streaming PCM chunks with 44-byte WAV header excluded;
- the first reference-audio request as **uncached**, subsequent requests as **cached**, over one persistent server process;
- process memory, per-request RTF, throughput, utterance duration and timing distributions;
- the same three models, two thread counts, two quantization modes, three passage lengths and two voice modes.

Run it after building Kyutai v4.0.1's official `pocket-tts` binary:
```bash
python3 benchmarks/benchmark_resident.py --binary /path/to/pocket-tts \
  --output resident.json --repeats 2
```
`resident.json` and `resident.md` retain per-trial measurements. The corresponding self-hosted Actions workflow is `benchmark-resident.yml`. Neither benchmark measures neural-quality/WER/speaker similarity or concurrent HTTP capacity; these require a separate evaluation protocol. 

## Verified resident run

[Resident run #40380071569](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40380071569) passed **144/144** real streaming HTTP generations, bringing the total to **288 successful trials** across both benchmark modes. [Download its JSON and Markdown results](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40380071569/artifacts/12677703174). The 7M model achieved **24 ms p50 preset-voice TTFA** at 4 CPU threads FP32 with a resident model, and **20 ms median TTFA for five cached reference-voice requests** after one uncached clone at 1,414 ms. For 18M REFL the corresponding values were 34/28 ms (first clone 1,543 ms), and default English 177/125 ms (first clone 1,789 ms).

The full interpreted results and 1-thread/INT8 comparisons are in [RESULTS_2026-10-26.md](RESULTS_2026-10-26.md). Both benchmark workflows run on manual dispatch only after the initial verification.
