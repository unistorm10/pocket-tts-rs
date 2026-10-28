# Direct Candle integration candidate

This independent Rust workspace is the first **in-process** adapter for the
audited community Candle implementation. It implements the backend-neutral
`SynthesisEngine` and `StreamingSynthesisEngine` traits from
`pocket-tts-model` without spawning a process or using an HTTP service.

It is **not** included in the root workspace until build, real-model inference,
speaker identity and audio parity gates pass. The validated XN backend remains
untouched.

## Build and real checkpoint probe

From repository root:

```bash
bash scripts/prepare-candle-v4.sh /tmp/pocket-tts-candle-v4-audited
bash scripts/attach-candle-inprocess.sh /tmp/pocket-tts-candle-v4-audited \
  integrations/candle-inprocess/vendor/pocket-tts
cargo test --manifest-path integrations/candle-inprocess/Cargo.toml
cargo run --release --manifest-path integrations/candle-inprocess/Cargo.toml --bin probe -- \
  probe-english /tmp/inprocess-candle.wav
```

The preparation script clones the **exact pinned audited commit** of
`stepupgaming/pocket-tts`, adds the three official full-model configuration
fixtures, and applies all nine v4 compatibility patches. The source remains
outside Git. This integration needs the gated full checkpoints and the
authorized `HF_TOKEN`. Encoder-disabled checkpoints are never selected.

**Implementation details:** One Candle `TTSModel` is resident across requests,
stable Hub conditioning is cached in-process, locally supplied recordings are
re-encoded when used, streaming yields real Mimi-decoded 16-bit PCM chunks,
and the file-producing API finalizes provisional WAV lengths. The backend
rejects unsupported language, model variants and unverified quantization.

**Limitations:** CPU and checkpoint validation first; no GPU parity or
production concurrency certification. WAV framing reports actual first PCM,
not merely a header. This is still an **experimental source-prepared**
integration rather than the final vendored standalone Candle engine.

Do not treat structural unit tests as speech quality acceptance. Real model
checks must pass independent ASR and speaker verification. Upstream Candle
source licensing is retained in the prepared source, not stripped.

## Private reference voices (five-recording corpus)

The actual repository is **`unistorm10/reference`** (singular, private), at
audited commit `10c52892ee68812682e14cebca3c3b596d3b5660`.
The five original WAVs are in `speech/reference_1.wav` through
`speech/reference_5.wav`; each has a paired `.txt` transcript.
Reference 1-4 also contain Qwen3-TTS `*_icl.json` state; those embeddings,
discrete codes and token IDs **cannot be supplied as Pocket TTS voice states**.
Pocket TTS conditions on the original WAVs through its complete Mimi encoder.

For CI, the reference repository must **not** be copied into this public source
repository or uploaded as artifacts. `.github/workflows/private-reference-voice-qa.yml`
is an explicitly dispatched, self-hosted Linux job. Set the following encrypted
repository secrets in `pocket-tts-rs` before dispatching it:

- `HF_TOKEN`: authorized access to official full checkpoints.
- `REFERENCE_REPO_READ_TOKEN`: a fine-grained GitHub token with **Contents:
  Read-only** restricted to the private `unistorm10/reference` repository.

Open GitHub Actions → **Private reference-voice cloning acceptance** → **Run
workflow**, select `feat/standalone-v4-foundation`, and choose `all` for
all three full checkpoints. The job pins the private reference commit and uses
no submodules. The standard `GITHUB_TOKEN` for `pocket-tts-rs` cannot be
assumed to read another private repository.

For local, trusted machines, prepare the audited Candle source and run:

```bash
cargo run --release --manifest-path integrations/candle-inprocess/Cargo.toml \
  --bin reference_probe -- \
  probe-english /path/to/private/reference/speech /tmp/pocket-private-qa
python benchmarks/private_reference_qa.py \
  --speech-dir /path/to/private/reference/speech \
  --generated-dir /tmp/pocket-private-qa \
  --variant probe-english \
  --output /tmp/pocket-private-qa/reference-quality.json
```

Create the output directory beforehand. The Rust test loads a resident Candle
model, then for **each of five source voices** generates two unseen utterances
conditioned directly on the local WAV (10 clips per model). The independent
SpeechBrain ECAPA speaker evaluator scores generated-vs-conditioning and
generated-vs-two-public-comparators; faster-whisper performs acoustic and
intelligibility checks. Logs contain only summary results; **no source audio,
source transcript, speaker embeddings or generated WAVs are uploaded**.
Per-sample measurements are kept only in the runner's temporary directory and
deleted after CI.

**Important:** Reference numbers are not speaker identity labels. Neither
different-speaker negatives nor same-speaker positives among these five are
assumed. Until identities are annotated and metrics calibrated, speaker scores
are diagnostic and no arbitrary numerical threshold certifies naturalness or
voice cloning. ASR/acoustic gates remain fail-closed.
