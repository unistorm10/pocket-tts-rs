# Rust-native port of kyutai-labs/pocket-tts

## Problem

Port pocket-tts (a CPU-friendly flow-matching TTS model: FlowLM transformer +
Mimi neural codec) from Python/PyTorch to a fully native Rust implementation
that:

- Uses **real downloaded HuggingFace weights** for inference at every stage —
  no mocked tensors, no synthetic/random weights standing in for real ones.
- Is deployable on any platform (Linux/macOS/Windows, x86_64/arm64) as a
  single static-ish binary.
- Ships as: a reusable Rust **library crate**, a **CLI** (`generate`, `serve`,
  voice import/export, quantize), and an **Axum HTTP server** replicating the
  FastAPI `serve` surface (incl. simple web UI).
- Reaches **full feature parity** with pocket-tts: voice cloning from an audio
  prompt, predefined voice library, multi-language configs, streaming
  generation, EOS detection, int8 quantization, safetensors voice
  export/import — plus forward-looking (SOTA/future-proofing) additions
  called out in Phase 9.

## Decisions locked in (from user)

1. **Tensor backend: [`candle`](https://github.com/huggingface/candle)**
   (pure Rust, native safetensors, CPU/CUDA/Metal backends).
2. **Deliverable surface**: library crate + CLI + Axum HTTP server + web UI.
3. **Scope**: full parity (voice cloning, predefined voices, multi-language,
   streaming, quantization) *plus* reasonable future-proofing (see Phase 9).
4. Weights are always downloaded from HuggingFace (`hf://kyutai/pocket-tts/...`
   and per-language repos) and loaded for real inference at every testable
   milestone. No phase may ship with random/mocked weights as a stand-in.

## Why a phased, agent-safe plan

Small local models (≤9B) drift when given large, vague, or multi-file tasks.
To prevent this, **every phase below is a single, narrowly-scoped unit of
work** with:
- An explicit **file list** (create/modify only these files).
- An explicit **"do not touch"** list.
- A **numeric reference** to the exact Python source lines/behavior being
  ported (file + purpose), so the agent translates rather than invents.
- A **concrete, runnable acceptance test** (`cargo test -p <crate> <test>`)
  that must pass before the phase is considered done.
- A **golden-value fixture** step: before writing Rust, dump reference
  tensors from the real Python implementation (using real downloaded
  weights) into a `.safetensors` fixture file; the Rust test loads that
  fixture and asserts numerical closeness (`max_abs_diff < tol`). This is
  the drift guard — it is impossible to silently diverge from the reference
  because the test compares against real numbers, not vibes.

Agents must follow this literally:
- **Never invent APIs** not present in this plan or in `candle`'s public
  docs — if something is missing, stop and ask rather than guessing.
- **Never mark a phase done without running its acceptance test.**
- **Never modify files outside the phase's file list.**
- **One phase, one PR/commit.** If a phase needs >1 sitting, checkpoint with
  passing tests before continuing.

## Repository layout (created in Phase 0)

```
pocket-tts-rs/
  Cargo.toml                     # workspace
  crates/
    pocket-tts-core/             # tensors, config, weight loading, streaming state
    pocket-tts-mimi/             # SEANet codec + Mimi model
    pocket-tts-flow/             # FlowLM transformer + flow-matching/LSD decode
    pocket-tts-model/            # TTSModel orchestration, voice state, text chunking
    pocket-tts-cli/              # CLI binary (generate, serve, voices, quantize)
    pocket-tts-server/           # Axum HTTP server + static web UI
  xtask/                         # dev tooling: golden fixture dumper (calls Python), fmt/lint/test runner
  fixtures/                      # golden .safetensors tensors + expected outputs (git-lfs or downloaded)
  tests/                         # workspace-level integration tests
  docs/
    PORTING_NOTES.md             # 1:1 map of Python file -> Rust file, kept updated every phase
```

## Golden-fixture workflow (used by every phase)

1. In a Python venv with the real `pocket-tts` package and real downloaded
   weights, run a small fixture script (checked into
   `xtask/fixtures/dump_<phase>.py`) that feeds a **fixed, hardcoded input**
   (fixed text, fixed random seed, fixed audio prompt file) through the
   *real* component being ported and saves inputs+outputs to
   `fixtures/<phase>.safetensors`.
2. The Rust test loads the same fixture, runs the Rust implementation on the
   saved inputs, and asserts outputs match within tolerance
   (`1e-3` relative for float32 unless noted).
3. Fixture scripts and files are committed so any agent/CI can re-verify
   without re-running Python.

## Phases

### Phase 0 — Workspace scaffolding & weight/download infra
Goal: empty-but-compiling Cargo workspace with the crate layout above, plus a
shared `pocket_tts_core::download` module that resolves `hf://owner/repo/path@revision`,
`http(s)://`, and local paths, caching under `~/.cache/pocket_tts_rs`
(mirrors `pocket_tts/utils/utils.py::download_if_necessary`).
Files: `Cargo.toml` (workspace), all crate skeletons with empty `lib.rs`,
`crates/pocket-tts-core/src/download.rs`, `crates/pocket-tts-core/src/config.rs`
(serde structs mirroring `pocket_tts/utils/config.py`'s `Config`, `FlowLMConfig`,
`MimiConfig`, `SEANetConfig`, `MimiTransformerConfig`, `QuantizerConfig`,
`FlowLMTransformerConfig`, `LookupTable`, `FlowConfig` — same field names/types,
`#[serde(deny_unknown_fields)]` to mirror `extra="forbid"`).
Acceptance test: `cargo test -p pocket-tts-core config_parses_english_yaml` —
downloads (or loads a committed copy of) `pocket_tts/config/english.yaml` from
the real repo and parses it into `Config` without error, asserting key fields
(`flow_lm.transformer.d_model == 1024`, `mimi.sample_rate == 24000`, etc.).
Do not touch: no model/tensor code yet.

### Phase 1 — Streaming state & primitive modules
Goal: port the streaming-state contract and low-level building blocks that
everything else depends on, using candle `Tensor`/`VarBuilder`.
Python reference: `pocket_tts/modules/stateful_module.py` (ModelState = map
of module-name -> tensor dict), `pocket_tts/modules/rope.py` (`apply_rope`),
`pocket_tts/modules/conv.py` (`StreamingConv1d`, `StreamingConvTranspose1d`,
`pad_for_conv1d`), `pocket_tts/modules/layer_scale.py`.
Files: `crates/pocket-tts-core/src/state.rs`, `.../rope.rs`, `.../conv.rs`,
`.../layer_scale.rs`.
Design constraints (do not deviate):
- `ModelState` = `HashMap<String, HashMap<String, Tensor>>`, keyed by
  absolute module path exactly like Python's `stamp_state_names`.
- RoPE: same frequency formula
  `freqs = exp(-log(max_period) * 2/D * arange(D/2))`, same interleaved
  real/imag layout as `apply_rope`.
- `StreamingConv1d`/`StreamingConvTranspose1d` state fields must be named
  `previous`/`first` and `partial` respectively, matching Python, so the
  golden-fixture safetensors (which store state keyed by these names) load
  directly.
Acceptance tests (fixture-based, golden values dumped from real Python):
`cargo test -p pocket-tts-core rope_matches_python`,
`conv1d_streaming_matches_python` (feed audio in 2 chunks, verify continuity
matches single-shot output), `convtranspose1d_streaming_matches_python`.
Do not touch: no full model code.

### Phase 2 — Attention with KV cache
Python reference: `pocket_tts/modules/attention.py` (`StreamingMultiheadAttention`,
`_LinearKVCacheBackend`, `_build_attention_mask`, `complete_kv`).
Files: `crates/pocket-tts-core/src/attention.rs`.
Constraints: same in_proj packing order (q,k,v interleaved per head via
`view(b,t,3,h,d)`), same causal+context masking semantics, must support both
stateless (training-style, full-sequence) and stateful (streaming, one step
at a time) forward passes as two code paths mirroring Python.
Acceptance test: golden fixture with a fixed small `d_model`/`num_heads`
config, comparing full-sequence output vs step-by-step streamed output vs
Python reference — all three must match within tolerance.

### Phase 3 — SEANet encoder/decoder (Mimi codec neural net)
Python reference: `pocket_tts/modules/seanet.py` (`SEANetEncoder`,
`SEANetDecoder`, `SEANetResnetBlock`), `pocket_tts/modules/resample.py`
(`ConvDownsample1d`, `ConvTrUpsample1d`), `pocket_tts/modules/dummy_quantizer.py`.
Files: `crates/pocket-tts-mimi/src/seanet.rs`, `.../resample.rs`,
`.../quantizer.rs`.
Constraints: exact layer ordering (ELU activations between convs, residual
add-back with shape assert), exact `ratios` reversal logic for encoder vs
decoder as in Python.
Acceptance test: load the **real** `mimi`-portion weights from the English
config's `weights_path`, run `encode_to_latent` on a fixed, short, real WAV
file fixture, compare against the golden fixture dumped from real Python
(`pocket_tts.models.mimi.MimiModel.encode_to_latent`). Must be run with real
weights, not random init.

### Phase 4 — Mimi transformer + full MimiModel
Python reference: `pocket_tts/modules/transformer.py`
(`StreamingTransformer`, `StreamingTransformerLayer`, `ProjectedTransformer`),
`pocket_tts/models/mimi.py` (`MimiModel`, `build_mimi`).
Files: `crates/pocket-tts-mimi/src/transformer.rs`, `.../mimi_model.rs`.
Acceptance test: full `MimiModel::encode_to_latent` then
`MimiModel::decode_from_latent` round trip on a real fixed WAV fixture using
real downloaded weights; compare decoded waveform against the Python-decoded
waveform (golden fixture) within tolerance (SNR-based comparison, not exact
equality, since floating point ops may reorder slightly across BLAS impls —
document the tolerance and rationale in code comments).

### Phase 5 — FlowLM: transformer, AdaLN flow net, text conditioner
Python reference: `pocket_tts/modules/mlp.py` (`SimpleMLPAdaLN`, `RMSNorm`,
`LayerNorm`, `TimestepEmbedder`, `ResBlock`, `FinalLayer`, `modulate`),
`pocket_tts/modules/text_conditioner.py` (`SentencePieceTokenizer`,
`LUTConditioner`), `pocket_tts/models/flow_lm.py` (`FlowLMModel`,
`lsd_decode`, `ot_decode`).
Files: `crates/pocket-tts-flow/src/mlp.rs`, `.../text_conditioner.rs`
(use the `sentencepiece` crate or `tokenizers`-compatible SP loader — must
load the real `tokenizer.model` file), `.../flow_lm.rs`.
Constraints: reproduce `lsd_decode` (2 time conditions, 1-step default) and
`ot_decode` (Euler integration, 1 time condition, needs the fixed step count
from config) exactly, including the "NaN marks BOS" convention
(`sequence = where(isnan(sequence), bos_emb, sequence)`).
Acceptance test: golden fixture feeding fixed text + fixed noise tensor
(seeded) through `FlowLMModel.forward` with real weights; assert output
latent and EOS logit match Python within tolerance.

### Phase 6 — Voice state & TTSModel orchestration
Python reference: `pocket_tts/models/model_state.py` (`export_model_state`,
`_import_model_state`, safetensors key format `"{module_name}/{key}"`, the
`offset`-from-`current_end` shape compatibility shim),
`pocket_tts/models/text_chunking.py` (`prepare_text_prompt`,
`split_into_best_sentences`), `pocket_tts/models/tts_model.py`
(`TTSModel.load_model`, `get_state_for_audio_prompt`, `generate_audio`,
`generate_audio_stream`, `_generate`, `_autoregressive_generation`,
`_estimate_max_gen_len`), `pocket_tts/default_parameters.py`.
Files: `crates/pocket-tts-model/src/{voice_state,text_chunking,tts_model}.rs`.
Constraints: safetensors voice-state files exported by this Rust code must be
byte-compatible enough to be loaded by the *Python* pocket-tts and vice
versa (interop test), preserving the `"{module}/{key}"` naming.
Acceptance tests:
- `voice_state_roundtrip`: export a voice state from Rust, re-import it in
  Rust, assert tensors match.
- `voice_state_interop_with_python`: import a voice-state `.safetensors`
  file produced by the **real** Python `export-voice` CLI command and
  generate audio from it in Rust; compare resulting waveform (golden
  fixture) within tolerance.
- `text_chunking_matches_python`: run `split_into_best_sentences` on several
  fixed strings (including the decimal-period edge case) and diff against
  Python's output list exactly (this is deterministic string logic — exact
  match required, no tolerance).
- `end_to_end_generate_matches_python`: full `generate_audio` on the
  **default English config + default voice**, fixed seed, comparing to a
  golden WAV dumped from real Python `pocket-tts generate`. This is the
  master "no mock" acceptance gate: it MUST use real downloaded weights end
  to end.

### Phase 7 — CLI
Python reference: `pocket_tts/main.py` `generate` command and helper
functions (not `serve`, that's Phase 8).
Files: `crates/pocket-tts-cli/src/main.rs`, `.../commands/generate.rs`,
`.../commands/voices.rs` (export/import voice state), `.../commands/quantize.rs`.
Acceptance test: `cargo run -p pocket-tts-cli -- generate --text "..." --voice alba --out out.wav`
against real downloaded weights, then an integration test asserts `out.wav`
exists, is valid WAV, non-silent, and RTF (real-time factor) is logged.

### Phase 8 — Quantization (int8 dynamic)
Python reference: `pocket_tts/quantization.py` (`apply_dynamic_int8`,
`RECOMMENDED_CONFIG = {"attention","ffn"}`).
Files: `crates/pocket-tts-flow/src/quantization.rs`.
Use candle's own int8/gguf-style quantized tensor support (or a hand-rolled
dynamic per-tensor int8 matmul if candle lacks a direct analog — this must
be decided by reading candle's quantized-tensor API, not invented).
Acceptance test: quantized `generate_audio` output vs float32 output: WER/
similarity proxy via simple energy+zero-crossing sanity checks, plus timing
showing speedup, run against real weights.

### Phase 9 — HTTP server (Axum) + web UI
Python reference: `pocket_tts/main.py` `serve` command, `web_app` FastAPI
routes (`/`, `/health`, `/tts`), `pocket_tts/static/index.html`.
Files: `crates/pocket-tts-server/src/{main,routes,state}.rs`,
`crates/pocket-tts-server/static/index.html` (ported/adapted).
Constraints: same route shapes (`GET /`, `GET /health`, `POST /tts` accepting
multipart `text`, `voice_url` OR `voice_wav`), same streamed
`audio/wav` chunked response behavior.
Acceptance test: start server in a test harness (`tokio::test` + `reqwest`),
POST to `/tts` with real text and the default voice, assert `200`, WAV
content-type, non-empty streamed body, real generated audio (again: real
weights, no mocks).

### Phase 10 — Cross-platform packaging & release
Goal: build matrix (Linux x86_64/arm64, macOS x86_64/arm64, Windows x86_64)
via GitHub Actions, static/portable binaries, Dockerfile parity with the
original `Dockerfile`.
Files: `.github/workflows/release.yml`, `Dockerfile`, `docs/INSTALL.md`.
Acceptance test: CI matrix green on all targets; each produces a binary that
runs `generate --help` successfully; at least the Linux x86_64 job also runs
the Phase 6 end-to-end real-weights generation test.

### Phase 11 — SOTA / future-proofing enhancements (stretch, after full parity)
Only start after Phases 0–10 are green. Each is its own sub-phase with the
same rigor (file list + golden test):
- GPU acceleration via candle's CUDA/Metal backends (feature-flagged,
  fallback to CPU), with a benchmark test.
- WebSocket streaming generation endpoint (lower latency than chunked HTTP).
- Batched generation (relaxing the Python code's batch-size-1 limitation),
  gated behind a feature flag and its own correctness tests.
- Pluggable voice-library manager (list/download/cache predefined voices,
  offline bundle mode).
- ONNX/GGUF export path for interop with other runtimes.
- Structured logging/tracing + Prometheus metrics for the server.
- Optional streaming SIMD/AVX2 kernels behind candle's `mkl`/`accelerate`
  feature flags for CPU speed parity or improvement over the Python RTF
  numbers reported in `AGENTS.md`.

## Anti-drift rules for every phase (repeat to each executing agent)

1. Read the "Python reference" file list for the phase FIRST, in full,
   before writing any Rust.
2. Only create/edit the files listed under "Files:" for that phase.
3. Do not invent config fields, function names, or behaviors not present in
   the referenced Python source — if the Rust design needs something extra,
   stop and flag it instead of guessing.
4. Every phase's "done" state requires the specified `cargo test` command(s)
   to pass, using real downloaded HF weights where specified — never stub
   weights with `Tensor::zeros`/random init as a substitute for real
   inference in acceptance tests.
5. Update `docs/PORTING_NOTES.md` with a one-line mapping entry
   (`python_file -> rust_file`) at the end of every phase.
6. If a golden fixture cannot be produced (e.g., no Python env available),
   the phase is blocked — do not fabricate expected values.
