# pocket-tts-rs

A **Rust-native** text-to-speech engine, ported from
[Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts) and built on
[Candle](https://github.com/huggingface/candle).

Runs entirely on CPU — no Python, no GPU required.

- **Pure Rust** — no Python runtime, no PyO3, just binaries and a library
- **CPU-first** — with optional Metal acceleration on macOS
- **Streaming** — stateful end-to-end streaming (FlowLM + Mimi) for low latency
- **Voice cloning** — from a WAV prompt, plus a predefined voice library
- **int8 quantisation** — smaller footprint, faster inference
- **WASM** — the full model runs in a browser
- **HTTP API + web UI** — Axum server with an OpenAI-compatible endpoint
- **Pause handling** — natural pauses and explicit `[pause:Xms]` markers
- **Multi-language** — YAML model configs per language/variant

## Workspace layout

| Crate | Responsibility |
|---|---|
| `pocket-tts-core` | Primitives: streaming `ModelState`, config, audio I/O, weight resolution, int8 quantisation, and the shared NN blocks (`attention`, `conv`, `mlp`, `rope`, `sdpa`, streaming `transformer`) |
| `pocket-tts-mimi` | Mimi neural audio codec — SEANet encoder/decoder + quantizer |
| `pocket-tts-flow` | FlowLM transformer, LSD decode, text conditioning |
| `pocket-tts-model` | `TTSModel` orchestration, text chunking, voice cloning, wasm bindings |
| `pocket-tts-server` | Axum HTTP API + embedded web UI |
| `pocket-tts-cli` | The `pocket-tts` binary (`generate`, `serve`) |

`pocket-tts-model` re-exports the `pocket-tts-core` APIs you need when driving
`TTSModel`, so most consumers only need that one crate.

## Quick start

```bash
# Build (the web UI ships prebuilt, so no Node toolchain is needed)
cargo build --release

# Generate audio
cargo run --release -p pocket-tts-cli -- generate --text "Hello, world!"

# Custom voice from a WAV file
cargo run --release -p pocket-tts-cli -- generate \
    --text "Hello, world!" --voice ./my_voice.wav --output output.wav

# Predefined voice
cargo run --release -p pocket-tts-cli -- generate --voice alba

# HTTP API server (http://localhost:8000)
cargo run --release -p pocket-tts-cli -- serve
```

Build without the bundled web UI:

```bash
cargo build --release --no-default-features
```

### Feature flags

| Feature | Effect |
|---|---|
| `metal` | Metal acceleration (macOS) |
| `quantized` | int8 quantised inference |
| `mkl` | Intel MKL BLAS backend |
| `wasm` | Browser build of `pocket-tts-model` |
| `web-ui` *(default, server)* | Embed the web UI assets |

## Library usage

```rust
use pocket_tts_model::TTSModel;

fn main() -> anyhow::Result<()> {
    let model = TTSModel::load("b6369a24")?;
    let voice_state = model.get_voice_state("voice.wav")?;
    let audio = model.generate("Hello, world!", &voice_state)?;

    // `pocket-tts-model` re-exports the core audio helpers.
    pocket_tts_model::audio::write_wav("output.wav", &audio, model.sample_rate as u32)?;
    Ok(())
}
```

A runnable version of this lives in
`crates/pocket-tts-model/examples/basic_generation.rs`:

Streaming:

```rust
for chunk in model.generate_stream("Long text here...", &voice_state) {
    let audio_chunk = chunk?;
    // play or process each chunk as it arrives
}
```

Tuning:

```rust
let model = TTSModel::load_with_params(
    "b6369a24",  // variant
    0.7,         // temperature (higher = more variation)
    1,           // lsd_decode_steps (more = better quality, slower)
    -4.0,        // eos_threshold (more negative = longer audio)
)?;
```

Model weights are fetched from Hugging Face on first use; set `HF_TOKEN` if the
model you use requires authentication.

## CLI reference

### `generate`

```
pocket-tts generate [OPTIONS]

  -t, --text <TEXT>            Text to synthesize
  -v, --voice <VOICE>          Predefined name, .wav file, or .safetensors
  -o, --output <PATH>          Output file            [default: output.wav]
      --variant <VARIANT>      Model variant         [default: b6369a24]
      --temperature <FLOAT>    Sampling temperature  [default: 0.7]
      --lsd-decode-steps <INT> LSD decode steps      [default: 1]
      --eos-threshold <FLOAT>  EOS threshold         [default: -4.0]
      --stream                 Stream raw PCM to stdout
  -q, --quiet                  Suppress output
      --use-metal              Metal acceleration (macOS)
```

Predefined voices: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`,
`eponine`, `azelma`.

### `serve`

```
pocket-tts serve [OPTIONS]

      --host <HOST>            Bind address          [default: 127.0.0.1]
  -p, --port <PORT>            Port                  [default: 8000]
      --voice <VOICE>          Default voice         [default: alba]
      --variant <VARIANT>      Model variant         [default: b6369a24]
      --temperature <FLOAT>    Temperature           [default: 0.7]
      --lsd-decode-steps <INT> LSD steps             [default: 1]
      --eos-threshold <FLOAT>  EOS threshold         [default: -4.0]
      --ui <UI>                standard | wasm-experimental
```

## WebAssembly

```bash
./scripts/build-wasm.sh
cargo run --release -p pocket-tts-cli -- serve --ui wasm-experimental
```

This requires the `wasm32-unknown-unknown` target and `wasm-bindgen-cli`. See
`docs/PORTING_NOTES.md` for details.

## Documentation

- `docs/generate.md` — generation options and audio quality
- `docs/serve.md` — HTTP API reference
- `docs/rust-api.md` — library API
- `docs/docker.md` — container usage
- `docs/PORTING_NOTES.md` — crate layout, lineage, and build notes

## Licence

Dual-licensed under `MIT OR Apache-2.0` — see `LICENSE`.
