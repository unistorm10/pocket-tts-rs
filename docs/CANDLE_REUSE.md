# Pinned Candle implementation reuse: actual compatibility gates

**Candidate:** [stepupgaming/pocket-tts](https://github.com/stepupgaming/pocket-tts/tree/dbf78c816cd83c89f11dbeb9d87290ad0a3dccd0) at `dbf78c816cd83c89f11dbeb9d87290ad0a3dccd0` (community Rust/Candle 0.6.2).

The upstream code already implements FlowLM, SEANet, Mimi, attention state, audio encoding, reference-voice extraction, real streaming and CLI/HTTP output. **Do not reimplement them wholesale.** Its model constructor was reviewed and compared directly with official Kyutai Pocket TTS v4.0.1 YAML fixtures in `pocket-tts-core::candle_compat`.

An upstream **compiling** Candle crate does not demonstrate that it loads or generates correct speech for the new full models. The compatibility analyzer must fail closed until the missing functionality is adapted.

| Capability | English (full) | 7M (October) | 18M REFL (October) |
| --- | --- | --- | --- |
| BOS before voice state | Adapt | Adapt | Adapt |
| Exact v4 text normalization | Adapt | Adapt | Adapt |
| Drifting flow sampler | Not used | Adapt | Adapt |
| Explicit `z_dim` latent | Not used | Adapt | Adapt |
| Grouped-query KV heads | Standard MHA | Adapt | Standard MHA |
| Attention projection width override | Not used | Adapt | Not used |
| Rank-factorized text embeddings | Not used | Adapt | Not used |
| Independent compact Mimi SEANet decoder | Not used | Adapt | Adapt |
| Independent Mimi decoder transformer | Not used | Adapt | Adapt |

**Note:** The table describes source-code gaps for the pinned *candidate* implementation. These are **not** limitations of the official Kyutai Rust/XN backend, which already passed full-model cloning tests. Additional tensor-level differences may be found while loading the full weights.

## Acceptance order
1. Compile unmodified pinned Candle library on the self-hosted Linux runner in a separate checkout, without adding Phonation/SOMA dependencies and without writing gated weights to source control.
2. Confirm actual builder APIs and isolate source modules for reuse. Record licensing and keep original copyright attribution.
3. Adapt the existing Candle code to accept full English v4.0.1 without silently ignoring `insert_bos_before_voice` or v4 character replacements. Produce genuine 24-kHz speech and reference-audio cloned output. Compare TTFA, RTF, memory and intelligibility against the [288-trial XN baseline](RESULTS_2026-10-26.md).
4. Adapt grouped-query attention, low-rank text lookup, drifting sampler, z-projection and asymmetric compact decoder for 7M and 18M REFL, using upstream implementations wherever possible.
5. Add a *true* embedded Candle `SynthesisEngine` backend with an in-process model lifecycle, not a shell/process wrapper. Run the exact gated full-weight checks and qualitative voice-cloning comparisons.
6. Only switch the default engine after actual parity and throughput measurements. Keep the working native Rust/XN fallback independently deployable.

The candidate is currently a **source reuse/build investigation**, not a finished integrated Candle engine. Its upstream source license is `MIT OR Apache-2.0` in its Cargo manifest, with an MIT root LICENSE; review and preserve upstream notices when vendoring specific modules.

## Local developer commands

```bash
cargo test -p pocket-tts-core candle_compat
git clone https://github.com/stepupgaming/pocket-tts.git /tmp/pocket-candle
git -C /tmp/pocket-candle checkout dbf78c816cd83c89f11dbeb9d87290ad0a3dccd0
cargo check --locked --manifest-path /tmp/pocket-candle/Cargo.toml -p pocket-tts
```
