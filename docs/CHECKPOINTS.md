# Verified Pocket TTS checkpoints (October 26, 2026)

The **primary** model source is [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), with full voice-cloning support. All full checkpoint references below are pinned to revision `db04f0e653473e5c7333c69686ac92b7460b0c1d`.

All three full models were fetched on the `elitedesk-mini` self-hosted Linux runner on October 26, 2026 and verified by direct safetensors inspection and a real Candle tensor computation in [Actions run #40358200381](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40358200381).

| Config variant | Model path (under `kyutai/pocket-tts`) | Bytes | SHA-256 | `mimi.encoder` |
| --- | --- | ---: | --- | --- |
| Default English | `languages/english/model.safetensors` | 219,029,196 | `bc96b82436c5ec00d3118ef3b01bc3358423e1766ecae3dca0aeea84789e7d74` | 42 of 42 active |
| English 7M | `languages/english_7m_2026-10/model.safetensors` | 66,482,920 | `1200a5d97d82b95804e76e3042ce88787773486a0d95e00737650667c2bee844` | 42 of 42 active |
| English 18M REFL | `languages/english_18m_refl_2026-10/model.safetensors` | 111,728,528 | `c99f4a179cffa070cdeb3ffc1ae950ebf16038c762c84dd79997868ac0a806c8` | 42 of 42 active |

## Full voice-cloning checkpoints only

All supported models **must** come from `kyutai/pocket-tts`; there is no
reduced-checkpoint mode, even for tests or CI. Checkpoint SHA-256 and all
42/42 active Mimi encoder tensors are checked before accepting a model.
`xtask prefetch` does not implement `--without-voice-cloning`.

All bundled tokenizer locations were normalized to the **full gated**
`kyutai/pocket-tts` repository at revision
`db04f0e653473e5c7333c69686ac92b7460b0c1d`.
The upstream configuration's obsolete reduced paths are not downloaded.
The full checkpoint and tokenizer are cached outside Git and require
authorized HF access; absence of authorization must fail.

**Scope:** Full models and voice-cloning encoders were verified. The native Rust/XN engine passed real reference-audio cloning; Candle output remains subject to independent intelligibility testing. Actual training runs are not yet verified.
