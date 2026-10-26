# Verified Pocket TTS checkpoints (October 26, 2026)

The **primary** model source is [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), with full voice-cloning support. All full checkpoint references below are pinned to revision `db04f0e653473e5c7333c69686ac92b7460b0c1d`.

All three full models were fetched on the `elitedesk-mini` self-hosted Linux runner on October 26, 2026 and verified by direct safetensors inspection and a real Candle tensor computation in [Actions run #40358200381](https://github.com/unistorm10/pocket-tts-rs/actions/runs/40358200381).

| Config variant | Model path (under `kyutai/pocket-tts`) | Bytes | SHA-256 | `mimi.encoder` |
| --- | --- | ---: | --- | --- |
| Default English | `languages/english/model.safetensors` | 219,029,196 | `bc96b82436c5ec00d3118ef3b01bc3358423e1766ecae3dca0aeea84789e7d74` | 42 of 42 active |
| English 7M | `languages/english_7m_2026-10/model.safetensors` | 66,482,920 | `1200a5d97d82b95804e76e3042ce88787773486a0d95e00737650667c2bee844` | 42 of 42 active |
| English 18M REFL | `languages/english_18m_refl_2026-10/model.safetensors` | 111,728,528 | `c99f4a179cffa070cdeb3ffc1ae950ebf16038c762c84dd79997868ac0a806c8` | 42 of 42 active |

The accompanying tokenizers are the official models named in Kyutai's v4.0.1 YAML configs. Their paths currently point to `kyutai/pocket-tts-without-voice-cloning` because the tokenizers are shared and **not** an encoder-disabled model checkpoint. English uses JSON tokenizers, compact 7M and 18M REFL use SentencePiece.

## Public reduced checkpoints

Only for non-cloning tests, reference `kyutai/pocket-tts-without-voice-cloning`, revision `a7056f505a4a8ca72d8b5fb1440071d6f1a35739`.

| Variant | SHA-256 | Voice encoder |
| --- | --- | --- |
| Default English | `4c56cbd171c63e8a26ea61f5d48b8a0cd8f3c51f4652c0ab9a19bfcced8c931d` | 42 of 42 zeroed |
| English 7M | `aea704a2989277c1a700abc16fb1442cb6baa2fa9ad32251ef7a5f91e4eb7d58` | 42 of 42 zeroed |

The reducer for English 18M REFL has not been hash-verified and is intentionally not claimed as such.

The `xtask prefetch` command validates the checksum against this known list, refuses full-checkpoint mode with inactive encoders, and never prints or writes `HF_TOKEN`. The full models reside in the runner's Hugging Face cache, **not** Git or GitHub Actions artifacts.

**Scope:** Checkpoint availability, encoder activation, and native Candle weight loading are established. No full synthesis, reference-audio cloning, or training result is implied. Reproducible synthesis remains an acceptance gate.
