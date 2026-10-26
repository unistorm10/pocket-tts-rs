#!/usr/bin/env python3
"""Reuse Kyutai v4's learned BOS-before-voice conditioning in Candle.

Kyutai v4.0.1 TTSModel.get_state_for_audio_prompt explicitly prepends
flow_lm.bos_before_voice when insert_bos_before_voice is true.
The historical Candle port silently omitted this learned prefix.
Keep all legacy configurations working when the flag is absent/false.
"""
import sys
from pathlib import Path
if len(sys.argv) != 2:
    raise SystemExit("usage: 0009_voice_bos_prefix.py /path/to/pinned-candle")
root = Path(sys.argv[1]) / "crates/pocket-tts/src"

def patch(path, old, new):
    file = root / path
    source = file.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"pinned upstream {path} shifted; {count} copies of {old[:60]!r}")
    file.write_text(source.replace(old, new, 1))

patch("config.rs",
    """    pub lookup_table: LookupTableConfig,
    #[serde(default)]
    pub z_dim:""",
    """    pub lookup_table: LookupTableConfig,
    #[serde(default)]
    pub insert_bos_before_voice: bool,
    #[serde(default)]
    pub z_dim:""")

patch("models/flow_lm.rs",
    """    pub bos_emb: Tensor,
    pub emb_mean: Tensor,""",
    """    pub bos_emb: Tensor,
    /// Full voice-cloning checkpoint's learned prefix, shape [1,1,d_model].
    pub bos_before_voice: Option<Tensor>,
    pub emb_mean: Tensor,""")

patch("models/flow_lm.rs",
    """            bos_emb,
            emb_mean,""",
    """            bos_emb,
            bos_before_voice: None,
            emb_mean,""")

patch("tts_model.rs",
    """        flow_lm.noise_clamp = noise_clamp;""",
    """        flow_lm.noise_clamp = noise_clamp;
        if config.flow_lm.insert_bos_before_voice {
            // Load the real, trained parameter. Never synthesize zero-prefixes.
            flow_lm.bos_before_voice = Some(
                vb.pp("flow_lm").get((1, 1, dim), "bos_before_voice")?
            );
        }""")

patch("tts_model.rs",
    """        let input = Tensor::cat(&[conditioning, &text_embeddings], 1)?;""",
    """        let audio_conditioning = match &self.flow_lm.bos_before_voice {
            Some(prefix) => Tensor::cat(&[prefix, conditioning], 1)?,
            None => conditioning.clone(),
        };
        let input = Tensor::cat(&[&audio_conditioning, &text_embeddings], 1)?;""")

patch("tts_model.rs",
    """        let increment_by = conditioning.dims()[1];""",
    """        let increment_by = audio_conditioning.dims()[1];""")

print("Loaded v4 learned BOS-before-voice weight and prepended it in voice-state prompting")
