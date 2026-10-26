#!/usr/bin/env python3
"""Fix v4 full English speaker projection width in pinned reused Candle code.

Must be applied after 0001_mimi_downsample.py. No weight rewriting.
"""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit("usage: 0002_speaker_projection.py UPSTREAM_CHECKOUT")
path = Path(sys.argv[1]) / "crates/pocket-tts/src/tts_model.rs"
source = path.read_text()
old = """        // Load speaker projection weight - uses mimi output dimension, not internal ldim
        let mimi_out_dim = config.mimi.quantizer.output_dimension;
        let speaker_proj_weight = vb.get((dim, mimi_out_dim), "flow_lm.speaker_proj_weight")?;"""
new = """        // The voice projection consumes the *downsampled encoder latent*,
        // not the 512-wide decoder-side quantizer output. The legacy code
        // accidentally equated these dimensions.
        let mimi_encoded_dim = config.mimi.quantizer.dimension;
        let speaker_proj_weight = vb.get((dim, mimi_encoded_dim), "flow_lm.speaker_proj_weight")?;"""
assert source.count(old) == 1, "upstream model constructor changed; re-audit"
path.write_text(source.replace(old,new))
print("Fixed v4 voice-conditioning 1024x32 projection width")
