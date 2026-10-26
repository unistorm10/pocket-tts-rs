#!/usr/bin/env python3
"""Enable independent Mimi decoder topology in pinned community Candle port.

Official v4 compact models use a 512-wide encoder with a distinct 128-wide
SEANet/transformer decoder. Reuse the existing modules with different configs;
when no decoder-specific config exists retain the legacy symmetric behavior.
"""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit("usage: 0006_asymmetric_mimi.py /path/to/pinned-candle")
root = Path(sys.argv[1]) / "crates/pocket-tts/src"
cfg = root / "config.rs"
s = cfg.read_text()
needle = """    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,"""
replacement = """    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,
    #[serde(default)]
    pub decoder_seanet: Option<SEANetConfig>,
    #[serde(default)]
    pub decoder_transformer: Option<MimiTransformerConfig>,"""
assert s.count(needle) == 1, "pinned Mimi config changed"
cfg.write_text(s.replace(needle, replacement))

path = root / "tts_model.rs"
s = path.read_text()
start = s.index("        let decoder = SEANetDecoder::new(")
end = s.index("        let mimi_tr_cfg", start)
old = s[start:end]
assert old.count("seanet_cfg.") >= 10, "pinned decoder config block changed"
new = """        let decoder_cfg = config.mimi.decoder_seanet.as_ref().unwrap_or(seanet_cfg);
""" + old.replace("seanet_cfg.", "decoder_cfg.")
s = s[:start] + new + s[end:]
start = s.index("        let decoder_transformer = ProjectedTransformer::new(")
end = s.index("        // Calculate encoder frame rate", start)
old = s[start:end]
assert old.count("mimi_tr_cfg.") >= 8, "pinned Mimi decoder transformer changed"
new = """        let decoder_tr_cfg = config
            .mimi
            .decoder_transformer
            .as_ref()
            .unwrap_or(mimi_tr_cfg);
""" + old.replace("mimi_tr_cfg.", "decoder_tr_cfg.")
s = s[:start] + new + s[end:]
path.write_text(s)
print("Reused existing SEANet/transformer with compact independent Mimi decoder topology")
