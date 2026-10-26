from pathlib import Path
import sys
p=Path(sys.argv[1])/"crates/pocket-tts/src/tts_model.rs"
s=p.read_text()
old="let mimi_out_dim = config.mimi.quantizer.output_dimension;"
new="let mimi_out_dim = config.mimi.quantizer.dimension;"
assert s.count(old)==1, "Pinned upstream projection constructor changed"
p.write_text(s.replace(old,new))
print("Aligned speaker projection with encoder latent dimensions")
