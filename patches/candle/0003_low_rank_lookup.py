#!/usr/bin/env python3
"""Add Kyutai v4 low-rank text embeddings to the pinned existing Candle port.

The official compact 7M config uses rank=64, dim=224:
embed.weight [4001,64], proj.weight [224,64].
Keep the old LUTConditioner::new API and all unranked behavior.
"""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit("usage: 0003_low_rank_lookup.py /path/to/pinned-candle")
root = Path(sys.argv[1]) / "crates/pocket-tts/src"
def patch(path, old, new, count=1):
    p = root / path
    s = p.read_text()
    assert s.count(old) == count, f"upstream source changed: {path}: {old[:50]}"
    p.write_text(s.replace(old, new, 1))

patch("config.rs",
      "    pub tokenizer_path: String,\n}",
      "    pub tokenizer_path: String,\n    #[serde(default)]\n    pub rank: Option<usize>,\n}")
patch("conditioners/text.rs",
      "use candle_nn::{Embedding, Module, VarBuilder};",
      "use candle_nn::{Embedding, Linear, Module, VarBuilder};")
patch("conditioners/text.rs",
      "    embed: Embedding,\n}",
      "    embed: Embedding,\n    proj: Option<Linear>,\n    output_dim: usize,\n}")
patch("conditioners/text.rs",
      """        _output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Load SentencePiece model using tokenizers crate""",
      """        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_rank(n_bins, tokenizer_path, dim, output_dim, None, vb)
    }

    pub fn new_with_rank(
        n_bins: usize,
        tokenizer_path: &Path,
        dim: usize,
        _output_dim: usize,
        rank: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Load SentencePiece model using tokenizers crate""")
patch("conditioners/text.rs",
      """        let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            embed,
        })""",
      """        let embed = candle_nn::embedding(n_bins + 1, rank.unwrap_or(dim), vb.pp("embed"))?;
        let proj = rank
            .map(|r| candle_nn::linear_no_bias(r, dim, vb.pp("proj")))
            .transpose()?;

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            embed,
            proj,
            output_dim: dim,
        })""", count=2)
patch("conditioners/text.rs",
      """        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            embed,
        })
    }

    pub fn prepare""",
      """        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            embed,
            proj: None,
            output_dim: dim,
        })
    }

    pub fn prepare""")
patch("conditioners/text.rs",
      """            let embed_dim = self.embed.embeddings().dims()[1];""",
      """            let embed_dim = self.output_dim;""")
patch("conditioners/text.rs",
      """        Ok(self.embed.forward(tokens)?)
    }

    /// Count tokens""",
      """        let embedded = self.embed.forward(tokens)?;
        match &self.proj {
            Some(proj) => Ok(proj.forward(&embedded)?),
            None => Ok(embedded),
        }
    }

    /// Count tokens""")
patch("tts_model.rs",
      """            let conditioner = LUTConditioner::new(
                config.flow_lm.lookup_table.n_bins,
                &tokenizer_path,
                config.flow_lm.lookup_table.dim,
                config.flow_lm.transformer.d_model,
                vb.pp("flow_lm.conditioner"),
            )?;""",
      """            let conditioner = LUTConditioner::new_with_rank(
                config.flow_lm.lookup_table.n_bins,
                &tokenizer_path,
                config.flow_lm.lookup_table.dim,
                config.flow_lm.transformer.d_model,
                config.flow_lm.lookup_table.rank,
                vb.pp("flow_lm.conditioner"),
            )?;""")
print("Enabled v4 rank-aware Candle text lookup without changing the legacy API")
