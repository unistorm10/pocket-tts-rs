#!/usr/bin/env python3
"""Add v4 grouped-query attention to the pinned community Candle implementation.

The 7M checkpoint has d_model=224, num_heads=8, attn_dim=512 and
num_kv_heads=2. Thus Q=512, KV=128 each, packed in_proj=768.
This alters real attention arithmetic and KV caches; no fake padding/tensors.
Legacy MHA call signatures delegate to the generalized implementation.
"""
from pathlib import Path
import sys

assert len(sys.argv) == 2, "usage: 0008_grouped_query_attention.py UPSTREAM_DIR"
root = Path(sys.argv[1]) / "crates/pocket-tts/src"

def patch(path, old, new, count=1):
    p = root / path
    s = p.read_text()
    assert s.count(old) == count, f"upstream changed ({path}) {old[:90]!r}: observed {s.count(old)}"
    p.write_text(s.replace(old, new, count))

patch("config.rs",
    """    pub num_heads: usize,
    pub num_layers: usize,
}""",
    """    pub num_heads: usize,
    pub num_layers: usize,
    #[serde(default)]
    pub num_kv_heads: Option<usize>,
    #[serde(default)]
    pub attn_dim: Option<usize>,
}""")

patch("modules/attention.rs",
    """    embed_dim: usize,
    num_heads: usize,
    rope: RotaryEmbedding,""",
    """    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    attn_dim: usize,
    rope: RotaryEmbedding,""")

patch("modules/attention.rs",
    """        // out_dim = embed_dim + 2 * kv_dim (GQA/MHA logic in original)""",
    """        Self::new_with_gqa(embed_dim, num_heads, num_heads, embed_dim, rope, context, name, vb)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_gqa(
        embed_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        attn_dim: usize,
        rope: RotaryEmbedding,
        context: Option<usize>,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0
            || num_heads % num_kv_heads != 0
            || attn_dim % num_heads != 0
        {
            return Err(candle_core::Error::Msg(
                "invalid GQA head count or attention width".to_owned()
            ));
        }
        // v4 packed weights use Q + K + V, where the KV heads may be fewer.
        let kv_dim = (attn_dim / num_heads) * num_kv_heads;
        // out_dim = embed_dim + 2 * kv_dim (GQA/MHA logic in original)""")

patch("modules/attention.rs",
    """let in_proj = candle_nn::linear_no_bias(embed_dim, 3 * embed_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("out_proj"))?;""",
    """let in_proj = candle_nn::linear_no_bias(embed_dim, attn_dim + 2*kv_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(attn_dim, embed_dim, vb.pp("out_proj"))?;""")

patch("modules/attention.rs",
    """            embed_dim,
            num_heads,
            rope,""",
    """            embed_dim,
            num_heads,
            num_kv_heads,
            attn_dim,
            rope,""")

patch("modules/attention.rs",
    """    pub fn init_state(
        &self,""",
    """    /// Repeat each KV head for its corresponding group of query heads.
    /// Keep compact KV in streaming caches; expand only the SDPA operands.
    fn expanded_kv(&self, kv: &Tensor) -> Result<Tensor> {
        let repeats = self.num_heads / self.num_kv_heads;
        if repeats == 1 {
            return Ok(kv.clone());
        }
        let heads: Vec<Tensor> = (0..self.num_heads)
            .map(|i| kv.narrow(1, i / repeats, 1))
            .collect::<Result<_>>()?;
        Tensor::cat(&heads, 1)
    }

    pub fn init_state(
        &self,""")

patch("modules/attention.rs",
    """let dim_per_head = self.embed_dim / self.num_heads;""",
    """let dim_per_head = self.attn_dim / self.num_heads;""")

# Both key and value buffers use compact num_kv_heads, rather than num_heads.
patch("modules/attention.rs",
    """(batch_size, self.num_heads, cap, dim_per_head)""",
    """(batch_size, self.num_kv_heads, cap, dim_per_head)""", 2)

patch("modules/attention.rs",
    """let d = self.embed_dim / self.num_heads;""",
    """let d = self.attn_dim / self.num_heads;""")

patch("modules/attention.rs",
    """        // Reshape to (b, t, 3, h, d)
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        let mut q = packed.narrow(2, 0, 1)?.squeeze(2)?; // (b, t, h, d)
        let mut k = packed.narrow(2, 1, 1)?.squeeze(2)?; // (b, t, h, d)
        let mut v = packed.narrow(2, 2, 1)?.squeeze(2)?; // (b, t, h, d)""",
    """        // Split packed Q=attn_dim; K,V each = kv_heads * head_dim.
        let kv_dim = self.num_kv_heads * d;
        let mut q = projected.narrow(2, 0, self.attn_dim)?
            .reshape((b, t, self.num_heads, d))?;
        let mut k = projected.narrow(2, self.attn_dim, kv_dim)?
            .reshape((b, t, self.num_kv_heads, d))?;
        let mut v = projected.narrow(2, self.attn_dim + kv_dim, kv_dim)?
            .reshape((b, t, self.num_kv_heads, d))?;""")

patch("modules/attention.rs",
    """(b, self.num_heads, initial_cap, d)""",
    """(b, self.num_kv_heads, initial_cap, d)""", 2)
patch("modules/attention.rs",
    """(b, self.num_heads, window_size - cap, d)""",
    """(b, self.num_kv_heads, window_size - cap, d)""")
patch("modules/attention.rs",
    """(b, self.num_heads, new_cap - cap, d)""",
    """(b, self.num_kv_heads, new_cap - cap, d)""")

patch("modules/attention.rs",
    """            k_chunks.push(k.clone());
            v_chunks.push(v.clone());

            let scale""",
    """            k_chunks.push(k.clone());
            v_chunks.push(v.clone());
            let k_attention: Vec<Tensor> = k_chunks.iter()
                .map(|chunk| self.expanded_kv(chunk)).collect::<Result<_>>()?;
            let v_attention: Vec<Tensor> = v_chunks.iter()
                .map(|chunk| self.expanded_kv(chunk)).collect::<Result<_>>()?;

            let scale""")

patch("modules/attention.rs",
    """            if k_chunks.len() == 1 {
                crate::modules::sdpa::sdpa(
                    &q,
                    &k_chunks[0],
                    &v_chunks[0],""",
    """            if k_attention.len() == 1 {
                crate::modules::sdpa::sdpa(
                    &q,
                    &k_attention[0],
                    &v_attention[0],""")

patch("modules/attention.rs",
    """                    &k_chunks,
                    &v_chunks,""",
    """                    &k_attention,
                    &v_attention,""")

patch("modules/attention.rs",
    """            crate::modules::sdpa::sdpa(&q, &kc, &vc, scale, true, self.context)?""",
    """            crate::modules::sdpa::sdpa(
                &q, &self.expanded_kv(&kc)?, &self.expanded_kv(&vc)?,
                scale, true, self.context
            )?""")

patch("modules/attention.rs",
    """reshape((b, t, self.embed_dim))?;""",
    """reshape((b, t, self.attn_dim))?;""")

# Preserve old constructor signatures for independent Mimi transformers;
# only flow_lm's constructor enables GQA and projection-width override.
patch("models/transformer.rs",
    """        let self_attn = StreamingMultiheadAttention::new(
            d_model,""",
    """        Self::new_with_attention(
            d_model, num_heads, num_heads, d_model, dim_feedforward,
            context, rope, layer_scale, _attention_kind, name, vb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_attention(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        attn_dim: usize,
        dim_feedforward: usize,
        context: Option<usize>,
        rope: RotaryEmbedding,
        layer_scale: Option<f32>,
        _attention_kind: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = StreamingMultiheadAttention::new_with_gqa(
            d_model,
            num_heads,
            num_kv_heads,
            attn_dim,""")

patch("models/transformer.rs",
    """        let rope = RotaryEmbedding::new(max_period, d_model / num_heads, vb.device())?;""",
    """        Self::new_with_attention(
            d_model, num_heads, num_heads, d_model, num_layers, layer_scale,
            dim_feedforward, context, max_period, kind, name, vb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_attention(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        attn_dim: usize,
        num_layers: usize,
        layer_scale: Option<f32>,
        dim_feedforward: usize,
        context: Option<usize>,
        max_period: f32,
        kind: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rope = RotaryEmbedding::new(max_period, attn_dim / num_heads, vb.device())?;""")

patch("models/transformer.rs",
    """            layers.push(StreamingTransformerLayer::new(
                d_model,
                num_heads,
                dim_feedforward,""",
    """            layers.push(StreamingTransformerLayer::new_with_attention(
                d_model,
                num_heads,
                num_kv_heads,
                attn_dim,
                dim_feedforward,""")

patch("tts_model.rs",
    """        let transformer = StreamingTransformer::new(
            dim,
            config.flow_lm.transformer.num_heads,
            config.flow_lm.transformer.num_layers,""",
    """        let transformer = StreamingTransformer::new_with_attention(
            dim,
            config.flow_lm.transformer.num_heads,
            config.flow_lm.transformer.num_kv_heads.unwrap_or(
                config.flow_lm.transformer.num_heads
            ),
            config.flow_lm.transformer.attn_dim.unwrap_or(dim),
            config.flow_lm.transformer.num_layers,""")

patch("models/transformer.rs",
    """            num_kv_heads,
            attn_dim,
            num_heads,
            rope,""",
    """            num_kv_heads,
            attn_dim,
            rope,""")

print("Applied real GQA: Q=512, KV=128×2 packed projection, compact streaming KV cache, and head expansion in SDPA")
