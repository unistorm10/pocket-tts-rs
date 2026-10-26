#!/usr/bin/env python3
"""Support official Kyutai v4 z_dim projected FlowLM conditioning.

The 2026 compact models project transformer hidden states to z_dim=64 for
both EOS prediction and their one-step flow head. Preserve the old
FlowLMModel::new API for models without z_dim.
"""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit("usage: 0005_z_projection.py /path/to/pinned-candle")
root = Path(sys.argv[1]) / "crates/pocket-tts/src"

def patch(path, old, new):
    p = root / path
    s = p.read_text()
    assert s.count(old) == 1, f"pinned upstream changed: {path} {old[:55]}"
    p.write_text(s.replace(old, new))

patch("config.rs",
      """    pub lookup_table: LookupTableConfig,
    #[serde(default)]
    pub weights_path:""",
      """    pub lookup_table: LookupTableConfig,
    #[serde(default)]
    pub z_dim: Option<usize>,
    #[serde(default)]
    pub weights_path:""")
patch("models/flow_lm.rs",
      """    pub out_eos: Linear,
    pub bos_emb: Tensor,""",
      """    pub out_eos: Linear,
    pub z_proj: Option<Linear>,
    pub bos_emb: Tensor,""")
patch("models/flow_lm.rs",
      """        vb: VarBuilder,
    ) -> Result<Self> {
        let input_linear = candle_nn::linear_no_bias(ldim, dim, vb.pp("input_linear"))?;""",
      """        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_z_dim(flow_net, transformer, ldim, dim, None, vb)
    }

    pub fn new_with_z_dim(
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        ldim: usize,
        dim: usize,
        z_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_linear = candle_nn::linear_no_bias(ldim, dim, vb.pp("input_linear"))?;""")
patch("models/flow_lm.rs",
      """        let out_eos = candle_nn::linear(dim, 1, vb.pp("out_eos"))?;""",
      """        let z_proj = z_dim
            .map(|z| candle_nn::linear_no_bias(dim, z, vb.pp("z_proj")))
            .transpose()?;
        let out_eos = candle_nn::linear(z_dim.unwrap_or(dim), 1, vb.pp("out_eos"))?;""")
patch("models/flow_lm.rs",
      """            out_eos,
            bos_emb,
            emb_mean,""",
      """            out_eos,
            z_proj,
            bos_emb,
            emb_mean,""")
patch("models/flow_lm.rs",
      """        let eos_score = self
            .out_eos
            .forward(&last_frame)?""",
      """        let projected = match &self.z_proj {
            Some(proj) => proj.forward(&last_frame)?,
            None => last_frame.clone(),
        };
        let eos_score = self
            .out_eos
            .forward(&projected)?""")
patch("models/flow_lm.rs",
      """        let c_emb = self.flow_net.embed_condition(&last_frame)?;""",
      """        let c_emb = self.flow_net.embed_condition(&projected)?;""")
patch("tts_model.rs",
      """            dim,                       // cond_channels (conditioning from transformer)""",
      """            config.flow_lm.z_dim.unwrap_or(dim), // projected head conditioning channels""")
patch("tts_model.rs",
      """        let mut flow_lm = FlowLMModel::new(flow_net, transformer, ldim, dim, vb.pp("flow_lm"))?;""",
      """        let mut flow_lm = FlowLMModel::new_with_z_dim(
            flow_net, transformer, ldim, dim, config.flow_lm.z_dim, vb.pp("flow_lm"),
        )?;""")
print("Enabled real v4 z-projection in reused Candle FlowLM with legacy fallback")
