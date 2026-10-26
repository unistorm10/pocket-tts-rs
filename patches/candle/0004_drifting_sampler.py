#!/usr/bin/env python3
"""Adapt the pinned Candle FlowLM to Kyutai's zero-timestep drifting head.

The released 7M and 18M REFL models have flow.type=drifting, so they lack
time_embed weights and use the network output as their one-step sample.
Leave legacy LSD generation intact. No synthetic tensors or fake weights.
"""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit("usage: 0004_drifting_sampler.py /path/to/pinned-candle")
root = Path(sys.argv[1]) / "crates/pocket-tts/src"

def patch(path, old, new):
    p = root / path
    s = p.read_text()
    assert s.count(old) == 1, f"pinned upstream changed: {path}: {old[:55]}"
    p.write_text(s.replace(old, new))

patch("config.rs",
      """pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
}""",
      """pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
    #[serde(rename = "type", default)]
    pub flow_type: Option<String>,
}""")
patch("modules/mlp.rs",
      """pub struct SimpleMLPAdaLN {
    time_embeds: Vec<TimestepEmbedder>,""",
      """pub struct SimpleMLPAdaLN {
    model_channels: usize,
    time_embeds: Vec<TimestepEmbedder>,""")
patch("modules/mlp.rs",
      """        Ok(Self {
            time_embeds,
            cond_embed,
            input_proj,""",
      """        Ok(Self {
            model_channels,
            time_embeds,
            cond_embed,
            input_proj,""")
patch("modules/mlp.rs",
      """    ) -> Result<Tensor> {
        let mut embeddings = Vec::with_capacity(num_steps);
        for i in 0..num_steps {""",
      """    ) -> Result<Tensor> {
        if self.num_time_conds == 0 {
            // Drifting and pointwise heads use conditioning only, no time embeds.
            return Ok(Tensor::zeros((1, self.model_channels), dtype, device)?);
        }
        let mut embeddings = Vec::with_capacity(num_steps);
        for i in 0..num_steps {""")
patch("models/flow_lm.rs",
      """    pub noise_clamp: Option<f32>,
}""",
      """    pub noise_clamp: Option<f32>,
    pub flow_type: String,
}""")
patch("models/flow_lm.rs",
      """            noise_clamp: None, // Default to no clamp
        })""",
      """            noise_clamp: None, // Default to no clamp
            flow_type: "lsd".to_owned(),
        })""")
patch("models/flow_lm.rs",
      """        let next_latent = lsd_decode(&self.flow_net, &modulations, &noise)?;""",
      """        let next_latent = match self.flow_type.as_str() {
            "drifting" => self.flow_net.forward_step_cached(&noise, &modulations[0])?,
            "pointwise" => {
                let direction = self.flow_net.forward_step_cached(&noise, &modulations[0])?;
                (noise + direction)?
            }
            "lsd" => lsd_decode(&self.flow_net, &modulations, &noise)?,
            other => return Err(candle_core::Error::Msg(format!("unsupported flow sampler: {other}"))),
        };""")
patch("tts_model.rs",
      """        let hidden_dim = dim * config.flow_lm.transformer.hidden_scale;

        // SimpleMLPAdaLN::new""",
      """        let hidden_dim = dim * config.flow_lm.transformer.hidden_scale;
        let flow_type = config.flow_lm.flow.flow_type.as_deref().unwrap_or("lsd");
        let time_conditions = match flow_type {
            "lsd" => 2,
            "drifting" | "pointwise" => 0,
            other => anyhow::bail!("unsupported flow sampler: {other}"),
        };

        // SimpleMLPAdaLN::new""")
patch("tts_model.rs",
      """            2,                         // num_time_conds (s and t)
            config.flow_lm.transformer.max_period as f32,""",
      """            time_conditions,           // 0 for drifting, 2 for LSD
            config.flow_lm.transformer.max_period as f32,""")
patch("tts_model.rs",
      """        flow_lm.noise_clamp = noise_clamp;""",
      """        flow_lm.noise_clamp = noise_clamp;
        flow_lm.flow_type = flow_type.to_owned();""")
print("Adapted existing Candle FlowLM to zero-time-condition drifting sampler")
