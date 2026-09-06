//! Configuration models mirroring `pocket_tts/utils/config.py`.
//!
//! Field names and nesting intentionally match the Python `pydantic` models
//! 1:1 so that YAML config files from the original project parse unchanged.
//! `deny_unknown_fields` mirrors `ConfigDict(extra="forbid")` in `StrictModel`.

use serde::Deserialize;
use std::path::Path;

/// Mirrors `pocket_tts.utils.config.FlowConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    pub dim: i64,
    pub depth: i64,
    #[serde(default = "default_flow_type")]
    pub r#type: String,
}

fn default_flow_type() -> String {
    "lsd".to_string()
}

/// Mirrors `pocket_tts.utils.config.FlowLMTransformerConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLMTransformerConfig {
    pub hidden_scale: i64,
    pub max_period: i64,
    pub d_model: i64,
    pub num_heads: i64,
    pub num_layers: i64,
}

/// Mirrors `pocket_tts.utils.config.LookupTable`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupTable {
    pub dim: i64,
    pub n_bins: i64,
    pub tokenizer: String,
    pub tokenizer_path: String,
}

/// Mirrors `pocket_tts.utils.config.FlowLMConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLMConfig {
    pub dtype: String,
    pub flow: FlowConfig,
    pub transformer: FlowLMTransformerConfig,
    pub lookup_table: LookupTable,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub insert_bos_before_voice: bool,
}

/// Mirrors `pocket_tts.utils.config.SEANetConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SEANetConfig {
    pub dimension: i64,
    pub channels: i64,
    pub n_filters: i64,
    pub n_residual_layers: i64,
    pub ratios: Vec<i64>,
    pub kernel_size: i64,
    pub residual_kernel_size: i64,
    pub last_kernel_size: i64,
    pub dilation_base: i64,
    pub pad_mode: String,
    pub compress: i64,
}

/// Mirrors `pocket_tts.utils.config.MimiTransformerConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiTransformerConfig {
    pub d_model: i64,
    pub input_dimension: i64,
    pub output_dimensions: Vec<i64>,
    pub num_heads: i64,
    pub num_layers: i64,
    pub layer_scale: f64,
    pub context: i64,
    #[serde(default = "default_max_period")]
    pub max_period: f64,
    pub dim_feedforward: i64,
}

fn default_max_period() -> f64 {
    10000.0
}

/// Mirrors `pocket_tts.utils.config.QuantizerConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizerConfig {
    pub dimension: i64,
    pub output_dimension: i64,
}

/// Mirrors `pocket_tts.utils.config.MimiConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiConfig {
    pub dtype: String,
    pub sample_rate: i64,
    pub channels: i64,
    pub frame_rate: f64,
    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer: QuantizerConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub inner_dim: Option<i64>,
    #[serde(default)]
    pub outer_dim: Option<i64>,
}

/// Mirrors `pocket_tts.utils.config.Config`, the top-level YAML document.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub weights_path_without_voice_cloning: Option<String>,
    #[serde(default)]
    pub pad_with_spaces_for_short_inputs: bool,
    #[serde(default)]
    pub remove_semicolons: bool,
    #[serde(default = "default_true")]
    pub append_terminal_punctuation: bool,
    #[serde(default)]
    pub model_recommended_frames_after_eos: Option<i64>,
    #[serde(default = "default_temperature")]
    pub default_temperature: f64,
}

fn default_true() -> bool {
    true
}

fn default_temperature() -> f64 {
    0.7
}

impl Config {
    /// Parse a `Config` from a YAML string. Mirrors
    /// `pocket_tts.utils.config.load_config`'s `yaml.safe_load` + pydantic
    /// validation step (the download step is handled separately by
    /// `crate::download::download_if_necessary`).
    pub fn from_yaml_str(yaml: &str) -> anyhow::Result<Self> {
        let config: Config = serde_yaml::from_str(yaml)?;
        Ok(config)
    }

    pub fn from_yaml_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_yaml_str(&text)
    }
}
