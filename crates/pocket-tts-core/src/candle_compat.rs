//! Audited feature comparison against the pinned community Candle model.
//!
//! Pinned candidate: stepupgaming/pocket-tts at
//! dbf78c816cd83c89f11dbeb9d87290ad0a3dccd0.
//! The candidate constructor and text preprocessing were reviewed at that
//! revision; these gates prevent old-model logic silently interpreting 2026
//! configs. This does not pretend to generate speech.

use crate::config::Config;

/// An architectural difference needing a tested upstream Candle adaptation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleGap {
    pub code: &'static str,
    pub detail: &'static str,
}

/// Report of verified model-shape compatibility, distinct from runtime
/// correctness or speech quality.
#[derive(Debug, Clone, Default)]
pub struct CandleReuseReport {
    pub gaps: Vec<CandleGap>,
}

impl CandleReuseReport {
    pub fn is_compatible(&self) -> bool {
        self.gaps.is_empty()
    }

    pub fn has_gap(&self, code: &str) -> bool {
        self.gaps.iter().any(|gap| gap.code == code)
    }
}

/// Compare a Kyutai v4.0.1 config with the *implemented* legacy Candle
/// constructor, not just the permissive upstream YAML parser.
pub fn analyze_legacy_candle(config: &Config) -> CandleReuseReport {
    let mut gaps = Vec::new();
    let mut add = |code, detail| gaps.push(CandleGap { code, detail });

    if config.flow_lm.flow.r#type != "lsd" {
        add(
            "flow_sampler",
            "Compact models use drifting rather than the legacy LSD decoder",
        );
    }
    if config.flow_lm.insert_bos_before_voice {
        add(
            "voice_bos",
            "Full v4.0.1 models require insert_bos_before_voice handling",
        );
    }
    if config.flow_lm.z_dim.is_some() {
        add(
            "flow_latent_projection",
            "The model has a distinct z_dim projection not built by the legacy constructor",
        );
    }
    let transformer = &config.flow_lm.transformer;
    if transformer
        .num_kv_heads
        .is_some_and(|kv| kv != transformer.num_heads)
    {
        add(
            "grouped_query_attention",
            "The model requires distinct grouped KV heads in flow attention",
        );
    }
    if transformer
        .attn_dim
        .is_some_and(|dim| dim != transformer.d_model)
    {
        add(
            "attention_projection_width",
            "The model has an explicit attention projection width",
        );
    }
    if transformer
        .dim_feedforward
        .is_some_and(|dim| dim != transformer.d_model * transformer.hidden_scale)
    {
        add(
            "feedforward_width",
            "The model overrides the default feedforward projection width",
        );
    }
    if config.flow_lm.lookup_table.rank.is_some() {
        add(
            "low_rank_text_embedding",
            "The tokenizer lookup uses a low-rank factorization",
        );
    }
    if config.mimi.decoder_seanet.is_some() {
        add(
            "asymmetric_seanet",
            "Mimi decoder uses a separate compact SEANet, not the encoder shape",
        );
    }
    if config.mimi.decoder_transformer.is_some() {
        add(
            "asymmetric_mimi_transformer",
            "Mimi decoder transformer uses an independent shape",
        );
    }
    if !config.replace_characters.is_empty() {
        add(
            "text_normalization",
            "The legacy text preprocessor does not apply the v4 character substitution map",
        );
    }
    if config
        .mimi
        .outer_dim
        .is_some_and(|outer| outer != config.mimi.quantizer.output_dimension)
    {
        add(
            "mimi_outer_projection",
            "Mimi outer dimension differs from the expected quantizer output dimension",
        );
    }
    CandleReuseReport { gaps }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(yaml: &str) -> Config {
        Config::from_yaml_str(yaml).expect("official YAML fixture must parse")
    }

    #[test]
    fn full_english_requires_voice_prefix_and_text_handling() {
        let report =
            analyze_legacy_candle(&config(include_str!("../tests/fixtures/english-v4.yaml")));
        assert!(report.has_gap("voice_bos"));
        assert!(report.has_gap("text_normalization"));
        assert!(!report.has_gap("asymmetric_seanet"));
        assert!(!report.has_gap("grouped_query_attention"));
    }

    #[test]
    fn october_7m_needs_drifting_gqa_and_decoder_reuse() {
        let report = analyze_legacy_candle(&config(include_str!(
            "../tests/fixtures/english-7m-v4.yaml"
        )));
        for code in [
            "flow_sampler",
            "voice_bos",
            "flow_latent_projection",
            "grouped_query_attention",
            "attention_projection_width",
            "low_rank_text_embedding",
            "asymmetric_seanet",
            "asymmetric_mimi_transformer",
            "text_normalization",
        ] {
            assert!(report.has_gap(code), "missing required gate: {code}");
        }
    }

    #[test]
    fn october_18m_refl_has_no_gqa_but_has_independent_mimi_decoder() {
        let report = analyze_legacy_candle(&config(include_str!(
            "../tests/fixtures/english-18m-refl-v4.yaml"
        )));
        assert!(report.has_gap("flow_sampler"));
        assert!(report.has_gap("asymmetric_seanet"));
        assert!(report.has_gap("asymmetric_mimi_transformer"));
        assert!(!report.has_gap("grouped_query_attention"));
        assert!(!report.has_gap("low_rank_text_embedding"));
    }
}
