use pocket_tts_core::config::Config;

/// Phase 0 acceptance test: `pocket-tts-core::config` must parse the real
/// `pocket_tts/config/english.yaml` from kyutai-labs/pocket-tts unchanged.
#[test]
fn config_parses_english_yaml() {
    let yaml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/english.yaml");
    let config = Config::from_yaml_file(std::path::Path::new(yaml_path))
        .expect("english.yaml should parse into Config");

    assert_eq!(config.flow_lm.transformer.d_model, 1024);
    assert_eq!(config.flow_lm.transformer.num_heads, 16);
    assert_eq!(config.flow_lm.transformer.num_layers, 6);
    assert_eq!(config.flow_lm.flow.dim, 512);
    assert_eq!(config.flow_lm.flow.depth, 6);
    assert_eq!(config.flow_lm.flow.r#type, "lsd");
    assert_eq!(config.flow_lm.lookup_table.n_bins, 4000);
    assert!(config.flow_lm.insert_bos_before_voice);

    assert_eq!(config.mimi.sample_rate, 24000);
    assert_eq!(config.mimi.channels, 1);
    assert_eq!(config.mimi.frame_rate, 12.5);
    assert_eq!(config.mimi.seanet.ratios, vec![6, 5, 4]);
    assert_eq!(config.mimi.transformer.d_model, 512);
    assert_eq!(config.mimi.transformer.context, 250);
    assert_eq!(config.mimi.quantizer.dimension, 32);
    assert_eq!(config.mimi.inner_dim, Some(32));
    assert_eq!(config.mimi.outer_dim, Some(512));

    assert_eq!(
        config.weights_path.as_deref(),
        Some("hf://kyutai/pocket-tts/languages/english/model.safetensors@39592ff23c9ef80098bb74895d104c26275fe2c9")
    );
    assert_eq!(config.default_temperature, 0.3);
}

/// Unknown fields must be rejected, mirroring `ConfigDict(extra="forbid")`
/// on the Python `StrictModel` base class.
#[test]
fn config_rejects_unknown_fields() {
    let yaml = r#"
flow_lm:
  dtype: float32
  not_a_real_field: 1
  flow: { dim: 1, depth: 1 }
  transformer: { hidden_scale: 1, max_period: 1, d_model: 1, num_heads: 1, num_layers: 1 }
  lookup_table: { dim: 1, n_bins: 1, tokenizer: sentencepiece, tokenizer_path: x }
mimi:
  dtype: float32
  sample_rate: 1
  channels: 1
  frame_rate: 1.0
  seanet: { dimension: 1, channels: 1, n_filters: 1, n_residual_layers: 1, ratios: [1], kernel_size: 1, residual_kernel_size: 1, last_kernel_size: 1, dilation_base: 1, pad_mode: constant, compress: 1 }
  transformer: { d_model: 1, input_dimension: 1, output_dimensions: [1], num_heads: 1, num_layers: 1, layer_scale: 0.0, context: 1, dim_feedforward: 1 }
  quantizer: { dimension: 1, output_dimension: 1 }
"#;
    assert!(Config::from_yaml_str(yaml).is_err());
}

/// Current upstream model configuration must deserialize without schema drift.
/// Fixture copied from kyutai-labs/pocket-tts tag v4.0.1.
#[test]
fn config_parses_current_english() {
    let config = Config::from_yaml_str(include_str!("fixtures/english-v4.yaml"))
        .expect("v4.0.1 default English configuration should parse");
    assert_eq!(config.default_temperature, 0.3);
    assert_eq!(config.default_eos_threshold, -4.0);
    assert!(config.capitalize_first_letter);
    assert_eq!(
        config.replace_characters.get("’").map(String::as_str),
        Some("'")
    );
    assert_eq!(config.flow_lm.lookup_table.tokenizer, "tokenizers");
    assert_eq!(config.flow_lm.transformer.attn_dim, None);
    assert_eq!(
        config.mimi.decoder_seanet.as_ref().map(|x| x.dimension),
        None
    );
}

#[test]
fn config_parses_compact_7m_with_asymmetric_decoder_and_gqa() {
    let config = Config::from_yaml_str(include_str!("fixtures/english-7m-v4.yaml"))
        .expect("v4.0.1 compact 7M configuration should parse");
    assert_eq!(config.flow_lm.flow.r#type, "drifting");
    assert_eq!(config.flow_lm.z_dim, Some(64));
    assert_eq!(config.flow_lm.transformer.attn_dim, Some(512));
    assert_eq!(config.flow_lm.transformer.num_kv_heads, Some(2));
    assert_eq!(config.flow_lm.lookup_table.rank, Some(64));
    assert_eq!(config.mimi.decoder_seanet.as_ref().unwrap().dimension, 128);
    assert_eq!(
        config.mimi.decoder_transformer.as_ref().unwrap().num_layers,
        12
    );
    assert_eq!(config.default_eos_threshold, -1.0);
}

#[test]
fn config_parses_compact_18m_with_explicit_null_overrides() {
    let config = Config::from_yaml_str(include_str!("fixtures/english-18m-refl-v4.yaml"))
        .expect("v4.0.1 compact 18M REFL configuration should parse");
    assert_eq!(config.flow_lm.z_dim, Some(64));
    assert_eq!(config.flow_lm.transformer.attn_dim, None);
    assert_eq!(config.flow_lm.transformer.num_kv_heads, None);
    assert_eq!(config.flow_lm.lookup_table.rank, None);
    assert!(config.mimi.decoder_transformer.is_some());
}
