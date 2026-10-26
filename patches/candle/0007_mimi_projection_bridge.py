#!/usr/bin/env python3
"""Preserve Mimi encoder width separately from the compact decoder width.

Kyutai 18M/7M Mimi:
 encoder transformer output 512 -> downsample 32 -> upsample 128 -> decoder 128.
Legacy Mimi used 512 for both encoder output and decoder input. Preserve the
old constructor API while adding an explicit encoder output dimension.
"""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit("usage: 0007_mimi_projection_bridge.py /path/to/pinned-candle")
root=Path(sys.argv[1])/"crates/pocket-tts/src"

def patch(file,old,new):
    p=root/file
    code=p.read_text()
    assert code.count(old)==1,f"upstream changed: {file} {old[:60]}"
    p.write_text(code.replace(old,new))

patch("models/mimi.rs",
      """        output_dimension: usize, // The decoder input dimension (512)
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let quantizer = Quantizer::new(dimension, output_dimension, vb.pp("quantizer"))?;""",
      """        output_dimension: usize, // Decoder input dimension
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_encoder_width(
            encoder, decoder, encoder_transformer, decoder_transformer,
            frame_rate, encoder_frame_rate, sample_rate, channels,
            dimension, output_dimension, output_dimension, name, vb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_encoder_width(
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        encoder_transformer: ProjectedTransformer,
        decoder_transformer: ProjectedTransformer,
        frame_rate: f64,
        encoder_frame_rate: f64,
        sample_rate: usize,
        channels: usize,
        dimension: usize,
        output_dimension: usize,
        encoder_output_dimension: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let quantizer = Quantizer::new(dimension, output_dimension, vb.pp("quantizer"))?;""")
patch("models/mimi.rs",
      """                Some(ConvDownsample1d::new_with_dims(
                    stride,
                    output_dimension,
                    dimension,""",
      """                Some(ConvDownsample1d::new_with_dims(
                    stride,
                    encoder_output_dimension,
                    dimension,""")
patch("tts_model.rs",
      """        let mimi = MimiModel::new(
            encoder,""",
      """        let mimi = MimiModel::new_with_encoder_width(
            encoder,""")
patch("tts_model.rs",
      """            config.mimi.quantizer.dimension,
            config.mimi.quantizer.output_dimension,
            "mimi",""",
      """            config.mimi.quantizer.dimension,
            config.mimi.quantizer.output_dimension,
            *config.mimi.transformer.output_dimensions.first()
                .ok_or_else(|| anyhow::anyhow!("Mimi encoder has no output dimensions"))?,
            "mimi",""")
print("Reused Mimi with independent 512-wide encoder and 128-wide compact decoder")
