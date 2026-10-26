#!/usr/bin/env python3
"""Adapt pinned community Candle Mimi downsample to Kyutai v4.0.1.

Apply only to stepupgaming/pocket-tts@dbf78c816cd83c89f11dbeb9d87290ad0a3dccd0.
This edits a throwaway, pinned upstream checkout on CI; never writes model weights.
Preserves the legacy constructor for backwards compatibility.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: 0001_mimi_downsample.py /path/to/pinned-candle")

root = Path(sys.argv[1]).resolve()
conv = root / "crates/pocket-tts/src/modules/conv.rs"
mimi = root / "crates/pocket-tts/src/models/mimi.rs"

source = conv.read_text()
old = """impl ConvDownsample1d {
    pub fn new(stride: usize, dimension: usize, name: &str, vb: VarBuilder) -> Result<Self> {
        let conv = StreamingConv1d::new(
            dimension,
            dimension,
"""
new = """impl ConvDownsample1d {
    pub fn new(stride: usize, dimension: usize, name: &str, vb: VarBuilder) -> Result<Self> {
        Self::new_with_dims(stride, dimension, dimension, name, vb)
    }

    /// v4.0.1 permits encoder output width and latent input width to differ.
    pub fn new_with_dims(
        stride: usize,
        input_dimension: usize,
        output_dimension: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = StreamingConv1d::new(
            input_dimension,
            output_dimension,
"""
assert source.count(old) == 1, "upstream ConvDownsample1d signature changed; re-audit"
source = source.replace(old, new)
conv.write_text(source)

source = mimi.read_text()
old_call = """                Some(ConvDownsample1d::new(
                    stride,
                    output_dimension,
"""
new_call = """                Some(ConvDownsample1d::new_with_dims(
                    stride,
                    output_dimension,
                    dimension,
"""
assert source.count(old_call) == 1, "upstream Mimi downsample call changed; re-audit"
source = source.replace(old_call, new_call)
mimi.write_text(source)
print("Applied upstream Candle v4 Mimi encoder 512->32 downsample patch (legacy API preserved)")
