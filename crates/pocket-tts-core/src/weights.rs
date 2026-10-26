//! Real, owned safetensors checkpoints exposed through Candle's native VarBuilder.
//! No memory mapping or unsafe code is needed for the initial CPU baseline.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;
use std::path::Path;

/// An owned checkpoint. This file remains usable without Phonation or networking.
#[derive(Debug)]
pub struct ModelWeights {
    data: Vec<u8>,
}

impl ModelWeights {
    /// Validates the safetensors header and keeps the checkpoint in memory.
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("reading model weights {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&data).context("invalid safetensors checkpoint")?;
        anyhow::ensure!(
            !tensors.names().is_empty(),
            "checkpoint has no named tensors"
        );
        Ok(Self { data })
    }

    /// Returns all tensor names found in the real checkpoint.
    pub fn names(&self) -> Result<Vec<String>> {
        let tensors = SafeTensors::deserialize(&self.data)?;
        Ok(tensors.names().into_iter().map(str::to_string).collect())
    }

    /// Produce a native Candle weight source; the checkpoint stays owned by this object.
    pub fn var_builder<'a>(
        &'a self,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<VarBuilder<'a>> {
        VarBuilder::from_slice_safetensors(&self.data, dtype, device)
    }

    /// Materialize a named real checkpoint tensor on the selected Candle device.
    pub fn get(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let safe = SafeTensors::deserialize(&self.data)?;
        let view = safe
            .tensor(name)
            .with_context(|| format!("tensor {name:?} is missing from checkpoint"))?;
        let shape = view.shape().to_vec();
        let vb = self.var_builder(dtype, device)?;
        vb.get(shape, name)
            .with_context(|| format!("loading real Candle tensor {name:?}"))
    }
}
