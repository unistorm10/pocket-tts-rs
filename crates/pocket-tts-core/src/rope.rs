//! Interleaved rotary positional embeddings matching Kyutai v4.0.1.
//! The input layout is [batch, time, heads, head_dim], with interleaved real/imag pairs.

use candle_core::{DType, Result, Tensor};

fn rotate(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, time, heads, dim) = input.dims4()?;
    let half = dim / 2;
    let dtype = input.dtype();
    let complex = input.reshape((batch, time, heads, half, 2))?;
    let real = complex.narrow(4, 0, 1)?.squeeze(4)?.to_dtype(DType::F32)?;
    let imag = complex.narrow(4, 1, 1)?.squeeze(4)?.to_dtype(DType::F32)?;
    let real_cos = real.broadcast_mul(cos)?;
    let imag_sin = imag.broadcast_mul(sin)?;
    let real_sin = real.broadcast_mul(sin)?;
    let imag_cos = imag.broadcast_mul(cos)?;
    let out_real = (real_cos - imag_sin)?.to_dtype(dtype)?;
    let out_imag = (real_sin + imag_cos)?.to_dtype(dtype)?;
    Tensor::stack(&[out_real, out_imag], 4)?.reshape((batch, time, heads, dim))
}

/// Rotate queries and keys, permitting different query and KV head counts.
///
/// `offset` is the number of tokens already processed in this streaming session.
pub fn apply_rope(
    queries: &Tensor,
    keys: &Tensor,
    offset: usize,
    max_period: f32,
) -> Result<(Tensor, Tensor)> {
    let (batch, time, heads, dim) = queries.dims4()?;
    let (key_batch, key_time, key_heads, key_dim) = keys.dims4()?;
    if batch != key_batch || time != key_time || dim != key_dim {
        candle_core::bail!("queries and keys must agree on batch, time and head_dim");
    }
    if dim == 0 || dim % 2 != 0 || heads == 0 || key_heads == 0 {
        candle_core::bail!("head count must be nonzero and head_dim must be positive and even");
    }
    if !max_period.is_finite() || max_period <= 0.0 {
        candle_core::bail!("max_period must be positive and finite");
    }
    let half = dim / 2;
    let frequency_scale = -max_period.ln() * 2.0 / (dim as f32);
    let mut cosine = Vec::with_capacity(time * half);
    let mut sine = Vec::with_capacity(time * half);
    for t in 0..time {
        let position = offset
            .checked_add(t)
            .ok_or_else(|| candle_core::Error::Msg("streaming RoPE position overflow".into()))?;
        for i in 0..half {
            let theta = (i as f32 * frequency_scale).exp() * (position as f32);
            cosine.push(theta.cos());
            sine.push(theta.sin());
        }
    }
    let shape = (1, time, 1, half);
    let cos = Tensor::from_vec(cosine, shape, queries.device())?;
    let sin = Tensor::from_vec(sine, shape, queries.device())?;
    Ok((rotate(queries, &cos, &sin)?, rotate(keys, &cos, &sin)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn zero_position_is_identity_even_with_grouped_kv_heads() {
        let dev = Device::Cpu;
        let q = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &dev)
            .unwrap()
            .reshape((1, 1, 1, 4))
            .unwrap();
        let k = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &dev)
            .unwrap()
            .reshape((1, 1, 2, 4))
            .unwrap();
        let (qo, ko) = apply_rope(&q, &k, 0, 10_000.0).unwrap();
        assert_eq!(
            qo.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 2., 3., 4.]
        );
        assert_eq!(
            ko.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 2., 3., 4., 5., 6., 7., 8.]
        );
    }

    #[test]
    fn streamed_offset_matches_full_sequence() {
        let dev = Device::Cpu;
        let q = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &dev)
            .unwrap()
            .reshape((1, 2, 1, 4))
            .unwrap();
        let (whole, _) = apply_rope(&q, &q, 0, 10_000.0).unwrap();
        let chunk = q.narrow(1, 1, 1).unwrap();
        let (partial, _) = apply_rope(&chunk, &chunk, 1, 10_000.0).unwrap();
        let expected = whole
            .narrow(1, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = partial.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert!((expected - actual).abs() < 1e-5);
        }
    }
}
