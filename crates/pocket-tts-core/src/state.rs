//! Persistent stream-local state, keyed by the absolute module paths used upstream.

use candle_core::Tensor;
use std::collections::HashMap;

/// Named tensors held by one StatefulModule, e.g. `previous` or `partial`.
pub type ModuleState = HashMap<String, Tensor>;

/// Independent state for one generation session.
///
/// Never share the mutable map across concurrent synthesis sessions.
pub type ModelState = HashMap<String, ModuleState>;

/// Return one module's state or a descriptive error instead of silently resetting it.
pub fn module_state<'a>(
    state: &'a mut ModelState,
    absolute_module_name: &str,
) -> Result<&'a mut ModuleState, String> {
    state
        .get_mut(absolute_module_name)
        .ok_or_else(|| format!("missing stream state for module {absolute_module_name:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn module_names_isolate_stream_caches() {
        let mut state = ModelState::new();
        let a = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::ones(3, DType::F32, &Device::Cpu).unwrap();
        state.insert("encoder.layers.0".into(), HashMap::from([("previous".into(), a)]));
        state.insert("decoder.layers.0".into(), HashMap::from([("partial".into(), b)]));
        assert_eq!(
            module_state(&mut state, "encoder.layers.0")
                .unwrap()["previous"]
                .elem_count(),
            2
        );
        assert_eq!(
            module_state(&mut state, "decoder.layers.0")
                .unwrap()["partial"]
                .elem_count(),
            3
        );
        assert!(module_state(&mut state, "missing.layers.0").is_err());
    }
}
