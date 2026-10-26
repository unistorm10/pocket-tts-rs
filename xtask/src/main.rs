//! Tooling for pinned, real Kyutai model checkpoint download and inspection.
//! No neural outputs are faked: this verifies actual safetensors weights.

use anyhow::{bail, Context, Result};
use pocket_tts_core::config::Config;
use pocket_tts_core::download::download_if_necessary;
use pocket_tts_core::weights::ModelWeights;
use safetensors::SafeTensors;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn usage() -> &'static str {
    "Usage: cargo run -p xtask -- prefetch <english|english-7m|english-18m-refl>\n       cargo run -p xtask -- inspect <checkpoint.safetensors>"
}

fn config_yaml(model: &str) -> Result<&'static str> {
    match model {
        "english" => Ok(include_str!(
            "../../crates/pocket-tts-core/tests/fixtures/english-v4.yaml"
        )),
        "english-7m" => Ok(include_str!(
            "../../crates/pocket-tts-core/tests/fixtures/english-7m-v4.yaml"
        )),
        "english-18m-refl" => Ok(include_str!(
            "../../crates/pocket-tts-core/tests/fixtures/english-18m-refl-v4.yaml"
        )),
        _ => bail!("unknown model {model:?}; {}", usage()),
    }
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut f = BufReader::new(f);
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 1024 * 1024];
    let mut bytes = 0_u64;
    loop {
        let count = f.read(&mut buf)?;
        if count == 0 {
            break;
        }
        hasher.update(&buf[..count]);
        bytes += count as u64;
    }
    Ok((format!("{:x}", hasher.finalize()), bytes))
}

fn inspect_weights(path: &Path, repo_ref: Option<&str>) -> Result<serde_json::Value> {
    let contents = std::fs::read(path)
        .with_context(|| format!("reading safetensors checkpoint: {}", path.display()))?;
    let safe = SafeTensors::deserialize(&contents)
        .with_context(|| format!("validating safetensors headers: {}", path.display()))?;
    let names = safe.names();
    anyhow::ensure!(!names.is_empty(), "model checkpoint contains no tensors");
    let mut parameters = 0_u64;
    let mut tensors = 0_u64;
    let mut nonzero_bytes = 0_u64;
    let mut encoder_tensor_count = 0_u64;
    let mut nonzero_encoder_tensors = 0_u64;
    for name in names {
        let tensor = safe.tensor(name)?;
        let elements = tensor.shape().iter().try_fold(1_u64, |acc, dim| {
            acc.checked_mul(*dim as u64)
                .context("tensor dimension overflow")
        })?;
        parameters = parameters
            .checked_add(elements)
            .context("parameter count overflow")?;
        nonzero_bytes += tensor.data().len() as u64;
        if name.starts_with("mimi.encoder") {
            encoder_tensor_count += 1;
            if tensor.data().iter().any(|byte| *byte != 0) {
                nonzero_encoder_tensors += 1;
            }
        }
        tensors += 1;
    }
    anyhow::ensure!(nonzero_bytes > 0, "checkpoint has no tensor payload data");
    anyhow::ensure!(
        encoder_tensor_count == 42 && nonzero_encoder_tensors == 42,
        "checkpoint is missing required active Mimi voice encoder (expected 42/42 tensors)"
    );
    let (digest, file_size) = sha256_file(path)?;
    Ok(json!({
        "path": path.display().to_string(),
        "reference": repo_ref,
        "sha256": digest,
        "file_bytes": file_size,
        "tensor_count": tensors,
        "parameter_count": parameters,
        "tensor_data_bytes": nonzero_bytes,
        "voice_encoder_tensor_count": encoder_tensor_count,
        "active_voice_encoder_tensors": nonzero_encoder_tensors,
        "voice_cloning_encoder_active": encoder_tensor_count > 0 && nonzero_encoder_tensors > 0,
        "validated": true
    }))
}

fn expected_checkpoint_sha256(model: &str) -> Result<&'static str> {
    match model {
        "english" => Ok("bc96b82436c5ec00d3118ef3b01bc3358423e1766ecae3dca0aeea84789e7d74"),
        "english-7m" => Ok("1200a5d97d82b95804e76e3042ce88787773486a0d95e00737650667c2bee844"),
        "english-18m-refl" => Ok("c99f4a179cffa070cdeb3ffc1ae950ebf16038c762c84dd79997868ac0a806c8"),
        _ => bail!("unsupported full voice-cloning model {model:?}"),
    }
}

fn prefetch(model: &str) -> Result<()> {
    let config = Config::from_yaml_str(config_yaml(model)?)?;
    let weights_ref = config.weights_path.as_deref()
        .context("full voice-cloning checkpoint path is required")?;
    anyhow::ensure!(
        weights_ref.starts_with("hf://kyutai/pocket-tts/"),
        "model must come from the full gated kyutai/pocket-tts repository"
    );
    println!("Downloading authenticated, full voice-cloning checkpoint for {model}");
    let checkpoint = download_if_necessary(weights_ref)
        .with_context(|| format!("downloading full checkpoint for {model}"))?;
    let inspection = inspect_weights(&checkpoint, Some(weights_ref))?;
    anyhow::ensure!(
        inspection["sha256"].as_str() == Some(expected_checkpoint_sha256(model)?),
        "unexpected checkpoint SHA-256 for {model}; refusing incomplete or substituted model"
    );
    anyhow::ensure!(
        inspection["voice_encoder_tensor_count"].as_u64() == Some(42)
            && inspection["active_voice_encoder_tensors"].as_u64() == Some(42),
        "full 42-tensor Mimi voice encoder is mandatory; rejecting this checkpoint"
    );
    println!("{}", serde_json::to_string_pretty(&inspection)?);

    // Force an actual checkpoint tensor through Candle's native safetensors loader
    // and arithmetic kernels. This is a weight-loading smoke test, not TTS inference.
    let weights = ModelWeights::from_file(&checkpoint)?;
    let safe_bytes = std::fs::read(&checkpoint)?;
    let safe = SafeTensors::deserialize(&safe_bytes)?;
    let first = weights
        .names()?
        .into_iter()
        .filter_map(|name| {
            let tensor = safe.tensor(&name).ok()?;
            let elements = tensor.shape().iter().product::<usize>();
            if elements > 0 && elements <= 4096 {
                Some((name, elements))
            } else {
                None
            }
        })
        .min_by_key(|(_, elements)| *elements)
        .context("checkpoint has no small tensors for Candle smoke test")?;
    let device = candle_core::Device::Cpu;
    let tensor = weights.get(&first.0, candle_core::DType::F32, &device)?;
    let sum_of_squares = tensor
        .sqr()?
        .sum_all()?
        .to_dtype(candle_core::DType::F32)?
        .to_scalar::<f32>()?;
    anyhow::ensure!(
        sum_of_squares.is_finite(),
        "Candle checkpoint tensor is non-finite"
    );
    println!(
        "Candle real-weight probe: tensor={:?} elements={} sum_of_squares={}",
        first.0, first.1, sum_of_squares
    );

    println!("Downloading pinned tokenizer");
    let tokenizer_ref = &config.flow_lm.lookup_table.tokenizer_path;
    let tokenizer = download_if_necessary(tokenizer_ref)
        .with_context(|| format!("downloading tokenizer for {model}"))?;
    let (tokenizer_hash, tokenizer_bytes) = sha256_file(&tokenizer)?;
    anyhow::ensure!(tokenizer_bytes > 0, "tokenizer is empty");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "tokenizer_kind": config.flow_lm.lookup_table.tokenizer,
            "reference": tokenizer_ref,
            "path": tokenizer.display().to_string(),
            "sha256": tokenizer_hash,
            "bytes": tokenizer_bytes,
            "validated": true
        }))?
    );
    println!("Real checkpoint and tokenizer validated. No inference was performed.");
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        bail!("{}", usage());
    };
    match command.as_str() {
        "prefetch" => {
            let model = args.next().context(usage())?;
            for arg in args {
                bail!("unsupported argument {arg:?}: encoder-disabled models are prohibited; {}", usage());
            }
            prefetch(&model)
        }
        "inspect" => {
            let path = args.next().context(usage())?;
            anyhow::ensure!(args.next().is_none(), "unexpected extra arguments");
            println!(
                "{}",
                serde_json::to_string_pretty(&inspect_weights(Path::new(&path), None)?)?
            );
            Ok(())
        }
        _ => bail!("unknown command {command:?}; {}", usage()),
    }
}
