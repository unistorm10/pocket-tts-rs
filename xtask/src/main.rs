//! Tooling for pinned, real Kyutai model checkpoint download and inspection.
//! No neural outputs are faked: this verifies actual safetensors weights.

use anyhow::{bail, Context, Result};
use pocket_tts_core::config::Config;
use pocket_tts_core::download::download_if_necessary;
use safetensors::SafeTensors;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn usage() -> &'static str {
    "Usage: cargo run -p xtask -- prefetch <english|english-7m|english-18m-refl> [--without-voice-cloning]\n       cargo run -p xtask -- inspect <checkpoint.safetensors>"
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
        tensors += 1;
    }
    anyhow::ensure!(nonzero_bytes > 0, "checkpoint has no tensor payload data");
    let (digest, file_size) = sha256_file(path)?;
    Ok(json!({
        "path": path.display().to_string(),
        "reference": repo_ref,
        "sha256": digest,
        "file_bytes": file_size,
        "tensor_count": tensors,
        "parameter_count": parameters,
        "tensor_data_bytes": nonzero_bytes,
        "validated": true
    }))
}

fn prefetch(model: &str, without_voice_cloning: bool) -> Result<()> {
    let config = Config::from_yaml_str(config_yaml(model)?)?;
    let weights_ref = if without_voice_cloning {
        config
            .weights_path_without_voice_cloning
            .as_deref()
            .context("config does not provide non-voice-cloning weights")?
    } else {
        config
            .weights_path
            .as_deref()
            .context("config does not provide model weights")?
    };
    println!("Downloading pinned checkpoint for {model} via hf-hub cache");
    let checkpoint = download_if_necessary(weights_ref)
        .with_context(|| format!("downloading pinned checkpoint for {model}"))?;
    let inspection = inspect_weights(&checkpoint, Some(weights_ref))?;
    println!("{}", serde_json::to_string_pretty(&inspection)?);

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
            let mut without_voice_cloning = false;
            for arg in args {
                if arg == "--without-voice-cloning" {
                    without_voice_cloning = true;
                } else {
                    bail!("unexpected argument {arg:?}; {}", usage());
                }
            }
            prefetch(&model, without_voice_cloning)
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
