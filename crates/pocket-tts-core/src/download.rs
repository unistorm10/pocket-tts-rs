//! Resolves and caches `hf://`, `http(s)://`, and local-path resources.
//!
//! Mirrors `pocket_tts.utils.utils.download_if_necessary`:
//! - `hf://owner/repo/path/to/file@revision` downloads via the HF Hub API
//!   (revision is optional, split on the last `@`).
//! - `http://` / `https://` URLs are downloaded once and cached under
//!   `~/.cache/pocket_tts_rs/<sha256(url)><suffix>`.
//! - Anything else is treated as a local filesystem path.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Mirrors `pocket_tts.utils.utils.make_cache_directory`, but uses a
/// Rust-specific cache directory name to avoid colliding with the Python
/// package's cache when both are installed on the same machine.
pub fn make_cache_directory() -> Result<PathBuf> {
    let base = dirs::home_dir().context("could not determine home directory")?;
    let cache_dir = base.join(".cache").join("pocket_tts_rs");
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Resolves `file_path` to a local filesystem path, downloading it first if
/// necessary. See module docs for the three supported schemes.
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    // A voice-cloning-capable model is mandatory, including in CI.
    // Block the encoder-disabled repository for checkpoints AND tokenizers.
    anyhow::ensure!(
        !file_path.contains("pocket-tts-without-voice-cloning"),
        "encoder-disabled repository is prohibited; use the full gated kyutai/pocket-tts source"
    );
    if file_path.starts_with("http://") || file_path.starts_with("https://") {
        download_http(file_path)
    } else if let Some(rest) = file_path.strip_prefix("hf://") {
        download_hf(rest)
    } else {
        Ok(PathBuf::from(file_path))
    }
}

fn download_http(url: &str) -> Result<PathBuf> {
    let cache_dir = make_cache_directory()?;
    let suffix = Path::new(url.split('?').next().unwrap_or(url))
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let digest = hasher.finalize();
    let hex = hex::encode(digest);
    let cached_file = cache_dir.join(format!("{hex}{suffix}"));
    if !cached_file.exists() {
        let response = reqwest::blocking::get(url)?.error_for_status()?;
        let bytes = response.bytes()?;
        std::fs::write(&cached_file, &bytes)?;
    }
    Ok(cached_file)
}

/// `path` is the part after `hf://`, e.g. `kyutai/pocket-tts/tts.safetensors@abcdef`.
fn download_hf(path: &str) -> Result<PathBuf> {
    let parts: Vec<&str> = path.splitn(3, '/').collect();
    anyhow::ensure!(
        parts.len() == 3,
        "hf:// path must be of the form owner/repo/filename[@revision], got {path:?}"
    );
    let repo_id = format!("{}/{}", parts[0], parts[1]);
    let (filename, revision) = match parts[2].split_once('@') {
        Some((f, r)) => (f.to_string(), Some(r.to_string())),
        None => (parts[2].to_string(), None),
    };

    let rev = revision.as_deref().unwrap_or("main");
    // hf-hub 0.3 reads cached tokens by default, but does not automatically
    // consume HF_TOKEN from CI. Pass that environment token explicitly for
    // access-controlled voice-cloning weights; never print or persist it.
    let mut builder = hf_hub::api::sync::ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            builder = builder.with_token(Some(token));
        }
    }
    let api = builder.build()?;
    let repo = match &revision {
        Some(rev) => api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Model,
            rev.clone(),
        )),
        None => api.repo(hf_hub::Repo::new(repo_id.clone(), hf_hub::RepoType::Model)),
    };

    match repo.get(&filename) {
        Ok(local_path) => Ok(local_path),
        Err(primary) if filename.ends_with(".json") => {
            // The older hf-hub client can fail to follow Hugging Face's relative
            // resolve-cache redirects for small Git-tracked JSON files.
            // The /raw/ endpoint serves these directly, with the same pinned revision.
            let url = format!("https://huggingface.co/{repo_id}/raw/{rev}/{filename}");
            download_http(&url).with_context(|| {
                format!("HF Hub JSON resolution failed ({primary}); raw fallback also failed")
            })
        }
        Err(primary) => Err(primary.into()),
    }
}

// Minimal hex encoding to avoid pulling in another dependency just for this.
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        let mut s = String::with_capacity(bytes.as_ref().len() * 2);
        for b in bytes.as_ref() {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::download_if_necessary;

    #[test]
    fn rejected_model_source_never_reaches_hf_or_http() {
        for path in [
            "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors",
            "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main/tokenizer.json",
            "pocket-tts-without-voice-cloning/model.safetensors",
        ] {
            assert!(download_if_necessary(path).is_err());
        }
    }
}
