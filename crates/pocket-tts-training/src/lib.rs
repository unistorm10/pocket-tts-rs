//! Standalone training orchestration for Kyutai's released Pocket TTS trainer.
//!
//! This is a Rust library/CLI that validates and executes the official pinned
//! PyTorch/CUDA scripts, NOT an implementation of backpropagation in Candle.
//! Synthesis in pocket-tts-rs remains fully independent of Python.

use anyhow::{bail, Context, Result};
use serde_yaml::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

/// All first-class recipes released with Kyutai Pocket TTS v4.0.1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Recipe {
    Scratch,
    Finetune,
    FinetuneLanguage,
    DepthDistill,
    Drifting,
}

impl Recipe {
    /// Official upstream training recipe path, relative to the checkout.
    pub fn path(self) -> &'static str {
        match self {
            Self::Scratch => "training/configs/scratch.yaml",
            Self::Finetune => "training/configs/finetune.yaml",
            Self::FinetuneLanguage => "training/configs/finetune_language.yaml",
            Self::DepthDistill => "training/configs/depth_distill.yaml",
            Self::Drifting => "training/configs/drifting.yaml",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "scratch" => Ok(Self::Scratch),
            "finetune" => Ok(Self::Finetune),
            "finetune-language" => Ok(Self::FinetuneLanguage),
            "depth-distill" => Ok(Self::DepthDistill),
            "drifting" => Ok(Self::Drifting),
            _ => bail!("unknown recipe {name:?}; expected scratch, finetune, finetune-language, depth-distill, drifting"),
        }
    }
}

/// An explicit local source checkout and recipe. There is no implicit cloning
/// or model download and no training process starts during construction.
#[derive(Debug)]
pub struct TrainingPlan {
    pub source: PathBuf,
    pub config: PathBuf,
    pub recipe: Recipe,
}

impl TrainingPlan {
    pub fn new(source: impl AsRef<Path>, recipe: Recipe) -> Result<Self> {
        let source = source.as_ref();
        anyhow::ensure!(source.is_dir(), "Kyutai source directory not found: {}", source.display());
        let source = source.canonicalize().context("canonicalizing Kyutai source path")?;
        anyhow::ensure!(
            source.join("training/train.py").is_file(),
            "training/train.py is missing: supply a checkout of kyutai-labs/pocket-tts at tag v4.0.1"
        );
        let config = source.join(recipe.path());
        anyhow::ensure!(
            config.is_file(),
            "training recipe is missing: {}",
            config.display()
        );
        Ok(Self { source, config, recipe })
    }

    /// Confirm the recipe references real prepared training/validation manifests.
    /// For distillation, ensure the teacher checkpoint exists as well.
    pub fn validate_inputs(&self) -> Result<()> {
        let config_str = std::fs::read_to_string(&self.config)
            .with_context(|| format!("reading recipe {}", self.config.display()))?;
        let config: Value = serde_yaml::from_str(&config_str)
            .with_context(|| format!("parsing recipe {}", self.config.display()))?;
        let dataset = config.get("data").context("training recipe lacks a data section")?;
        for field in ["train_jsonl", "valid_jsonl"] {
            let relative = dataset.get(field).and_then(Value::as_str)
                .with_context(|| format!("missing data.{field} in training recipe"))?;
            self.require_file(relative, field)?;
        }
        if self.recipe == Recipe::DepthDistill {
            let teacher = config.get("distill_teacher_weights").and_then(Value::as_str)
                .context("distillation recipe lacks distill_teacher_weights")?;
            self.require_file(teacher, "distillation teacher checkpoint")?;
        }
        let model_config = config.get("model_config").and_then(Value::as_str)
            .context("recipe lacks model_config")?;
        self.require_file(model_config, "model configuration")?;
        Ok(())
    }

    fn require_file(&self, reference: &str, purpose: &str) -> Result<()> {
        let path = Path::new(reference);
        // hf:// and https:// references resolve via the official Pocket TTS
        // loader; local paths must exist before invoking training.
        if reference.starts_with("hf://") || reference.starts_with("https://") {
            return Ok(());
        }
        let resolved = if path.is_absolute() { path.to_path_buf() } else { self.source.join(path) };
        anyhow::ensure!(resolved.is_file(), "{purpose} missing: {}", resolved.display());
        Ok(())
    }

    /// Execute the official Python/CUDA training. Never run implicitly:
    /// callers must explicitly invoke this after setting up the dataset.
    ///
    /// Inherits stdout/stderr and environment, including any HF_TOKEN.
    /// No secrets are echoed or passed as command-line arguments.
    pub fn execute(&self) -> Result<ExitStatus> {
        self.validate_inputs()?;
        let uv = Command::new("uv")
            .arg("--version")
            .current_dir(&self.source)
            .output()
            .context("uv not available; install uv in the training environment")?;
        anyhow::ensure!(uv.status.success(), "uv --version failed");
        let status = Command::new("uv")
            .arg("run")
            .arg("training/train.py")
            .arg(self.recipe.path())
            .current_dir(&self.source)
            .status()
            .context("could not launch official training script")?;
        if !status.success() {
            bail!("training process exited with {status}");
        }
        Ok(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_recipes_are_mapped_without_modifying_upstream() {
        assert_eq!(Recipe::parse("scratch").unwrap().path(), "training/configs/scratch.yaml");
        assert_eq!(Recipe::parse("finetune").unwrap().path(), "training/configs/finetune.yaml");
        assert_eq!(Recipe::parse("finetune-language").unwrap().path(), "training/configs/finetune_language.yaml");
        assert_eq!(Recipe::parse("depth-distill").unwrap().path(), "training/configs/depth_distill.yaml");
        assert_eq!(Recipe::parse("drifting").unwrap().path(), "training/configs/drifting.yaml");
        assert!(Recipe::parse("lora").is_err()); // upstream does not ship a LoRA recipe
    }

    #[test]
    fn missing_checkout_cannot_launch_training() {
        let missing = std::env::temp_dir().join("pocket-tts-rs-intentionally-missing-source-9dd1a294");
        assert!(TrainingPlan::new(missing, Recipe::Scratch).is_err());
    }
}
