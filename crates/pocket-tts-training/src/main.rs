//! Explicit executable frontend to the standalone Rust training launcher.

use anyhow::{bail, Context, Result};
use pocket_tts_training::{Recipe, TrainingPlan};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut source: Option<PathBuf> = None;
    let mut recipe = Recipe::Finetune;
    let mut run = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--source" => {
                source = Some(PathBuf::from(
                    args.next().context("--source needs a directory")?,
                ))
            }
            "--recipe" => recipe = Recipe::parse(&args.next().context("--recipe needs a name")?)?,
            "--run" => run = true,
            "--help" | "-h" => {
                println!("Usage: pocket-tts-train --source <kyutai-v4.0.1-checkout> [--recipe scratch|finetune|finetune-language|depth-distill|drifting] [--run]");
                println!("Without --run, validates the checkout and recipe only. --run checks dataset inputs and launches official uv/PyTorch training.");
                return Ok(());
            }
            _ => bail!("unexpected argument: {arg} (use --help)"),
        }
    }
    let source = source.context("missing --source (use --help)")?;
    let plan = TrainingPlan::new(source, recipe)?;
    println!(
        "Validated upstream training source: {}",
        plan.source.display()
    );
    println!("Recipe: {}", plan.config.display());
    if run {
        plan.execute()?;
    } else {
        println!("Dry run only. Dataset and GPU not checked; use --run to start actual training.");
    }
    Ok(())
}
