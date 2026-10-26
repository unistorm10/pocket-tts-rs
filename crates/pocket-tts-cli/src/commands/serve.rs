//! Serve command implementation
//!
//! Provides `pocket-tts serve` for HTTP API server.
//!
//! The argument surface (`ServeArgs`, `UiMode`) and the endpoint banner live in
//! `pocket-tts-server`, so that crate can be embedded as a library without
//! pulling in the CLI. This module only wires the CLI up to it.

use anyhow::Result;
use owo_colors::OwoColorize;

pub use pocket_tts_server::config::{ServeArgs, UiMode, print_endpoints};

pub async fn run(args: ServeArgs) -> Result<()> {
    print_banner();

    println!(
        "{} Loading model variant: {}",
        "->".cyan(),
        args.variant.yellow()
    );

    println!("{} UI mode: {}", "->".cyan(), args.ui.as_str().yellow());

    pocket_tts_server::start_server(args).await
}

fn print_banner() {
    println!();
    println!(
        "  {}  {} {}",
        "[]".bold(),
        "Pocket TTS".bold().cyan(),
        "API Server".bold()
    );
    println!(
        "      {} {}",
        "Rust/Candle port".dimmed(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!();
}
