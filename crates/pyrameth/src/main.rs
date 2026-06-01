//! PyraMeth Rust CLI (`pyrameth`)
//!
//! Sub-commands:
//!   pyrameth call_mods  — detect methylation (signal → per-read probabilities)
//!   pyrameth call_freq  — aggregate per-read calls to genome-level frequency
//!   pyrameth extract    — extract features to TSV (for downstream training/testing)

mod call_freq;
mod call_mods;
mod extract;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name    = "pyrameth",
    version = env!("CARGO_PKG_VERSION"),
    about   = "PyraMeth — fast nanopore methylation calling (Rust)",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect DNA methylation from signal files or pre-extracted TSV.
    CallMods(call_mods::CallModsArgs),
    /// Aggregate per-read calls to genome-level modification frequency.
    CallFreq(call_freq::CallFreqArgs),
    /// Extract signal features to TSV (for training / evaluation).
    Extract(extract::ExtractArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::CallMods(args) => call_mods::run(args),
        Commands::CallFreq(args) => call_freq::run(args),
        Commands::Extract(args)  => extract::run(args),
    }
}
