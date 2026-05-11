//! `ds3 call_mods` — unified inference sub-command.
//!
//! Mirrors Python `call_modifications.py::inference_ultra` CLI arguments.

use std::path::PathBuf;

use clap::Args;
use ds3_core::signal::NormalizeMethod;
use ds3_model::pipeline::{InferenceConfig, ModelClass, run_inference};

/// Arguments for `ds3 call_mods`.
#[derive(Args, Debug)]
pub struct CallModsArgs {
    // ── INPUT ──────────────────────────────────────────────────────────────
    /// Signal directory (pod5/slow5) or pre-extracted TSV file.
    #[arg(short = 'i', long)]
    pub input_path: PathBuf,

    /// Indexed BAM file (required for signal input, not needed for TSV).
    #[arg(long)]
    pub bam: Option<PathBuf>,

    // ── OUTPUT ─────────────────────────────────────────────────────────────
    /// Output file for per-read modification probabilities.
    #[arg(short = 'o', long)]
    pub result_file: PathBuf,

    // ── MODEL ──────────────────────────────────────────────────────────────
    /// TorchScript model file (.pt), exported by scripts/export_torchscript.py.
    #[arg(short = 'm', long)]
    pub model_path: PathBuf,

    /// Model architecture: 'mtm' (default) or 'bilstm'.
    #[arg(long, default_value = "mtm")]
    pub model_class: String,

    // ── MODEL HYPER-PARAMS ─────────────────────────────────────────────────
    /// k-mer window length (must be odd).
    #[arg(long, default_value_t = 21)]
    pub seq_len: usize,

    /// Signals per base in the rectangular matrix.
    #[arg(long, default_value_t = 15)]
    pub signal_len: usize,

    /// Base embedding dimension (n_embed). Must match the exported model.
    #[arg(long, default_value_t = 4)]
    pub n_embed: i64,

    // ── EXTRACTION ─────────────────────────────────────────────────────────
    /// Methylation motif(s), comma-separated. Default: CG (CpG).
    #[arg(long, default_value = "CG")]
    pub motifs: String,

    /// 0-based offset of modification base within the motif.
    #[arg(long, default_value_t = 0)]
    pub mod_loc: usize,

    /// Signal normalisation method: 'mad' (default) or 'zscore'.
    #[arg(long, default_value = "mad")]
    pub normalize_method: String,

    /// Minimum mapping quality.
    #[arg(long, default_value_t = 1)]
    pub mapq: u8,

    /// Minimum aligned/query length ratio.
    #[arg(long, default_value_t = 0.5)]
    pub coverage_ratio: f64,

    /// Minimum alignment identity (1 - NM/aln_len).
    #[arg(long, default_value_t = 0.0)]
    pub identity: f64,

    /// Modification label (0=unmod, 1=mod). Stamped into features; ignored at inference.
    #[arg(long, default_value_t = 1)]
    pub methy_label: u8,

    /// Optional TSV file of positions to restrict calling (chrom, pos, strand).
    #[arg(long)]
    pub positions: Option<PathBuf>,

    // ── PERFORMANCE ────────────────────────────────────────────────────────
    /// Batch size per GPU forward pass.
    #[arg(short = 'b', long, default_value_t = 500)]
    pub batch_size: usize,

    /// Number of IO / feature-extraction threads.
    #[arg(short = 'p', long, default_value_t = 10)]
    pub nproc: usize,

    /// Force CPU inference even when GPUs are present.
    #[arg(long)]
    pub use_cpu: bool,
}

pub fn run(args: CallModsArgs) -> anyhow::Result<()> {
    let model_class: ModelClass = args.model_class.parse().map_err(anyhow::Error::msg)?;
    let normalize_method: NormalizeMethod = args.normalize_method.parse().map_err(anyhow::Error::msg)?;

    let cfg = InferenceConfig {
        model_path:       args.model_path,
        model_class,
        n_embed:          args.n_embed,
        input_path:       args.input_path,
        bam_path:         args.bam,
        file_type:        None,
        result_file:      args.result_file,
        seq_len:          args.seq_len,
        signal_len:       args.signal_len,
        motifs:           args.motifs,
        mod_loc:          args.mod_loc,
        normalize_method,
        mapq:             args.mapq,
        coverage_ratio:   args.coverage_ratio,
        identity:         args.identity,
        methy_label:      args.methy_label,
        positions_file:   args.positions,
        batch_size:       args.batch_size,
        nproc_io:         args.nproc,
        use_cpu:          args.use_cpu,
    };

    run_inference(&cfg)
}
