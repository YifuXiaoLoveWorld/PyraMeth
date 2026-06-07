//! `pyrameth extract` — extract signal features to TSV.
//!
//! Mirrors Python `extract_features_pod5.py::extract_features`.
//! Output: 12-column TSV suitable for training / evaluation.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use clap::Args;
use rayon::prelude::*;

use pyrameth_core::{
    error::Result as Ds3Result,
    features::{
        bilstm_feature_to_tsv, process_data_bilstm, ExtractionArgs,
    },
    io::{
        bam::{BamSearcher, ReadIndexedBam},
        pod5::iter_pod5_filtered,
        slow5::read_slow5,
        RawRead,
    },
    kmer::get_motif_seqs,
    signal::NormalizeMethod,
};

// ─── CLI args ─────────────────────────────────────────────────────────────────

/// Arguments for `pyrameth extract`.
#[derive(Args, Debug)]
pub struct ExtractArgs {
    // ── INPUT ──────────────────────────────────────────────────────────────
    /// Directory of signal files (pod5 / slow5 / blow5).
    #[arg(short = 'i', long)]
    pub input_dir: PathBuf,

    /// Indexed BAM file.
    #[arg(long)]
    pub bam: PathBuf,

    // ── OUTPUT ─────────────────────────────────────────────────────────────
    /// Output feature TSV file.
    #[arg(short = 'o', long)]
    pub write_path: PathBuf,

    // ── EXTRACTION ─────────────────────────────────────────────────────────
    /// Signal normalisation method.
    #[arg(long, default_value = "mad")]
    pub normalize_method: String,

    /// Modification label (0=unmod, 1=mod).
    #[arg(long, default_value_t = 1)]
    pub methy_label: u8,

    /// k-mer window length (must be odd).
    #[arg(long, default_value_t = 21)]
    pub seq_len: usize,

    /// Signals per base in rectangular matrix.
    #[arg(long, default_value_t = 15)]
    pub signal_len: usize,

    /// Methylation motif(s), comma-separated.
    #[arg(long, default_value = "CG")]
    pub motifs: String,

    /// 0-based offset of modification base within the motif.
    #[arg(long, default_value_t = 0)]
    pub mod_loc: usize,

    /// Optional position filter file.
    #[arg(long)]
    pub positions: Option<PathBuf>,

    // ── MAPPING FILTERS ────────────────────────────────────────────────────
    /// Minimum mapping quality.
    #[arg(long, default_value_t = 1)]
    pub mapq: u8,

    /// Minimum alignment identity.
    #[arg(long, default_value_t = 0.0)]
    pub identity: f64,

    /// Minimum aligned/query length ratio.
    #[arg(long, default_value_t = 0.5)]
    pub coverage_ratio: f64,

    // ── PERFORMANCE ────────────────────────────────────────────────────────
    /// Number of parallel extraction threads.
    #[arg(short = 'p', long, default_value_t = 10)]
    pub nproc: usize,
}

// ─── entry point ─────────────────────────────────────────────────────────────

pub fn run(args: ExtractArgs) -> anyhow::Result<()> {
    let normalize_method: NormalizeMethod = args.normalize_method.parse().map_err(anyhow::Error::msg)?;
    let motif_seqs = get_motif_seqs(&args.motifs);

    // Load optional position filter
    let positions = args
        .positions
        .as_ref()
        .map(|p| load_positions(p))
        .transpose()?;

    let ext_args = Arc::new(ExtractionArgs {
        seq_len:          args.seq_len,
        signal_len:       args.signal_len,
        mapq:             args.mapq,
        coverage_ratio:   args.coverage_ratio,
        identity:         args.identity,
        mod_loc:          args.mod_loc,
        motif_seqs,
        normalize_method,
        methy_label:      args.methy_label,
        positions,
    });

    // Build BAM index
    log::info!("Building BAM index …");
    let bam_index = Arc::new(ReadIndexedBam::open(&args.bam)?);
    log::info!("BAM index built ({} reads).", bam_index.num_records);

    // Detect file type and collect files
    let (file_type, signal_files) = detect_and_collect(&args.input_dir)?;
    log::info!("Found {} {} files.", signal_files.len(), file_type);

    // Shared writer (protected by mutex for parallel writes)
    let writer = Arc::new(Mutex::new(BufWriter::new(File::create(&args.write_path)?)));

    // Configure rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.nproc)
        .build_global()
        .ok(); // ignore if already configured

    // Process files in parallel.
    // Each task creates its own BamSearcher — one BAM file open per pod5 file
    // instead of one open per read (the previous bottleneck).
    signal_files
        .par_iter()
        .try_for_each(|file| -> anyhow::Result<()> {
            let mut searcher = BamSearcher::new(&bam_index)?;
            match file_type.as_str() {
                // Hand the BAM-membership test to the POD5 reader so signal for
                // reads absent from the BAM is never decompressed; the iterator
                // decodes lazily, one read at a time.
                "pod5" => process_reads(
                    iter_pod5_filtered(file, |rid| bam_index.contains(rid))?,
                    &mut searcher,
                    &ext_args,
                    &writer,
                ),
                "slow5" => process_reads(
                    read_slow5(file)?.into_iter().map(|r| -> Ds3Result<RawRead> { Ok(r) }),
                    &mut searcher,
                    &ext_args,
                    &writer,
                ),
                other => anyhow::bail!("unsupported file type '{other}'"),
            }
        })?;

    // Ensure final flush
    writer.lock().unwrap().flush()?;
    log::info!("Extract finished → {:?}", args.write_path);
    Ok(())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn process_reads(
    reads: impl Iterator<Item = Ds3Result<RawRead>>,
    searcher: &mut BamSearcher,
    ext_args: &ExtractionArgs,
    writer: &Arc<Mutex<BufWriter<File>>>,
) -> anyhow::Result<()> {
    // Reuse a single String buffer across reads to avoid per-read allocation.
    // The mutex is acquired once per read instead of once per output line,
    // which dramatically reduces lock contention across parallel threads.
    let mut local_buf = String::with_capacity(8192);

    for raw_read_result in reads {
        let raw_read = raw_read_result?;
        let alignments = match searcher.get_alignments(&raw_read.read_id) {
            Ok(a) if a.is_empty() => continue,
            Ok(a)  => a,
            Err(_) => continue,
        };

        local_buf.clear();
        for aln in &alignments {
            let feats = process_data_bilstm(&raw_read.signal, aln, ext_args)?;
            for feat in &feats {
                let k_mer: Vec<u8> = feat
                    .k_seq
                    .iter()
                    .filter_map(|&c| pyrameth_core::kmer::code_to_base(c as u64))
                    .collect();
                local_buf.push_str(&bilstm_feature_to_tsv(feat, &k_mer));
                local_buf.push('\n');
            }
        }
        if !local_buf.is_empty() {
            let mut wf = writer.lock().unwrap();
            wf.write_all(local_buf.as_bytes())?;
        }
    }
    Ok(())
}

fn detect_and_collect(input_dir: &PathBuf) -> anyhow::Result<(String, Vec<PathBuf>)> {
    let mut pod5 = Vec::new();
    let mut slow5 = Vec::new();

    for entry in walkdir::WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let p = entry.path().to_path_buf();
        let name = p.to_string_lossy();
        if name.ends_with(".pod5")  { pod5.push(p); }
        else if name.ends_with(".slow5") || name.ends_with(".blow5") { slow5.push(p); }
    }

    if !pod5.is_empty() && slow5.is_empty() {
        Ok(("pod5".to_string(), pod5))
    } else if !slow5.is_empty() && pod5.is_empty() {
        Ok(("slow5".to_string(), slow5))
    } else if pod5.is_empty() && slow5.is_empty() {
        anyhow::bail!("No signal files (pod5/slow5/blow5) found in {:?}", input_dir)
    } else {
        anyhow::bail!("Mixed pod5 and slow5 files in {:?}", input_dir)
    }
}

fn load_positions(
    path: &PathBuf,
) -> anyhow::Result<std::collections::HashSet<String>> {
    use std::io::{BufRead, BufReader};
    let mut set = std::collections::HashSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let cols: Vec<&str> = line.trim().splitn(4, '\t').collect();
        if cols.len() >= 3 {
            set.insert(format!("{}\t{}\t{}", cols[0], cols[1], cols[2]));
        }
    }
    Ok(set)
}
