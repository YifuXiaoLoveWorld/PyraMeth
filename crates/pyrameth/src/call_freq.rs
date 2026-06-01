//! `pyrameth call_freq` — aggregate per-read modification calls to genome-level
//! modification frequency.
//!
//! Two modes:
//!
//! | Mode          | Flag             | Description                              |
//! |---------------|------------------|------------------------------------------|
//! | Count-based   | *(default)*      | Simple met/unmet ratio per site          |
//! | AggrAttRNN    | `--aggre_model`  | Neural-network refined frequency (bedMethyl) |
//!
//! Mirrors Python `call_mods_freq.py`.

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use clap::Args;
use pyrameth_model::{normalized_histogram, run_aggr_model};

// ─── CLI args ─────────────────────────────────────────────────────────────────

/// Arguments for `pyrameth call_freq`.
#[derive(Args, Debug)]
pub struct CallFreqArgs {
    /// Input file(s) or directory from `pyrameth call_mods` output.
    /// Can be specified multiple times.
    #[arg(short = 'i', long = "input_path", required = true, num_args = 1..)]
    pub input_paths: Vec<PathBuf>,

    /// Output file path.
    #[arg(short = 'o', long)]
    pub result_file: PathBuf,

    /// Remove ambiguous calls where |prob1 - prob0| < prob_cf.
    #[arg(long, default_value_t = 0.0)]
    pub prob_cf: f64,

    /// Write output in bedMethyl format (count mode only; aggre mode always uses bedMethyl).
    #[arg(long)]
    pub bed: bool,

    /// Sort output by chromosome and position.
    #[arg(long)]
    pub sort: bool,

    /// Filter files in a directory by this substring.
    #[arg(long)]
    pub file_uid: Option<String>,

    // ── AggrAttRNN options ────────────────────────────────────────────────────

    /// Path to AggrAttRNN TorchScript model (.pt).
    /// When provided, uses neural-network frequency refinement and writes bedMethyl.
    /// Export via: python scripts/export_torchscript.py --model_class aggr
    #[arg(long)]
    pub aggre_model: Option<PathBuf>,

    /// Minimum read coverage per site for aggregate mode.
    #[arg(long, default_value_t = 4)]
    pub cov_cf: usize,

    /// Histogram bin count for aggregate mode (must match model training).
    #[arg(long, default_value_t = 20)]
    pub bin_size: usize,

    /// Sliding window length for aggregate mode (must match model training).
    #[arg(long, default_value_t = 11)]
    pub aggr_seq_len: usize,

    /// Inference batch size for aggregate mode.
    #[arg(long, default_value_t = 1024)]
    pub aggr_batch_size: usize,
}

// ─── entry point ──────────────────────────────────────────────────────────────

pub fn run(args: CallFreqArgs) -> anyhow::Result<()> {
    let files = collect_files(&args.input_paths, args.file_uid.as_deref())?;
    log::info!("Aggregating {} input file(s)…", files.len());

    if let Some(ref model_path) = args.aggre_model {
        run_aggre_mode(&args, &files, model_path)
    } else {
        run_count_mode(&args, &files)
    }
}

// ─── count-based mode (default) ───────────────────────────────────────────────

fn run_count_mode(args: &CallFreqArgs, files: &[PathBuf]) -> anyhow::Result<()> {
    let site_stats = calculate_freq(files, args.prob_cf)?;
    log::info!("{} sites found, writing output…", site_stats.len());
    write_count_stats(&site_stats, &args.result_file, args.sort, args.bed)?;
    log::info!("Done.");
    Ok(())
}

// ─── AggrAttRNN mode ──────────────────────────────────────────────────────────

fn run_aggre_mode(
    args: &CallFreqArgs,
    files: &[PathBuf],
    model_path: &PathBuf,
) -> anyhow::Result<()> {
    log::info!("Loading AggrAttRNN model from {:?}…", model_path);
    let device = tch::Device::Cpu;
    let model  = tch::CModule::load(model_path)
        .map_err(|e| anyhow::anyhow!("failed to load aggre model: {e}"))?;

    log::info!("Reading input files…");
    let track_map = prepare_site_data(files, args.prob_cf, args.cov_cf, args.bin_size)?;
    log::info!("{} chromosome-strand tracks found.", track_map.len());

    log::info!("Running AggrAttRNN inference…");
    let mut refined: HashMap<TrackKey, Vec<f32>> = HashMap::new();
    for (key, track) in &track_map {
        let probs = run_aggr_model(
            &track.positions,
            &track.histograms,
            &model,
            device,
            args.aggr_seq_len,
            args.bin_size,
            args.aggr_batch_size,
        )?;
        refined.insert(key.clone(), probs);
    }

    log::info!("Writing bedMethyl…");
    write_bedmethyl_aggr(&track_map, &refined, &args.result_file, args.sort)?;
    log::info!("Done.");
    Ok(())
}

// ─── site data preparation ────────────────────────────────────────────────────

/// Chromosome + strand identifier.
type TrackKey = (String, char);

/// Per-site data for one chromosome-strand track.
struct SiteTrack {
    /// Sorted genomic positions (0-based).
    positions: Vec<i64>,
    /// L2-normalised probability histograms, one per site.
    histograms: Vec<Vec<f32>>,
    /// Total read coverage per site (for bedMethyl output).
    coverages: Vec<u32>,
    /// Strand position string per site (for bedMethyl output).
    pos_in_strands: Vec<String>,
}

/// Read per-read TSV, aggregate per site, build histograms.
///
/// Input line format (from `pyrameth call_mods` output):
/// `chrom\tpos\tstrand\tpos_in_strand\treadname\tread_loc\tprob0\tprob1\tpred\tkmer5`
fn prepare_site_data(
    files:    &[PathBuf],
    prob_cf:  f64,
    cov_cf:   usize,
    bin_size: usize,
) -> anyhow::Result<HashMap<TrackKey, SiteTrack>> {
    // First pass: accumulate per-site prob_1 values.
    // Key: (chrom, strand) → BTreeMap<pos, (pos_in_strand, Vec<prob_1>)>
    type PosMap = std::collections::BTreeMap<i64, (String, Vec<f32>)>;
    let mut raw: HashMap<TrackKey, PosMap> = HashMap::new();

    for path in files {
        let reader = BufReader::new(File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() { continue; }

            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 9 { continue; }

            let prob_0: f64 = cols[6].parse().unwrap_or(0.5);
            let prob_1: f64 = cols[7].parse().unwrap_or(0.5);

            // Confidence filter
            if (prob_1 - prob_0).abs() < prob_cf {
                continue;
            }

            let chrom         = cols[0].to_owned();
            let pos: i64      = cols[1].parse().unwrap_or(-1);
            let strand        = cols[2].chars().next().unwrap_or('+');
            let pos_in_strand = cols[3].to_owned();

            raw.entry((chrom, strand))
                .or_default()
                .entry(pos)
                .or_insert_with(|| (pos_in_strand, Vec::new()))
                .1
                .push(prob_1 as f32);
        }
    }

    // Second pass: filter by cov_cf and build histograms.
    let mut tracks: HashMap<TrackKey, SiteTrack> = HashMap::new();

    for (key, pos_map) in raw {
        let mut positions    = Vec::new();
        let mut histograms   = Vec::new();
        let mut coverages    = Vec::new();
        let mut pos_in_strands = Vec::new();

        for (pos, (pis, probs)) in pos_map {
            if probs.len() < cov_cf { continue; }
            positions.push(pos);
            coverages.push(probs.len() as u32);
            pos_in_strands.push(pis);
            histograms.push(normalized_histogram(&probs, bin_size));
        }

        if !positions.is_empty() {
            tracks.insert(key, SiteTrack { positions, histograms, coverages, pos_in_strands });
        }
    }

    Ok(tracks)
}

// ─── bedMethyl output (aggre mode) ────────────────────────────────────────────

fn write_bedmethyl_aggr(
    tracks:      &HashMap<TrackKey, SiteTrack>,
    refined:     &HashMap<TrackKey, Vec<f32>>,
    result_file: &PathBuf,
    is_sort:     bool,
) -> anyhow::Result<()> {
    // Collect all rows: (chrom, pos, strand, cov, freq_pct, pos_in_strand)
    struct Row {
        chrom:          String,
        pos:            i64,
        strand:         char,
        cov:            u32,
        freq_pct:       u32,
        _pos_in_strand: String,
    }

    let mut rows: Vec<Row> = Vec::new();

    for ((chrom, strand), track) in tracks {
        let probs = match refined.get(&(chrom.clone(), *strand)) {
            Some(p) => p,
            None    => continue,
        };
        for (i, (&pos, &cov)) in track
            .positions
            .iter()
            .zip(track.coverages.iter())
            .enumerate()
        {
            let freq   = *probs.get(i).unwrap_or(&0.0);
            let pct    = (freq * 100.0 + 0.001).round() as u32;
            rows.push(Row {
                chrom:          chrom.clone(),
                pos,
                strand:         *strand,
                cov,
                freq_pct:       pct,
                _pos_in_strand: track.pos_in_strands[i].clone(),
            });
        }
    }

    if is_sort {
        rows.sort_by(|a, b| {
            a.chrom.cmp(&b.chrom)
                .then_with(|| a.pos.cmp(&b.pos))
        });
    }

    let mut wf = BufWriter::new(File::create(result_file)?);
    for r in &rows {
        writeln!(
            wf,
            "{}\t{}\t{}\t.\t{}\t{}\t{}\t{}\t0,0,0\t{}\t{}",
            r.chrom, r.pos, r.pos + 1,
            r.cov, r.strand,
            r.pos, r.pos + 1,
            r.cov, r.freq_pct,
        )?;
    }
    Ok(())
}

// ─── count-based aggregation (unchanged logic) ────────────────────────────────

#[derive(Default)]
struct SiteStats {
    strand:        char,
    pos_in_strand: String,
    kmer:          String,
    prob_0:        f64,
    prob_1:        f64,
    met:           u64,
    unmet:         u64,
}

impl SiteStats {
    fn coverage(&self) -> u64 { self.met + self.unmet }
    fn methylation_freq(&self) -> f64 {
        if self.coverage() == 0 { 0.0 } else { self.met as f64 / self.coverage() as f64 }
    }
}

fn split_key(key: &str) -> (String, i64) {
    let mut parts = key.splitn(3, "||");
    let chrom = parts.next().unwrap_or("").to_string();
    let pos: i64 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    (chrom, pos)
}

fn calculate_freq(files: &[PathBuf], prob_cf: f64) -> anyhow::Result<HashMap<String, SiteStats>> {
    let mut stats: HashMap<String, SiteStats> = HashMap::new();
    let mut total = 0u64;
    let mut used  = 0u64;

    for path in files {
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            if line.is_empty() { continue; }
            total += 1;

            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 9 { continue; }

            let prob_0: f64 = cols[6].parse().unwrap_or(0.5);
            let prob_1: f64 = cols[7].parse().unwrap_or(0.5);
            let pred:   i32 = cols[8].parse().unwrap_or(0);

            if (prob_1 - prob_0).abs() < prob_cf { continue; }
            used += 1;

            let chrom         = cols[0];
            let pos: i64      = cols[1].parse().unwrap_or(-1);
            let strand        = cols[2];
            let pos_in_strand = cols[3].to_string();
            let kmer5         = cols.get(9).copied().unwrap_or("").to_string();

            let key = format!("{chrom}||{pos}||{strand}");
            let s = stats.entry(key).or_insert_with(|| SiteStats {
                strand: strand.chars().next().unwrap_or('+'),
                pos_in_strand,
                kmer: kmer5,
                ..Default::default()
            });
            s.prob_0 += prob_0;
            s.prob_1 += prob_1;
            if pred == 1 { s.met += 1; } else { s.unmet += 1; }
        }
    }

    if total > 0 {
        log::info!(
            "{:.2}% ({}/{}) calls used after filtering.",
            used as f64 / total as f64 * 100.0, used, total,
        );
    }
    Ok(stats)
}

fn write_count_stats(
    stats:       &HashMap<String, SiteStats>,
    result_file: &PathBuf,
    is_sort:     bool,
    is_bed:      bool,
) -> anyhow::Result<()> {
    let mut keys: Vec<&str> = stats.keys().map(String::as_str).collect();
    if is_sort { keys.sort_by_key(|k| split_key(k)); }

    let mut wf = BufWriter::new(File::create(result_file)?);
    for key in keys {
        let (chrom, pos) = split_key(key);
        let s   = &stats[key];
        let cov = s.coverage();
        if cov == 0 { continue; }
        let rmet = s.methylation_freq();

        if is_bed {
            writeln!(
                wf,
                "{chrom}\t{pos}\t{}\t.\t{cov}\t{}\t{pos}\t{}\t0,0,0\t{cov}\t{}",
                pos + 1, s.strand, pos + 1,
                (rmet * 100.0 + 0.001).round() as u32,
            )?;
        } else {
            writeln!(
                wf,
                "{chrom}\t{pos}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{cov}\t{:.4}\t{}",
                s.strand, s.pos_in_strand,
                s.prob_0, s.prob_1, s.met, s.unmet, rmet, s.kmer,
            )?;
        }
    }
    Ok(())
}

// ─── file collection ──────────────────────────────────────────────────────────

fn collect_files(input_paths: &[PathBuf], file_uid: Option<&str>) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for ipath in input_paths {
        let abs = ipath.canonicalize()?;
        if abs.is_file() {
            files.push(abs);
        } else if abs.is_dir() {
            for entry in std::fs::read_dir(&abs)? {
                let e    = entry?;
                let name = e.file_name().to_string_lossy().into_owned();
                if file_uid.map_or(true, |uid| name.contains(uid)) {
                    files.push(e.path());
                }
            }
        } else {
            anyhow::bail!("input_path not found: {:?}", ipath);
        }
    }
    Ok(files)
}
