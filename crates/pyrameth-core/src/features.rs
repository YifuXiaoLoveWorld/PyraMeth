//! Per-read feature extraction for modelMTM and ModelBiLSTM.
//!
//! Exact port of `utils_dataloader.py::process_data_fast` (MTM) and
//! `utils_dataloader.py::process_data_bilstm` (BiLSTM).

use crate::{
    cigar::get_q2tloc_from_cigar,
    error::Result,
    io::bam::AlnRecord,
    kmer::{encode_kmer, find_motif_sites},
    signal::{build_signal_rect, group_signals_by_movetable, normalize_signals, NormalizeMethod},
};

// ─── feature item types ───────────────────────────────────────────────────────

/// Feature item produced for modelMTM inference.
///
/// Matches the tuple `(sampleinfo, k_seq, k_signals_rect, label, tag)` from
/// Python `process_data_fast`.
#[derive(Debug, Clone)]
pub struct MtmFeature {
    /// Tab-separated `chrom\tpos\tstrand\t.\tread_name\t.`
    pub sample_info: String,
    /// Integer-encoded k-mer, shape `[seq_len]`, dtype i64.
    pub k_seq: Vec<i64>,
    /// Rectangular signal matrix, shape `[seq_len][signal_len]`, f32 (may contain NaN).
    pub k_signals: Vec<Vec<f32>>,
    /// Label (0 or 1; passed through from args, discarded at inference time).
    pub label: u8,
    /// Proximity tag: 1 if another motif site is within 10 bp, else 0.
    pub tag: u8,
}

/// Feature item produced for ModelBiLSTM inference.
///
/// Matches the tuple `(sampleinfo, k_seq, means, stds, lens, k_signals_rect, label)`
/// from Python `process_data_bilstm`.
#[derive(Debug, Clone)]
pub struct BiLstmFeature {
    /// Tab-separated `chrom\tpos\tstrand\t.\tread_name\t.`
    pub sample_info: String,
    /// Integer-encoded k-mer, shape `[seq_len]`, dtype i64.
    pub k_seq: Vec<i64>,
    /// Per-base signal mean, shape `[seq_len]`, f32.
    pub means: Vec<f32>,
    /// Per-base signal std, shape `[seq_len]`, f32.
    pub stds: Vec<f32>,
    /// Per-base signal length, shape `[seq_len]`, i32.
    pub lens: Vec<i32>,
    /// Rectangular signal matrix, shape `[seq_len][signal_len]`, f32 (may contain NaN).
    pub k_signals: Vec<Vec<f32>>,
    /// Label (0 or 1).
    pub label: u8,
}

// ─── extraction parameters ────────────────────────────────────────────────────

/// Shared extraction parameters (mirrors CLI args struct).
#[derive(Debug, Clone)]
pub struct ExtractionArgs {
    /// k-mer window length (must be odd).
    pub seq_len: usize,
    /// Signals per base in the rectangular matrix.
    pub signal_len: usize,
    /// Minimum mapping quality filter.
    pub mapq: u8,
    /// Minimum aligned-to-query-length ratio.
    pub coverage_ratio: f64,
    /// Minimum alignment identity (`1 - NM / aln_len`).
    pub identity: f64,
    /// 0-based offset of the modification base within the motif.
    pub mod_loc: usize,
    /// Expanded concrete motif sequences (from `kmer::get_motif_seqs`).
    pub motif_seqs: Vec<Vec<u8>>,
    /// Signal normalisation method.
    pub normalize_method: NormalizeMethod,
    /// Label stamped onto every feature (0=unmod, 1=mod; training only).
    pub methy_label: u8,
    /// Optional set of `"chrom\tpos\tstrand"` strings to restrict output.
    pub positions: Option<std::collections::HashSet<String>>,
}

// ─── shared pre-processing ────────────────────────────────────────────────────

/// Normalise, trim, and build the signal rectangle.
/// Returns `None` on any validation failure (matching Python's `return []`).
fn preprocess_signal(
    raw_signal: &[f32],
    aln: &AlnRecord,
    args: &ExtractionArgs,
) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
    // Trim signal (ts + sp tags → num_trimmed)
    let num_trimmed = aln.num_trimmed;
    let trimmed: &[f32] = if num_trimmed >= 0 {
        let t = num_trimmed as usize;
        if t >= raw_signal.len() { return None; }
        &raw_signal[t..]
    } else {
        let t = (-num_trimmed) as usize;
        if t >= raw_signal.len() { return None; }
        &raw_signal[..raw_signal.len() - t]
    };

    // Normalise
    let norm_signal = normalize_signals(trimmed, args.normalize_method);

    // Build rectangular signal matrix (NaN-padded)
    let signal_rect = build_signal_rect(&norm_signal, &aln.movetable, aln.mv_stride, args.signal_len);

    Some((norm_signal, signal_rect))
}

// ─── modelMTM feature extraction ─────────────────────────────────────────────

/// Extract MTM features for one (signal, BAM-record) pair.
///
/// Exact port of Python `process_data_fast`.
pub fn process_data_mtm(
    raw_signal: &[f32],
    aln: &AlnRecord,
    args: &ExtractionArgs,
) -> Result<Vec<MtmFeature>> {
    // mapq filter
    if aln.mapq < args.mapq {
        return Ok(Vec::new());
    }

    // Preprocess signal
    let Some((_, signal_rect)) = preprocess_signal(raw_signal, aln, args) else {
        return Ok(Vec::new());
    };

    // Sequence and motif sites
    let seq = &aln.forward_seq;
    let tsite_locs = find_motif_sites(seq, &args.motif_seqs, args.mod_loc);
    if tsite_locs.is_empty() {
        return Ok(Vec::new());
    }

    // Coverage ratio filter
    if !aln.is_unmapped() {
        let aln_len = aln.qa_end - aln.qa_start;
        if aln.query_length() == 0
            || (aln_len as f64 / aln.query_length() as f64) < args.coverage_ratio
        {
            return Ok(Vec::new());
        }
    }

    let num_bases = (args.seq_len - 1) / 2;

    // Reference coordinate mapping
    let coords = compute_ref_coords(aln, seq);

    let strand = if aln.is_unmapped() {
        '.'
    } else {
        aln.strand()
    };
    let ref_name = if aln.is_unmapped() { "." } else { &aln.ref_name };

    let mut out = Vec::new();

    for (i, &loc) in tsite_locs.iter().enumerate() {
        if !(num_bases <= loc && loc + num_bases < seq.len()) {
            continue;
        }

        let ref_pos = resolve_ref_pos(loc, &coords, aln);
        if ref_pos == i64::MIN {
            // insertion or out of alignment range
            continue;
        }

        // Position filter
        if let Some(ref pos_set) = args.positions {
            let key = format!("{ref_name}\t{ref_pos}\t{strand}");
            if !pos_set.contains(&key) {
                continue;
            }
        }

        // Proximity tag: 1 if another motif is within 10 bp
        let tag = if (i > 0 && loc - tsite_locs[i - 1] <= 10)
            || (i + 1 < tsite_locs.len() && tsite_locs[i + 1] - loc <= 10)
        {
            1u8
        } else {
            0u8
        };

        let k_mer = &seq[loc - num_bases..=loc + num_bases];
        let k_seq = encode_kmer(k_mer)?;
        let k_signals = signal_rect[loc - num_bases..=loc + num_bases].to_vec();

        let sample_info = format!(
            "{ref_name}\t{ref_pos}\t{strand}\t.\t{}\t.",
            String::from_utf8_lossy(aln.read_id.as_bytes()),
        );

        out.push(MtmFeature { sample_info, k_seq, k_signals, label: args.methy_label, tag });
    }

    Ok(out)
}

// ─── ModelBiLSTM feature extraction ──────────────────────────────────────────

/// Extract BiLSTM features for one (signal, BAM-record) pair.
///
/// Exact port of Python `process_data_bilstm`.
pub fn process_data_bilstm(
    raw_signal: &[f32],
    aln: &AlnRecord,
    args: &ExtractionArgs,
) -> Result<Vec<BiLstmFeature>> {
    if aln.mapq < args.mapq {
        return Ok(Vec::new());
    }

    let Some((norm_signal, signal_rect)) = preprocess_signal(raw_signal, aln, args) else {
        return Ok(Vec::new());
    };

    // Variable-length signal groups for mean/std/len
    let signal_group = match group_signals_by_movetable(&norm_signal, &aln.movetable, aln.mv_stride) {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };

    let seq = &aln.forward_seq;
    let tsite_locs = find_motif_sites(seq, &args.motif_seqs, args.mod_loc);
    if tsite_locs.is_empty() {
        return Ok(Vec::new());
    }

    if !aln.is_unmapped() {
        let aln_len = aln.qa_end - aln.qa_start;
        if aln.query_length() == 0
            || (aln_len as f64 / aln.query_length() as f64) < args.coverage_ratio
        {
            return Ok(Vec::new());
        }
    }

    let num_bases = (args.seq_len - 1) / 2;
    let coords = compute_ref_coords(aln, seq);

    let strand = if aln.is_unmapped() { '.' } else { aln.strand() };
    let ref_name = if aln.is_unmapped() { "." } else { &aln.ref_name };

    let mut out = Vec::new();

    for &loc in &tsite_locs {
        if !(num_bases <= loc && loc + num_bases < seq.len()) {
            continue;
        }

        let ref_pos = resolve_ref_pos(loc, &coords, aln);
        if ref_pos == i64::MIN {
            continue;
        }

        if let Some(ref pos_set) = args.positions {
            let key = format!("{ref_name}\t{ref_pos}\t{strand}");
            if !pos_set.contains(&key) {
                continue;
            }
        }

        let k_mer = &seq[loc - num_bases..=loc + num_bases];
        let k_seq = encode_kmer(k_mer)?;

        // Variable-length groups for this k-mer window
        let k_sigs_v = &signal_group[loc - num_bases..=loc + num_bases];
        let means: Vec<f32> = k_sigs_v.iter().map(|g| mean_f32(g)).collect();
        let stds:  Vec<f32> = k_sigs_v.iter().map(|g| std_f32(g)).collect();
        let lens:  Vec<i32> = k_sigs_v.iter().map(|g| g.len() as i32).collect();
        let k_signals = signal_rect[loc - num_bases..=loc + num_bases].to_vec();

        let sample_info = format!(
            "{ref_name}\t{ref_pos}\t{strand}\t.\t{}\t.",
            String::from_utf8_lossy(aln.read_id.as_bytes()),
        );

        out.push(BiLstmFeature {
            sample_info, k_seq, means, stds, lens, k_signals,
            label: args.methy_label,
        });
    }

    Ok(out)
}

// ─── statistics helpers ───────────────────────────────────────────────────────

/// Arithmetic mean of a signal group (matches `np.mean`).
fn mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() { return 0.0; }
    v.iter().map(|&x| x as f64).sum::<f64>() as f32 / v.len() as f32
}

/// Population standard deviation (matches `np.std`, ddof=0).
fn std_f32(v: &[f32]) -> f32 {
    if v.len() < 2 { return 0.0; }
    let m = mean_f32(v) as f64;
    let var = v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len() as f64;
    var.sqrt() as f32
}

// ─── reference coordinate resolution ─────────────────────────────────────────

/// Pre-computed reference coordinate info for a read.
struct RefCoords {
    q_to_r_poss: Vec<i32>,
    seq_start: usize,
    seq_end: usize,
}

/// Compute reference coordinate mapping for a mapped read.
/// Returns `None` for unmapped reads.
fn compute_ref_coords(aln: &AlnRecord, seq: &[u8]) -> Option<RefCoords> {
    if aln.is_unmapped() {
        return None;
    }

    let strand_code: i8 = if aln.is_reverse() { -1 } else { 1 };
    let seq_len = aln.qa_end - aln.qa_start;

    let q_to_r_poss = get_q2tloc_from_cigar(&aln.cigar, strand_code, seq_len).ok()?;

    Some(RefCoords {
        q_to_r_poss,
        seq_start: aln.qa_start,
        seq_end: aln.qa_end,
    })
}

/// Resolve a query position `loc` to a reference position.
///
/// Returns `i64::MIN` for:
/// * insertion into reference (`-1` sentinel from CIGAR)
/// * `loc` outside the aligned region
/// * unmapped read
fn resolve_ref_pos(loc: usize, coords: &Option<RefCoords>, aln: &AlnRecord) -> i64 {
    let Some(c) = coords else {
        return -1; // unmapped → ref_pos = -1 (Python default)
    };

    if !(c.seq_start <= loc && loc < c.seq_end) {
        return i64::MIN; // outside alignment region → skip
    }

    let rpos = c.q_to_r_poss[loc - c.seq_start];
    if rpos == -1 {
        // insertion
        return i64::MIN;
    }

    if aln.is_reverse() {
        aln.ref_end - 1 - rpos as i64
    } else {
        aln.ref_start + rpos as i64
    }
}

// ─── TSV serialisation (for `extract` subcommand) ────────────────────────────

/// Serialise a BiLSTM feature to the 12-column TSV format used by `extract`.
///
/// Columns: chrom, pos, strand, pos_in_strand, readname, read_loc,
///          k_mer, signal_means, signal_stds, signal_lens, k_signals_rect, methy_label.
///
/// Matches Python `_features_to_str` in `extract_features_pod5.py`.
/// NaN padding in `k_signals_rect` is replaced with `0.0` for TSV
/// compatibility (same as Python `np.where(np.isnan(...), 0.0, ...)`).
pub fn bilstm_feature_to_tsv(f: &BiLstmFeature, k_mer: &[u8]) -> String {
    let parts: Vec<&str> = f.sample_info.splitn(6, '\t').collect();
    let (chrom, pos, strand) = (
        parts.get(0).copied().unwrap_or("."),
        parts.get(1).copied().unwrap_or("-1"),
        parts.get(2).copied().unwrap_or("."),
    );

    let means_text = f
        .means
        .iter()
        .map(|x| format!("{:.6}", x))
        .collect::<Vec<_>>()
        .join(",");
    let stds_text = f
        .stds
        .iter()
        .map(|x| format!("{:.6}", x))
        .collect::<Vec<_>>()
        .join(",");
    let lens_text = f
        .lens
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let signals_text = f
        .k_signals
        .iter()
        .map(|row| {
            row.iter()
                .map(|&v| format!("{:.6}", if v.is_nan() { 0.0 } else { v }))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join(";");

    format!(
        "{chrom}\t{pos}\t{strand}\t.\t{readname}\t.\t{kmer}\t{means}\t{stds}\t{lens}\t{sigs}\t{label}",
        readname = parts.get(4).copied().unwrap_or("."),
        kmer     = String::from_utf8_lossy(k_mer),
        means    = means_text,
        stds     = stds_text,
        lens     = lens_text,
        sigs     = signals_text,
        label    = f.label,
    )
}
