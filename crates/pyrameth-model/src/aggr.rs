//! AggrAttRNN frequency-refinement inference.
//!
//! Mirrors Python `call_mods_freq.py::_run_aggr_model`.
//!
//! # Model overview
//!
//! For every genomic site the model looks at a sliding window of `seq_len`
//! (default 11) neighbouring sites on the same chromosome+strand.  Each site
//! is represented by:
//!
//! - a **normalised probability histogram** (`binsize` bins, L2-normalised)
//! - the **absolute position distance** to the centre site (scalar)
//!
//! These are concatenated → `(N, seq_len, binsize + 1)` and fed through a
//! bidirectional GRU with Bahdanau attention → a refined methylation frequency
//! ∈ [0, 1] per site.
//!
//! # TorchScript calling convention
//!
//! ```text
//! model.forward(offsets: Tensor[N, L], histos: Tensor[N, L, binsize])
//!              -> Tensor[N, 1]   (clamped to [0, 1])
//! ```
//!
//! Export via `scripts/export_torchscript.py --model_class aggr`.

use anyhow::Context;
use tch::{Device, Kind, Tensor, no_grad};

// ─── public API ──────────────────────────────────────────────────────────────

/// Run AggrAttRNN on one chromosome-strand track.
///
/// * `positions`  — sorted genomic positions of sites in this track
/// * `histograms` — per-site L2-normalised probability histograms;
///                  `histograms[i].len() == binsize`
/// * `model`      — TorchScript `CModule` loaded from `.pt`
/// * `device`     — `Device::Cpu` or `Device::Cuda(i)`
/// * `seq_len`    — context window (default 11; must match model)
/// * `binsize`    — histogram bins (default 20; must match model)
/// * `batch_size` — inference batch size (default 1024)
///
/// Returns one refined frequency per site (same order as `positions`).
pub fn run_aggr_model(
    positions:  &[i64],
    histograms: &[Vec<f32>],
    model:      &tch::CModule,
    device:     Device,
    seq_len:    usize,
    binsize:    usize,
    batch_size: usize,
) -> anyhow::Result<Vec<f32>> {
    let n = positions.len();
    assert_eq!(histograms.len(), n, "positions/histograms length mismatch");

    if n == 0 {
        return Ok(Vec::new());
    }

    let pad = seq_len / 2; // = 5 for seq_len=11

    // ── build padded position windows ────────────────────────────────────────
    // Python: np.pad(pos_arr, (pad, pad), constant_values=(pos[0]-10000, pos[-1]+10000))
    let first = positions[0];
    let last  = positions[n - 1];
    let mut pos_padded: Vec<i64> = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad { pos_padded.push(first - 10000 * (pad - i) as i64); }
    pos_padded.extend_from_slice(positions);
    for i in 1..=pad { pos_padded.push(last + 10000 * i as i64); }

    // ── build padded histogram windows ────────────────────────────────────────
    // Python: np.pad(hist_mat, ((pad, pad), (0, 0)))
    let zero_hist = vec![0.0f32; binsize];
    let mut hist_padded: Vec<&Vec<f32>> = Vec::with_capacity(n + 2 * pad);
    for _ in 0..pad { hist_padded.push(&zero_hist); }
    for h in histograms { hist_padded.push(h); }
    for _ in 0..pad { hist_padded.push(&zero_hist); }

    // ── sliding windows (N, seq_len) ─────────────────────────────────────────
    // hist_windows[i] = hist_padded[i .. i+seq_len]
    // pos_dist[i][j]  = |pos_padded[i+j] - positions[i]|
    let mut hist_windows: Vec<f32> = Vec::with_capacity(n * seq_len * binsize);
    let mut pos_dist:     Vec<f32> = Vec::with_capacity(n * seq_len);

    for i in 0..n {
        let center = positions[i];
        for j in 0..seq_len {
            // position distance
            pos_dist.push((pos_padded[i + j] - center).unsigned_abs() as f32);
            // histogram
            hist_windows.extend_from_slice(hist_padded[i + j]);
        }
    }

    // ── batched TorchScript inference ─────────────────────────────────────────
    let l   = seq_len as i64;
    let bs  = binsize as i64;
    let mut results: Vec<f32> = Vec::with_capacity(n);

    for chunk_start in (0..n).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(n);
        let b = (chunk_end - chunk_start) as i64;

        // Slice the pre-flattened arrays.
        let hist_slice = &hist_windows[chunk_start * seq_len * binsize
                                       ..chunk_end * seq_len * binsize];
        let pos_slice  = &pos_dist[chunk_start * seq_len..chunk_end * seq_len];

        let histos_t  = Tensor::from_slice(hist_slice)
            .reshape([b, l, bs])
            .to_device(device)
            .to_kind(Kind::Float);
        let offsets_t = Tensor::from_slice(pos_slice)
            .reshape([b, l])
            .to_device(device)
            .to_kind(Kind::Float);

        let prob_tensor: Tensor = no_grad(|| {
            model
                .forward_ts(&[offsets_t, histos_t])
                .context("AggrAttRNN forward pass failed")
        })?;

        let probs: Vec<f32> = Vec::try_from(
            prob_tensor
                .clamp(0.0, 1.0)
                .to_kind(Kind::Float)
                .to_device(Device::Cpu)
                .flatten(0, -1),
        )
        .context("failed to convert model output to Vec<f32>")?;

        results.extend(probs);
    }

    Ok(results)
}

// ─── histogram helpers ────────────────────────────────────────────────────────

/// Build a L2-normalised probability histogram, matching Python:
/// ```python
/// hist, _ = np.histogram(probs, bins=binsize, range=[0., 1.])
/// norm = np.linalg.norm(hist)
/// return np.round(hist / (norm + 1e-8), 6)
/// ```
pub fn normalized_histogram(probs: &[f32], binsize: usize) -> Vec<f32> {
    let mut hist = vec![0u32; binsize];
    for &p in probs {
        // numpy histogram: bins are half-open [lo, hi) except the last which is [lo, hi]
        let idx = if p >= 1.0 {
            binsize - 1
        } else {
            (p * binsize as f32).floor() as usize
        };
        hist[idx.min(binsize - 1)] += 1;
    }
    // L2 norm of integer counts (same as numpy.linalg.norm on integer array)
    let norm = (hist.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sqrt();
    hist.iter()
        .map(|&v| {
            let r = v as f64 / (norm + 1e-8);
            (r * 1e6).round() as f32 / 1e6
        })
        .collect()
}
