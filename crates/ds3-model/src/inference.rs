//! Single-GPU batch inference helpers for modelMTM and ModelBiLSTM.
//!
//! Mirrors Python `_run_batch_mtm` and `_run_batch_bilstm` in
//! `call_modifications.py`.  All tensor operations match the Python logic
//! exactly — same reshape, expand, and mask construction.

use tch::{Device, Kind, Tensor, no_grad};

use ds3_core::{
    features::{BiLstmFeature, MtmFeature},
    kmer::{code_to_base, central5mer},
};

/// Result line in the output TSV format:
/// `chrom\tpos\tstrand\t.\treadname\t.\tprob0\tprob1\tpred_label\tkmer5`
pub type ResultLine = String;

// ─── modelMTM inference ───────────────────────────────────────────────────────

/// Run one forward pass for modelMTM on a batch of MTM features.
///
/// Exact port of Python `_run_batch_mtm`.
///
/// Input shapes:
/// * `k_seq`    : `[B, L]`      i64
/// * `k_signals`: `[B, L, S]`   f32 (may contain NaN)
/// * `tag`      : `[B]`         i64 (proximity tag: 0 or 1)
///
/// The function expands the kmer codes to match the signal time axis
/// (`L*S`) and builds a NaN mask for padding, exactly as in Python.
pub fn run_batch_mtm(
    batch: &[MtmFeature],
    model: &tch::CModule,
    device: Device,
    n_embed: i64,
) -> Vec<ResultLine> {
    if batch.is_empty() {
        return Vec::new();
    }

    let b = batch.len() as i64;
    let l = batch[0].k_seq.len() as i64;          // seq_len
    let s = batch[0].k_signals[0].len() as i64;   // signal_len

    // ── stack raw arrays into tensors ────────────────────────────────────
    let kmers_flat: Vec<i64> = batch.iter().flat_map(|f| f.k_seq.iter().copied()).collect();
    let kmers = Tensor::from_slice(&kmers_flat).reshape([b, l]);

    // signals: replace NaN with 0.0 for tensor stacking; NaN mask is built separately
    let sigs_flat: Vec<f32> = batch
        .iter()
        .flat_map(|f| f.k_signals.iter().flat_map(|row| row.iter().copied()))
        .collect();
    let signals_raw = Tensor::from_slice(&sigs_flat)
        .reshape([b, l, s])
        .to_dtype(Kind::Float, false, false);

    let tags_flat: Vec<i64> = batch.iter().map(|f| f.tag as i64).collect();
    let tags = Tensor::from_slice(&tags_flat);

    // ── move to device ───────────────────────────────────────────────────
    let signals_dev = signals_raw.to(device);
    let kmers_dev   = kmers.to(device);
    let tags_dev    = tags.to(device);

    // Reshape signals: (B, L, S) → (B, L*S, 1)   [matches Python]
    let signals_rs = signals_dev.reshape([b, l * s, 1]);

    // kmer_expand: (B, L) → (B, L*S)  [each kmer code repeated S times]
    let kmer_expand = kmers_dev
        .unsqueeze(2)            // (B, L, 1)
        .expand(&[b, l, s], false)
        .reshape([b, l * s]);

    // x_mask: (B, L*S, 1+n_embed)  — NaN positions + false for embedding dims
    let nan_mask = signals_rs.isnan();                                // (B, L*S, 1)
    let false_mask = Tensor::zeros([b, l * s, n_embed], (Kind::Bool, device));
    let x_mask = Tensor::cat(&[nan_mask, false_mask], 2);            // (B, L*S, 1+n_embed)

    // tpos: (B, L*S) absolute position indices
    let tpos = Tensor::arange(l * s, (Kind::Int64, device))
        .unsqueeze(0)
        .expand(&[b, l * s], false);

    // x_static: (B, 1) proximity tag
    let x_static = tags_dev.unsqueeze(1).to_dtype(Kind::Int64, false, false);

    // ── forward pass ─────────────────────────────────────────────────────
    let (probs, pred, kmers_cpu) = no_grad(|| {
        // Replace NaN with 0 before model forward (NaN propagates through ops)
        let signals_clean = signals_rs.nan_to_num(0.0, f64::INFINITY, f64::NEG_INFINITY);

        let logits = model
            .forward_ts(&[signals_clean, kmer_expand.shallow_clone(), x_mask, tpos, x_static])
            .expect("modelMTM forward failed");

        let probs = logits.softmax(-1, Kind::Float).to(Device::Cpu);
        let pred  = logits.argmax(1, false).to(Device::Cpu);
        let kmers_cpu = kmers_dev.to(Device::Cpu);
        (probs, pred, kmers_cpu)
    });

    // ── decode outputs ────────────────────────────────────────────────────
    let probs_data: Vec<f32> = Vec::try_from(probs.flatten(0, -1)).expect("probs tensor conversion failed");
    let pred_data:  Vec<i64> = Vec::try_from(pred.flatten(0, -1)).expect("pred tensor conversion failed");
    let kmers_data: Vec<i64> = Vec::try_from(kmers_cpu.flatten(0, -1)).expect("kmers tensor conversion failed");

    batch
        .iter()
        .enumerate()
        .map(|(i, feat)| {
            let p0 = (probs_data[i * 2]     as f64 * 1e6).round() / 1e6;
            let p1 = (probs_data[i * 2 + 1] as f64 * 1e6).round() / 1e6;
            let pred_label = pred_data[i];

            let kmer_bytes: Vec<u8> = (0..l as usize)
                .map(|j| code_to_base(kmers_data[i * l as usize + j] as u64).unwrap_or(b'N'))
                .collect();
            let k5 = String::from_utf8_lossy(&central5mer(&kmer_bytes)).into_owned();

            format!("{}\t{p0:.6}\t{p1:.6}\t{pred_label}\t{k5}", feat.sample_info)
        })
        .collect()
}

// ─── ModelBiLSTM inference ────────────────────────────────────────────────────

/// Run one forward pass for ModelBiLSTM on a batch of BiLSTM features.
///
/// Exact port of Python `_run_batch_bilstm`.
pub fn run_batch_bilstm(
    batch: &[BiLstmFeature],
    model: &tch::CModule,
    device: Device,
) -> Vec<ResultLine> {
    if batch.is_empty() {
        return Vec::new();
    }

    let b = batch.len() as i64;
    let l = batch[0].k_seq.len() as i64;
    let s = batch[0].k_signals[0].len() as i64;

    let kmers_flat: Vec<i64>  = batch.iter().flat_map(|f| f.k_seq.iter().copied()).collect();
    let means_flat: Vec<f32>  = batch.iter().flat_map(|f| f.means.iter().copied()).collect();
    let stds_flat:  Vec<f32>  = batch.iter().flat_map(|f| f.stds.iter().copied()).collect();
    let lens_flat:  Vec<f32>  = batch.iter().flat_map(|f| f.lens.iter().map(|&x| x as f32)).collect();
    let sigs_flat:  Vec<f32>  = batch.iter().flat_map(|f| f.k_signals.iter().flat_map(|r| r.iter().copied())).collect();

    let kmers   = Tensor::from_slice(&kmers_flat).reshape([b, l]).to(device);
    let means   = Tensor::from_slice(&means_flat).reshape([b, l]).to(device);
    let stds    = Tensor::from_slice(&stds_flat ).reshape([b, l]).to(device);
    let lens    = Tensor::from_slice(&lens_flat ).reshape([b, l]).to(device);
    let signals = Tensor::from_slice(&sigs_flat ).reshape([b, l, s]).to(device);

    let (probs, pred, kmers_cpu) = no_grad(|| {
        // BiLSTM forward returns (logits, softmax_probs) — we use index 1
        let out = model
            .forward_ts(&[kmers.shallow_clone(), means, stds, lens, signals])
            .expect("BiLSTM forward failed");
        // TorchScript returns a tuple; index 1 = softmax probs
        let probs = out.get(1).softmax(-1, Kind::Float).to(Device::Cpu);
        let pred  = probs.argmax(1, false).to(Device::Cpu);
        let kmers_cpu = kmers.to(Device::Cpu);
        (probs, pred, kmers_cpu)
    });

    let probs_data: Vec<f32> = Vec::try_from(probs.flatten(0, -1)).expect("probs tensor conversion failed");
    let pred_data:  Vec<i64> = Vec::try_from(pred.flatten(0, -1)).expect("pred tensor conversion failed");
    let kmers_data: Vec<i64> = Vec::try_from(kmers_cpu.flatten(0, -1)).expect("kmers tensor conversion failed");

    batch
        .iter()
        .enumerate()
        .map(|(i, feat)| {
            let p0 = (probs_data[i * 2]     as f64 * 1e6).round() / 1e6;
            let p1 = (probs_data[i * 2 + 1] as f64 * 1e6).round() / 1e6;
            let pred_label = pred_data[i];

            let kmer_bytes: Vec<u8> = (0..l as usize)
                .map(|j| code_to_base(kmers_data[i * l as usize + j] as u64).unwrap_or(b'N'))
                .collect();
            let k5 = String::from_utf8_lossy(&central5mer(&kmer_bytes)).into_owned();

            format!("{}\t{p0:.6}\t{p1:.6}\t{pred_label}\t{k5}", feat.sample_info)
        })
        .collect()
}
