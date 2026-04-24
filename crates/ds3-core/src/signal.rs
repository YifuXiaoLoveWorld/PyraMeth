//! Signal normalisation and base-signal alignment.
//!
//! All algorithms are bit-for-bit equivalent to the Python reference in
//! `utils/process_utils.py` and `utils_dataloader.py`.

/// Constant used by statsmodels `robust.mad`: the 75th-percentile of N(0,1).
/// Replicates `statsmodels.robust.mad(x)` which divides by this value.
const MAD_SCALE: f64 = 0.674_489_750_196_082;

// ─── median helpers ───────────────────────────────────────────────────────────

/// Sort `values` and return the median.  NaN values are excluded.
/// Matches `np.median` semantics (promotes to f64, then converts back).
fn median_f64(values: &mut Vec<f64>) -> f64 {
    // Remove NaN (shouldn't appear in raw signals but be defensive)
    values.retain(|x| !x.is_nan());
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("NaN already removed"));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

// ─── normalisation ────────────────────────────────────────────────────────────

/// Normalise a raw signal slice using the MAD method.
///
/// Equivalent to Python:
/// ```python
/// sshift = np.median(signals)
/// sscale = float(robust.mad(signals))   # statsmodels
/// norm   = (signals - sshift) / sscale
/// np.around(norm, decimals=6)
/// ```
///
/// Computation is performed in f64 (matching numpy's internal promotion)
/// and the result is rounded to 6 decimal places before converting to f32.
pub fn normalize_mad(signals: &[f32]) -> Vec<f32> {
    let mut f64s: Vec<f64> = signals.iter().map(|&x| x as f64).collect();
    let median = median_f64(&mut f64s.clone());

    let mut abs_devs: Vec<f64> = f64s.iter().map(|x| (x - median).abs()).collect();
    let mad = median_f64(&mut abs_devs) / MAD_SCALE;

    if mad == 0.0 {
        // Degenerate case: return signals unchanged (as in Python)
        return signals.to_vec();
    }

    signals
        .iter()
        .map(|&x| {
            let normed = (x as f64 - median) / mad;
            // Round to 6 decimal places — matches np.around(..., decimals=6).
            // Note: np.around uses banker's rounding (round-half-to-even);
            // f64::round() uses round-half-away-from-zero.  The difference only
            // arises at exact halfway values (e.g. x.5000000), which is
            // astronomically unlikely for real signal data.
            (normed * 1e6).round() / 1e6
        })
        .map(|x| x as f32)
        .collect()
}

/// Normalise a raw signal slice using the z-score method.
///
/// Equivalent to Python:
/// ```python
/// sshift = np.mean(signals)
/// sscale = float(np.std(signals))
/// norm   = (signals - sshift) / sscale
/// np.around(norm, decimals=6)
/// ```
pub fn normalize_zscore(signals: &[f32]) -> Vec<f32> {
    let f64s: Vec<f64> = signals.iter().map(|&x| x as f64).collect();
    let n = f64s.len() as f64;
    let mean = f64s.iter().sum::<f64>() / n;
    let variance = f64s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std == 0.0 {
        return signals.to_vec();
    }

    signals
        .iter()
        .map(|&x| {
            let normed = (x as f64 - mean) / std;
            (normed * 1e6).round() / 1e6
        })
        .map(|x| x as f32)
        .collect()
}

/// Dispatch to the requested normalisation method.
pub fn normalize_signals(signals: &[f32], method: NormalizeMethod) -> Vec<f32> {
    match method {
        NormalizeMethod::Mad    => normalize_mad(signals),
        NormalizeMethod::Zscore => normalize_zscore(signals),
    }
}

/// Normalisation method selector (mirrors Python `--normalize_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizeMethod {
    /// Median Absolute Deviation (default, robust to outliers).
    Mad,
    /// Z-score (mean / standard deviation).
    Zscore,
}

impl std::str::FromStr for NormalizeMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mad"    => Ok(Self::Mad),
            "zscore" => Ok(Self::Zscore),
            other    => Err(format!("unknown normalize_method '{other}'")),
        }
    }
}

// ─── signal rectangularisation ────────────────────────────────────────────────

/// Build a `(num_events, signals_len)` rectangular signal matrix from a move
/// table, matching `build_signal_rect_from_movetable` in `utils_dataloader.py`.
///
/// Padding strategy (identical to Python):
/// * Short event (L ≤ signals_len): centre-pad with `f32::NAN`.
/// * Long  event (L > signals_len): downsample with `np.linspace` indices
///   (integer *truncation*, i.e. `as i32 as usize`, not rounding).
///
/// The NaN padding is intentional: for modelMTM the caller converts it to a
/// boolean mask (`x_mask = torch.isnan(signals)`).
pub fn build_signal_rect(
    norm_signal: &[f32],
    movetable: &[i32],
    stride: usize,
    signals_len: usize,
) -> Vec<Vec<f32>> {
    // Positions of 1s in the move table (base boundaries)
    let move_positions: Vec<usize> = movetable
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
        .collect();

    let n_events = move_positions.len();
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_events);

    for (ev_idx, &mp) in move_positions.iter().enumerate() {
        let start = mp * stride;
        let end = if ev_idx + 1 < move_positions.len() {
            move_positions[ev_idx + 1] * stride
        } else {
            movetable.len() * stride
        };

        let sig = &norm_signal[start.min(norm_signal.len())..end.min(norm_signal.len())];
        let l = sig.len();

        let mut row = vec![f32::NAN; signals_len];
        if l == 0 {
            // empty event — all NaN
        } else if l <= signals_len {
            // centre-pad: matches Python `pad_left = (signals_len - L) // 2`
            let pad_left = (signals_len - l) / 2;
            row[pad_left..pad_left + l].copy_from_slice(sig);
        } else {
            // downsample: matches Python
            //   idx = np.linspace(0, L-1, signals_len).astype(np.int32)
            // np.int32 truncates (same as `as i32 as usize` in Rust)
            for (k, row_val) in row.iter_mut().enumerate() {
                let frac = k as f64 / (signals_len - 1) as f64 * (l - 1) as f64;
                let idx = frac as i32 as usize; // truncation, not rounding
                *row_val = sig[idx];
            }
        }
        out.push(row);
    }
    out
}

/// Variable-length signal grouping for ModelBiLSTM.
///
/// Mirrors `_group_signals_by_movetable_v2` in `utils_dataloader.py`.
/// Returns one `Vec<f32>` per base (variable length).
///
/// # Errors
/// Returns an error string if the move table does not start with 1
/// or if the trimmed signal is shorter than expected.
pub fn group_signals_by_movetable(
    norm_signal: &[f32],
    movetable: &[i32],
    stride: usize,
) -> Result<Vec<Vec<f32>>, String> {
    if movetable.first() != Some(&1) {
        return Err(format!(
            "move table must start with 1, got {:?}",
            movetable.first()
        ));
    }
    let expected_len = movetable.len() * stride;
    if norm_signal.len() < expected_len {
        return Err(format!(
            "trimmed signal length ({}) < expected ({} × {} = {})",
            norm_signal.len(),
            movetable.len(),
            stride,
            expected_len
        ));
    }

    let move_pos: Vec<usize> = movetable
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
        .collect();

    let mut groups: Vec<Vec<f32>> = Vec::with_capacity(move_pos.len());
    for i in 0..move_pos.len() {
        let s = move_pos[i] * stride;
        let e = if i + 1 < move_pos.len() {
            move_pos[i + 1] * stride
        } else {
            movetable.len() * stride
        };
        groups.push(norm_signal[s..e.min(norm_signal.len())].to_vec());
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_mad_zero_variance_returns_original() {
        let sig = vec![1.0_f32; 100];
        let out = normalize_mad(&sig);
        assert_eq!(out, sig);
    }

    #[test]
    fn normalize_mad_unit_signal_is_bounded() {
        let sig: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let out = normalize_mad(&sig);
        // After MAD normalisation the centre value should be near 0
        let mid = out[50];
        assert!(mid.abs() < 1.0, "mid={mid}");
    }

    #[test]
    fn build_signal_rect_short_event_is_nan_padded() {
        let sig: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mv = vec![1_i32, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let rect = build_signal_rect(&sig, &mv, 1, 8);
        // Event 0: samples 0..5, len=5, signals_len=8 → pad_left=1
        assert!(rect[0][0].is_nan());           // padding
        assert!((rect[0][1] - 0.0).abs() < 1e-6); // first sample
    }

    #[test]
    fn build_signal_rect_long_event_is_downsampled() {
        let sig: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let mv = vec![1_i32; 20];
        let rect = build_signal_rect(&sig, &mv, 1, 4);
        // Each event is one sample, no downsampling needed (len=1 <= 4)
        assert!(!rect[0][0].is_nan());
    }
}
