//! POD5 signal file reader.
//!
//! POD5 is Oxford Nanopore's current standard signal format.
//! Specification: <https://github.com/nanoporetech/pod5-file-format>
//!
//! # Implementation
//!
//! Built on the `pod5-polars` crate from <https://github.com/bsaintjo/pod5-rs>,
//! the same interface the `extract_pod5` reference uses.  The crate exposes a
//! [`Reader`](pod5::reader::Reader) over the two Arrow tables inside a POD5
//! file:
//!
//! | Table  | Accessor              | Columns we read                         |
//! |--------|-----------------------|-----------------------------------------|
//! | reads  | `reader.read_dfs()`   | `read_id` (UUID str), `signal` (row idx)|
//! | signal | `reader.signal_dfs()` | `signal` (VBZ-decompressed i16 chunks)  |
//!
//! VBZ decompression (zstd + streamvbyte + zigzag + delta) is handled inside
//! the crate (`signal_dfs().next()` already yields decompressed `i16`).  All
//! pure Rust; no C library is required.
//!
//! # Reading strategy (mirrors `extract_pod5`)
//!
//! 1. Walk the reads table and keep only the reads of interest — the
//!    [`read_pod5_filtered`] caller passes a `keep` predicate, typically
//!    `|read_id| bam_index.contains(read_id)`.  For each kept read we record
//!    its list of *signal row indices*.
//! 2. Walk the signal table and decompress **only the rows the kept reads
//!    actually need**, stopping as soon as every needed chunk has been
//!    collected.
//! 3. Concatenate each read's chunks into its full signal.
//!
//! Because reads not present in the BAM never have their signal decompressed,
//! this is substantially cheaper than decoding the whole file and filtering
//! afterwards.
//!
//! # Signal units
//!
//! Signals are returned as **raw ADC** values (`i16` cast to `f32`), exactly
//! matching the Python pipeline — `extract_features_pod5.py` / `call_mods_bam.py`
//! feed `pod5_record.signal` (raw ADC, **not** `signal_pa`) — and the Slow5
//! reader, which also returns raw ADC.  No picoamp calibration is applied, so
//! MAD / z-score normalisation downstream is bit-for-bit consistent with Python.
//!
//! # Feature flag
//!
//! Enabled with `--features pod5-pure`.  Without it the public functions return
//! a clear `Err` at runtime.
//!
//! ```bash
//! cargo build --features pod5-pure   # native POD5 reading
//! cargo build                        # POD5 disabled; Slow5/BAM still work
//! ```

use std::{collections::HashMap, path::Path};

use crate::error::Result;
use super::RawRead;

// ─── public API ──────────────────────────────────────────────────────────────

/// Read every read from a POD5 file.
///
/// Requires the `pod5-pure` Cargo feature.
pub fn read_pod5(path: impl AsRef<Path>) -> Result<Vec<RawRead>> {
    read_pod5_filtered(path, |_| true)
}

/// Read only the reads for which `keep(read_id)` returns `true`.
///
/// Signal for reads rejected by `keep` is never decompressed, which is the
/// main acceleration over reading the whole file: callers pass
/// `|read_id| bam_index.contains(read_id)` so reads missing from the BAM cost
/// nothing beyond a hash lookup.
///
/// Requires the `pod5-pure` Cargo feature.
pub fn read_pod5_filtered(
    path: impl AsRef<Path>,
    keep: impl Fn(&str) -> bool,
) -> Result<Vec<RawRead>> {
    #[cfg(feature = "pod5-pure")]
    {
        native::read_pod5_filtered_impl(path.as_ref(), keep)
    }
    #[cfg(not(feature = "pod5-pure"))]
    {
        let _ = (path, keep);
        Err(pod5_disabled())
    }
}

/// Iterator over every read in a POD5 file.
///
/// The whole file is read eagerly (a read's signal chunks may be spread across
/// signal batches, so they must all be collected before any read can be
/// assembled); the returned iterator simply walks the resulting `Vec`.  Prefer
/// [`read_pod5_filtered`] when you can cheaply tell which reads you need.
///
/// Requires the `pod5-pure` Cargo feature.
pub fn iter_pod5(
    path: impl AsRef<Path>,
) -> Result<Box<dyn Iterator<Item = Result<RawRead>> + Send>> {
    let reads = read_pod5(path)?;
    Ok(Box::new(reads.into_iter().map(Ok)))
}

/// Build a `HashMap<read_id, signal>` for O(1) lookup.
pub fn index_pod5(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<f32>>> {
    Ok(read_pod5(path)?
        .into_iter()
        .map(|r| (r.read_id, r.signal))
        .collect())
}

#[cfg(not(feature = "pod5-pure"))]
fn pod5_disabled() -> crate::error::Ds3Error {
    crate::error::Ds3Error::SignalFile(
        "POD5 reading is disabled. Rebuild with `--features pod5-pure`. \
         See BUILD.md for instructions."
            .into(),
    )
}

// ─── native implementation ────────────────────────────────────────────────────

#[cfg(feature = "pod5-pure")]
mod native {
    use std::{
        collections::{HashMap, HashSet},
        fs::File,
        path::Path,
    };

    use pod5::{
        polars::prelude::{DataType, Series},
        reader::Reader,
    };

    use crate::error::{Ds3Error, Result};
    use super::super::RawRead;

    /// Map any pod5/polars error into our signal-file error variant.
    fn sig_err(context: &str, err: impl std::fmt::Display) -> Ds3Error {
        Ds3Error::SignalFile(format!("{context}: {err}"))
    }

    /// One kept read plus the signal-table row indices that make up its signal.
    struct ReadWorkItem {
        read_id: String,
        signal_indices: Vec<usize>,
    }

    pub(super) fn read_pod5_filtered_impl(
        path: &Path,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Vec<RawRead>> {
        let file = File::open(path)
            .map_err(|e| sig_err(&format!("cannot open {path:?}"), e))?;
        // One reader suffices: each `read_dfs()` / `signal_dfs()` call buffers
        // its table into an owned cursor, so they can be (re)invoked freely.
        let mut reader =
            Reader::from_reader(file).map_err(|e| sig_err("POD5 footer parse error", e))?;

        let read_items = collect_relevant_reads(&mut reader, &keep)?;
        if read_items.is_empty() {
            return Ok(Vec::new());
        }

        let needed: HashSet<usize> = read_items
            .iter()
            .flat_map(|item| item.signal_indices.iter().copied())
            .collect();

        let signal_chunks = collect_signal_chunks(&mut reader, &needed)?;

        let mut out = Vec::with_capacity(read_items.len());
        for item in read_items {
            let signal = assemble_signal(&item, &signal_chunks)?;
            out.push(RawRead { read_id: item.read_id, signal });
        }
        Ok(out)
    }

    /// Walk the reads table, keeping reads for which `keep(read_id)` is true and
    /// recording each one's list of signal-table row indices.
    fn collect_relevant_reads(
        reader: &mut Reader<File>,
        keep: &impl Fn(&str) -> bool,
    ) -> Result<Vec<ReadWorkItem>> {
        let mut items = Vec::new();
        for read_df in reader.read_dfs().map_err(|e| sig_err("POD5 reads table", e))? {
            let df = read_df.map_err(|e| sig_err("POD5 reads batch", e))?.into_inner();

            let read_ids = df
                .column("read_id")
                .map_err(|e| sig_err("reads table missing 'read_id'", e))?
                .str()
                .map_err(|e| sig_err("'read_id' column was not utf8", e))?;
            let signal_lists = df
                .column("signal")
                .map_err(|e| sig_err("reads table missing 'signal'", e))?
                .list()
                .map_err(|e| sig_err("reads 'signal' column was not a list", e))?;

            for row in 0..df.height() {
                let Some(read_id) = read_ids.get(row) else { continue };
                if !keep(read_id) {
                    continue;
                }
                let signal_indices = signal_lists
                    .get_as_series(row)
                    .as_ref()
                    .map(series_to_usize_vec)
                    .transpose()?
                    .unwrap_or_default();
                items.push(ReadWorkItem {
                    read_id: read_id.to_owned(),
                    signal_indices,
                });
            }
        }
        Ok(items)
    }

    /// Decompress only the signal rows in `needed`, keyed by their global row
    /// index, stopping once all are collected.
    fn collect_signal_chunks(
        reader: &mut Reader<File>,
        needed: &HashSet<usize>,
    ) -> Result<HashMap<usize, Vec<f32>>> {
        let mut chunks = HashMap::with_capacity(needed.len());
        let mut global_row = 0usize;

        for signal_df in reader.signal_dfs().map_err(|e| sig_err("POD5 signal table", e))? {
            // `next()` already decompressed the batch to raw `i16` ADC values;
            // we keep them as-is (matching Python's `pod5_record.signal`) and let
            // downstream MAD / z-score normalisation handle scaling.
            let df = signal_df
                .map_err(|e| sig_err("POD5 signal batch", e))?
                .into_inner();
            let signals = df
                .column("signal")
                .map_err(|e| sig_err("signal table missing 'signal'", e))?
                .list()
                .map_err(|e| sig_err("signal 'signal' column was not a list", e))?;

            for row in 0..df.height() {
                let chunk_idx = global_row + row;
                if !needed.contains(&chunk_idx) {
                    continue;
                }
                let Some(series) = signals.get_as_series(row) else { continue };
                chunks.insert(chunk_idx, series_to_f32_vec(&series)?);
            }

            global_row += df.height();
            if chunks.len() == needed.len() {
                break;
            }
        }

        Ok(chunks)
    }

    /// Concatenate a read's signal chunks in order.
    fn assemble_signal(
        item: &ReadWorkItem,
        chunks: &HashMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let total: usize = item
            .signal_indices
            .iter()
            .map(|idx| {
                chunks.get(idx).map(Vec::len).ok_or_else(|| {
                    Ds3Error::SignalFile(format!(
                        "missing signal chunk {idx} for read {}",
                        item.read_id
                    ))
                })
            })
            .sum::<Result<usize>>()?;

        let mut signal = Vec::with_capacity(total);
        for idx in &item.signal_indices {
            let chunk = chunks.get(idx).ok_or_else(|| {
                Ds3Error::SignalFile(format!(
                    "missing signal chunk {idx} for read {}",
                    item.read_id
                ))
            })?;
            signal.extend_from_slice(chunk);
        }
        Ok(signal)
    }

    // ─── polars Series → Vec converters ────────────────────────────────────────

    fn series_to_f32_vec(series: &Series) -> Result<Vec<f32>> {
        match series.dtype() {
            DataType::Float32 => Ok(series
                .f32()
                .map_err(|e| sig_err("signal as f32", e))?
                .into_iter()
                .flatten()
                .collect()),
            DataType::Float64 => Ok(series
                .f64()
                .map_err(|e| sig_err("signal as f64", e))?
                .into_iter()
                .flatten()
                .map(|v| v as f32)
                .collect()),
            DataType::Int16 => Ok(series
                .i16()
                .map_err(|e| sig_err("signal as i16", e))?
                .into_iter()
                .flatten()
                .map(f32::from)
                .collect()),
            DataType::Int32 => Ok(series
                .i32()
                .map_err(|e| sig_err("signal as i32", e))?
                .into_iter()
                .flatten()
                .map(|v| v as f32)
                .collect()),
            other => Err(Ds3Error::SignalFile(format!(
                "unsupported signal dtype: {other:?}"
            ))),
        }
    }

    fn series_to_usize_vec(series: &Series) -> Result<Vec<usize>> {
        let to_usize = |v: i128| -> Result<usize> {
            usize::try_from(v).map_err(|_| {
                Ds3Error::SignalFile(format!("signal index {v} does not fit in usize"))
            })
        };
        match series.dtype() {
            DataType::UInt64 => series
                .u64()
                .map_err(|e| sig_err("signal indices as u64", e))?
                .into_iter()
                .flatten()
                .map(|v| to_usize(v as i128))
                .collect(),
            DataType::UInt32 => Ok(series
                .u32()
                .map_err(|e| sig_err("signal indices as u32", e))?
                .into_iter()
                .flatten()
                .map(|v| v as usize)
                .collect()),
            DataType::Int64 => series
                .i64()
                .map_err(|e| sig_err("signal indices as i64", e))?
                .into_iter()
                .flatten()
                .map(|v| to_usize(v as i128))
                .collect(),
            DataType::Int32 => series
                .i32()
                .map_err(|e| sig_err("signal indices as i32", e))?
                .into_iter()
                .flatten()
                .map(|v| to_usize(v as i128))
                .collect(),
            other => Err(Ds3Error::SignalFile(format!(
                "unsupported signal index dtype: {other:?}"
            ))),
        }
    }
}
