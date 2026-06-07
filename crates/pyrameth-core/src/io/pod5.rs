//! POD5 signal file reader.
//!
//! POD5 is Oxford Nanopore's current standard signal format.
//! Specification: <https://github.com/nanoporetech/pod5-file-format>
//!
//! # Implementation
//!
//! Built on three crates from <https://github.com/bsaintjo/pod5-rs>:
//!
//! | Crate        | Role                                               |
//! |--------------|----------------------------------------------------|
//! | `pod5-format`| FlatBuffers footer → byte offsets of Arrow tables |
//! | `svb16`      | VBZ decompression (zstd + SVB16 + zigzag + delta)  |
//! | `arrow`      | Arrow IPC `FileReader` for reads/signal tables     |
//!
//! All three are pure Rust; no C library is required.
//!
//! # Feature flag
//!
//! Enabled with `--features pod5-pure`.  Without it [`read_pod5`] returns
//! a clear `Err` at runtime.
//!
//! ```bash
//! cargo build --features pod5-pure   # native POD5 reading
//! cargo build                        # POD5 disabled; Slow5/BAM still work
//! ```

use std::{collections::HashMap, path::Path};

use crate::error::{Ds3Error, Result};
use super::RawRead;

// ─── public API ──────────────────────────────────────────────────────────────

/// Read all reads from a POD5 file.
///
/// Requires the `pod5-pure` Cargo feature.
pub fn read_pod5(path: impl AsRef<Path>) -> Result<Vec<RawRead>> {
    #[cfg(feature = "pod5-pure")]
    {
        native::iter_pod5_impl(path.as_ref())?.collect()
    }
    #[cfg(not(feature = "pod5-pure"))]
    {
        let _ = path;
        Err(Ds3Error::SignalFile(
            "POD5 reading is disabled. Rebuild with `--features pod5-pure`. \
             See BUILD.md for instructions."
                .into(),
        ))
    }
}

/// Streaming iterator over reads in a POD5 file.
///
/// Yields one `Result<RawRead>` per sequencing read.  Signal decompression
/// happens read-by-read, keeping peak memory proportional to the largest read
/// rather than the whole file.
///
/// Requires the `pod5-pure` Cargo feature.
pub fn iter_pod5(
    path: impl AsRef<Path>,
) -> Result<Box<dyn Iterator<Item = Result<RawRead>> + Send>> {
    #[cfg(feature = "pod5-pure")]
    {
        Ok(Box::new(native::iter_pod5_impl(path.as_ref())?))
    }
    #[cfg(not(feature = "pod5-pure"))]
    {
        let _ = path;
        Err(Ds3Error::SignalFile(
            "POD5 reading is disabled. Rebuild with `--features pod5-pure`. \
             See BUILD.md for instructions."
                .into(),
        ))
    }
}

/// Read only the reads for which `keep(read_id)` returns `true`.
///
/// Only the signal rows belonging to kept reads are svb16-decompressed; reads
/// rejected by `keep` cost nothing beyond the (cheap) reads-table scan.  Callers
/// pass `|read_id| bam_index.contains(read_id)` so reads missing from the BAM
/// are skipped before any signal is decoded.
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
        Err(Ds3Error::SignalFile(
            "POD5 reading is disabled. Rebuild with `--features pod5-pure`. \
             See BUILD.md for instructions."
                .into(),
        ))
    }
}

/// Build a `HashMap<read_id, signal>` for O(1) lookup.
pub fn index_pod5(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<f32>>> {
    Ok(read_pod5(path)?
        .into_iter()
        .map(|r| (r.read_id, r.signal))
        .collect())
}

// ─── native implementation ────────────────────────────────────────────────────

#[cfg(feature = "pod5-pure")]
mod native {
    use std::{fs::File, io::Cursor, path::Path};

    use arrow::{
        array::{
            Array, BinaryArray, FixedSizeBinaryArray, LargeBinaryArray, LargeListArray,
            ListArray, RecordBatch, UInt32Array, UInt64Array,
        },
        ipc::reader::FileReader,
    };
    use memmap2::MmapOptions;
    use pod5_format::ParsedFooter;

    use crate::error::{Ds3Error, Result};
    use super::super::RawRead;

    // ─── public entry point ───────────────────────────────────────────────────

    pub(super) fn iter_pod5_impl(path: &Path) -> Result<Pod5Iter> {
        let mut file = File::open(path)
            .map_err(|e| Ds3Error::SignalFile(format!("cannot open {:?}: {e}", path)))?;

        let footer = ParsedFooter::read_footer(&mut file)
            .map_err(|e| Ds3Error::SignalFile(format!("POD5 footer parse error: {e}")))?;

        // Map the whole file read-only. The OS pages in only what we touch,
        // and Arrow copies batch data into its own buffers, so we can drop
        // the mmap immediately after loading (see end of this function).
        //
        // Safety: we never mutate the mapping and hold the File open while
        // the Mmap lives (required on Windows).
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| Ds3Error::SignalFile(format!("mmap {:?}: {e}", path)))?;

        let reads_meta   = load_reads_table(&mmap, &footer)?;
        let signal_table = load_signal_table(&mmap, &footer)?;

        // mmap (and file) are dropped here. RecordBatches inside signal_table
        // own their Arrow buffers independently.
        Ok(Pod5Iter {
            reads: reads_meta.into_iter(),
            signal_table,
        })
    }

    /// Like [`iter_pod5_impl`] but keeps only reads passing `keep`, and decodes
    /// *only those reads'* signal rows.  The reads table is scanned and filtered
    /// before the signal table is touched, so rejected reads never trigger an
    /// svb16 decode.
    pub(super) fn read_pod5_filtered_impl(
        path: &Path,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Vec<RawRead>> {
        let mut file = File::open(path)
            .map_err(|e| Ds3Error::SignalFile(format!("cannot open {:?}: {e}", path)))?;

        let footer = ParsedFooter::read_footer(&mut file)
            .map_err(|e| Ds3Error::SignalFile(format!("POD5 footer parse error: {e}")))?;

        // Safety: as in `iter_pod5_impl` — the mapping is never mutated and the
        // File is held open for the mmap's lifetime.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| Ds3Error::SignalFile(format!("mmap {:?}: {e}", path)))?;

        // Filter the (cheap) reads table first; only kept reads are decoded.
        let kept: Vec<ReadMeta> = load_reads_table(&mmap, &footer)?
            .into_iter()
            .filter(|m| keep(&m.read_id))
            .collect();
        if kept.is_empty() {
            return Ok(Vec::new());
        }

        let signal_table = load_signal_table(&mmap, &footer)?;

        // mmap/file dropped at end of scope; signal_table owns its Arrow buffers.
        kept.into_iter()
            .map(|meta| decode_read(meta, &signal_table))
            .collect()
    }

    // ─── streaming iterator ───────────────────────────────────────────────────

    pub(super) struct Pod5Iter {
        reads:        std::vec::IntoIter<ReadMeta>,
        signal_table: SignalTable,
    }

    impl Iterator for Pod5Iter {
        type Item = Result<RawRead>;

        fn next(&mut self) -> Option<Self::Item> {
            let meta = self.reads.next()?;
            Some(decode_read(meta, &self.signal_table))
        }
    }

    // ─── reads table ──────────────────────────────────────────────────────────

    struct ReadMeta {
        read_id:     String,
        signal_rows: Vec<u64>,
    }

    fn load_reads_table(mmap: &memmap2::Mmap, footer: &ParsedFooter) -> Result<Vec<ReadMeta>> {
        let section = footer
            .read_table()
            .map_err(|e| Ds3Error::SignalFile(format!("reads table missing from footer: {e}")))?;

        let offset = section.as_ref().offset() as usize;
        let length = section.as_ref().length() as usize;

        // Zero-copy: Cursor wraps a slice of the mmap; no heap Vec<u8> needed.
        let cursor = Cursor::new(&mmap[offset..offset + length]);
        let mut reader = FileReader::try_new(cursor, None)
            .map_err(|e| Ds3Error::SignalFile(format!("reads IPC parse: {e}")))?;

        // Column indices are the same for every batch; look them up once.
        let schema  = reader.schema();
        let rid_idx = schema.index_of("read_id").map_err(|_| {
            Ds3Error::SignalFile("reads table missing 'read_id' column".into())
        })?;
        let sr_idx = schema
            .index_of("signal_rows")
            .or_else(|_| schema.index_of("signal_row_count"))
            .or_else(|_| schema.index_of("signal"))
            .map_err(|_| {
                let cols: Vec<&str> =
                    schema.fields().iter().map(|f| f.name().as_str()).collect();
                Ds3Error::SignalFile(format!(
                    "reads table missing signal-rows column; available columns: {cols:?}"
                ))
            })?;

        let mut out = Vec::new();
        for batch_res in &mut reader {
            let batch = batch_res
                .map_err(|e| Ds3Error::SignalFile(format!("reads batch: {e}")))?;

            for row in 0..batch.num_rows() {
                let read_id     = decode_uuid(batch.column(rid_idx).as_ref(), row)?;
                let signal_rows = decode_signal_rows(batch.column(sr_idx).as_ref(), row)?;
                out.push(ReadMeta { read_id, signal_rows });
            }
        }

        Ok(out)
    }

    // ─── signal table ─────────────────────────────────────────────────────────

    /// Arrow batches from the signal table plus a cumulative row index.
    ///
    /// `batch_starts[i]` is the global signal-row index of the first row in
    /// batch `i` (i.e. the number of rows in batches `0..i`).  This makes the
    /// `signal_row → (batch, offset)` mapping robust to **non-uniform batch
    /// sizes**: we never assume every batch has the same number of rows, only
    /// that batches appear in order (which the Arrow IPC reader guarantees).
    struct SignalTable {
        batches:      Vec<RecordBatch>,
        batch_starts: Vec<u64>,
        sig_col_idx:  usize,
        cnt_col_idx:  usize,
    }

    impl SignalTable {
        /// Decompress one signal chunk, borrowing its bytes directly from the
        /// Arrow buffer (no extra heap copy before decompression).
        fn decode_row(&self, row_idx: u64) -> Result<Vec<i16>> {
            // The target batch is the last one whose start index is <= row_idx.
            // `batch_starts` is sorted ascending, so a partition point finds it
            // in O(log n) without assuming a fixed batch width.
            let batch_idx = self
                .batch_starts
                .partition_point(|&start| start <= row_idx)
                .checked_sub(1)
                .ok_or_else(|| {
                    Ds3Error::SignalFile(format!(
                        "signal row {row_idx} precedes the first signal batch \
                         ({} batches)",
                        self.batches.len()
                    ))
                })?;

            let batch = self.batches.get(batch_idx).ok_or_else(|| {
                Ds3Error::SignalFile(format!(
                    "signal row {row_idx} → batch {batch_idx} out of range \
                     ({} batches)",
                    self.batches.len(),
                ))
            })?;

            let row_in_batch = (row_idx - self.batch_starts[batch_idx]) as usize;
            if row_in_batch >= batch.num_rows() {
                return Err(Ds3Error::SignalFile(format!(
                    "signal row {row_idx} is beyond the end of the signal table \
                     (batch {batch_idx} has {} rows)",
                    batch.num_rows()
                )));
            }

            // Borrow compressed bytes directly from Arrow's buffer — no copy.
            let sig_bytes = extract_binary(batch.column(self.sig_col_idx).as_ref(), row_in_batch)?;
            let count = batch
                .column(self.cnt_col_idx)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Ds3Error::SignalFile("samples column: not UInt32".into()))?
                .value(row_in_batch);

            svb16::decode(sig_bytes, count as usize)
                .map_err(|e| Ds3Error::SignalFile(format!("VBZ decode (row {row_idx}): {e}")))
        }
    }

    fn load_signal_table(mmap: &memmap2::Mmap, footer: &ParsedFooter) -> Result<SignalTable> {
        let section = footer
            .signal_table()
            .map_err(|e| Ds3Error::SignalFile(format!("signal table missing from footer: {e}")))?;

        let offset = section.as_ref().offset() as usize;
        let length = section.as_ref().length() as usize;

        let cursor = Cursor::new(&mmap[offset..offset + length]);
        let mut reader = FileReader::try_new(cursor, None)
            .map_err(|e| Ds3Error::SignalFile(format!("signal IPC parse: {e}")))?;

        let schema      = reader.schema();
        let sig_col_idx = schema.index_of("signal").map_err(|_| {
            Ds3Error::SignalFile("signal table missing 'signal' column".into())
        })?;
        let cnt_col_idx = schema.index_of("samples").map_err(|_| {
            Ds3Error::SignalFile("signal table missing 'samples' column".into())
        })?;

        let mut batches      = Vec::with_capacity(reader.num_batches());
        let mut batch_starts = Vec::with_capacity(reader.num_batches());
        let mut cumulative   = 0u64;

        for batch_res in &mut reader {
            let batch = batch_res
                .map_err(|e| Ds3Error::SignalFile(format!("signal batch: {e}")))?;
            batch_starts.push(cumulative);
            cumulative += batch.num_rows() as u64;
            batches.push(batch);
        }

        Ok(SignalTable { batches, batch_starts, sig_col_idx, cnt_col_idx })
    }

    // ─── per-read assembly ────────────────────────────────────────────────────

    fn decode_read(meta: ReadMeta, signal_table: &SignalTable) -> Result<RawRead> {
        let mut signal: Vec<f32> = Vec::new();
        for &row_idx in &meta.signal_rows {
            let chunk = signal_table.decode_row(row_idx)?;
            signal.extend(chunk.iter().map(|&v| v as f32));
        }
        Ok(RawRead { read_id: meta.read_id, signal })
    }

    // ─── column decoders ──────────────────────────────────────────────────────

    fn decode_uuid(col: &dyn Array, row: usize) -> Result<String> {
        let bytes = col
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .map(|a| a.value(row))
            .or_else(|| col.as_any().downcast_ref::<BinaryArray>().map(|a| a.value(row)))
            .or_else(|| {
                col.as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .map(|a| a.value(row))
            })
            .ok_or_else(|| Ds3Error::SignalFile("read_id: unexpected Arrow type".into()))?;

        if bytes.len() == 16 {
            let b = bytes;
            Ok(format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
                u16::from_be_bytes([b[4], b[5]]),
                u16::from_be_bytes([b[6], b[7]]),
                u16::from_be_bytes([b[8], b[9]]),
                {
                    let mut hi = [0u8; 8];
                    hi[2..].copy_from_slice(&b[10..16]);
                    u64::from_be_bytes(hi)
                }
            ))
        } else {
            std::str::from_utf8(bytes)
                .map(|s| s.to_owned())
                .map_err(|_| Ds3Error::SignalFile("read_id: not 16-byte UUID nor UTF-8".into()))
        }
    }

    fn decode_signal_rows(col: &dyn Array, row: usize) -> Result<Vec<u64>> {
        if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
            let offsets = list.offsets();
            let start   = offsets[row] as usize;
            let end     = offsets[row + 1] as usize;
            if let Some(v) = list.values().as_any().downcast_ref::<UInt64Array>() {
                return Ok((start..end).map(|i| v.value(i)).collect());
            }
            if let Some(v) = list.values().as_any().downcast_ref::<UInt32Array>() {
                return Ok((start..end).map(|i| v.value(i) as u64).collect());
            }
        }
        if let Some(list) = col.as_any().downcast_ref::<LargeListArray>() {
            let offsets = list.offsets();
            let start   = offsets[row] as usize;
            let end     = offsets[row + 1] as usize;
            if let Some(v) = list.values().as_any().downcast_ref::<UInt64Array>() {
                return Ok((start..end).map(|i| v.value(i)).collect());
            }
        }
        if let Some(v) = col.as_any().downcast_ref::<UInt64Array>() {
            return Ok(vec![v.value(row)]);
        }
        Err(Ds3Error::SignalFile("signal_rows: unrecognised Arrow column type".into()))
    }

    /// Borrow compressed signal bytes from an Arrow Binary column (zero-copy).
    fn extract_binary<'a>(col: &'a dyn Array, row: usize) -> Result<&'a [u8]> {
        if let Some(a) = col.as_any().downcast_ref::<LargeBinaryArray>() {
            return Ok(a.value(row));
        }
        if let Some(a) = col.as_any().downcast_ref::<BinaryArray>() {
            return Ok(a.value(row));
        }
        Err(Ds3Error::SignalFile(
            "signal column: expected LargeBinary or Binary".into(),
        ))
    }
}
