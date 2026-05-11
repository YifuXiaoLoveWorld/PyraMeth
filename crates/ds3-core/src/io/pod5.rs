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
        native::read_pod5_impl(path.as_ref())
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
    use std::{
        fs::File,
        io::{Cursor, Read, Seek, SeekFrom},
        path::Path,
    };

    use arrow::{
        array::{
            Array, BinaryArray, FixedSizeBinaryArray, LargeBinaryArray, LargeListArray,
            ListArray, UInt32Array, UInt64Array,
        },
        ipc::reader::FileReader,
    };
    use pod5_format::ParsedFooter;

    use crate::error::{Ds3Error, Result};
    use super::super::RawRead;

    // ─── entry point ─────────────────────────────────────────────────────────

    pub(super) fn read_pod5_impl(path: &Path) -> Result<Vec<RawRead>> {
        let mut file = File::open(path)
            .map_err(|e| Ds3Error::SignalFile(format!("cannot open {:?}: {e}", path)))?;

        // pod5-format parses the FlatBuffers footer and returns section offsets.
        let footer = ParsedFooter::read_footer(&mut file)
            .map_err(|e| Ds3Error::SignalFile(format!("POD5 footer parse error: {e}")))?;

        let reads_meta   = load_reads_table(&mut file, &footer)?;
        let signal_table = load_signal_table(&mut file, &footer)?;

        assemble_reads(reads_meta, signal_table)
    }

    // ─── reads table ──────────────────────────────────────────────────────────

    struct ReadMeta {
        read_id:     String,
        signal_rows: Vec<u64>,
    }

    fn load_reads_table(file: &mut File, footer: &ParsedFooter) -> Result<Vec<ReadMeta>> {
        let section = footer
            .read_table()
            .map_err(|e| Ds3Error::SignalFile(format!("reads table missing from footer: {e}")))?;

        let offset = section.as_ref().offset() as u64;
        let length = section.as_ref().length() as u64;

        let ipc_bytes = read_bytes_at(file, offset, length)?;
        let cursor = Cursor::new(ipc_bytes);
        let mut reader = FileReader::try_new(cursor, None)
            .map_err(|e| Ds3Error::SignalFile(format!("reads IPC parse: {e}")))?;

        let mut out = Vec::new();

        for batch_res in &mut reader {
            let batch = batch_res
                .map_err(|e| Ds3Error::SignalFile(format!("reads batch: {e}")))?;
            let schema = batch.schema();

            let rid_idx = schema.index_of("read_id").map_err(|_| {
                Ds3Error::SignalFile("reads table missing 'read_id' column".into())
            })?;
            let sr_idx = schema
                .index_of("signal_rows")
                .or_else(|_| schema.index_of("signal_row_count"))
                .or_else(|_| schema.index_of("signal"))
                .map_err(|_| {
                    let cols: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
                    Ds3Error::SignalFile(format!(
                        "reads table missing signal-rows column; available columns: {:?}",
                        cols
                    ))
                })?;

            let rid_col = batch.column(rid_idx);
            let sr_col  = batch.column(sr_idx);

            for row in 0..batch.num_rows() {
                let read_id     = decode_uuid(rid_col.as_ref(), row)?;
                let signal_rows = decode_signal_rows(sr_col.as_ref(), row)?;
                out.push(ReadMeta { read_id, signal_rows });
            }
        }

        Ok(out)
    }

    /// Decode a UUID from a FixedSizeBinary(16) column row → hyphenated string.
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
            // Some versions store UUID as UTF-8 text.
            std::str::from_utf8(bytes)
                .map(|s| s.to_owned())
                .map_err(|_| Ds3Error::SignalFile("read_id is not 16-byte UUID or UTF-8".into()))
        }
    }

    /// Extract signal row indices from a ListArray column.
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
        // Scalar fallback: one row = one signal section row.
        if let Some(v) = col.as_any().downcast_ref::<UInt64Array>() {
            return Ok(vec![v.value(row)]);
        }
        Err(Ds3Error::SignalFile(
            "signal_rows: unrecognised Arrow type".into(),
        ))
    }

    // ─── signal table ─────────────────────────────────────────────────────────

    struct SignalTable {
        /// (compressed_bytes, sample_count) per row.
        rows: Vec<(Vec<u8>, u32)>,
    }

    fn load_signal_table(file: &mut File, footer: &ParsedFooter) -> Result<SignalTable> {
        let section = footer
            .signal_table()
            .map_err(|e| Ds3Error::SignalFile(format!("signal table missing from footer: {e}")))?;

        let offset = section.as_ref().offset() as u64;
        let length = section.as_ref().length() as u64;

        let ipc_bytes = read_bytes_at(file, offset, length)?;
        let cursor = Cursor::new(ipc_bytes);
        let mut reader = FileReader::try_new(cursor, None)
            .map_err(|e| Ds3Error::SignalFile(format!("signal IPC parse: {e}")))?;

        let mut rows = Vec::new();

        for batch_res in &mut reader {
            let batch = batch_res
                .map_err(|e| Ds3Error::SignalFile(format!("signal batch: {e}")))?;
            let schema = batch.schema();

            let sig_idx = schema.index_of("signal").map_err(|_| {
                Ds3Error::SignalFile("signal table missing 'signal' column".into())
            })?;
            let cnt_idx = schema.index_of("samples").map_err(|_| {
                Ds3Error::SignalFile("signal table missing 'samples' column".into())
            })?;

            let sig_col = batch.column(sig_idx);
            let cnt_col = batch
                .column(cnt_idx)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Ds3Error::SignalFile("samples: not UInt32".into()))?;

            for row in 0..batch.num_rows() {
                let compressed = extract_binary(sig_col.as_ref(), row)?.to_vec();
                let samples    = cnt_col.value(row);
                rows.push((compressed, samples));
            }
        }

        Ok(SignalTable { rows })
    }

    fn extract_binary<'a>(col: &'a dyn Array, row: usize) -> Result<&'a [u8]> {
        if let Some(a) = col.as_any().downcast_ref::<LargeBinaryArray>() {
            return Ok(a.value(row));
        }
        if let Some(a) = col.as_any().downcast_ref::<BinaryArray>() {
            return Ok(a.value(row));
        }
        Err(Ds3Error::SignalFile(
            "signal: expected LargeBinary or Binary column".into(),
        ))
    }

    // ─── assembly ─────────────────────────────────────────────────────────────

    fn assemble_reads(
        reads_meta:   Vec<ReadMeta>,
        signal_table: SignalTable,
    ) -> Result<Vec<RawRead>> {
        let mut out = Vec::with_capacity(reads_meta.len());

        for meta in reads_meta {
            let mut signal: Vec<f32> = Vec::new();

            for &row_idx in &meta.signal_rows {
                let row = row_idx as usize;
                if row >= signal_table.rows.len() {
                    return Err(Ds3Error::SignalFile(format!(
                        "read '{}': signal row {row} out of range ({} rows)",
                        meta.read_id,
                        signal_table.rows.len()
                    )));
                }
                let (ref compressed, sample_count) = signal_table.rows[row];
                // svb16::decode handles the full VBZ pipeline:
                //   zstd → StreamVByte16 → zigzag → delta → Vec<i16>
                let chunk = svb16::decode(compressed, sample_count as usize)
                    .map_err(|e| {
                        Ds3Error::SignalFile(format!(
                            "VBZ decompression failed (row {row}): {e}"
                        ))
                    })?;
                signal.extend(chunk.iter().map(|&v| v as f32));
            }

            out.push(RawRead { read_id: meta.read_id, signal });
        }

        Ok(out)
    }

    // ─── utilities ────────────────────────────────────────────────────────────

    fn read_bytes_at(file: &mut File, offset: u64, len: u64) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len as usize];
        file.seek(SeekFrom::Start(offset))
            .and_then(|_| file.read_exact(&mut buf))
            .map_err(|e| {
                Ds3Error::SignalFile(format!(
                    "read_bytes_at offset={offset} len={len}: {e}"
                ))
            })?;
        Ok(buf)
    }
}
