//! Slow5 / Blow5 signal file reader — pure-Rust implementation.
//!
//! Specification: <https://hasindu2008.github.io/slow5specs/>
//!
//! | Extension | Format                        |
//! |-----------|-------------------------------|
//! | `.slow5`  | Plain-text tab-separated      |
//! | `.blow5`  | BGZF-compressed slow5 text    |
//!
//! No external C library is required.  BGZF decompression for `.blow5` is
//! handled by the `noodles-bgzf` crate (already a transitive dependency).

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use noodles::bgzf;

use crate::error::{Ds3Error, Result};

use super::RawRead;

// ─── public API ──────────────────────────────────────────────────────────────

/// Read all reads from a `.slow5` or `.blow5` file.
///
/// Returns one [`RawRead`] per sequencing read.  Signal values are the raw
/// ADC integers cast to `f32` (normalisation is applied later).
pub fn read_slow5(path: impl AsRef<Path>) -> Result<Vec<RawRead>> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "blow5" => read_slow5_inner(bgzf::Reader::new(open(path)?)),
        _ => read_slow5_inner(open(path)?),
    }
}

/// Build a `HashMap<read_id, signal>` for O(1) lookup.
pub fn index_slow5(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<f32>>> {
    Ok(read_slow5(path)?
        .into_iter()
        .map(|r| (r.read_id, r.signal))
        .collect())
}

// ─── internal ────────────────────────────────────────────────────────────────

fn open(path: &Path) -> Result<File> {
    File::open(path)
        .map_err(|e| Ds3Error::SignalFile(format!("cannot open {:?}: {e}", path)))
}

/// Generic slow5 parser over any `Read` source (plain or BGZF-decompressed).
fn read_slow5_inner<R: Read>(reader: R) -> Result<Vec<RawRead>> {
    let mut buf = BufReader::new(reader);
    let col = parse_header(&mut buf)?;
    parse_body(&mut buf, &col)
}

// ─── Slow5 header ────────────────────────────────────────────────────────────

/// Column indices resolved from the slow5 header.
#[derive(Debug)]
struct ColumnMap {
    read_id: usize,
    raw_signal: usize,
}

impl Default for ColumnMap {
    /// Standard slow5 column order when no `@col_names` line is present.
    ///
    /// Position 0 = read_id, 7 = raw_signal.
    fn default() -> Self {
        Self { read_id: 0, raw_signal: 7 }
    }
}

/// Parse slow5 header lines (those starting with `@` or `#`).
///
/// Returns the column map once `@end_header` is reached.
fn parse_header<R: BufRead>(reader: &mut R) -> Result<ColumnMap> {
    let mut col = ColumnMap::default();
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| Ds3Error::SignalFile(format!("header read error: {e}")))?;
        if n == 0 {
            return Err(Ds3Error::SignalFile(
                "slow5 file ended before @end_header".into(),
            ));
        }

        let trimmed = line.trim_end_matches(['\n', '\r']);

        if trimmed.starts_with("@end_header") {
            return Ok(col);
        }

        // @col_names line (slow5 v0.x) or @data_types line (v1.x prelude)
        // Both tab-separate the field names; the leading token is the directive.
        if trimmed.starts_with("@col_names") || trimmed.starts_with("@data_types") {
            let fields: Vec<&str> = trimmed.splitn(256, '\t').collect();
            // fields[0] is the directive token; fields[1..] are column names.
            for (i, name) in fields.iter().skip(1).enumerate() {
                match *name {
                    "read_id" => col.read_id = i,
                    "raw_signal" => col.raw_signal = i,
                    _ => {}
                }
            }
        }
        // Lines starting with '#' are type annotations; skip.
    }
}

// ─── Slow5 body ──────────────────────────────────────────────────────────────

/// Parse data lines after `@end_header`.
fn parse_body<R: BufRead>(reader: &mut R, col: &ColumnMap) -> Result<Vec<RawRead>> {
    let mut reads = Vec::new();
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| Ds3Error::SignalFile(format!("body read error: {e}")))?;
        if n == 0 {
            break; // EOF
        }

        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() || trimmed.starts_with('@') || trimmed.starts_with('#') {
            continue;
        }

        // Split on tabs; limit to max(read_id, raw_signal) + 2 fields to avoid
        // allocating the entire line into many fragments needlessly.
        let max_col = col.read_id.max(col.raw_signal) + 1;
        let fields: Vec<&str> = trimmed.splitn(max_col + 1, '\t').collect();

        if fields.len() <= col.raw_signal {
            log::warn!("slow5 line has too few columns ({} ≤ {}): skipping", fields.len(), col.raw_signal);
            continue;
        }

        let read_id = fields[col.read_id].to_owned();
        let signal = parse_signal(fields[col.raw_signal])?;

        reads.push(RawRead { read_id, signal });
    }

    Ok(reads)
}

/// Parse a comma-separated sequence of int16 ADC values into `Vec<f32>`.
///
/// The raw signal in slow5 is stored as `int16_t*`; we cast directly to f32
/// without any calibration — calibration (offset/range/digitisation) is
/// deliberately omitted because PyraMeth performs its own MAD normalisation.
#[inline]
fn parse_signal(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .filter(|tok| !tok.is_empty())
        .map(|tok| {
            tok.trim()
                .parse::<i16>()
                .map(|v| v as f32)
                .map_err(|e| Ds3Error::SignalFile(format!("bad signal value '{tok}': {e}")))
        })
        .collect()
}
