//! BAM file indexing and record retrieval via noodles.
//!
//! Mirrors Python `utils/bam_reader.py::ReadIndexedBam`.
//! Stores virtual (BGZF) positions per read-ID for memory-efficient random access.

use std::{
    collections::HashMap,
    fs::File,
    io::BufReader,
    path::Path,
};

use noodles::{
    bam,
    bgzf::VirtualPosition,
    sam::alignment::record::data::field::Tag,
};

use crate::{
    cigar::{query_alignment_bounds, CigarOp, CigarTuple},
    error::{Ds3Error, Result},
    kmer::reverse_complement,
};

// ─── tag constants ────────────────────────────────────────────────────────────

const TAG_MV: Tag = Tag::new(b'm', b'v'); // move table
const TAG_TS: Tag = Tag::new(b't', b's'); // trimmed signal samples
const TAG_SP: Tag = Tag::new(b's', b'p'); // split-read trim offset
const TAG_NM: Tag = Tag::new(b'N', b'M'); // edit distance
const TAG_PI: Tag = Tag::new(b'p', b'i'); // parent read ID (child-read mode)

// ─── types ───────────────────────────────────────────────────────────────────

/// A decoded BAM alignment record ready for feature extraction.
///
/// All expensive heap allocations are done once at read time.
#[derive(Debug, Clone)]
pub struct AlnRecord {
    /// Read name (or parent ID if `pi` tag present).
    pub read_id: String,

    /// Forward-strand query sequence (5'→3').
    /// For reverse-strand reads this is the reverse complement of the stored seq.
    pub forward_seq: Vec<u8>,

    /// Mapping quality (0–255, 255 = not available).
    pub mapq: u8,

    /// Alignment flags.
    pub flags: u16,

    /// Reference sequence name (chromosome), empty string if unmapped.
    pub ref_name: String,

    /// 0-based reference start (inclusive).
    pub ref_start: i64,

    /// 0-based reference end (exclusive), computed from CIGAR.
    pub ref_end: i64,

    /// Start of the aligned portion within the *forward* sequence.
    pub qa_start: usize,
    /// End of the aligned portion within the *forward* sequence.
    pub qa_end: usize,

    /// Decoded CIGAR tuples in the order they appear in the BAM record.
    pub cigar: Vec<CigarTuple>,

    // ── BAM tags needed by feature extraction ──────────────────────────────

    /// Stride (first element of the `mv` tag array).
    pub mv_stride: usize,

    /// Move table (elements 1.. of the `mv` tag array).
    pub movetable: Vec<i32>,

    /// Number of trimmed signal samples (`ts` tag, plus `sp` if present).
    pub num_trimmed: i64,

    /// Alignment identity proxy: `Some(nm)` from the `NM` tag.
    pub nm: Option<i32>,
}

impl AlnRecord {
    /// Strand character, `'+'` or `'-'`.
    pub fn strand(&self) -> char {
        if self.flags & 0x10 != 0 { '-' } else { '+' }
    }

    /// Whether the read maps to the reverse strand.
    pub fn is_reverse(&self) -> bool {
        self.flags & 0x10 != 0
    }

    /// Whether the read is unmapped.
    pub fn is_unmapped(&self) -> bool {
        self.flags & 0x04 != 0
    }

    /// Total query length (length of `forward_seq`).
    pub fn query_length(&self) -> usize {
        self.forward_seq.len()
    }
}

// ─── ReadIndexedBam ───────────────────────────────────────────────────────────

/// BAM file reader with in-memory index from read-ID → BGZF virtual positions.
///
/// Mirrors Python `ReadIndexedBam`:
/// * Skips supplementary (`0x800`) and secondary (`0x100`) alignments.
/// * Supports the `pi` tag (parent-read mode): indexes by the parent ID.
/// * Supports multiple alignments per read (chimeric / multi-map).
pub struct ReadIndexedBam {
    bam_path: std::path::PathBuf,
    /// read_id  →  list of BGZF virtual positions in the BAM file
    index: HashMap<String, Vec<VirtualPosition>>,
    /// Total number of indexed records.
    pub num_records: usize,
}

impl ReadIndexedBam {
    /// Open `bam_path`, scan all records once, and build the in-memory index.
    pub fn open(bam_path: impl AsRef<Path>) -> Result<Self> {
        let path = bam_path.as_ref().to_path_buf();
        let mut reader = bam::io::Reader::new(BufReader::new(
            File::open(&path).map_err(Ds3Error::Bam)?,
        ));
        reader.read_header().map_err(|e| Ds3Error::Bam(e))?;

        let mut index: HashMap<String, Vec<VirtualPosition>> = HashMap::new();
        let mut num_records: usize = 0;
        let mut record = bam::Record::default();

        loop {
            // Capture virtual position BEFORE reading the record
            let vpos = reader.get_ref().virtual_position();
            match reader.read_record(&mut record) {
                Ok(0) => break, // EOF
                Ok(_) => {}
                Err(e) => return Err(Ds3Error::Bam(e)),
            }

            // Skip supplementary and secondary
            let flags = record.flags().bits();
            if flags & 0x0900 != 0 {
                continue;
            }

            // Determine index key: use `pi` tag (parent ID) if present, else query name
            let read_id = get_parent_id(&record)?;
            index.entry(read_id).or_default().push(vpos);
            num_records += 1;
        }

        log::info!(
            "BAM index built: {} unique read IDs, {} records total",
            index.len(),
            num_records
        );

        Ok(Self { bam_path: path, index, num_records })
    }

    /// Check whether a read ID is present in the index.
    pub fn contains(&self, read_id: &str) -> bool {
        self.index.contains_key(read_id)
    }

    /// Retrieve all alignment records for `read_id`.
    ///
    /// Opens the BAM file and seeks to each stored virtual position.
    /// Returns an empty vec if the read ID is not in the index.
    pub fn get_alignments(&self, read_id: &str) -> Result<Vec<AlnRecord>> {
        let Some(positions) = self.index.get(read_id) else {
            return Ok(Vec::new());
        };

        let file = File::open(&self.bam_path).map_err(Ds3Error::Bam)?;
        let mut reader = bam::io::Reader::new(BufReader::new(file));
        let header = reader.read_header().map_err(|e| Ds3Error::Bam(e))?;

        let ref_seqs = header.reference_sequences();
        let mut out = Vec::with_capacity(positions.len());

        for &vpos in positions {
            reader.get_mut().seek(vpos).map_err(|e| Ds3Error::Bam(e))?;
            let mut raw = bam::Record::default();
            reader.read_record(&mut raw).map_err(|e| Ds3Error::Bam(e))?;
            let aln = decode_record(&raw, ref_seqs, read_id)?;
            out.push(aln);
        }
        Ok(out)
    }

    /// All read IDs in the index.
    pub fn read_ids(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }
}

// ─── record decoding helpers ─────────────────────────────────────────────────

/// Extract the index key for a record.
/// Uses the `pi` (parent ID) tag when present, otherwise the query name.
fn get_parent_id(
    record: &bam::Record,
) -> Result<String> {
    // Try pi tag first
    if let Some(Ok(field)) = record.data().get(&TAG_PI) {
        if let noodles::sam::alignment::record::data::field::Value::String(s) = field {
            return Ok(String::from_utf8_lossy(&**s).into_owned());
        }
    }
    // Fall back to query name
    Ok(record
        .name()
        .map(|n| std::str::from_utf8(n.as_ref()).unwrap_or("").to_string())
        .unwrap_or_default())
}

/// Decode a raw `bam::Record` into a fully-parsed [`AlnRecord`].
fn decode_record(
    record: &bam::Record,
    ref_seqs: &noodles::sam::header::ReferenceSequences,
    read_id: &str,
) -> Result<AlnRecord> {
    let flags = record.flags().bits();

    // ── reference name ────────────────────────────────────────────────────
    let ref_name = record
        .reference_sequence_id()
        .and_then(|r| r.ok())
        .and_then(|id| ref_seqs.get_index(id))
        .map(|(name, _)| name.to_string())
        .unwrap_or_default();

    // ── mapping quality ───────────────────────────────────────────────────
    let mapq = record
        .mapping_quality()
        .map(|mq| mq.get())
        .unwrap_or(0);

    // ── reference start / end ─────────────────────────────────────────────
    let ref_start = record
        .alignment_start()
        .and_then(|r| r.ok())
        .map(|p| p.get() as i64 - 1) // noodles is 1-based
        .unwrap_or(-1);

    // ── CIGAR ─────────────────────────────────────────────────────────────
    let cigar: Vec<CigarTuple> = record
        .cigar()
        .iter()
        .filter_map(|op| op.ok())
        .map(|op| CigarTuple {
            op: cigar_kind_to_op(op.kind()),
            len: op.len() as u32,
        })
        .collect();

    // ref_end = ref_start + reference-consuming length of CIGAR
    let ref_consuming: i64 = cigar
        .iter()
        .filter(|c| {
            matches!(
                c.op,
                CigarOp::Match
                    | CigarOp::Deletion
                    | CigarOp::Skip
                    | CigarOp::SeqMatch
                    | CigarOp::SeqMismatch
            )
        })
        .map(|c| c.len as i64)
        .sum();
    let ref_end = ref_start + ref_consuming;

    // ── query alignment bounds ────────────────────────────────────────────
    let (qa_start_raw, qa_end_raw) = query_alignment_bounds(&cigar);

    // ── forward sequence ──────────────────────────────────────────────────
    // The stored sequence may or may not be in sequencing orientation.
    // `get_forward_sequence()` in pysam returns the sequence in the
    // orientation of the *reference* (i.e. RC if the read maps to `-` strand).
    let raw_seq: Vec<u8> = record
        .sequence()
        .iter()
        .map(|b| b as u8)
        .collect();

    let is_reverse = flags & 0x10 != 0;
    let forward_seq = if is_reverse {
        reverse_complement(&raw_seq)
    } else {
        raw_seq
    };

    // Adjust qa bounds for reverse-strand reads (mirrors Python)
    let (qa_start, qa_end) = if is_reverse {
        let len = forward_seq.len();
        (len - qa_end_raw, len - qa_start_raw)
    } else {
        (qa_start_raw, qa_end_raw)
    };

    // ── tags: mv, ts, sp, NM ─────────────────────────────────────────────
    let data = record.data();

    // mv tag: [stride, m0, m1, m2, ...]
    let (mv_stride, movetable) = decode_mv_tag(&data, read_id)?;

    // ts tag (num trimmed samples)
    let ts = get_tag_i64(&data, TAG_TS, read_id, "ts")?;

    // sp tag (optional additional trim)
    let sp = get_tag_i64_opt(&data, TAG_SP).unwrap_or(0);

    let num_trimmed = ts + sp;

    // NM tag (edit distance, optional)
    let nm = get_tag_i64_opt(&data, TAG_NM).map(|v| v as i32);

    Ok(AlnRecord {
        read_id: read_id.to_string(),
        forward_seq,
        mapq,
        flags,
        ref_name,
        ref_start,
        ref_end,
        qa_start,
        qa_end,
        cigar,
        mv_stride,
        movetable,
        num_trimmed,
        nm,
    })
}

// ─── tag decoders ─────────────────────────────────────────────────────────────

/// Decode the `mv` tag into (stride, movetable).
fn decode_mv_tag(
    data: &impl noodles::sam::alignment::record::Data,
    read_id: &str,
) -> Result<(usize, Vec<i32>)> {
    use noodles::sam::alignment::record::data::field::Value;

    let field = data
        .get(&TAG_MV)
        .ok_or_else(|| Ds3Error::MissingTag { read_id: read_id.to_string(), tag: "mv" })?
        .map_err(|e| Ds3Error::Noodles(e.to_string()))?;

    let values: Vec<i32> = match field {
        Value::Array(arr) => {
            use noodles::sam::alignment::record::data::field::value::Array as Arr;
            macro_rules! to_i32 {
                ($vals:expr) => {
                    $vals.iter()
                        .map(|r| r.map(|x| x as i32).map_err(|e| Ds3Error::Noodles(e.to_string())))
                        .collect::<Result<Vec<_>>>()?
                };
            }
            match arr {
                Arr::Int8(v)   => to_i32!(v),
                Arr::UInt8(v)  => to_i32!(v),
                Arr::Int16(v)  => to_i32!(v),
                Arr::UInt16(v) => to_i32!(v),
                Arr::Int32(v)  => to_i32!(v),
                Arr::UInt32(v) => to_i32!(v),
                _ => return Err(Ds3Error::MissingTag { read_id: read_id.to_string(), tag: "mv" }),
            }
        }
        _ => {
            return Err(Ds3Error::MissingTag { read_id: read_id.to_string(), tag: "mv" });
        }
    };

    if values.is_empty() {
        return Err(Ds3Error::InvalidMoveTable {
            read_id: read_id.to_string(),
            reason: "mv tag is empty".to_string(),
        });
    }

    let stride = values[0] as usize;
    let movetable = values[1..].to_vec();
    Ok((stride, movetable))
}

/// Get an integer tag value, returning an error if absent.
fn get_tag_i64(
    data: &impl noodles::sam::alignment::record::Data,
    tag: Tag,
    read_id: &str,
    tag_name: &'static str,
) -> Result<i64> {
    use noodles::sam::alignment::record::data::field::Value;

    let field = data
        .get(&tag)
        .ok_or_else(|| Ds3Error::MissingTag { read_id: read_id.to_string(), tag: tag_name })?
        .map_err(|e| Ds3Error::Noodles(e.to_string()))?;

    match field {
        Value::Int8(v)  => Ok(v as i64),
        Value::UInt8(v) => Ok(v as i64),
        Value::Int16(v) => Ok(v as i64),
        Value::UInt16(v)=> Ok(v as i64),
        Value::Int32(v) => Ok(v as i64),
        Value::UInt32(v)=> Ok(v as i64),
        _ => Err(Ds3Error::MissingTag { read_id: read_id.to_string(), tag: tag_name }),
    }
}

/// Get an optional integer tag value, returning `None` if absent.
fn get_tag_i64_opt(
    data: &impl noodles::sam::alignment::record::Data,
    tag: Tag,
) -> Option<i64> {
    use noodles::sam::alignment::record::data::field::Value;
    let field = data.get(&tag)?.ok()?;
    match field {
        Value::Int8(v)  => Some(v as i64),
        Value::UInt8(v) => Some(v as i64),
        Value::Int16(v) => Some(v as i64),
        Value::UInt16(v)=> Some(v as i64),
        Value::Int32(v) => Some(v as i64),
        Value::UInt32(v)=> Some(v as i64),
        _ => None,
    }
}

// ─── CIGAR kind conversion ────────────────────────────────────────────────────

fn cigar_kind_to_op(kind: noodles::sam::alignment::record::cigar::op::Kind) -> CigarOp {
    use noodles::sam::alignment::record::cigar::op::Kind;
    match kind {
        Kind::Match            => CigarOp::Match,
        Kind::Insertion        => CigarOp::Insertion,
        Kind::Deletion         => CigarOp::Deletion,
        Kind::Skip             => CigarOp::Skip,
        Kind::SoftClip         => CigarOp::SoftClip,
        Kind::HardClip         => CigarOp::HardClip,
        Kind::Pad              => CigarOp::Padding,
        Kind::SequenceMatch    => CigarOp::SeqMatch,
        Kind::SequenceMismatch => CigarOp::SeqMismatch,
    }
}
