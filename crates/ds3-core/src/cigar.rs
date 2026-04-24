//! CIGAR → query-to-reference position mapping.
//!
//! Exact port of `get_q2tloc_from_cigar` in `utils_dataloader.py`.

/// Sentinel: query position falls in an insertion into the reference.
pub const INS_SENTINEL: i32 = -1;

/// Sentinel: position is outside the alignment / deleted from reference.
pub const DEL_SENTINEL: i32 = -2;

/// CIGAR operation codes (pysam / SAM spec integers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CigarOp {
    /// `M` — alignment match (may be seq-match or mismatch)
    Match = 0,
    /// `I` — insertion relative to reference
    Insertion = 1,
    /// `D` — deletion from reference
    Deletion = 2,
    /// `N` — skipped region from reference (intron / splice)
    Skip = 3,
    /// `S` — soft clip (query consumed, ref not)
    SoftClip = 4,
    /// `H` — hard clip
    HardClip = 5,
    /// `P` — padding
    Padding = 6,
    /// `=` — sequence match
    SeqMatch = 7,
    /// `X` — sequence mismatch
    SeqMismatch = 8,
    /// Any other code
    Other = 255,
}

impl CigarOp {
    /// Parse from pysam integer code (0–8).
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Match,
            1 => Self::Insertion,
            2 => Self::Deletion,
            3 => Self::Skip,
            4 => Self::SoftClip,
            5 => Self::HardClip,
            6 => Self::Padding,
            7 => Self::SeqMatch,
            8 => Self::SeqMismatch,
            _ => Self::Other,
        }
    }
}

/// A single decoded CIGAR element.
#[derive(Debug, Clone, Copy)]
pub struct CigarTuple {
    /// Operation code.
    pub op: CigarOp,
    /// Run length.
    pub len: u32,
}

/// Map query positions to reference positions via CIGAR.
///
/// Returns an array of length `seq_len + 1`:
/// * `≥ 0`               → reference offset from alignment start
/// * [`INS_SENTINEL`]    → query position falls in an insertion
/// * [`DEL_SENTINEL`]    → position outside valid range
///
/// The `strand` parameter matches pysam convention:
/// * `1`  → forward (CIGAR applied left-to-right)
/// * `-1` → reverse (CIGAR applied right-to-left)
///
/// # Errors
/// Returns an error string if the CIGAR does not account for `seq_len` query
/// positions (indicates a malformed alignment record).
pub fn get_q2tloc_from_cigar(
    cigar: &[CigarTuple],
    strand: i8,
    seq_len: usize,
) -> Result<Vec<i32>, String> {
    let mut q_to_r_poss: Vec<i32> = vec![DEL_SENTINEL; seq_len + 1];
    let mut curr_r: i32 = 0;
    let mut curr_q: usize = 0;

    // Reverse CIGAR for reverse-strand reads (matches Python `[::-1]`)
    let cigar_ops: Vec<CigarTuple> = if strand == -1 {
        cigar.iter().rev().copied().collect()
    } else {
        cigar.to_vec()
    };

    for CigarTuple { op, len } in &cigar_ops {
        let l = *len as usize;
        match op {
            CigarOp::Insertion => {
                // insertion into ref: mark query positions as -1
                for q in curr_q..curr_q + l {
                    if q < q_to_r_poss.len() {
                        q_to_r_poss[q] = INS_SENTINEL;
                    }
                }
                curr_q += l;
            }
            CigarOp::Deletion | CigarOp::Skip => {
                // ref advances, query stays
                curr_r += l as i32;
            }
            CigarOp::Match | CigarOp::SeqMatch | CigarOp::SeqMismatch => {
                for off in 0..l {
                    let q = curr_q + off;
                    if q < q_to_r_poss.len() {
                        q_to_r_poss[q] = curr_r + off as i32;
                    }
                }
                curr_q += l;
                curr_r += l as i32;
            }
            // Soft-clips consume query bases but are not in the alignment region.
            // Hard clips and padding don't consume query or ref here.
            CigarOp::SoftClip | CigarOp::HardClip | CigarOp::Padding | CigarOp::Other => {}
        }
    }

    // Final position (matches Python `q_to_r_poss[curr_q_pos] = curr_r_pos`)
    if curr_q < q_to_r_poss.len() {
        q_to_r_poss[curr_q] = curr_r;
    }

    if q_to_r_poss[seq_len] == DEL_SENTINEL {
        return Err(format!(
            "CIGAR implied {curr_r} ref positions but seq_len={seq_len}"
        ));
    }

    Ok(q_to_r_poss)
}

/// Compute (`query_alignment_start`, `query_alignment_end`) from a CIGAR.
///
/// These match pysam's `query_alignment_start` / `query_alignment_end`:
/// * `start`: number of leading hard/soft-clipped query bases (0-based)
/// * `end`:   `start + number of query bases consumed by alignment ops`
pub fn query_alignment_bounds(cigar: &[CigarTuple]) -> (usize, usize) {
    let mut start: usize = 0;
    let mut align_len: usize = 0;
    let mut seen_align = false;

    for CigarTuple { op, len } in cigar {
        let l = *len as usize;
        match op {
            // Clips at the start (before any match) shift the start pointer
            CigarOp::SoftClip | CigarOp::HardClip if !seen_align => {
                start += l;
            }
            CigarOp::Match
            | CigarOp::SeqMatch
            | CigarOp::SeqMismatch
            | CigarOp::Insertion => {
                seen_align = true;
                align_len += l;
            }
            // Deletions / skips don't consume query bases
            CigarOp::Deletion | CigarOp::Skip => {}
            // Trailing soft/hard clips after alignment
            CigarOp::SoftClip | CigarOp::HardClip => {}
            CigarOp::Padding | CigarOp::Other => {}
        }
    }

    (start, start + align_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ct(op: CigarOp, len: u32) -> CigarTuple {
        CigarTuple { op, len }
    }

    #[test]
    fn simple_match_forward() {
        // 10M: positions 0..10 map to ref 0..10
        let cigar = vec![ct(CigarOp::Match, 10)];
        let poss = get_q2tloc_from_cigar(&cigar, 1, 10).unwrap();
        assert_eq!(&poss[..10], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(poss[10], 10);
    }

    #[test]
    fn insertion_marked_as_minus_one() {
        // 5M 2I 5M
        let cigar = vec![
            ct(CigarOp::Match, 5),
            ct(CigarOp::Insertion, 2),
            ct(CigarOp::Match, 5),
        ];
        let poss = get_q2tloc_from_cigar(&cigar, 1, 12).unwrap();
        assert_eq!(poss[5], INS_SENTINEL);
        assert_eq!(poss[6], INS_SENTINEL);
        assert_eq!(poss[7], 5); // first base after insertion
    }

    #[test]
    fn deletion_skips_ref() {
        // 5M 2D 5M: ref advances by 2 at deletion
        let cigar = vec![
            ct(CigarOp::Match, 5),
            ct(CigarOp::Deletion, 2),
            ct(CigarOp::Match, 5),
        ];
        let poss = get_q2tloc_from_cigar(&cigar, 1, 10).unwrap();
        assert_eq!(poss[5], 7); // after 5M + 2D
    }

    #[test]
    fn query_bounds_with_soft_clip() {
        // 3S 10M 2S
        let cigar = vec![
            ct(CigarOp::SoftClip, 3),
            ct(CigarOp::Match, 10),
            ct(CigarOp::SoftClip, 2),
        ];
        let (start, end) = query_alignment_bounds(&cigar);
        assert_eq!(start, 3);
        assert_eq!(end, 13);
    }
}
