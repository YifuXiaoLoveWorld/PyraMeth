//! k-mer encoding, motif finding, and IUPAC expansion.
//!
//! Mirrors `process_utils.py`: `base2code_dna`, `get_refloc_of_methysite_in_motif`,
//! `get_motif_seqs`, `complement_seq`.

use crate::error::Ds3Error;

// ─── base encoding ────────────────────────────────────────────────────────────

/// Encode a DNA base character to its integer code (0-based).
///
/// Matches Python `base2code_dna`:
/// A=0, C=1, G=2, T=3, N=4, W=5, S=6, M=7, K=8, R=9, Y=10,
/// B=11, V=12, D=13, H=14, Z=15.
pub fn base_to_code(b: u8) -> Result<u64, Ds3Error> {
    Ok(match b.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' | b'U' => 3,
        b'N' => 4,
        b'W' => 5,
        b'S' => 6,
        b'M' => 7,
        b'K' => 8,
        b'R' => 9,
        b'Y' => 10,
        b'B' => 11,
        b'V' => 12,
        b'D' => 13,
        b'H' => 14,
        b'Z' => 15,
        ch  => return Err(Ds3Error::UnknownBase { ch: ch as char }),
    })
}

/// Decode an integer code back to a DNA base character.
///
/// Matches Python `code2base_dna`.
pub fn code_to_base(code: u64) -> Option<u8> {
    const TABLE: &[u8] = b"ACGTNWSMKRYBVDHZ";
    TABLE.get(code as usize).copied()
}

/// Encode a k-mer string as an `i64` array (shape `[seq_len]`).
///
/// Matches the `np.fromiter((base2code_dna[x] for x in k_mer), dtype=np.int64)` pattern.
pub fn encode_kmer(kmer: &[u8]) -> Result<Vec<i64>, Ds3Error> {
    kmer.iter().map(|&b| base_to_code(b).map(|c| c as i64)).collect()
}

/// Decode an integer-encoded k-mer back to a byte string.
pub fn decode_kmer(codes: &[i64]) -> Vec<u8> {
    codes.iter().map(|&c| code_to_base(c as u64).unwrap_or(b'N')).collect()
}

/// Extract the central 5-mer from a decoded k-mer (for output column `kmer5`).
///
/// Matches Python:
/// ```python
/// c = len(kseq) // 2
/// kmer5 = kseq[max(c - 2, 0):c + 3]
/// ```
pub fn central5mer(kmer_bytes: &[u8]) -> Vec<u8> {
    let c = kmer_bytes.len() / 2;
    let start = c.saturating_sub(2);
    let end = (c + 3).min(kmer_bytes.len());
    kmer_bytes[start..end].to_vec()
}

// ─── complement / reverse-complement ─────────────────────────────────────────

/// DNA complement of a single base (IUPAC-aware).
///
/// Matches Python `basepairs` dict.
pub fn complement_base(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        b'N' => b'N',
        b'W' => b'W',
        b'S' => b'S',
        b'M' => b'K',
        b'K' => b'M',
        b'R' => b'Y',
        b'Y' => b'R',
        b'B' => b'V',
        b'V' => b'B',
        b'D' => b'H',
        b'H' => b'D',
        b'Z' => b'Z',
        other => other,
    }
}

/// Reverse complement of a DNA sequence.
///
/// Matches Python `complement_seq(base_seq, seq_type="DNA")`.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| complement_base(b)).collect()
}

// ─── motif expansion (IUPAC) ─────────────────────────────────────────────────

/// Expand an IUPAC base character to all concrete bases.
///
/// Matches Python `iupac_alphabets`.
fn iupac_expand(b: u8) -> &'static [u8] {
    match b.to_ascii_uppercase() {
        b'A' => b"A",
        b'C' => b"C",
        b'G' => b"G",
        b'T' => b"T",
        b'R' => b"AG",
        b'M' => b"AC",
        b'S' => b"CG",
        b'Y' => b"CT",
        b'K' => b"GT",
        b'W' => b"AT",
        b'B' => b"CGT",
        b'D' => b"AGT",
        b'H' => b"ACT",
        b'V' => b"ACG",
        _    => b"ACGT", // N and unknowns
    }
}

/// Convert a single IUPAC motif string (e.g. `"CG"`) to all concrete
/// base sequences (e.g. `["CG"]`).
///
/// Matches Python `_convert_motif_seq`.
pub fn expand_motif(motif: &str) -> Vec<Vec<u8>> {
    let mut results: Vec<Vec<u8>> = vec![Vec::new()];
    for &b in motif.to_ascii_uppercase().as_bytes() {
        let expansions = iupac_expand(b);
        let mut new_results: Vec<Vec<u8>> = Vec::new();
        for prefix in &results {
            for &exp_b in expansions {
                let mut new_seq = prefix.clone();
                new_seq.push(exp_b);
                new_results.push(new_seq);
            }
        }
        results = new_results;
    }
    results
}

/// Parse a comma-separated motif list (e.g. `"CG,CHG,CHH"`) and expand all
/// IUPAC codes, returning a sorted-deduped list of concrete motif strings.
///
/// Matches Python `get_motif_seqs(motifs, is_dna=True)`.
pub fn get_motif_seqs(motifs: &str) -> Vec<Vec<u8>> {
    let mut all: Vec<Vec<u8>> = Vec::new();
    for motif in motifs.split(',') {
        all.extend(expand_motif(motif.trim()));
    }
    all.sort();
    all.dedup();
    all
}

// ─── motif site finding ───────────────────────────────────────────────────────

/// Find all positions in `seq` where one of `motif_seqs` occurs.
///
/// Returns a list of 0-based positions of the **modification base** within
/// `seq` (offset by `methyloc_in_motif`).
///
/// Matches Python `get_refloc_of_methysite_in_motif(seqstr, motifset, methyloc_in_motif)`.
pub fn find_motif_sites(
    seq: &[u8],
    motif_seqs: &[Vec<u8>],
    methyloc_in_motif: usize,
) -> Vec<usize> {
    if motif_seqs.is_empty() {
        return Vec::new();
    }
    let motif_len = motif_seqs[0].len();
    let seq_upper: Vec<u8> = seq.iter().map(|&b| b.to_ascii_uppercase()).collect();

    let mut sites: Vec<usize> = Vec::new();
    if seq_upper.len() < motif_len {
        return sites;
    }
    for i in 0..=seq_upper.len() - motif_len {
        let window = &seq_upper[i..i + motif_len];
        if motif_seqs.iter().any(|m| m.as_slice() == window) {
            sites.push(i + methyloc_in_motif);
        }
    }
    sites
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let kmer = b"ACGTN";
        let codes = encode_kmer(kmer).unwrap();
        assert_eq!(codes, vec![0, 1, 2, 3, 4]);
        let decoded = decode_kmer(&codes);
        assert_eq!(decoded, b"ACGTN");
    }

    #[test]
    fn central5mer_from_13mer() {
        let kmer = b"AAACGCGTTTAAA";
        let c5 = central5mer(kmer);
        assert_eq!(c5, b"GCGTT");
    }

    #[test]
    fn reverse_complement_basic() {
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
        assert_eq!(reverse_complement(b"AAAA"), b"TTTT");
        assert_eq!(reverse_complement(b"GCGC"), b"GCGC");
    }

    #[test]
    fn expand_cg_motif() {
        let seqs = expand_motif("CG");
        assert_eq!(seqs, vec![b"CG".to_vec()]);
    }

    #[test]
    fn find_cpg_sites() {
        let seq = b"ACGCGT";
        let motifs = get_motif_seqs("CG");
        let sites = find_motif_sites(seq, &motifs, 0);
        assert_eq!(sites, vec![1, 3]);
    }
}
