//! Unified error type for pyrameth-core.

use thiserror::Error;

/// All errors that can occur in pyrameth-core.
#[derive(Debug, Error)]
pub enum Ds3Error {
    /// BAM / alignment file errors.
    #[error("BAM I/O error: {0}")]
    Bam(#[from] std::io::Error),

    /// BAM record is missing a required tag.
    #[error("BAM record '{read_id}' is missing required tag '{tag}'")]
    MissingTag {
        /// The read ID of the offending record.
        read_id: String,
        /// The two-letter BAM tag that was expected.
        tag: &'static str,
    },

    /// Move table validation failed.
    #[error("Invalid move table for read '{read_id}': {reason}")]
    InvalidMoveTable {
        /// The read ID of the offending record.
        read_id: String,
        /// Human-readable description of why the move table is invalid.
        reason: String,
    },

    /// CIGAR parsing produced an inconsistent mapping.
    #[error("CIGAR parse error for read '{read_id}': {reason}")]
    CigarError {
        /// The read ID of the offending record.
        read_id: String,
        /// Human-readable description of the CIGAR inconsistency.
        reason: String,
    },

    /// Signal file (POD5 / Slow5) error.
    #[error("Signal file error: {0}")]
    SignalFile(String),

    /// k-mer contains an unknown base character.
    #[error("Unknown base character '{ch}' in k-mer")]
    UnknownBase {
        /// The unrecognised character.
        ch: char,
    },

    /// Noodles codec / parse error (wrapped as string to stay Send).
    #[error("Noodles error: {0}")]
    Noodles(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Ds3Error>;
