//! Unified error type for ds3-core.

use thiserror::Error;

/// All errors that can occur in ds3-core.
#[derive(Debug, Error)]
pub enum Ds3Error {
    /// BAM / alignment file errors.
    #[error("BAM I/O error: {0}")]
    Bam(#[from] std::io::Error),

    /// BAM record is missing a required tag.
    #[error("BAM record '{read_id}' is missing required tag '{tag}'")]
    MissingTag { read_id: String, tag: &'static str },

    /// Move table validation failed.
    #[error("Invalid move table for read '{read_id}': {reason}")]
    InvalidMoveTable { read_id: String, reason: String },

    /// CIGAR parsing produced an inconsistent mapping.
    #[error("CIGAR parse error for read '{read_id}': {reason}")]
    CigarError { read_id: String, reason: String },

    /// Signal file (POD5 / Slow5) error.
    #[error("Signal file error: {0}")]
    SignalFile(String),

    /// k-mer contains an unknown base character.
    #[error("Unknown base character '{ch}' in k-mer")]
    UnknownBase { ch: char },

    /// Noodles codec / parse error (wrapped as string to stay Send).
    #[error("Noodles error: {0}")]
    Noodles(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Ds3Error>;
