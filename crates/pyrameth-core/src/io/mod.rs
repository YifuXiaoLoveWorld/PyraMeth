//! File I/O adapters for signal and alignment files.
//!
//! | Module   | Format              | Native?                         |
//! |----------|---------------------|---------------------------------|
//! | `bam`    | BAM / BGZF          | Always (noodles-bam)            |
//! | `slow5`  | Slow5 / Blow5       | Always (pure Rust)              |
//! | `pod5`   | POD5 (Arrow + VBZ)  | `--features pod5-pure` required |

pub mod bam;
pub mod pod5;
pub mod slow5;

/// A decoded, ready-to-use sequencing read.
///
/// Owned struct so it can be sent across threads freely.
#[derive(Debug, Clone)]
pub struct RawRead {
    /// ONT read identifier (UUID string).
    pub read_id: String,
    /// Raw (un-normalised) ADC signal values as f32.
    pub signal: Vec<f32>,
}
