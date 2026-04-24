//! ds3-core — signal processing, file I/O, and feature extraction
//! for the deepsignal3 methylation calling pipeline.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod cigar;
pub mod error;
pub mod features;
pub mod io;
pub mod kmer;
pub mod signal;
