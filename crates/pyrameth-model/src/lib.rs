//! pyrameth-model — TorchScript model loading and multi-GPU inference pipeline.
//!
//! # Design
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────────┐
//!  │  Producer threads  (rayon pool, one shard per IO thread)         │
//!  │  Read POD5/Slow5 → extract features → send batches              │
//!  └───────────────────────┬──────────────────────────────────────────┘
//!                          │ crossbeam_channel (per GPU)
//!                          ▼
//!  ┌──────────┐   ┌──────────┐       ┌──────────┐
//!  │ GPU-0    │   │ GPU-1    │  ...  │ GPU-N    │   inference threads
//!  │ CModule  │   │ CModule  │       │ CModule  │   (one per device)
//!  └────┬─────┘   └────┬─────┘       └────┬─────┘
//!       │              │                   │
//!       └──────────────┼───────────────────┘
//!                      │ pred_tx  (single channel)
//!                      ▼
//!               Writer thread  → output TSV
//! ```
//!
//! Each GPU thread loads its own copy of the TorchScript model so there is no
//! locking contention on model weights.

pub mod aggr;
pub mod inference;
pub mod pipeline;

pub use aggr::{normalized_histogram, run_aggr_model};
pub use pipeline::{InferenceConfig, ModelClass, run_inference};
