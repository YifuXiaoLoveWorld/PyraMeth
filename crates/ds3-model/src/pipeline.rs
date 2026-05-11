//! Multi-GPU inference pipeline.
//!
//! Architecture (mirrors Python `inference_ultra` in `call_modifications.py`):
//!
//! ```text
//!  nproc_io × Producer threads
//!    └─ read signal files, extract features, send to per-GPU channels
//!
//!  n_gpu × GPU inference threads
//!    └─ accumulate batches → forward pass → send result lines to writer
//!
//!  1 × Writer thread
//!    └─ write result lines to output TSV
//! ```
//!
//! Concurrency primitives:
//! * `crossbeam_channel::bounded`  — back-pressure between producers and GPU threads
//! * `std::thread` for GPU threads (each thread owns its device and model)
//! * `rayon::ThreadPool` for producer parallelism

use std::{
    collections::HashSet,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

use crossbeam_channel::{bounded, Receiver, Sender};
use rayon::prelude::*;
use tch::Device;

use ds3_core::{
    features::{BiLstmFeature, ExtractionArgs, MtmFeature},
    io::{
        bam::ReadIndexedBam,
        pod5::index_pod5,
        slow5::index_slow5,
    },
    kmer::get_motif_seqs,
    signal::NormalizeMethod,
};

use crate::inference::{run_batch_bilstm, run_batch_mtm, ResultLine};

// ─── public configuration ─────────────────────────────────────────────────────

/// Which model architecture to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelClass {
    /// Multi-scale Temporal Mixer (default).
    Mtm,
    /// Bidirectional LSTM.
    BiLstm,
}

impl std::str::FromStr for ModelClass {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mtm"    => Ok(Self::Mtm),
            "bilstm" => Ok(Self::BiLstm),
            other    => Err(format!("unknown model_class '{other}'")),
        }
    }
}

/// All parameters required to run the inference pipeline.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    // ── model ──────────────────────────────────────────────────────────────
    /// Path to TorchScript `.pt` model file (exported by `export_torchscript.py`).
    pub model_path: PathBuf,
    pub model_class: ModelClass,
    /// Embedding dimension (`n_embed`), needed for MTM mask shape.
    pub n_embed: i64,

    // ── input ──────────────────────────────────────────────────────────────
    /// Signal directory (POD5 or Slow5) or pre-extracted TSV.
    pub input_path: PathBuf,
    /// BAM file (required for signal input; unused for TSV input).
    pub bam_path: Option<PathBuf>,
    /// "pod5" | "slow5" | "tsv" — auto-detected if None.
    pub file_type: Option<String>,

    // ── output ─────────────────────────────────────────────────────────────
    pub result_file: PathBuf,

    // ── extraction ─────────────────────────────────────────────────────────
    pub seq_len: usize,
    pub signal_len: usize,
    pub motifs: String,
    pub mod_loc: usize,
    pub normalize_method: NormalizeMethod,
    pub mapq: u8,
    pub coverage_ratio: f64,
    pub identity: f64,
    pub methy_label: u8,
    /// Optional set of "chrom\tpos\tstrand" strings.
    pub positions_file: Option<PathBuf>,

    // ── performance ────────────────────────────────────────────────────────
    /// Batch size per GPU forward pass.
    pub batch_size: usize,
    /// Number of IO / feature-extraction threads.
    pub nproc_io: usize,
    /// Force CPU even when GPUs are present.
    pub use_cpu: bool,
}

// ─── batch item enums ─────────────────────────────────────────────────────────

enum Batch {
    Mtm(Vec<MtmFeature>),
    BiLstm(Vec<BiLstmFeature>),
}

// ─── CUDA backend preload ────────────────────────────────────────────────────

/// dlopen libtorch_cuda.so so its static initialisers register the CUDA backend.
///
/// The linker's --as-needed drops the library from the NEEDED list because no
/// Rust symbol directly references it.  Calling dlopen() here mirrors what
/// Python does when `import torch.cuda` is executed.
///
/// libcuda.so (the driver API) is found by libcudart.so via NVIDIA's kernel
/// module paths at runtime — it does not need to be in ldconfig or LD_LIBRARY_PATH.
#[cfg(target_os = "linux")]
fn preload_torch_cuda() {
    extern "C" {
        fn dlopen(
            filename: *const std::os::raw::c_char,
            flag: std::os::raw::c_int,
        ) -> *mut std::os::raw::c_void;
        fn dlerror() -> *const std::os::raw::c_char;
    }

    let libtorch = match std::env::var("LIBTORCH") {
        Ok(p) => p,
        Err(_) => {
            log::debug!("LIBTORCH not set — skipping libtorch_cuda.so preload");
            return;
        }
    };

    let path = format!("{libtorch}/lib/libtorch_cuda.so\0");
    let handle = unsafe {
        dlopen(
            path.as_ptr() as *const std::os::raw::c_char,
            2 | 0x100, // RTLD_NOW | RTLD_GLOBAL
        )
    };

    if handle.is_null() {
        let msg = unsafe {
            let p = dlerror();
            if p.is_null() { "unknown error".to_string() }
            else { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        log::warn!("dlopen libtorch_cuda.so failed: {msg} — will run on CPU");
    } else {
        log::debug!("libtorch_cuda.so preloaded via dlopen");
    }
}

// ─── entry point ─────────────────────────────────────────────────────────────

/// Run the full inference pipeline (equivalent to Python `inference_ultra`).
pub fn run_inference(cfg: &InferenceConfig) -> anyhow::Result<()> {
    // On Linux the linker's --as-needed drops libtorch_cuda.so because no Rust
    // symbol directly references it.  Its static initialisers register PyTorch's
    // CUDA dispatch backend; without them tch::Cuda::device_count() returns 0.
    // Preload it via dlopen(), exactly as Python's `import torch.cuda` does.
    #[cfg(target_os = "linux")]
    preload_torch_cuda();

    // ── device selection ──────────────────────────────────────────────────
    let cuda_count = tch::Cuda::device_count();
    log::info!("CUDA devices visible to tch: {cuda_count}  (use_cpu={})", cfg.use_cpu);
    let devices: Vec<Device> = if cfg.use_cpu || cuda_count == 0 {
        log::warn!(
            "Running on CPU — {}",
            if cfg.use_cpu { "--use-cpu flag set" }
            else { "no CUDA device found; check LIBTORCH is a CUDA build and LD_LIBRARY_PATH includes $LIBTORCH/lib" }
        );
        vec![Device::Cpu]
    } else {
        let n = cuda_count as usize;
        log::info!("Using {n} GPU(s).");
        (0..n).map(Device::Cuda).collect()
    };
    let n_workers = devices.len();

    // ── channels ──────────────────────────────────────────────────────────
    // One bounded channel per GPU (back-pressure = 256 batches)
    let (batch_txs, batch_rxs): (Vec<_>, Vec<_>) =
        (0..n_workers).map(|_| bounded::<Option<Batch>>(256)).unzip();

    // Single result channel
    let (result_tx, result_rx) = bounded::<Option<Vec<ResultLine>>>(1024);

    // ── load optional position filter ────────────────────────────────────
    let positions: Option<HashSet<String>> = cfg
        .positions_file
        .as_ref()
        .map(|p| load_positions(p))
        .transpose()?;

    // ── expansion args shared by all producers ────────────────────────────
    let motif_seqs = get_motif_seqs(&cfg.motifs);
    let ext_args = Arc::new(ExtractionArgs {
        seq_len:          cfg.seq_len,
        signal_len:       cfg.signal_len,
        mapq:             cfg.mapq,
        coverage_ratio:   cfg.coverage_ratio,
        identity:         cfg.identity,
        mod_loc:          cfg.mod_loc,
        motif_seqs,
        normalize_method: cfg.normalize_method,
        methy_label:      cfg.methy_label,
        positions,
    });

    // ── spawn GPU inference threads ───────────────────────────────────────
    let gpu_handles: Vec<_> = devices
        .into_iter()
        .zip(batch_rxs.into_iter())
        .enumerate()
        .map(|(rank, (device, rx))| {
            let model_path = cfg.model_path.clone();
            let model_class = cfg.model_class;
            let batch_size = cfg.batch_size;
            let n_embed = cfg.n_embed;
            let result_tx = result_tx.clone();

            thread::spawn(move || {
                gpu_worker(rank, device, &model_path, model_class, batch_size,
                           n_embed, rx, result_tx);
            })
        })
        .collect();

    // ── spawn writer thread ───────────────────────────────────────────────
    let result_file = cfg.result_file.clone();
    let writer_handle = thread::spawn(move || {
        writer_worker(&result_file, result_rx).expect("writer failed");
    });

    // ── run producers (blocking, uses rayon for IO parallelism) ──────────
    let file_type = detect_file_type(cfg)?;
    match file_type.as_str() {
        "tsv" => {
            tsv_producer(&cfg.input_path, cfg.model_class, &batch_txs, cfg.batch_size)?;
        }
        "pod5" | "slow5" => {
            signal_producers(cfg, &file_type, &ext_args, &batch_txs)?;
        }
        other => anyhow::bail!("unsupported file_type '{other}'"),
    }

    // ── send termination signals ──────────────────────────────────────────
    for tx in &batch_txs {
        // Send one None per IO worker to each GPU (mirrors Python end-count logic)
        for _ in 0..cfg.nproc_io {
            let _ = tx.send(None);
        }
    }

    for h in gpu_handles {
        h.join().expect("GPU thread panicked");
    }

    // Signal writer to finish
    let _ = result_tx.send(None);
    writer_handle.join().expect("Writer thread panicked");

    log::info!("ALL DONE");
    Ok(())
}

// ─── GPU worker ───────────────────────────────────────────────────────────────

fn gpu_worker(
    rank: usize,
    device: Device,
    model_path: &Path,
    model_class: ModelClass,
    batch_size: usize,
    n_embed: i64,
    rx: Receiver<Option<Batch>>,
    result_tx: Sender<Option<Vec<ResultLine>>>,
) {
    log::info!("[Worker-{rank}({device:?})] loading model …");
    let mut model = tch::CModule::load_on_device(model_path, device)
        .unwrap_or_else(|e| panic!("[Worker-{rank}] failed to load model: {e}"));
    model.set_eval();
    log::info!("[Worker-{rank}({device:?})] model loaded.");

    // ── batch buffers ─────────────────────────────────────────────────────
    let mut mtm_buf:    Vec<MtmFeature>    = Vec::with_capacity(batch_size);
    let mut bilstm_buf: Vec<BiLstmFeature> = Vec::with_capacity(batch_size);
    let mut _end_count = 0usize;

    let flush_mtm = |buf: &mut Vec<MtmFeature>| -> Vec<ResultLine> {
        if buf.is_empty() { return Vec::new(); }
        let lines = run_batch_mtm(buf, &model, device, n_embed);
        buf.clear();
        lines
    };

    let flush_bilstm = |buf: &mut Vec<BiLstmFeature>| -> Vec<ResultLine> {
        if buf.is_empty() { return Vec::new(); }
        let lines = run_batch_bilstm(buf, &model, device);
        buf.clear();
        lines
    };

    loop {
        let item = rx.recv().unwrap_or(None);

        if item.is_none() {
            _end_count += 1;
            // Python sends one None per IO producer per GPU queue
            // We mirror the same count: break when all producers have sent None
            // (The pipeline sends cfg.nproc_io Nones per GPU channel)
            // Here we just count and flush when all signals received.
            // The channel is bounded so we process remaining items first.
            let lines = match model_class {
                ModelClass::Mtm    => flush_mtm(&mut mtm_buf),
                ModelClass::BiLstm => flush_bilstm(&mut bilstm_buf),
            };
            if !lines.is_empty() {
                let _ = result_tx.send(Some(lines));
            }
            // We don't know nproc_io here; the channel will be disconnected
            // when all producers have finished and all Nones are consumed.
            // Use the channel disconnection as the termination signal.
            if rx.is_empty() {
                break;
            }
            continue;
        }

        match item.unwrap() {
            Batch::Mtm(feats) => {
                for f in feats {
                    mtm_buf.push(f);
                    if mtm_buf.len() >= batch_size {
                        let lines = flush_mtm(&mut mtm_buf);
                        if !lines.is_empty() {
                            let _ = result_tx.send(Some(lines));
                        }
                    }
                }
            }
            Batch::BiLstm(feats) => {
                for f in feats {
                    bilstm_buf.push(f);
                    if bilstm_buf.len() >= batch_size {
                        let lines = flush_bilstm(&mut bilstm_buf);
                        if !lines.is_empty() {
                            let _ = result_tx.send(Some(lines));
                        }
                    }
                }
            }
        }
    }

    log::info!("[Worker-{rank}({device:?})] done.");
}

// ─── writer worker ────────────────────────────────────────────────────────────

fn writer_worker(
    result_file: &Path,
    result_rx: Receiver<Option<Vec<ResultLine>>>,
) -> std::io::Result<()> {
    let mut wf = BufWriter::new(File::create(result_file)?);
    loop {
        match result_rx.recv() {
            Ok(Some(lines)) => {
                for line in lines {
                    writeln!(wf, "{line}")?;
                }
            }
            Ok(None) | Err(_) => break,
        }
    }
    wf.flush()?;
    Ok(())
}

// ─── signal producers ─────────────────────────────────────────────────────────

/// Producer path: signal files (POD5 / Slow5) + BAM → features → batch channels.
fn signal_producers(
    cfg: &InferenceConfig,
    file_type: &str,
    ext_args: &Arc<ExtractionArgs>,
    batch_txs: &[Sender<Option<Batch>>],
) -> anyhow::Result<()> {
    let bam_path = cfg
        .bam_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--bam is required for signal input"))?;

    log::info!("Building BAM index …");
    let bam_index = Arc::new(ReadIndexedBam::open(bam_path)?);
    log::info!("BAM index built ({} reads).", bam_index.num_records);

    // Collect signal files
    let signal_files = collect_signal_files(&cfg.input_path, file_type)?;
    log::info!("Found {} {} files.", signal_files.len(), file_type);

    let n_workers = batch_txs.len();
    let batch_size = cfg.batch_size;
    let model_class = cfg.model_class;

    // Distribute files across nproc_io rayon threads
    signal_files
        .par_chunks(
            (signal_files.len() / cfg.nproc_io.max(1)).max(1),
        )
        .enumerate()
        .try_for_each(|(shard_id, files)| -> anyhow::Result<()> {
            // Each shard gets its own BAM reader (cloned index, separate file handle)
            let bam = bam_index.clone();
            let args = ext_args.clone();
            let ft = file_type.to_string();

            // Round-robin GPU assignment for this shard
            let tx = &batch_txs[shard_id % n_workers];

            let mut mtm_batch    = Vec::with_capacity(batch_size);
            let mut bilstm_batch = Vec::with_capacity(batch_size);

            for file in files {
                let signal_map: std::collections::HashMap<String, Vec<f32>> = match ft.as_str() {
                    "pod5"  => index_pod5(file.clone())?,
                    "slow5" => index_slow5(file.clone())?,
                    _       => anyhow::bail!("unknown file type"),
                };

                for (read_id, signal) in &signal_map {
                    let alignments = match bam.get_alignments(read_id) {
                        Ok(a) => a,
                        Err(_) => continue,
                    };
                    for aln in &alignments {
                        match model_class {
                            ModelClass::Mtm => {
                                if let Ok(feats) = ds3_core::features::process_data_mtm(
                                    signal, aln, &args,
                                ) {
                                    mtm_batch.extend(feats);
                                    if mtm_batch.len() >= batch_size {
                                        let _ = tx.send(Some(Batch::Mtm(
                                            std::mem::replace(&mut mtm_batch, Vec::with_capacity(batch_size)),
                                        )));
                                    }
                                }
                            }
                            ModelClass::BiLstm => {
                                if let Ok(feats) = ds3_core::features::process_data_bilstm(
                                    signal, aln, &args,
                                ) {
                                    bilstm_batch.extend(feats);
                                    if bilstm_batch.len() >= batch_size {
                                        let _ = tx.send(Some(Batch::BiLstm(
                                            std::mem::replace(&mut bilstm_batch, Vec::with_capacity(batch_size)),
                                        )));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Flush remaining
            if !mtm_batch.is_empty() {
                let _ = tx.send(Some(Batch::Mtm(mtm_batch)));
            }
            if !bilstm_batch.is_empty() {
                let _ = tx.send(Some(Batch::BiLstm(bilstm_batch)));
            }

            Ok(())
        })?;

    Ok(())
}

// ─── TSV producer ─────────────────────────────────────────────────────────────

/// Producer path: pre-extracted TSV → feature items → batch channels.
///
/// Mirrors Python `tsv_producer` in `call_modifications.py`.
fn tsv_producer(
    tsv_path: &Path,
    model_class: ModelClass,
    batch_txs: &[Sender<Option<Batch>>],
    batch_size: usize,
) -> anyhow::Result<()> {
    use ds3_core::kmer::base_to_code;
    use std::io::{BufRead, BufReader};

    let n_workers = batch_txs.len();
    let file: Box<dyn std::io::Read> = if tsv_path.extension().map_or(false, |e| e == "gz") {
        // gzip support via flate2 — add to Cargo.toml if needed
        // For now just open as plain
        Box::new(File::open(tsv_path)?)
    } else {
        Box::new(File::open(tsv_path)?)
    };

    let reader = BufReader::new(file);
    let mut mtm_buf:    Vec<MtmFeature>    = Vec::with_capacity(batch_size);
    let mut bilstm_buf: Vec<BiLstmFeature> = Vec::with_capacity(batch_size);
    let mut rng_state: usize = 0;

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() { continue; }
        let words: Vec<&str> = line.splitn(13, '\t').collect();
        if words.len() < 12 { continue; }

        let sample_info = words[..6].join("\t");
        let k_mer = words[6].as_bytes();
        let k_seq: Vec<i64> = k_mer
            .iter()
            .filter_map(|&b| base_to_code(b).ok().map(|c| c as i64))
            .collect();

        let k_signals: Vec<Vec<f32>> = words[10]
            .split(';')
            .map(|row| row.split(',').filter_map(|v| v.parse().ok()).collect())
            .collect();

        let label: u8 = words[11].trim().parse().unwrap_or(0);

        // Round-robin worker assignment
        rng_state = rng_state.wrapping_add(1);
        let qid = rng_state % n_workers;

        match model_class {
            ModelClass::Mtm => {
                mtm_buf.push(MtmFeature {
                    sample_info,
                    k_seq,
                    k_signals,
                    label,
                    tag: 1, // no proximity info in TSV → default 1
                });
                if mtm_buf.len() >= batch_size {
                    batch_txs[qid].send(Some(Batch::Mtm(
                        std::mem::replace(&mut mtm_buf, Vec::with_capacity(batch_size)),
                    )))?;
                }
            }
            ModelClass::BiLstm => {
                let means: Vec<f32> = words[7].split(',').filter_map(|v| v.parse().ok()).collect();
                let stds:  Vec<f32> = words[8].split(',').filter_map(|v| v.parse().ok()).collect();
                let lens:  Vec<i32> = words[9].split(',').filter_map(|v| v.parse().ok()).collect();
                bilstm_buf.push(BiLstmFeature {
                    sample_info, k_seq, means, stds, lens, k_signals, label,
                });
                if bilstm_buf.len() >= batch_size {
                    batch_txs[qid].send(Some(Batch::BiLstm(
                        std::mem::replace(&mut bilstm_buf, Vec::with_capacity(batch_size)),
                    )))?;
                }
            }
        }
    }

    // Flush
    let qid = rng_state % n_workers;
    if !mtm_buf.is_empty() {
        batch_txs[qid].send(Some(Batch::Mtm(mtm_buf)))?;
    }
    if !bilstm_buf.is_empty() {
        batch_txs[qid].send(Some(Batch::BiLstm(bilstm_buf)))?;
    }

    log::info!("[TSV-Producer] done.");
    Ok(())
}

// ─── utilities ────────────────────────────────────────────────────────────────

fn detect_file_type(cfg: &InferenceConfig) -> anyhow::Result<String> {
    if let Some(ref ft) = cfg.file_type {
        return Ok(ft.clone());
    }
    let p = &cfg.input_path;
    if p.is_file() {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        return Ok(match ext {
            "tsv" | "gz" => "tsv",
            "pod5"       => "pod5",
            "slow5" | "blow5" => "slow5",
            _ => anyhow::bail!("cannot detect file type for {:?}", p),
        }
        .to_string());
    }
    // Directory: look at first file extension
    for entry in std::fs::read_dir(p)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".pod5")  { return Ok("pod5".to_string()); }
        if name.ends_with(".slow5") || name.ends_with(".blow5") {
            return Ok("slow5".to_string());
        }
    }
    anyhow::bail!("no signal files found in {:?}", p)
}

fn collect_signal_files(dir: &Path, file_type: &str) -> anyhow::Result<Vec<PathBuf>> {
    let ext = match file_type {
        "pod5"  => ".pod5",
        "slow5" => ".slow5",
        _       => ".blow5",
    };
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let p = entry.path();
        if p.to_string_lossy().ends_with(ext) {
            files.push(p.to_path_buf());
        }
    }
    Ok(files)
}

fn load_positions(path: &Path) -> anyhow::Result<HashSet<String>> {
    use std::io::{BufRead, BufReader};
    let mut set = HashSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let words: Vec<&str> = line.trim().splitn(4, '\t').collect();
        if words.len() >= 3 {
            set.insert(format!("{}\t{}\t{}", words[0], words[1], words[2]));
        }
    }
    Ok(set)
}
