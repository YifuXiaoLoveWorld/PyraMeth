# pyrameth-rs — Build & Setup Guide

## Prerequisites

### 1. Rust toolchain
```powershell
# Install rustup (https://rustup.rs)
winget install Rustlang.Rustup
rustup default stable
```

### 2. LibTorch (required for tch-rs / model inference)
Download the matching LibTorch for your CUDA version from https://pytorch.org/

```powershell
# Example: CUDA 12.1
$env:LIBTORCH = "C:\libtorch"
$env:Path     = "$env:LIBTORCH\lib;$env:Path"
```
Or set these permanently in System → Environment Variables.

### 3. Python dependencies (for TorchScript export only)
```bash
pip install torch   # for export_torchscript.py
```

Slow5/Blow5 reading is pure Rust — no Python or C library needed.

POD5 reading is native Rust when built with `--features pod5-pure`
(`pod5-format` + `svb16` + `arrow` from bsaintjo/pod5-rs — pure Rust, no C
library, no polars).  It decodes signal straight to raw ADC and only
decompresses the rows belonging to reads present in the BAM index.
Without that feature the binary exits with a clear error message.

---

## Step 1 — Export model to TorchScript

```bash
cd e:/code/pyrameth-rs

# modelMTM (default, recommended)
python scripts/export_torchscript.py \
    --model_path  ../model/human_r1041_5khz_CG_epoch5.ckpt \
    --output_path ../model/human_r1041_5khz_CG_epoch5.pt   \
    --model_class mtm \
    --seq_len 21 --signal_len 15

# ModelBiLSTM
python scripts/export_torchscript.py \
    --model_path  ../model/plant_r1041_5khz_C_epoch4.ckpt \
    --output_path ../model/plant_r1041_5khz_C_epoch4.pt  \
    --model_class bilstm
```

---

## Step 2 — Build

```bash
cd e:/code/pyrameth-rs

# Debug build (fast compile); Slow5/Blow5 native, POD5 disabled
cargo build

# Release build with native POD5 support (recommended for production)
cargo build --release --features pod5-pure

# Release build without POD5 (Slow5/BAM only)
cargo build --release
```

The binary appears at `target/release/pyrameth` (Linux/macOS) or
`target\release\pyrameth.exe` (Windows).

> **POD5 feature note**: `--features pod5-pure` enables native POD5 reading via
> `pod5-format` (FlatBuffers footer), `svb16` (VBZ decompression), and `arrow`
> (IPC table reader) — all pure Rust, no C library, no polars.  Signal is decoded
> to raw ADC i16 (matching Python's `pod5_record.signal`), and only the rows of
> reads present in the BAM index are decompressed.  Without this flag, running
> `pyrameth` on POD5 files prints an informative error and exits.

---

## Step 3 — Usage

### call-mods (methylation inference)
```bash
./target/release/pyrameth call-mods \
    --input-path /data/pod5/ \
    --bam        /data/aligned.bam \
    --model-path ../model/human_r1041_5khz_CG_epoch5.pt \
    --model-class mtm \
    --result-file /data/mods.tsv \
    --seq-len 21 --signal-len 15 \
    --batch-size 512 --nproc 8
```

### call-freq (genome-level frequency)
```bash
# Count-based TSV
./target/release/pyrameth call-freq \
    --input-path /data/mods.tsv \
    --result-file /data/freq.tsv \
    --prob-cf 0.5 --sort

# Count-based bedMethyl
./target/release/pyrameth call-freq \
    --input-path /data/mods.tsv \
    --result-file /data/freq.bed \
    --bed --sort

# AggrAttRNN neural-network refinement (always writes bedMethyl)
# Step 1: export AggrAttRNN model
python scripts/export_torchscript.py \
    --model_path  ../model/aggr_model.ckpt \
    --output_path ../model/aggr_model.pt   \
    --model_class aggr

# Step 2: run refined frequency calling
./target/release/pyrameth call-freq \
    --input-path  /data/mods.tsv \
    --result-file /data/freq_aggr.bed \
    --aggre-model ../model/aggr_model.pt \
    --cov-cf 4 --bin-size 20 --sort
```

### extract (feature extraction to TSV)
```bash
./target/release/pyrameth extract \
    --input-dir /data/pod5/ \
    --bam       /data/aligned.bam \
    --write-path /data/features.tsv \
    --motifs CG --seq-len 21 --signal-len 15 \
    --nproc 16
```

---

## Feature extraction correctness

The Rust feature extraction is designed to produce **bit-for-bit identical
output** to Python for the following operations:

| Algorithm | Python reference | Rust implementation |
|-----------|-----------------|---------------------|
| MAD normalisation | `statsmodels.robust.mad` (÷ 0.6744897501960817) | `signal::normalize_mad` |
| Signal rounding | `np.around(..., decimals=6)` | `(x * 1e6).round() / 1e6` |
| Signal rect | `build_signal_rect_from_movetable` | `signal::build_signal_rect` |
| Downsampling | `np.linspace().astype(np.int32)` (truncation) | `frac as i32 as usize` |
| CIGAR parsing | `get_q2tloc_from_cigar` | `cigar::get_q2tloc_from_cigar` |
| k-mer encoding | `base2code_dna` | `kmer::base_to_code` |

> **Validation tip**: Run both Python and Rust on the same small BAM+POD5
> pair, then `diff` the feature TSVs.  Any difference will be limited to the
> 6th decimal place due to f32↔f64 promotion differences, which is negligible
> for model inference.

---

## Roadmap

- [ ] Native POD5 reading via `pod5` crate (eliminates Python subprocess)
- [ ] Native Slow5 reading via `slow5` crate
- [ ] AggrAttRNN frequency refinement (call_freq --aggre_model)
- [ ] gzip output support for large result files
- [ ] Benchmarking harness vs Python baseline
