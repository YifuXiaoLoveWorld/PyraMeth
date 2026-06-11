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

## Available models

Checkpoints live in `model/`.  Per-read **call** models (`*_5mC`) are used by
`call-mods`; site-level **aggregation** models (`*_5mC_site`) are used by
`call-freq --aggre-model` for neural-network frequency refinement.

| Model file | Chemistry | Sample rate | Modification | Stage |
|------------|-----------|-------------|--------------|-------|
| `r1041_4khz_5mC.ckpt`      | R10.4.1 | 4 kHz | 5mC | call-mods (per-read) |
| `r1041_5khz_5mC.ckpt`      | R10.4.1 | 5 kHz | 5mC | call-mods (per-read) |
| `r1041_4khz_5mC_site.ckpt` | R10.4.1 | 4 kHz | 5mC | call-freq (site aggregation) |
| `r1041_5khz_5mC_site.ckpt` | R10.4.1 | 5 kHz | 5mC | call-freq (site aggregation) |

Each `.ckpt` is exported to a TorchScript `.pt` before use — see
[Exporting models to TorchScript](#exporting-models-to-torchscript).

---

## Step 1 — Build

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

## Step 2 — Usage

### call-mods (methylation inference)
```bash
./target/release/pyrameth call-mods \
    --input-path /data/pod5/ \
    --bam        /data/aligned.bam \
    --model-path model/r1041_5khz_5mC.pt \
    --result-file /data/mods.tsv \
    --seq-len 21 --signal-len 15 \
    --batch-size 512 --nproc 8
```

### call-freq (site-level frequency)
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

# site-level neural-network frequency estimation (always writes bedMethyl)
# uses the exported site model (see "Exporting models to TorchScript" below)
./target/release/pyrameth call-freq \
    --input-path  /data/mods.tsv \
    --result-file /data/freq_aggr.bed \
    --aggre-model model/r1041_5khz_5mC_site.pt \
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

## Exporting models to TorchScript

The Rust binary loads TorchScript `.pt` files.  Convert each `.ckpt` from the
[Available models](#available-models) table with `scripts/export_torchscript.py`
(requires `pip install torch`).

```bash
cd e:/code/pyrameth-rs

# Per-read call model (used by call-mods)
python scripts/export_torchscript.py \
    --model_path  model/r1041_5khz_5mC.ckpt \
    --output_path model/r1041_5khz_5mC.pt   \
    --seq_len 21 --signal_len 15

# Site-level aggregation model (used by call-freq --aggre-model)
python scripts/export_torchscript.py \
    --model_path  model/r1041_5khz_5mC_site.ckpt \
    --output_path model/r1041_5khz_5mC_site.pt   \
    --model_class aggr
```

---

## Roadmap

- [ ] Native POD5 reading via `pod5` crate (eliminates Python subprocess)
- [ ] Native Slow5 reading via `slow5` crate
- [ ] site-level neural-network frequency estimation (call_freq --aggre_model)
- [ ] gzip output support for large result files
- [ ] Benchmarking harness vs Python baseline
