# PyraMeth

## PyraMeth: a lightweight framework for low-coverage nanopore 5mC methylation quantification.

PyraMeth is a two-phase framework for accurate CpG methylation calling from Oxford Nanopore R10.4.1 reads, with a particular focus on 5×–10× low-coverage regimes. The read-level phase uses a **Hierarchical Temporal Encoder (HTE)** (~1.48 M parameters) that captures ionic current features across multiple temporal scales. The site-level phase uses a **site-level estimation module** that integrates the distribution of per-read probabilities at a CpG and its local methylation context to estimate methylation frequency without reducing probabilistic evidence to hard read counts.

## Contents

- [Installation](#Installation)
- [Trained models](#Trained-models)
- [Quick start](#Quick-start)
- [Usage](#Usage)
  - [1. Basecall](#1-basecall)
  - [2. call modifications](#2-call-modifications)
  - [3. call frequency of modifications](#3-call-frequency-of-modifications)
  - [4. extract features](#4-extract-features)
  - [5. train new models](#5-train-new-models)
- [Appendix](#Appendix)

## Installation

PyraMeth is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/).

- Prerequisites:\
   [Python3.\*](https://www.python.org/) (version >=3.12) \
   [Dorado](https://github.com/nanoporetech/dorado)
- Dependencies: \
   [numpy](http://www.numpy.org/) \
   [h5py](https://github.com/h5py/h5py) \
   [statsmodels](https://github.com/statsmodels/statsmodels/) \
   [scikit-learn](https://scikit-learn.org/stable/) \
   [mappy](https://github.com/lh3/minimap2/tree/master/python) \
   [pysam](https://github.com/pysam-developers/pysam) \
   [pod5](https://github.com/nanoporetech/pod5-file-format) \
   [pyslow5](https://github.com/hasindu2008/slow5lib) \
   [PyTorch](https://pytorch.org/) (version >=2.0.0)

#### 1. Create an environment

We highly recommend using a virtual environment for the installation of PyraMeth and its dependencies. A virtual environment can be created and (de)activated as follows using [conda](https://conda.io/docs/):

```bash
# create (recommended: use environment.yml for exact dependency resolution)
conda env create -f environment.yml
# or create manually
conda create -n pyrameth python=3.12
# activate
conda activate pyrameth
# deactivate
conda deactivate
```

The virtual environment can also be created using [virtualenv](https://github.com/pypa/virtualenv/).

#### 2. Install PyraMeth

After creating and activating the environment, download PyraMeth (**latest version**) from GitHub:

```bash
git clone https://github.com/PengNi/PyraMeth.git
cd PyraMeth
pip install -e .
```

[PyTorch](https://pytorch.org/) should be installed to match your CUDA version. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/):

```bash
# example: CUDA 11.8
conda install pytorch=2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# or via pip
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

## Trained models

Currently, the following models are available:

- [human_r1041_4khz_CG.ckpt](model/human_r1041_4khz_CG_epoch7.ckpt): HTE model trained on human **R10.4.1 (4 kHz)** data aligned to CHM13 v2.0, for detecting 5mC at CpG sites.
- [human_r1041_5khz_CG.ckpt](model/human_r1041_5khz_CG_epoch5.ckpt): HTE model trained on human **R10.4.1 (5 kHz)** data aligned to CHM13 v2.0, for detecting 5mC at CpG sites.
- [human_r1041_4khz_CG_site.ckpt](model/human_r1041_4khz_CG_site.ckpt): Site-level estimation model trained on human **R10.4.1 (4 kHz)** data aligned to CHM13 v2.0, for aggregating per-read 5mC probabilities to site-level frequency.
- [human_r1041_5khz_CG_site.ckpt](model/human_r1041_5khz_CG_site.ckpt): Site-level estimation model trained on human **R10.4.1 (5 kHz)** data aligned to CHM13 v2.0, for aggregating per-read 5mC probabilities to site-level frequency.

## Example data

Example data (training and test sets) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GNkT0a8-jNdNJe1Wx2eI5hJY_Zv9bXqF). The example data are from the human genome HG002.

## Quick start

Raw POD5 files must be basecalled with [Dorado](https://github.com/nanoporetech/dorado) using the `--emit-moves` flag to retain the move table.

**POD5/SloW5/BloW5 → per-read TSV then frequency:**

```bash
# 1. Dorado basecall with move table
dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 --emit-moves --device cuda:all pod5/ --reference chm13v2.0.fa > demo.bam

# 2. Phase 1 — read-level calling (HTE model)
pyrameth call_mods --input_path pod5/ --bam demo.bam --model_path *.ckpt \
    --result_file pod5.CG.call_mods.tsv --nproc 32 --nproc_gpu 4 --seq_len 21 --signal_len 15 -b 8192

# 3a. Phase 2 — count-based frequency (default)
pyrameth call_freq --input_path pod5.CG.call_mods.tsv --result_file pod5.CG.call_mods.frequency.tsv

# 3b. Phase 2 — site-level neural-network frequency estimation (requires site-level model checkpoint)
pyrameth call_freq --input_path pod5.CG.call_mods.tsv --result_file pod5.CG.aggregate.bed -m human_r1041_4khz_CG_site.ckpt
```

**POD5/SloW5/BloW5 → ModBAM (MM/ML tags):**

```bash
pyrameth call_mods_bam --input_path pod5/ --bam demo.bam --model_path *.ckpt \
    --output_bam pod5.CG.mods.bam --nproc 32
```

## Usage

#### 1. Basecall

For POD5 input, basecall with [Dorado](https://github.com/nanoporetech/dorado). The `--emit-moves` flag is required for signal-to-base alignment:

```bash
# GPU
dorado basecaller dna_r10.4.1_e8.2_400bps_sup@v4.1.0 --device cuda:0 --emit-moves pod5/ --reference reference.fa > example.bam
# CPU
dorado basecaller dna_r10.4.1_e8.2_400bps_sup@v4.1.0 --device cpu   --emit-moves pod5/ --reference reference.fa > example.bam
```

#### 2. call modifications

`call_mods` accepts either raw signal files (POD5/SloW5/BloW5) or a pre-extracted feature TSV as input and writes a per-read TSV. `call_mods_bam` is an alternative that writes a ModBAM file with MM/ML tags directly.

```bash
# pod5/slow5/blow5 → TSV, GPU
pyrameth call_mods --input_path pod5/ --bam demo.bam --model_path human.r10.4.CG.ckpt \
    --result_file pod5.CG.call_mods.tsv --nproc 32 --nproc_gpu 4 --seq_len 21 --signal_len 15 -b 8192

# pod5/slow5/blow5 → ModBAM (MM/ML tags, sorted and indexed)
pyrameth call_mods_bam --input_path pod5/ --bam demo.bam --model_path human.r10.4.CG.ckpt \
    --output_bam pod5.CG.mods.bam --nproc 32

# pre-extracted feature TSV → TSV (skip signal reading)
pyrameth call_mods --input_path pod5s.CG.features.tsv --model_path human.r10.4.CG.ckpt \
    --result_file pod5s.CG.call_mods.tsv --motifs CG --nproc 32 --nproc_gpu 4 -b 8192
```

The per-read modification call file is a tab-delimited text file with the following columns:

- **chrom**: chromosome name
- **pos**: 0-based position of the targeted base in the chromosome
- **strand**: +/−, aligned strand of the read to the reference
- **pos_in_strand**: 0-based position in the aligned strand
- **readname**: read name
- **read_strand**: t/c, template or complement
- **prob_0**: [0, 1], probability of unmethylated
- **prob_1**: [0, 1], probability of methylated
- **called_label**: 0/1, unmethylated/methylated
- **k_mer**: sequence context around the targeted base

#### 3. call frequency of modifications

`call_freq` supports two modes controlled by whether `--aggre_model` is provided:

**Count mode** (default) — count-based aggregation:

```bash
# TSV output
pyrameth call_freq --input_path pod5s.CG.call_mods.tsv --result_file pod5s.CG.call_mods.frequency.tsv
# bedMethyl output
pyrameth call_freq --input_path pod5s.CG.call_mods.tsv --result_file pod5s.CG.call_mods.frequency.bed --bed
# sorted bedMethyl
pyrameth call_freq --input_path pod5s.CG.call_mods.tsv --result_file pod5s.CG.call_mods.frequency.bed --bed --sort
```

Default TSV output columns:

- **chrom**, **pos**, **strand**, **pos_in_strand**
- **prob_0_sum**: sum of unmethylated probabilities across reads
- **prob_1_sum**: sum of methylated probabilities across reads
- **count_modified**: reads called as modified
- **count_unmodified**: reads called as unmodified
- **coverage**: total aligned reads at this site
- **modification_frequency**: methylation frequency
- **k_mer**: sequence context

**Aggregate mode** (`--aggre_model`) — site-level neural-network frequency estimation, always outputs bedMethyl:

```bash
pyrameth call_freq \
    --input_path pod5s.CG.call_mods.tsv \
    --result_file pod5s.CG.aggregate.bed \
    --aggre_model human_r1041_4khz_CG_site.ckpt \
    --cov_cf 4 \
    --bin_size 20 \
    --sort
```

Aggregate-mode parameters:
- **--aggre_model / -m**: site-level estimation model checkpoint (.ckpt)
- **--cov_cf**: minimum read coverage per site (default: 4)
- **--bin_size**: histogram bin count for the per-read probability distribution (default: 20)

#### 4. extract features

Feature extraction from signal files, primarily used for training. By default, PyraMeth extracts 21-mer sequence and 21×15-signal features at each CpG motif:

```bash
pyrameth extract -i pod5/ --bam example.bam --reference_path chm13v2.0.fa \
    -o pod5.CG.features.tsv --nproc 30 --motifs CG
```

Extracted feature file columns:

- **chrom**, **pos**, **strand**, **pos_in_strand**, **readname**, **read_strand**
- **k_mer**: sequence context around the targeted base
- **signal_means**: per-base signal means in the k-mer
- **signal_stds**: per-base signal standard deviations
- **signal_lens**: per-base signal lengths
- **raw_signals**: raw signal values per base, separated by ';'
- **methy_label**: 0/1 ground-truth label (for training)

#### 5. train new models

```bash
# requires two independent datasets for training and validation
# use pyrameth train -h for full options
pyrameth train --train_file /path/to/train/file --valid_file /path/to/valid/file \
    --model_dir /dir/to/save/the/new/model
```

## Todo

- [ ] add tqdm for progress bar
