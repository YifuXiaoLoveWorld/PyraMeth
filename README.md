# PyraMeth

**A lightweight framework for low-coverage nanopore 5mC methylation quantification.**

PyraMeth is a two-phase framework for accurate CpG methylation calling from Oxford
Nanopore R10.4.1 reads, with a particular focus on 5×–10× low-coverage regimes.
The read-level phase uses a **Hierarchical Temporal Encoder (HTE)** (~1.48 M
parameters) to capture ionic-current features across multiple temporal scales. The
site-level phase integrates per-read probability distributions and local methylation
context instead of reducing the evidence to hard read counts.

## Contents

- [Recommended workflow: end-to-end pipeline](#recommended-workflow-end-to-end-pipeline)
- [Installation](#installation)
- [Trained models](#trained-models)
- [Example data](#example-data)
- [Advanced usage: individual steps](#advanced-usage-individual-steps)
  - [1. Basecall raw reads](#1-basecall-raw-reads)
  - [2. Call read-level modifications](#2-call-read-level-modifications)
  - [3. Estimate site-level frequencies](#3-estimate-site-level-frequencies)
  - [4. Extract features](#4-extract-features)
  - [5. Train new models](#5-train-new-models)

## Recommended workflow: end-to-end pipeline

For normal inference, use `pyrameth pipeline`. It runs read-level modification
calling and site-level frequency estimation in one command. Selecting `4khz` or
`5khz` automatically loads the matching bundled models and their fixed parameters.

### 1. Basecall with move tables

Raw signal files must first be basecalled with [Dorado](https://github.com/nanoporetech/dorado).
The `--emit-moves` flag is required for signal-to-base alignment:

```bash
dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 \
    --emit-moves --device cuda:all pod5/ --reference chm13v2.0.fa > demo.bam
```

### 2. Run the pipeline

```bash
pyrameth pipeline \
    --input_path pod5/ \
    --bam demo.bam \
    --result_file sample.frequency.bed \
    --platform 4khz \
    --batch_size 800
```

Use `--platform 4khz` for 4 kHz data or `--platform 5khz` for 5 kHz data.
Apart from the required input/output paths and platform, only `--batch_size` normally
needs tuning. Its default value is `500`.

The command produces two files in the output directory:

| Output | Description |
| --- | --- |
| `sample.frequency.bed` | Sorted site-level methylation frequencies in bedMethyl format |
| `sample.frequency.read_calls.tsv` | Retained per-read modification probabilities |

The pipeline also accepts a pre-extracted feature TSV. In that case, `--bam` is not
required:

```bash
pyrameth pipeline -i sample.features.tsv -o sample.frequency.bed \
    --platform 4khz -b 800
```

Use the individual commands documented under
[Advanced usage](#advanced-usage-individual-steps) only when you need to run one
phase separately, customize low-level parameters, or generate ModBAM output.

## Installation

PyraMeth is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/).

Requirements:

- [Python](https://www.python.org/) >= 3.12
- [PyTorch](https://pytorch.org/) >= 2.0
- [Dorado](https://github.com/nanoporetech/dorado) for basecalling raw reads

The remaining Python dependencies are installed from `requirements.txt` or
`environment.yml`.

### 1. Create an environment

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

### 2. Install PyraMeth

After creating and activating the environment, download PyraMeth (**latest version**) from GitHub:

```bash
git clone https://github.com/YifuXiaoLoveWorld/PyraMeth.git
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

The checkpoints are bundled with PyraMeth. `pyrameth pipeline` selects both models
in the matching row automatically:

| Platform | Read-level HTE model | Site-level estimation model |
| --- | --- | --- |
| R10.4.1 4 kHz | [`r1041_4khz_5mC.ckpt`](pyrameth/model/r1041_4khz_5mC.ckpt) | [`r1041_4khz_5mC_site.ckpt`](pyrameth/model/r1041_4khz_5mC_site.ckpt) |
| R10.4.1 5 kHz | [`r1041_5khz_5mC.ckpt`](pyrameth/model/r1041_5khz_5mC.ckpt) | [`r1041_5khz_5mC_site.ckpt`](pyrameth/model/r1041_5khz_5mC_site.ckpt) |

All four models were trained on human data aligned to CHM13 v2.0.

## Example data

Example data (training and test sets) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GNkT0a8-jNdNJe1Wx2eI5hJY_Zv9bXqF). The example data are from the human genome HG002.

## Advanced usage: individual steps

The commands below expose each phase separately. Most users can skip this section
and use the end-to-end pipeline above.

### 1. Basecall raw reads

For POD5 input, basecall with [Dorado](https://github.com/nanoporetech/dorado). The `--emit-moves` flag is required for signal-to-base alignment:

```bash
# GPU
dorado basecaller dna_r10.4.1_e8.2_400bps_sup@v4.1.0 --device cuda:0 --emit-moves pod5/ --reference reference.fa > example.bam
# CPU
dorado basecaller dna_r10.4.1_e8.2_400bps_sup@v4.1.0 --device cpu   --emit-moves pod5/ --reference reference.fa > example.bam
```

### 2. Call read-level modifications

`call_mods` accepts either raw signal files (POD5/SloW5/BloW5) or a pre-extracted feature TSV as input and writes a per-read TSV. `call_mods_bam` is an alternative that writes a ModBAM file with MM/ML tags directly.

```bash
# pod5/slow5/blow5 → TSV, GPU
pyrameth call_mods --input_path pod5/ --bam demo.bam --model_path r1041_4khz_5mC.ckpt \
    --result_file pod5.CG.call_mods.tsv --nproc 32 --seq_len 21 --signal_len 15 -b 800

# pod5/slow5/blow5 → ModBAM (MM/ML tags, sorted and indexed)
pyrameth call_mods_bam --input_path pod5/ --bam demo.bam --model_path r1041_4khz_5mC.ckpt \
    --output_bam pod5.CG.mods.bam --nproc 32

# pre-extracted feature TSV → TSV (skip signal reading)
pyrameth call_mods --input_path pod5s.CG.features.tsv --model_path r1041_4khz_5mC.ckpt \
    --result_file pod5s.CG.call_mods.tsv --motifs CG --nproc 32 -b 800
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

### 3. Estimate site-level frequencies

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
    --aggre_model r1041_4khz_5mC_site.ckpt \
    --cov_cf 4 \
    --bin_size 20 \
    --sort
```

Aggregate-mode parameters:
- **--aggre_model / -m**: site-level estimation model checkpoint (.ckpt)
- **--cov_cf**: minimum read coverage per site (default: 4)
- **--bin_size**: histogram bin count for the per-read probability distribution (default: 20)

### 4. Extract features

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

### 5. Train new models

```bash
# requires two independent datasets for training and validation
# use pyrameth trainm -h for full options
pyrameth trainm --train_file /path/to/train/file --valid_file /path/to/valid/file \
    --model_dir /dir/to/save/the/new/model
```

## Todo

- [ ] add tqdm for progress bar
