"""End-to-end read- and site-level methylation calling.

The public pipeline intentionally exposes only the inputs that vary between runs.
Model checkpoints and their matching hyperparameters are selected from the
sequencing platform.
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path


_MODEL_FILES = {
    "4khz": ("r1041_4khz_5mC.ckpt", "r1041_4khz_5mC_site.ckpt"),
    "5khz": ("r1041_5khz_5mC.ckpt", "r1041_5khz_5mC_site.ckpt"),
}


def get_platform_models(platform: str) -> tuple[Path, Path]:
    """Return the bundled read-level and site-level models for *platform*."""
    try:
        read_model, site_model = _MODEL_FILES[platform.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("platform must be one of: 4khz, 5khz") from exc

    model_dir = Path(__file__).resolve().parent / "model"
    paths = model_dir / read_model, model_dir / site_model
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Bundled pipeline model(s) not found: {}".format(", ".join(missing))
        )
    return paths


def _read_call_args(args, model_path: Path, result_file: Path) -> Namespace:
    """Build the fixed, trained configuration expected by inference_ultra."""
    return Namespace(
        input_path=str(args.input_path),
        bam=getattr(args, "bam", None),
        result_file=str(result_file),
        model_path=str(model_path),
        batch_size=args.batch_size,
        model_class="mtm",
        seq_len=21,
        signal_len=15,
        class_num=2,
        dropout_rate=0.0,
        n_vocab=16,
        n_embed=4,
        mtm_num_base_features=1,
        mtm_hid_rnn=128,
        mtm_d_static=1,
        mtm_ratios=[2, 2, 2, 2],
        mtm_r_hid=4,
        mtm_norm_first="True",
        mtm_down_mode="concat",
        mtm_temporal_depth=2,
        use_compile=False,
        use_cpu=False,
        nproc_cpu=1,
        nproc=min(10, os.cpu_count() or 1),
        motifs="CG",
        mod_loc=0,
        positions=None,
        chrom=None,
        normalize_method="mad",
        methy_label=1,
        mapq=1,
        identity=0.0,
        coverage_ratio=0.5,
        plant=False,
        single=False,
        recursively="yes",
    )


def _frequency_args(result_file: Path, read_calls: Path, model_path: Path) -> Namespace:
    """Build the fixed site-level estimation configuration."""
    return Namespace(
        input_path=[str(read_calls)],
        result_file=str(result_file),
        file_uid=None,
        bed=True,
        sort=True,
        prob_cf=0.0,
        aggre_model=str(model_path),
        cov_cf=4,
        bin_size=20,
        aggre_hidden=32,
    )


def _call_read_modifications(args: Namespace) -> None:
    from .call_modifications import inference_ultra

    inference_ultra(args)


def _call_frequencies(args: Namespace) -> None:
    from .call_mods_freq import call_mods_frequency_to_file

    call_mods_frequency_to_file(args)


def _read_calls_path(result_file: Path) -> Path:
    """Derive a stable per-read TSV path next to the final frequency file."""
    return result_file.with_name("{}.read_calls.tsv".format(result_file.stem))


def run_pipeline(args) -> None:
    """Run read-level calling followed by site-level frequency estimation."""
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError("input path not found: {}".format(input_path))

    is_feature_tsv = input_path.is_file() and input_path.name.endswith(
        (".tsv", ".tsv.gz")
    )
    bam = getattr(args, "bam", None)
    if not is_feature_tsv:
        if not bam:
            raise ValueError("--bam is required for raw signal input")
        if not Path(bam).expanduser().is_file():
            raise FileNotFoundError("BAM file not found: {}".format(bam))

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero")

    result_file = Path(args.result_file).expanduser().resolve()
    result_file.parent.mkdir(parents=True, exist_ok=True)
    read_model, site_model = get_platform_models(args.platform)
    read_calls = _read_calls_path(result_file)

    print("[pipeline] platform: {}".format(args.platform))
    print("[pipeline] read-level model: {}".format(read_model.name))
    print("[pipeline] site-level model: {}".format(site_model.name))
    print("[pipeline] phase 1/2: calling read-level modifications")
    _call_read_modifications(_read_call_args(args, read_model, read_calls))

    print("[pipeline] phase 2/2: estimating site-level frequencies")
    _call_frequencies(_frequency_args(result_file, read_calls, site_model))

    print("[pipeline] read-level calls: {}".format(read_calls))
    print("[pipeline] done: {}".format(result_file))
