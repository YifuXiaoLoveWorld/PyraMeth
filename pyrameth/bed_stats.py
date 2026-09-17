#! /usr/bin/env python
"""Summarise methylation BED / TSV files and explain size differences."""

from __future__ import absolute_import

import argparse
import gzip
import os
import sys
from collections import Counter


CANONICAL_CHROMS = {"chr{}".format(i) for i in range(1, 23)} | {"chrX", "chrY", "chrM", "chrMT"}
CANONICAL_CHROMS |= {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "M"}


def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _filesize(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _human_bytes(n):
    if n is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{} B".format(n)


def detect_format(fields):
    """Classify a methylation text row.

    Returns one of: bedmethyl, freq_tsv, per_read, unknown.
    """
    n = len(fields)
    if n < 6:
        return "unknown"
    strand_like = {"+", "-", "."}
    if n >= 11 and fields[3] == "." and fields[5] in strand_like:
        return "bedmethyl"
    if n >= 10 and fields[2] in strand_like:
        # per-read:  chrom pos strand pos_in_strand readname t/c prob0 prob1 label kmer
        # freq TSV:  chrom pos strand pos_in_strand p0sum p1sum met unmet cov freq kmer
        if fields[5] in {"t", "c", "T", "C"}:
            return "per_read"
        if _is_float(fields[4]) and _is_float(fields[5]):
            return "freq_tsv"
        if not _is_float(fields[4]):
            return "per_read"
    return "unknown"


def _is_float(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _coverage_from_row(fmt, fields):
    try:
        if fmt == "bedmethyl":
            return int(fields[4])
        if fmt == "freq_tsv":
            return int(fields[8])
        if fmt == "per_read":
            return 1
    except (ValueError, IndexError):
        return None
    return None


def _site_key(fmt, fields):
    try:
        chrom = fields[0]
        pos = fields[1]
        if fmt == "bedmethyl":
            return chrom, pos, fields[5]
        if fmt in {"freq_tsv", "per_read"}:
            return chrom, pos, fields[2]
        return chrom, pos, "."
    except IndexError:
        return None


def summarize_file(path, max_examples=5, collect_sites=False, progress_every=5000000):
    """Stream a BED/TSV and collect size / format / chromosome / coverage stats."""
    stats = {
        "path": path,
        "exists": os.path.isfile(path),
        "bytes": _filesize(path) if os.path.isfile(path) else None,
        "compressed": path.endswith(".gz"),
        "format": None,
        "n_lines": 0,
        "n_skipped": 0,
        "n_cols": None,
        "examples": [],
        "chroms": Counter(),
        "strands": Counter(),
        "coverage_sum": 0,
        "coverage_n": 0,
        "coverage_min": None,
        "coverage_max": None,
        "cov1": 0,
        "cov_lt4": 0,
        "canonical_lines": 0,
        "other_lines": 0,
        "sites": set() if collect_sites else None,
        "n_unique_sites": None,
        "error": None,
    }
    if not stats["exists"]:
        stats["error"] = "file not found"
        return stats

    fmt = None
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#") or line.startswith("track"):
                stats["n_skipped"] += 1
                continue
            fields = line.rstrip("\n").split("\t")
            stats["n_lines"] += 1
            if fmt is None:
                fmt = detect_format(fields)
                stats["format"] = fmt
                stats["n_cols"] = len(fields)
            if len(stats["examples"]) < max_examples:
                stats["examples"].append(fields[: min(12, len(fields))])
            chrom = fields[0] if fields else ""
            stats["chroms"][chrom] += 1
            if chrom in CANONICAL_CHROMS:
                stats["canonical_lines"] += 1
            else:
                stats["other_lines"] += 1
            key = _site_key(fmt, fields)
            if key is not None:
                stats["strands"][key[2]] += 1
                if collect_sites:
                    stats["sites"].add(key)
            cov = _coverage_from_row(fmt, fields)
            if cov is not None:
                stats["coverage_sum"] += cov
                stats["coverage_n"] += 1
                stats["coverage_min"] = cov if stats["coverage_min"] is None else min(stats["coverage_min"], cov)
                stats["coverage_max"] = cov if stats["coverage_max"] is None else max(stats["coverage_max"], cov)
                if cov <= 1:
                    stats["cov1"] += 1
                if cov < 4:
                    stats["cov_lt4"] += 1
            if stats["n_lines"] % progress_every == 0:
                print(
                    "[bed_stats] {} scanned {} lines..".format(os.path.basename(path), stats["n_lines"]),
                    flush=True,
                )

    if collect_sites:
        stats["n_unique_sites"] = len(stats["sites"])
    if stats["coverage_n"]:
        stats["coverage_mean"] = stats["coverage_sum"] / float(stats["coverage_n"])
    else:
        stats["coverage_mean"] = None
    return stats


def _format_report(stats):
    lines = []
    lines.append("file: {}".format(stats["path"]))
    if stats["error"]:
        lines.append("  error: {}".format(stats["error"]))
        return "\n".join(lines)
    lines.append("  size: {} ({})".format(_human_bytes(stats["bytes"]), stats["bytes"]))
    lines.append("  gzip: {}".format(stats["compressed"]))
    lines.append("  format: {} ({} columns)".format(stats["format"], stats["n_cols"]))
    lines.append("  data lines: {}".format(stats["n_lines"]))
    if stats["n_unique_sites"] is not None:
        lines.append("  unique chrom:pos:strand: {}".format(stats["n_unique_sites"]))
        if stats["n_lines"] and stats["n_unique_sites"] < stats["n_lines"]:
            lines.append(
                "  duplicate rows: {}".format(stats["n_lines"] - stats["n_unique_sites"])
            )
        if stats["format"] == "per_read":
            lines.append(
                "  NOTE: this looks like a per-read call file, not a site-level frequency BED."
            )
    chroms = stats["chroms"]
    lines.append("  chromosomes: {} distinct".format(len(chroms)))
    lines.append("  canonical chr1-22/X/Y/M lines: {}".format(stats["canonical_lines"]))
    lines.append("  other contig lines: {}".format(stats["other_lines"]))
    top_chroms = chroms.most_common(8)
    if top_chroms:
        lines.append(
            "  top chroms: "
            + ", ".join("{}={}".format(name, count) for name, count in top_chroms)
        )
    if stats["strands"]:
        lines.append(
            "  strands: "
            + ", ".join("{}={}".format(k, v) for k, v in sorted(stats["strands"].items()))
        )
    if stats["coverage_n"]:
        lines.append(
            "  coverage: n={} min={} max={} mean={:.2f} cov<=1={} cov<4={}".format(
                stats["coverage_n"],
                stats["coverage_min"],
                stats["coverage_max"],
                stats["coverage_mean"],
                stats["cov1"],
                stats["cov_lt4"],
            )
        )
    if stats["examples"]:
        lines.append("  example row: " + "\t".join(stats["examples"][0]))
    return "\n".join(lines)


def explain_size_gap(left, right):
    """Return human-readable reasons why *left* is larger than *right*."""
    reasons = []
    if not left["exists"] or not right["exists"]:
        return reasons

    l_bytes, r_bytes = left["bytes"] or 0, right["bytes"] or 0
    if l_bytes and r_bytes and l_bytes > r_bytes:
        reasons.append(
            "raw size ratio: {:.1f}x ({} vs {})".format(
                l_bytes / float(r_bytes), _human_bytes(l_bytes), _human_bytes(r_bytes)
            )
        )
    if left["n_lines"] and right["n_lines"]:
        reasons.append(
            "line-count ratio: {:.1f}x ({} vs {})".format(
                left["n_lines"] / float(right["n_lines"]),
                left["n_lines"],
                right["n_lines"],
            )
        )

    if left["format"] == "per_read" and right["format"] in {"bedmethyl", "freq_tsv"}:
        reasons.append(
            "FORMAT: the larger file is per-read (one row per CpG per read), "
            "while the smaller file is site-level frequency. That alone can be "
            "tens to hundreds of times larger."
        )
    elif left["format"] == "freq_tsv" and right["format"] == "bedmethyl":
        reasons.append(
            "FORMAT: the larger file looks like call_freq TSV (includes k-mer and "
            "probability sums), not compact bedMethyl. Running call_freq with --bed "
            "produces the smaller 11-column bedMethyl layout."
        )
    elif left["format"] and right["format"] and left["format"] != right["format"]:
        reasons.append(
            "FORMAT mismatch: {} vs {}.".format(left["format"], right["format"])
        )

    if not left["compressed"] and right["compressed"]:
        reasons.append("COMPRESSION: larger file is uncompressed, smaller file is gzipped.")

    if left["other_lines"] and right["n_lines"]:
        extra_frac = left["other_lines"] / float(left["n_lines"])
        right_extra = right["other_lines"] / float(right["n_lines"]) if right["n_lines"] else 0
        if extra_frac > right_extra + 0.05:
            reasons.append(
                "CONTIGS: {:.1f}% of larger-file rows are non-canonical chromosomes "
                "vs {:.1f}% in the smaller file (alt/decoy/unplaced contigs).".format(
                    100 * extra_frac, 100 * right_extra
                )
            )

    if left["coverage_n"] and right["coverage_n"]:
        l1 = left["cov1"] / float(left["coverage_n"])
        r1 = right["cov1"] / float(right["coverage_n"])
        if l1 > r1 + 0.1:
            reasons.append(
                "COVERAGE: {:.1f}% of larger-file sites have coverage <=1 vs {:.1f}% "
                "in the smaller file. Unfiltered / non-Q10 / non-downsampled reads "
                "add many singleton sites.".format(100 * l1, 100 * r1)
            )
        l4 = left["cov_lt4"] / float(left["coverage_n"])
        r4 = right["cov_lt4"] / float(right["coverage_n"])
        if l4 > r4 + 0.1:
            reasons.append(
                "COVERAGE: {:.1f}% of larger-file sites have coverage <4 vs {:.1f}% "
                "in the smaller file.".format(100 * l4, 100 * r4)
            )

    left_name = os.path.basename(left["path"])
    right_name = os.path.basename(right["path"])
    if "all" in left_name.lower() and any(tag in right_name.lower() for tag in ("q10", "ds")):
        reasons.append(
            "READ FILTER: filename {} is unfiltered (.all), while {} is Q10 "
            "and/or downsampled (ds). That is the comparison this Q10 filter "
            "step is meant to correct.".format(left_name, right_name)
        )

    if left["sites"] is not None and right["sites"] is not None:
        only_left = len(left["sites"] - right["sites"])
        only_right = len(right["sites"] - left["sites"])
        both = len(left["sites"] & right["sites"])
        reasons.append(
            "SITE OVERLAP: shared={}, only-in-larger={}, only-in-smaller={}.".format(
                both, only_left, only_right
            )
        )

    if not reasons:
        reasons.append("No single obvious format bug; compare chromosome sets and coverage next.")
    return reasons


def write_report(paths, output_file=None, collect_sites=False):
    reports = []
    stats_list = []
    for path in paths:
        print("[bed_stats] summarising {}".format(path), flush=True)
        stats = summarize_file(path, collect_sites=collect_sites)
        stats_list.append(stats)
        reports.append(_format_report(stats))

    if len(stats_list) >= 2:
        reports.append("why the first file is larger than the second:")
        for reason in explain_size_gap(stats_list[0], stats_list[1]):
            reports.append("  - {}".format(reason))

    text = "\n\n".join(reports) + "\n"
    print(text, end="")
    if output_file:
        out_dir = os.path.dirname(os.path.abspath(output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w") as handle:
            handle.write(text)
        print("[bed_stats] wrote {}".format(output_file), flush=True)
    return stats_list


def bed_stats_from_args(args):
    return write_report(
        args.input_path,
        output_file=args.result_file,
        collect_sites=args.sites,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarise BED/TSV methylation files and compare sizes."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        action="append",
        required=True,
        help="BED or TSV path. Pass twice to compare a large file against a small one.",
    )
    parser.add_argument("--result_file", "-o", default=None, help="optional report output path")
    parser.add_argument(
        "--sites",
        action="store_true",
        default=False,
        help="collect unique chrom:pos:strand keys (uses more RAM, needed for overlap)",
    )
    args = parser.parse_args(argv)
    bed_stats_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
