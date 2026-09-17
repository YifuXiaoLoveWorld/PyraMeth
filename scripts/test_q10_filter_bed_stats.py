#!/usr/bin/env python
"""Synthetic checks for filter_reads and bed_stats."""

from __future__ import print_function

import gzip
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pyrameth.filter_reads import filter_read_calls, load_read_ids
from pyrameth.bed_stats import detect_format, explain_size_gap, summarize_file


def _write_gz(path, lines):
    with gzip.open(path, "wt") as handle:
        handle.write("".join(lines))


def test_filter_reads():
    tmp = tempfile.mkdtemp(prefix="pyrameth_filter_")
    tsv = os.path.join(tmp, "calls.tsv.gz")
    ids = os.path.join(tmp, "ids.txt.gz")
    out = os.path.join(tmp, "kept.tsv.gz")
    rows = [
        "chr1\t10\t+\t10\treadA\tt\t0.9\t0.1\t0\tACGCG\n",
        "chr1\t11\t-\t11\treadB\tc\t0.2\t0.8\t1\tCGCGT\n",
        "chr1\t12\t+\t12\treadC\tt\t0.5\t0.5\t0\tGCGTA\n",
        "chr2\t20\t+\t20\treadA\tt\t0.1\t0.9\t1\tTTTTC\n",
    ]
    _write_gz(tsv, rows)
    _write_gz(ids, ["read_id\n", "readA\n", "readC\n"])
    allow = load_read_ids(ids)
    assert allow == {"readA", "readC"}, allow
    n_in, n_out, n_bad = filter_read_calls(tsv, allow, out)
    assert n_in == 4 and n_out == 3 and n_bad == 0, (n_in, n_out, n_bad)
    with gzip.open(out, "rt") as handle:
        kept = handle.readlines()
    assert len(kept) == 3
    assert all("readB" not in line for line in kept)
    print("test_filter_reads ok")


def test_format_detection():
    bed = "chr1\t10\t11\t.\t8\t+\t10\t11\t0,0,0\t8\t75".split("\t")
    freq = "chr1\t10\t+\t10\t1.2\t6.8\t7\t1\t8\t0.8750\tACGCG".split("\t")
    per_read = "chr1\t10\t+\t10\treadA\tt\t0.1\t0.9\t1\tACGCG".split("\t")
    assert detect_format(bed) == "bedmethyl"
    assert detect_format(freq) == "freq_tsv"
    assert detect_format(per_read) == "per_read"
    print("test_format_detection ok")


def test_bed_stats_reason():
    tmp = tempfile.mkdtemp(prefix="pyrameth_bed_")
    big = os.path.join(tmp, "HBA20-2.dorado_0.7.1.pyrameth.all.bed")
    small = os.path.join(tmp, "HBA20-2.dorado_0.7.1.sup.per_read.Q10.ds.CG.freq.bed")
    per_read_rows = [
        "chr1\t10\t+\t10\treadA\tt\t0.1\t0.9\t1\tACGCG\n",
        "chr1\t10\t+\t10\treadB\tt\t0.2\t0.8\t1\tACGCG\n",
        "chrUn_decoy\t1\t+\t1\treadC\tt\t0.3\t0.7\t1\tTTTTT\n",
    ]
    bed_rows = [
        "chr1\t10\t11\t.\t2\t+\t10\t11\t0,0,0\t2\t80\n",
    ]
    with open(big, "w") as handle:
        handle.writelines(per_read_rows)
    with open(small, "w") as handle:
        handle.writelines(bed_rows)
    left = summarize_file(big)
    right = summarize_file(small)
    assert left["format"] == "per_read"
    assert right["format"] == "bedmethyl"
    reasons = " ".join(explain_size_gap(left, right))
    assert "per-read" in reasons
    assert "READ FILTER" in reasons
    print("test_bed_stats_reason ok")


def test_call_freq_count_mode():
    from argparse import Namespace
    from pyrameth.call_mods_freq import call_mods_frequency_to_file

    tmp = tempfile.mkdtemp(prefix="pyrameth_freq_")
    tsv = os.path.join(tmp, "q10.tsv")
    bed = os.path.join(tmp, "q10.bed")
    with open(tsv, "w") as handle:
        handle.write("chr1\t10\t+\t10\treadA\tt\t0.9\t0.1\t0\tACGCG\n")
        handle.write("chr1\t10\t+\t10\treadC\tt\t0.2\t0.8\t1\tACGCG\n")
        handle.write("chr1\t20\t-\t20\treadA\tc\t0.1\t0.9\t1\tGCGTA\n")
    args = Namespace(
        input_path=[tsv],
        result_file=bed,
        file_uid=None,
        bed=True,
        sort=True,
        prob_cf=0.0,
        aggre_model=None,
    )
    call_mods_frequency_to_file(args)
    with open(bed) as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
    assert len(rows) == 2, rows
    by_pos = {int(row[1]): row for row in rows}
    assert by_pos[10][4] == "2"
    assert by_pos[10][10] == "50"
    assert by_pos[20][4] == "1"
    assert by_pos[20][10] == "100"
    print("test_call_freq_count_mode ok")


if __name__ == "__main__":
    test_filter_reads()
    test_format_detection()
    test_bed_stats_reason()
    test_call_freq_count_mode()
    print("all tests passed")
