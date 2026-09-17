#! /usr/bin/env python
"""Stream-filter call_mods TSV files by a read-ID allowlist."""

from __future__ import absolute_import

import argparse
import gzip
import os
import sys
import time


def _open_text(path, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def _nth_field(line, n, sep="\t"):
    start = 0
    for _ in range(n):
        idx = line.find(sep, start)
        if idx < 0:
            return None
        start = idx + 1
    end = line.find(sep, start)
    if end < 0:
        return line[start:].rstrip("\r\n")
    return line[start:end]


def load_read_ids(path):
    """Load a set of read IDs from a (gzipped) text file.

    Uses the first whitespace-separated field of each non-empty line.
    A header named read_id / readname / qname is skipped.
    """
    ids = set()
    n_lines = 0
    with _open_text(path) as handle:
        for i, raw in enumerate(handle):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rid = line.split()[0]
            if i == 0 and rid.lower() in {
                "read_id", "readid", "read_name", "readname", "qname", "read",
            }:
                continue
            ids.add(rid)
            n_lines += 1
            if n_lines % 2000000 == 0:
                print("[filter_reads] loaded {} read ids..".format(n_lines), flush=True)
    print("[filter_reads] {} unique read ids from {}".format(len(ids), path), flush=True)
    return ids


def filter_read_calls(input_path, read_ids, output_path, read_col=4, progress_every=2000000):
    """Keep call_mods rows whose read name (0-based column *read_col*) is in *read_ids*."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError("input TSV not found: {}".format(input_path))
    if not read_ids:
        raise ValueError("read-id allowlist is empty")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_in = n_out = n_bad = 0
    start = time.time()
    print("[filter_reads] {} -> {}".format(input_path, output_path), flush=True)
    with _open_text(input_path) as inf, _open_text(output_path, "wt") as outf:
        for line in inf:
            n_in += 1
            rid = _nth_field(line, read_col)
            if rid is None:
                n_bad += 1
                continue
            if rid in read_ids:
                outf.write(line)
                n_out += 1
            if n_in % progress_every == 0:
                elapsed = time.time() - start
                print(
                    "[filter_reads] scanned {} calls, kept {}, {:.1f}s".format(
                        n_in, n_out, elapsed
                    ),
                    flush=True,
                )

    elapsed = time.time() - start
    kept_pct = (100.0 * n_out / n_in) if n_in else 0.0
    print(
        "[filter_reads] done: kept {:.2f}% ({} of {}) calls, {} malformed, {:.1f}s".format(
            kept_pct, n_out, n_in, n_bad, elapsed
        ),
        flush=True,
    )
    return n_in, n_out, n_bad


def filter_read_calls_from_args(args):
    read_ids = load_read_ids(args.read_ids)
    return filter_read_calls(
        args.input_path,
        read_ids,
        args.result_file,
        read_col=args.read_col,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Keep call_mods TSV rows whose read names are in a read-ID list."
    )
    parser.add_argument("--input_path", "-i", required=True, help="call_mods TSV (.tsv or .tsv.gz)")
    parser.add_argument(
        "--read_ids",
        required=True,
        help="allowlist of read IDs, one per line (.txt or .txt.gz)",
    )
    parser.add_argument("--result_file", "-o", required=True, help="filtered TSV output path")
    parser.add_argument(
        "--read_col",
        type=int,
        default=4,
        help="0-based column of the read name in the TSV, default 4",
    )
    args = parser.parse_args(argv)
    filter_read_calls_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
