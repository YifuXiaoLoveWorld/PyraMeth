"""
utils_dataloader.py
Core data processing functions and IO producer for inference.

Active components:
  - build_signal_rect_from_movetable  : fast vectorised signal-to-base mapping
  - get_q2tloc_from_cigar             : CIGAR → query-to-ref position mapping
  - _group_signals_by_movetable_v2    : variable-length signal grouping
  - _get_signals_rect                 : pad/trim signal windows to fixed length
  - process_data_fast                 : feature extraction for modelMTM
  - producer                          : multi-process IO worker (pod5 / slow5)
"""

import json
import os
import time as _time

import numpy as np
from numba import jit

import pod5
import pyslow5

from .utils.process_utils import get_logger
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import compute_proximity_tag
from .utils.process_utils import normalize_signals
from .utils.process_utils import base2code_dna
from .utils import bam_reader

LOGGER = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal ↔ base alignment utilities
# ─────────────────────────────────────────────────────────────────────────────

@jit(nopython=True, cache=True)
def _q2tloc_jit(cigar_ops, cigar_lens, seq_len, forward):
    q_to_r_poss = np.full(seq_len + 1, np.int32(-2), dtype=np.int32)
    curr_r_pos = np.int32(0)
    curr_q_pos = np.int32(0)
    n = len(cigar_ops)
    for ii in range(n):
        i = ii if forward else (n - 1 - ii)
        op     = cigar_ops[i]
        op_len = cigar_lens[i]
        if op == 1:                         # insertion
            for q_pos in range(curr_q_pos, curr_q_pos + op_len):
                q_to_r_poss[q_pos] = np.int32(-1)
            curr_q_pos += op_len
        elif op == 2 or op == 3:            # deletion / skip
            curr_r_pos += op_len
        elif op == 0 or op == 7 or op == 8: # match / seq-match / seq-mismatch
            for off in range(op_len):
                q_to_r_poss[curr_q_pos + off] = curr_r_pos + off
            curr_q_pos += op_len
            curr_r_pos += op_len
    q_to_r_poss[curr_q_pos] = curr_r_pos
    return q_to_r_poss


def get_q2tloc_from_cigar(r_cigar_tuple, strand, seq_len):
    """
    Map query positions to reference positions via CIGAR.
    Returns an array of length seq_len+1.
      -1  → insertion into ref
      -2  → deletion / invalid
    """
    ops  = np.array([op  for op, _  in r_cigar_tuple], dtype=np.int32)
    lens = np.array([ln  for _,  ln in r_cigar_tuple], dtype=np.int32)
    q_to_r_poss = _q2tloc_jit(ops, lens, seq_len, strand == 1)
    if q_to_r_poss[-1] == -2:
        raise ValueError(
            f"Invalid CIGAR: ref_len={seq_len}, cigar did not cover full query"
        )
    return q_to_r_poss


def _group_signals_by_movetable_v2(trimed_signals, movetable, stride):
    """
    Group raw signals per base using the move table (Python-loop version).
    Used by feature extraction to obtain variable-length per-base signals
    for mean / std / len computation.
    """
    if movetable[0] != 1:
        raise ValueError(
            f"move table must start with 1 (a move), got {movetable[0]}"
        )
    if len(trimed_signals) < len(movetable) * stride:
        raise ValueError(
            f"trimmed signal length ({len(trimed_signals)}) is shorter than "
            f"expected ({len(movetable)} moves × stride {stride} = {len(movetable) * stride})"
        )
    move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
    signal_group = []
    for i in range(len(move_pos) - 1):
        s, e = move_pos[i], move_pos[i + 1]
        signal_group.append(trimed_signals[s * stride: e * stride].tolist())
    assert len(signal_group) == int(np.sum(movetable))
    return signal_group



@jit(nopython=True, cache=True)
def _build_signal_rect_jit(sig, starts, ends, signals_len):
    """JIT kernel: fills rect with 0.0 and marks valid positions in a bool mask."""
    N   = len(starts)
    out   = np.zeros((N, signals_len), dtype=np.float32)
    valid = np.ones((N, signals_len),  dtype=np.bool_)
    for i in range(N):
        s = starts[i]
        e = ends[i]
        L = e - s
        if L == 0:
            for j in range(signals_len):
                valid[i, j] = False
            continue
        if L <= signals_len:
            pad_left = (signals_len - L) // 2
            for j in range(pad_left):
                valid[i, j] = False
            for j in range(L):
                out[i, pad_left + j] = sig[s + j]
            for j in range(pad_left + L, signals_len):
                valid[i, j] = False
        else:
            # downsample: evenly-spaced indices across [s, e)
            for j in range(signals_len):
                idx = s + int(j * (L - 1) / (signals_len - 1))
                out[i, j] = sig[idx]
    return out, valid


def build_signal_rect_from_movetable(trimed_signals, movetable, stride, signals_len=16):
    """
    Build (num_events, signals_len) rect array from move table.
    NaN-padded for short events; downsampled for long events.
    Used by process_data_fast (MTM).
    """
    move_idx = np.flatnonzero(movetable == 1)
    move_idx = np.append(move_idx, len(movetable))
    starts = (move_idx[:-1] * stride).astype(np.int64)
    ends   = (move_idx[1:]  * stride).astype(np.int64)

    sig = np.ascontiguousarray(trimed_signals, dtype=np.float32)
    out, valid = _build_signal_rect_jit(sig, starts, ends, signals_len)
    out[~valid] = np.nan   # restore NaN padding for MTM mask (torch.isnan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (per-read, called from producer)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bam_read(signal, seq_read, args, seq=None):
    """
    Shared pre-processing: trim, normalise, build rect signal matrix.
    Returns (seq, norm_signal, signal_rect, movetable, stride) or None on failure.
    """
    if seq is None:
        seq = seq_read.get_forward_sequence()
    if seq is None:
        return None

    if not seq_read.has_tag("mv") or not seq_read.has_tag("ts"):
        return None

    mv     = np.asarray(seq_read.get_tag("mv"), dtype=np.int32)
    stride = int(mv[0])
    movetable = mv[1:]

    num_trimmed = int(seq_read.get_tag("ts"))
    if seq_read.has_tag("sp"):
        num_trimmed += int(seq_read.get_tag("sp"))

    sig_trimmed = signal[num_trimmed:] if num_trimmed >= 0 else signal[:num_trimmed]
    norm_signal  = normalize_signals(sig_trimmed, args.normalize_method)
    signal_rect  = build_signal_rect_from_movetable(norm_signal, movetable, stride, args.signal_len)

    return seq, norm_signal, signal_rect, movetable, stride


def _get_ref_coords(seq_read, seq):
    """
    Compute strand, ref coords, and q→r mapping for a mapped read.
    Returns dict with keys: strand, ref_name, ref_start, ref_end,
    seq_start, seq_end, q_to_r_poss.
    Returns None if read is unmapped.
    """
    if seq_read.is_unmapped:
        return None

    strand     = "-" if seq_read.is_reverse else "+"
    strand_code = -1 if seq_read.is_reverse else 1
    ref_name   = seq_read.reference_name or "."
    ref_start  = seq_read.reference_start
    ref_end    = seq_read.reference_end

    qa_start = seq_read.query_alignment_start
    qa_end   = seq_read.query_alignment_end

    if seq_read.is_reverse:
        seq_start = len(seq) - qa_end
        seq_end   = len(seq) - qa_start
    else:
        seq_start, seq_end = qa_start, qa_end

    q_to_r_poss = get_q2tloc_from_cigar(
        seq_read.cigartuples, strand_code, seq_end - seq_start
    )
    return dict(
        strand=strand, ref_name=ref_name,
        ref_start=ref_start, ref_end=ref_end,
        seq_start=seq_start, seq_end=seq_end,
        q_to_r_poss=q_to_r_poss,
    )


def process_data_fast(signal, seq_read, motif_seqs, positions, args):
    """
    Extract features for modelMTM inference.
    Output per site: (sampleinfo, k_seq[int64], k_signals_rect[float32], label, tag)
    No mean/std/len – fastest path.

    args.plant (bool):
        True  → tag counts any C within ±10 bp (plant / multi-motif mode)
        False → tag counts only same-motif sites within ±10 bp (human / CG mode)
    """
    if seq_read.mapping_quality < args.mapq:
        return []

    if not seq_read.is_unmapped:
        qa_start = seq_read.query_alignment_start
        qa_end   = seq_read.query_alignment_end
        if seq_read.query_length and (qa_end - qa_start) / seq_read.query_length < args.coverage_ratio:
            return []

    ref_name = (seq_read.reference_name or ".") if not seq_read.is_unmapped else "."
    chrom_args = getattr(args, "chrom", None) or []
    _excl = {c[2:] for c in chrom_args if c.startswith("no")}
    _incl = {c for c in chrom_args if not c.startswith("no")}
    if (_excl and ref_name in _excl) or (_incl and ref_name not in _incl):
        return []

    seq = seq_read.get_forward_sequence()
    if seq is None:
        return []

    tsite_locs = get_refloc_of_methysite_in_motif(seq, motif_seqs, args.mod_loc)
    if not tsite_locs:
        return []

    parsed = _parse_bam_read(signal, seq_read, args, seq=seq)
    if parsed is None:
        return []
    seq, _, signal_rect, _, _ = parsed

    num_bases = (args.seq_len - 1) // 2
    coords    = _get_ref_coords(seq_read, seq)

    strand   = coords["strand"]   if coords else "."
    ref_name = coords["ref_name"] if coords else ref_name

    # Pre-compute tag_locs once per read
    plant = getattr(args, "plant", False)
    if plant:
        tag_locs = [i for i, b in enumerate(seq) if b == "C"]
    else:
        tag_locs = tsite_locs  # already sorted

    out = []
    for loc in tsite_locs:
        if not (num_bases <= loc < len(seq) - num_bases):
            continue

        ref_pos = -1
        if coords:
            s, e = coords["seq_start"], coords["seq_end"]
            if not (s <= loc < e):
                continue
            rpos = coords["q_to_r_poss"][loc - s]
            if rpos == -1:
                continue
            ref_pos = (
                coords["ref_end"] - 1 - rpos if strand == "-"
                else coords["ref_start"] + rpos
            )

        if positions is not None:
            if f"{ref_name}\t{ref_pos}\t{strand}" not in positions:
                continue

        tag = compute_proximity_tag(loc, tag_locs, window=10)

        k_mer = seq[loc - num_bases: loc + num_bases + 1]
        k_seq = np.fromiter(
            (base2code_dna[x] for x in k_mer),
            dtype=np.int64, count=args.seq_len,
        )
        k_signals = signal_rect[loc - num_bases: loc + num_bases + 1]
        sampleinfo = f"{ref_name}\t{ref_pos}\t{strand}\t.\t{seq_read.query_name}\t."

        out.append((sampleinfo, k_seq, k_signals, args.methy_label, tag))

    return out


FEATURE_BATCH = "feature_batch"


def pack_feature_batch(items):
    """Pack site tuples into contiguous arrays for cheaper multiprocessing IPC."""
    infos = [it[0] for it in items]
    k_arr = np.stack([np.asarray(it[1], dtype=np.int64) for it in items])
    s_arr = np.stack([np.asarray(it[2], dtype=np.float32) for it in items])
    labels = np.asarray([it[3] for it in items], dtype=np.int64)
    tags = np.asarray([it[4] for it in items], dtype=np.int64)
    return (FEATURE_BATCH, infos, k_arr, s_arr, labels, tags)


def iter_feature_items(item):
    """Yield (sampleinfo, k_seq, k_signals, label, tag) from a queue payload."""
    if isinstance(item, tuple) and item and item[0] == FEATURE_BATCH:
        _, infos, k_arr, s_arr, labels, tags = item
        for i, info in enumerate(infos):
            yield (info, k_arr[i], s_arr[i], int(labels[i]), int(tags[i]))
        return
    items = item if isinstance(item, list) else [item]
    for sub in items:
        yield sub


class FeatureBatchBuffer:
    """Pass full packed batches through; join only partial batches."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.size = 0
        self.infos = []
        self.kmers, self.signals, self.tags = [], [], []

    def finish(self):
        if not self.size:
            return None
        def joined(parts):
            return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        batch = (self.infos, joined(self.kmers), joined(self.signals), joined(self.tags))
        self.size = 0
        self.infos = []
        self.kmers, self.signals, self.tags = [], [], []
        return batch

    def push(self, item):
        if not (isinstance(item, tuple) and item and item[0] == FEATURE_BATCH):
            item = pack_feature_batch(list(iter_feature_items(item)))
        _, infos, kmers, signals, _, tags = item
        offset = 0
        while offset < len(infos):
            remaining = len(infos) - offset
            if self.size == 0 and remaining >= self.batch_size:
                stop = offset + self.batch_size
                yield infos[offset:stop], kmers[offset:stop], signals[offset:stop], tags[offset:stop]
            else:
                stop = offset + min(self.batch_size - self.size, remaining)
                self.infos.extend(infos[offset:stop])
                self.kmers.append(kmers[offset:stop])
                self.signals.append(signals[offset:stop])
                self.tags.append(tags[offset:stop])
                self.size += stop - offset
                if self.size == self.batch_size:
                    yield self.finish()
            offset = stop


def _profile_enabled():
    return os.environ.get("PYRAMETH_PROFILE", os.environ.get("DEEPSIGNAL_PROFILE", "")) == "1"


def _append_profile(role, payload):
    profile_dir = os.environ.get("PYRAMETH_PROFILE_DIR")
    if not profile_dir:
        return
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, "%s_%s.jsonl" % (role, os.getpid()))
    payload = dict(payload)
    payload.setdefault("pid", os.getpid())
    payload.setdefault("role", role)
    payload.setdefault("time", _time.time())
    with open(path, "a") as fh:
        fh.write(json.dumps(payload) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# IO Producer  (one process per worker_id shard)
# ─────────────────────────────────────────────────────────────────────────────

def producer(worker_id, files, queues, args, motif_seqs, positions,
             file_type, num_workers, nproc_io):
    """
    Read signal files (pod5 / slow5), extract features, distribute to model workers.

    Each worker handles files[worker_id::nproc_io] (round-robin sharding).
    Items are buffered and sent in batches of BUF_SIZE to reduce IPC overhead.

    Supports:
      - pod5  : pod5.Reader
      - slow5 : pyslow5.Open (includes .blow5)
    """
    my_files = files[worker_id::nproc_io]
    print(f"[Producer-{worker_id}] {len(my_files)} {file_type} files", flush=True)

    bam_index = bam_reader.ReadIndexedBam(
        args.bam, fallback_dir=getattr(args, "bam_index_dir", None)
    )

    process_fn = process_data_fast

    BUF_SIZE = max(1, int(getattr(args, "batch_size", 128) or 128))
    read_batch = max(1, int(os.environ.get("PYRAMETH_READ_BATCH", "64")))
    buffers  = [[] for _ in range(num_workers)]
    rr = worker_id % max(1, num_workers)

    _profile = _profile_enabled()
    _t_proc = _t_bam = _t_put = 0.0
    _n_reads = _n_sites = 0

    def _flush_buffer(qid):
        nonlocal _t_put
        if not buffers[qid]:
            return
        payload = pack_feature_batch(buffers[qid])
        t0 = _time.perf_counter() if _profile else None
        queues[qid].put(payload)
        if _profile:
            _t_put += _time.perf_counter() - t0
        buffers[qid] = []

    def _emit_features(feats):
        nonlocal rr, _n_sites
        for feat in feats:
            qid = rr % num_workers
            rr += 1
            buffers[qid].append(feat)
            _n_sites += 1
            if len(buffers[qid]) >= BUF_SIZE:
                _flush_buffer(qid)

    def _handle_read(signal, read_name, ptrs=None):
        nonlocal _t_proc, _t_bam, _t_put, _n_reads
        try:
            t0 = _time.perf_counter() if _profile else None
            if ptrs is None:
                ptrs = bam_index.get_offsets(read_name)
            aligns = list(bam_index.iter_alignments_at(ptrs))
            if _profile:
                _t_bam += _time.perf_counter() - t0
            for seq_read in aligns:
                t1 = _time.perf_counter() if _profile else None
                feats = process_fn(signal, seq_read, motif_seqs, positions, args)
                if _profile:
                    _t_proc += _time.perf_counter() - t1
                _emit_features(feats)
            if _profile:
                _n_reads += 1
                if _n_reads % 500 == 0:
                    rec = {
                        "reads": _n_reads,
                        "sites": _n_sites,
                        "bam_ms_per_read": _t_bam * 1e3 / 500,
                        "process_ms_per_read": _t_proc * 1e3 / 500,
                        "queue_put_ms_per_500": _t_put * 1e3,
                    }
                    print(
                        f"[Producer-{worker_id}] {_n_reads} reads | "
                        f"bam_lookup {rec['bam_ms_per_read']:.2f} ms/r | "
                        f"process_fn {rec['process_ms_per_read']:.2f} ms/r | "
                        f"queue_put {rec['queue_put_ms_per_500']:.1f} ms/500r",
                        flush=True,
                    )
                    _append_profile("producer", rec)
                    _t_proc = _t_bam = _t_put = 0.0
        except KeyError:
            pass  # read not in BAM – skip silently

    def _handle_read_batch(pending):
        keyed = []
        for signal, read_name in pending:
            try:
                ptrs = bam_index.get_offsets(read_name)
            except KeyError:
                continue
            first = int(ptrs[0]) if len(ptrs) else 0
            keyed.append((first, signal, read_name, ptrs))
        keyed.sort(key=lambda row: row[0])
        for _, signal, read_name, ptrs in keyed:
            _handle_read(signal, read_name, ptrs=ptrs)

    for file in my_files:
        pending = []
        try:
            if file_type == "pod5":
                with pod5.Reader(file) as reader:
                    for read in reader.reads():
                        pending.append((read.signal, str(read.read_id)))
                        if len(pending) >= read_batch:
                            _handle_read_batch(pending)
                            pending = []
                    if pending:
                        _handle_read_batch(pending)

            elif file_type in ("slow5", "blow5"):
                s5 = pyslow5.Open(file, "r")
                try:
                    for read in s5.seq_reads():
                        pending.append((read["signal"], read["read_id"]))
                        if len(pending) >= read_batch:
                            _handle_read_batch(pending)
                            pending = []
                    if pending:
                        _handle_read_batch(pending)
                finally:
                    s5.close()

            elif file_type == "fast5":
                from .utils import fast5_reader
                is_single = getattr(args, "single", False)
                if is_single:
                    f5 = fast5_reader.SingleFast5(file, is_single=True)
                    try:
                        sig = f5.rescale_signals(f5.get_raw_signal())
                        _handle_read(sig, f5.get_readid())
                    finally:
                        f5.close()
                else:
                    mf = fast5_reader.MultiFast5(file)
                    try:
                        for rname in mf:
                            f5 = fast5_reader.SingleFast5(mf[rname], readname=rname)
                            sig = f5.rescale_signals(f5.get_raw_signal())
                            pending.append((sig, f5.get_readid()))
                            if len(pending) >= read_batch:
                                _handle_read_batch(pending)
                                pending = []
                        if pending:
                            _handle_read_batch(pending)
                    finally:
                        mf.close()

        except Exception as e:
            print(f"[Producer-{worker_id}] error on {file}: {e}", flush=True)

    for qid in range(num_workers):
        _flush_buffer(qid)

    if _profile:
        _append_profile("producer", {"event": "done", "reads": _n_reads, "sites": _n_sites})
    print(f"[Producer-{worker_id}] done", flush=True)
