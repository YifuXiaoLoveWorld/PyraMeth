import json
import hashlib
import os
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pysam

from .process_utils import get_logger

LOGGER = get_logger(__name__)

_INDEX_VERSION = 2


def get_parent_id(bam_read):
    try:
        # if pi tag is present this is a child read
        return bam_read.get_tag("pi")
    except KeyError:
        # else this is the parent read so return query_name
        return bam_read.query_name


def bam_fingerprint(bam_path):
    st = os.stat(bam_path)
    return int(st.st_size), int(st.st_mtime_ns)


def default_index_cache_path(bam_path, fallback_dir=None):
    env = os.environ.get("PYRAMETH_BAM_INDEX")
    if env:
        return env
    sidecar = bam_path + ".pyrameth.ridx.npz"
    parent = os.path.dirname(os.path.abspath(bam_path)) or "."
    if os.access(parent, os.W_OK):
        return sidecar
    if fallback_dir:
        os.makedirs(fallback_dir, exist_ok=True)
        key = hashlib.sha256(os.path.realpath(bam_path).encode()).hexdigest()[:16]
        return os.path.join(fallback_dir, os.path.basename(bam_path) + "." + key + ".pyrameth.ridx.npz")
    return sidecar


@contextmanager
def _exclusive_lock(lock_path):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        except ImportError:
            pass
        os.close(fd)


def _encode_ids(read_ids):
    width = max((len(rid.encode("ascii")) for rid in read_ids), default=36)
    width = max(width, 36)
    return np.array(read_ids, dtype="S%d" % width)


def _index_arrays_from_dict(idx_dict):
    read_ids = list(idx_dict.keys())
    ids = _encode_ids(read_ids)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    starts = np.empty(len(order), dtype=np.int64)
    counts = np.empty(len(order), dtype=np.int32)
    ptr_chunks = []
    cursor = 0
    for i, src in enumerate(order):
        ptrs = np.asarray(idx_dict[read_ids[src]], dtype=np.int64)
        starts[i] = cursor
        counts[i] = ptrs.size
        ptr_chunks.append(ptrs)
        cursor += ptrs.size
    ptrs = np.concatenate(ptr_chunks) if ptr_chunks else np.zeros(0, dtype=np.int64)
    return ids, starts, counts, ptrs


def _save_index_npz(path, ids, starts, counts, ptrs, bam_size, bam_mtime, num_records):
    # Write via a file object. np.savez(filename) appends ".npz" when the
    # temporary name does not already end with that suffix, which made the
    # subsequent os.replace look for a file that was never created.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            ids=ids,
            starts=starts,
            counts=counts,
            ptrs=ptrs,
            bam_size=np.asarray([bam_size], dtype=np.int64),
            bam_mtime=np.asarray([bam_mtime], dtype=np.int64),
            num_records=np.asarray([num_records], dtype=np.int64),
            version=np.asarray([_INDEX_VERSION], dtype=np.int32),
        )
    os.replace(tmp, path)
    meta_path = path + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump(
            {
                "version": _INDEX_VERSION,
                "bam_size": bam_size,
                "bam_mtime": bam_mtime,
                "num_reads": int(ids.shape[0]),
                "num_records": int(num_records),
            },
            fh,
        )


def _load_index_npz(path, bam_size, bam_mtime):
    with np.load(path, allow_pickle=False) as data:
        version = int(data["version"][0])
        if version not in (1, _INDEX_VERSION):
            return None
        # Legacy NPZ caches stored seconds. Reuse them once during migration;
        # all new manifests store nanoseconds and the canonical BAM path.
        expected_mtime = bam_mtime // 1_000_000_000 if version == 1 else bam_mtime
        if int(data["bam_size"][0]) != bam_size or int(data["bam_mtime"][0]) != expected_mtime:
            return None
        return (
            data["ids"].copy(),
            data["starts"].copy(),
            data["counts"].copy(),
            data["ptrs"].copy(),
            int(data["num_records"][0]),
        )


_ARRAY_NAMES = ("ids", "starts", "counts", "ptrs")


def _load_mmap_index(path, bam_path, bam_size, bam_mtime):
    with open(path + ".mmap.json") as fh:
        meta = json.load(fh)
    if (meta["version"] != _INDEX_VERSION or meta["bam_path"] != os.path.realpath(bam_path)
            or meta["bam_size"] != bam_size or meta["bam_mtime_ns"] != bam_mtime):
        return None
    directory = os.path.join(os.path.dirname(os.path.abspath(path)), meta["directory"])
    arrays = tuple(np.load(os.path.join(directory, name + ".npy"), mmap_mode="r", allow_pickle=False)
                   for name in _ARRAY_NAMES)
    return (*arrays, meta["num_records"])


def _save_mmap_index(path, bam_path, arrays, bam_size, bam_mtime, num_records):
    parent = os.path.dirname(os.path.abspath(path))
    directory = tempfile.mkdtemp(prefix=os.path.basename(path) + ".arrays-", dir=parent)
    for name, arr in zip(_ARRAY_NAMES, arrays):
        np.save(os.path.join(directory, name + ".npy"), arr, allow_pickle=False)
    meta = dict(version=_INDEX_VERSION, bam_path=os.path.realpath(bam_path),
                bam_size=bam_size, bam_mtime_ns=bam_mtime, num_records=int(num_records),
                directory=os.path.basename(directory))
    # Publish only after all arrays exist. Existing readers keep their mapping
    # of the previous generation, so never overwrite its array files in place.
    temporary = os.path.join(directory, "manifest.json")
    with open(temporary, "w") as fh:
        json.dump(meta, fh)
    os.replace(temporary, path + ".mmap.json")


def ensure_read_index(bam_path, cache_path=None, fallback_dir=None):
    """Build or reuse the read-id index in the parent process before workers spawn."""
    index = ReadIndexedBam(bam_path, cache_path=cache_path, fallback_dir=fallback_dir)
    return index._cache_path + ".mmap.json"


@dataclass
class ReadIndexedBam:
    bam_path: str
    cache_path: str = None
    fallback_dir: str = None

    @property
    def filename(self):
        """Alias to mimic AlignmentFile attribute"""
        return self.bam_path

    def __post_init__(self):
        self.num_reads = None
        self.num_records = 0
        self.bam_fh = None
        self._iter = None
        self._ids = None
        self._starts = None
        self._counts = None
        self._ptrs = None
        self._idx_dict = None
        self._cache_path = self.cache_path or default_index_cache_path(
            self.bam_path, fallback_dir=self.fallback_dir
        )
        self._load_or_build_index()

    def has_index(self):
        """Alias to mimic AlignmentFile attribute"""
        if self.bam_fh is None:
            self.open()
        return self.bam_fh.has_index()

    def open(self):
        # hide warnings for no index when using unmapped or unsorted files
        self.pysam_save = pysam.set_verbosity(0)
        self.bam_fh = pysam.AlignmentFile(self.bam_path, mode="rb", check_sq=False)
        return self

    def close(self):
        self.bam_fh.close()
        self.bam_fh = None
        pysam.set_verbosity(self.pysam_save)

    def _load_or_build_index(self):
        bam_size, bam_mtime = bam_fingerprint(self.bam_path)

        def load_mapped():
            try:
                return _load_mmap_index(self._cache_path, self.bam_path, bam_size, bam_mtime)
            except (OSError, ValueError, KeyError):
                return None

        loaded = load_mapped()
        if loaded is not None:
            self._ids, self._starts, self._counts, self._ptrs, self.num_records = loaded
            self.num_reads = len(self._ids)
            return
        lock_path = self._cache_path + ".lock"
        os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)) or ".", exist_ok=True)
        with _exclusive_lock(lock_path):
            loaded = load_mapped()
            if loaded is None:
                legacy = None
                # Once a manifest exists, an invalid fingerprint means rebuild;
                # never resurrect an older NPZ after a BAM change.
                if not os.path.exists(self._cache_path + ".mmap.json") and os.path.isfile(self._cache_path):
                    try:
                        legacy = _load_index_npz(self._cache_path, bam_size, bam_mtime)
                    except (OSError, ValueError, KeyError):
                        pass
                if legacy is not None:
                    LOGGER.info("Migrating existing BAM index to shared read-only arrays")
                    arrays, self.num_records = legacy[:4], legacy[4]
                else:
                    started = time.perf_counter()
                    LOGGER.info("Building BAM read-id index from %s", self.bam_path)
                    self.compute_read_index()
                    arrays = _index_arrays_from_dict(self._idx_dict)
                    self._idx_dict = None
                    LOGGER.info("Built BAM index in %.1fs", time.perf_counter() - started)
                _save_mmap_index(self._cache_path, self.bam_path, arrays, bam_size, bam_mtime, self.num_records)
                del arrays
                legacy = None
                loaded = load_mapped()
            self._ids, self._starts, self._counts, self._ptrs, self.num_records = loaded
            self.num_reads = len(self._ids)

    def compute_read_index(self):
        bam_was_closed = self.bam_fh is None
        if bam_was_closed:
            self.open()
        self._idx_dict = defaultdict(list)
        self.num_records = 0
        while True:
            read_ptr = self.bam_fh.tell()
            try:
                read = next(self.bam_fh)
            except StopIteration:
                break
            index_read_id = get_parent_id(read)
            if read.is_supplementary or read.is_secondary:
                continue
            self.num_records += 1
            self._idx_dict[index_read_id].append(read_ptr)
        if bam_was_closed:
            self.close()
        self._idx_dict = dict(self._idx_dict)
        self.num_reads = len(self._idx_dict)

    def _id_key(self, read_id):
        return np.bytes_(read_id)

    def _search_id(self, read_id):
        if self._ids is None:
            raise RuntimeError("BAM index not initialized")
        key = self._id_key(read_id)
        idx = int(np.searchsorted(self._ids, key))
        if idx >= self._ids.shape[0] or self._ids[idx] != key:
            raise KeyError(read_id)
        return idx

    def get_offsets(self, read_id):
        """Return BAM file offsets for *read_id*. Raises KeyError if missing."""
        if self._idx_dict is not None:
            return self._idx_dict[read_id]
        idx = self._search_id(read_id)
        start = int(self._starts[idx])
        count = int(self._counts[idx])
        return self._ptrs[start:start + count]

    def iter_alignments_at(self, ptrs):
        if self.bam_fh is None:
            self.open()
        for read_ptr in ptrs:
            self.bam_fh.seek(int(read_ptr))
            try:
                bam_read = next(self.bam_fh)
            except OSError as e:
                LOGGER.warning("Failed to read BAM record at offset %d: %s", int(read_ptr), e)
                continue
            yield bam_read

    def get_alignments(self, read_id):  # 多重序列比对，一条read可能map到多个位置
        return self.iter_alignments_at(self.get_offsets(read_id))

    def get_first_alignment(self, read_id):
        return next(self.get_alignments(read_id))

    def __contains__(self, read_id):
        try:
            self.get_offsets(read_id)
            return True
        except KeyError:
            return False

    def __getitem__(self, read_id):
        ptrs = self.get_offsets(read_id)
        return ptrs.tolist() if hasattr(ptrs, "tolist") else list(ptrs)

    def __del__(self):
        if self.bam_fh is not None:
            self.bam_fh.close()

    @cached_property
    def _bam_idx(self):
        if self._idx_dict is not None:
            return self._idx_dict
        decoded = [rid.split(b"\0", 1)[0].decode("ascii") for rid in self._ids]
        return {
            rid: self._ptrs[int(self._starts[i]):int(self._starts[i]) + int(self._counts[i])].tolist()
            for i, rid in enumerate(decoded)
        }

    @cached_property
    def read_ids(self):
        if self._idx_dict is not None:
            return list(self._idx_dict.keys())
        return [rid.split(b"\0", 1)[0].decode("ascii") for rid in self._ids]

    def __iter__(self):
        if self.bam_fh is None:
            self.open()
        self.bam_fh.reset()
        self._iter = iter(self.bam_fh)
        return self._iter

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self.bam_fh)
        return next(self._iter)


def get_read_ids(bam_idx, pod5_dr, num_reads=None, return_num_bam_reads=False):
    """Get overlapping read ids from bam index and pod5 file

    Args:
        bam_idx (ReadIndexedBam): Read indexed BAM
        pod5_dr (pod5.DatasetReader): POD5 Dataset Reader
        num_reads (int): Maximum number of reads, or None for no max
        return_num_child_reads (bool): Return the number of bam records (child
            reads and multiple mappings) with a parent read ID. When set to
            False the number of parent read IDs is returned.
    """
    if isinstance(pod5_dr, str):
        both_read_ids = list(bam_idx.read_ids)
    else:
        pod5_read_ids = set(pod5_dr.read_ids)
        both_read_ids = list(pod5_read_ids.intersection(bam_idx.read_ids))
    num_both_read_ids = sum(
        len(bam_idx.get_offsets(parent_read_id)) for parent_read_id in both_read_ids
    )
    print(
        f"Found {bam_idx.num_records:,} valid BAM records. Found signal "
        f"in POD5 for {num_both_read_ids / bam_idx.num_records:.2%} of BAM "
        "records."
    )
    if not return_num_bam_reads:
        num_both_read_ids = len(both_read_ids)
    if num_reads is None:
        num_reads = num_both_read_ids
    else:
        num_reads = min(num_reads, num_both_read_ids)
    return both_read_ids, num_reads


class Read:
    def __init__(self, pod5_record, bam_record, read_id):  # pysam.AlignedSegment
        self._readid = read_id
        self._tag = dict(bam_record.tags)
        self._signal = np.asarray(pod5_record.signal)
        self._seq = bam_record.query_sequence
        self._num_trimmed = self._tag["ts"]
        self._norm_shift = self._tag["sm"]
        self._norm_scale = self._tag["sd"]

        self._ref_name = bam_record.reference_name
        self._strand = "-" if bam_record.is_reverse else "+"
        self._read_start = bam_record.query_alignment_start
        self._read_end = bam_record.query_alignment_end
        self._ref_start = bam_record.reference_start
        self._ref_end = bam_record.reference_end
        # self._ref_poses=bam_record.get_reference_positions()
        # self._read_poses=bam_record.positions

    def get_readid(self):
        return self._readid

    def get_raw_signal(self):
        return self._signal

    def get_seq(self):
        return self._seq.strip()

    def get_move(self):
        return np.asarray(self._tag["mv"][1:])

    def get_stride(self):
        return int(self._tag["mv"][0])

    def rescale_signals(self):
        num_trimmed = self._num_trimmed
        signal = self._signal
        if num_trimmed >= 0:
            self._signal = (signal[num_trimmed:] - self._norm_shift) / self._norm_scale
        else:
            self._signal = (signal[:num_trimmed] - self._norm_shift) / self._norm_scale
        return self._signal

    def check_signal(self):
        assert self._signal is not None

    def check_seq(self):
        assert self._seq is not None

    def check_map(self, bam_record):
        assert bam_record.is_unmapped is False

    def get_map_info(self, bam_record):
        cigars = bam_record.cigarstring
        chrom_strands = (self._ref_name, self._strand)
        frags = (
            self._read_start,
            self._read_end,
            self._ref_start,
            self._ref_end,
        )  # add tuple will occur error
        mapinfo = []
        mapinfo.append((cigars, chrom_strands, frags))
        return mapinfo
