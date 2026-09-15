"""On-disk store for prefix-cache entries (``--prefix-disk-cache``).

One entry is everything a request needs to resume a prompt at one boundary ``L`` without
recomputing ``[0, L)``: the token ids, the KV pages that hold ``[0, L)``, and (hybrid GDN
models) the recurrent-state snapshot at ``L``. This module knows nothing about pools, trees or
the GPU -- it stores named CPU tensors under a key and gives them back only when they are
provably the ones that were written, by the same configuration.

What "provably" means here:

* **Fingerprint.** Every configuration that changes what the stored bytes mean (model and
  weights, dtypes, ``--kv-cache-dtype``, page size, ``--dense-quant``, the rank split, the pool
  layout, the code version) is folded into one digest. Entries live in a directory named after
  it, and every file header repeats the full digest; a header that disagrees is never read.
* **Key.** ``sha256(fingerprint digest || int32 token ids of [0, L))``. The ids are stored in
  the file too and compared on load, so a hash collision cannot hand back another prompt.
* **Integrity.** A CRC32 per tensor. A file is only ever visible under its final name after a
  complete write (tmp + fsync + rename), so a crash mid-write leaves a ``.tmp`` that the next
  start removes, never a short entry.

Capacity is one byte budget over the whole cache root (entries of other fingerprints count
too, so switching models cannot grow the directory past it); the oldest file by mtime goes
first, and a hit bumps the mtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

MAGIC = b"FTPFXC01"
FORMAT_VERSION = 1
SUFFIX = ".ftpc"
_HEADER_LEN = struct.Struct("<I")
_MAX_HEADER = 1 << 20
_IO_CHUNK = 64 << 20

# dtypes an entry may hold; anything else in a header is a reason not to read it
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}


def dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"prefix disk cache cannot store dtype {dtype}")
    return name


def fingerprint_digest(fingerprint: dict) -> str:
    blob = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _ids_bytes(ids: torch.Tensor) -> bytes:
    return ids.detach().to("cpu", torch.int32).contiguous().numpy().astype("<i4", copy=False).tobytes()


def prefix_keys(digest: str, ids: torch.Tensor, lengths: Iterable[int]) -> Dict[int, str]:
    """Keys of ``ids[:L]`` for every ``L`` in ``lengths``, in one pass over the ids (each key
    is the running hash, copied at its length)."""
    raw = _ids_bytes(ids)
    h = hashlib.sha256(bytes.fromhex(digest))
    out: Dict[int, str] = {}
    pos = 0
    for length in sorted(set(lengths)):
        if length < 0 or length * 4 > len(raw):
            raise ValueError(f"prefix length {length} outside the {len(raw) // 4} ids given")
        h.update(raw[pos * 4 : length * 4])
        pos = length
        out[length] = h.copy().hexdigest()
    return out


def prefix_key(digest: str, ids: torch.Tensor) -> str:
    return prefix_keys(digest, ids, [len(ids)])[len(ids)]


@dataclass
class DiskEntry:
    key: str
    length: int
    nbytes: int
    path: str
    mtime: float


class PrefixDiskStore:
    """The entry files of one fingerprint, plus the capacity account of the whole root.

    Thread-safe: the scheduler thread looks up, one writer thread saves, one reader thread
    loads. Only index updates take the lock; file I/O runs outside it (an entry that is
    evicted while a reader has it open is still read whole on POSIX)."""

    def __init__(
        self,
        root: str,
        capacity_bytes: int,
        fingerprint: dict,
        *,
        log: Callable[[str], None] | None = None,
        drop_page_cache: bool = True,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError(f"prefix disk cache capacity must be positive, got {capacity_bytes}")
        self.root = os.path.abspath(os.path.expanduser(root))
        self.capacity = int(capacity_bytes)
        self.fingerprint = fingerprint
        self.digest = fingerprint_digest(fingerprint)
        self.dir = os.path.join(self.root, self.digest[:16])
        self._log = log or (lambda msg: None)
        # Written files are dropped from the page cache (posix_fadvise DONTNEED) after the
        # fsync: on a --moe-bank-ram machine the page cache IS the expert bank's cold half, and
        # a gigabyte of entry written through it pushes a gigabyte of experts out.
        self._drop_page_cache = drop_page_cache
        self._lock = threading.Lock()
        self._entries: Dict[str, DiskEntry] = {}
        self._lengths: Dict[int, int] = {}  # length -> number of entries with it
        self._foreign: Dict[str, Tuple[float, int]] = {}  # other fingerprints' files: mtime, size
        self.stats = {"removed_corrupt": 0, "removed_tmp": 0, "evicted": 0}
        os.makedirs(self.dir, exist_ok=True)
        self._write_fingerprint_file()
        self._scan()
        with self._lock:
            self._evict_to_fit_locked(0)

    # ------------------------------------------------------------------ index
    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_locked()

    def _total_locked(self) -> int:
        return sum(e.nbytes for e in self._entries.values()) + sum(
            size for _, size in self._foreign.values()
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def entries(self) -> List[DiskEntry]:
        with self._lock:
            return list(self._entries.values())

    def _add_locked(self, entry: DiskEntry) -> None:
        old = self._entries.get(entry.key)
        if old is not None:
            self._drop_locked(old)
        self._entries[entry.key] = entry
        self._lengths[entry.length] = self._lengths.get(entry.length, 0) + 1

    def _drop_locked(self, entry: DiskEntry) -> None:
        if self._entries.get(entry.key) is not entry:
            return
        del self._entries[entry.key]
        n = self._lengths.get(entry.length, 0) - 1
        if n <= 0:
            self._lengths.pop(entry.length, None)
        else:
            self._lengths[entry.length] = n

    def lookup(self, ids: torch.Tensor, max_len: int | None = None) -> Optional[DiskEntry]:
        """The deepest entry whose key is a prefix of ``ids`` (and no longer than ``max_len``).
        A key match is not yet a guarantee -- ``load`` compares the stored ids."""
        limit = len(ids) if max_len is None else min(max_len, len(ids))
        with self._lock:
            lengths = [n for n in self._lengths if 0 < n <= limit]
        if not lengths:
            return None
        keys = prefix_keys(self.digest, ids, lengths)
        with self._lock:
            for n in sorted(lengths, reverse=True):
                entry = self._entries.get(keys[n])
                if entry is not None:
                    return entry
        return None

    def touch(self, entry: DiskEntry) -> None:
        now = time.time()
        try:
            os.utime(entry.path, (now, now))
        except OSError:
            return
        with self._lock:
            entry.mtime = now

    def remove(self, entry: DiskEntry) -> None:
        with self._lock:
            self._drop_locked(entry)
        _unlink_quiet(entry.path)

    # ------------------------------------------------------------------ capacity
    def _evict_to_fit_locked(self, incoming: int) -> None:
        while self._total_locked() + incoming > self.capacity:
            own = min(self._entries.values(), key=lambda e: e.mtime, default=None)
            foreign = min(self._foreign.items(), key=lambda kv: kv[1][0], default=None)
            if own is None and foreign is None:
                return
            if foreign is not None and (own is None or foreign[1][0] <= own.mtime):
                path = foreign[0]
                del self._foreign[path]
            else:
                path = own.path
                self._drop_locked(own)
            _unlink_quiet(path)
            self.stats["evicted"] += 1

    # ------------------------------------------------------------------ save
    def save(
        self, key: str, length: int, tensors: Sequence[Tuple[str, torch.Tensor]]
    ) -> Optional[DiskEntry]:
        """Write one entry atomically. None when it cannot fit the capacity at all."""
        specs, blobs, offset = [], [], 0
        for name, t in tensors:
            t = t.detach().to("cpu").contiguous()
            flat = t.reshape(-1).view(torch.uint8) if t.numel() else torch.empty(0, dtype=torch.uint8)
            mv = memoryview(flat.numpy())
            specs.append({
                "name": name, "dtype": dtype_name(t.dtype), "shape": list(t.shape),
                "offset": offset, "nbytes": len(mv), "crc32": zlib.crc32(mv),
            })
            blobs.append(mv)
            offset += len(mv)
        header = json.dumps({
            "version": FORMAT_VERSION, "fingerprint": self.digest, "key": key,
            "length": int(length), "tensors": specs, "created": time.time(),
        }, separators=(",", ":")).encode()
        nbytes = len(MAGIC) + _HEADER_LEN.size + len(header) + offset
        if nbytes > self.capacity:
            return None
        with self._lock:
            self._evict_to_fit_locked(nbytes)
        final = os.path.join(self.dir, key + SUFFIX)
        tmp = os.path.join(self.dir, f".{key}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(MAGIC)
                f.write(_HEADER_LEN.pack(len(header)))
                f.write(header)
                for mv in blobs:
                    for i in range(0, len(mv), _IO_CHUNK):
                        f.write(mv[i : i + _IO_CHUNK])
                f.flush()
                os.fsync(f.fileno())
                self._advise_dontneed(f.fileno())
            os.replace(tmp, final)
            _fsync_dir(self.dir)
        except BaseException:
            _unlink_quiet(tmp)
            raise
        entry = DiskEntry(key, int(length), nbytes, final, os.stat(final).st_mtime)
        with self._lock:
            self._add_locked(entry)
            self._foreign.pop(final, None)
        return entry

    # ------------------------------------------------------------------ load
    def load(
        self, entry: DiskEntry, expect_ids: torch.Tensor | None = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """The entry's tensors, or None if the file is gone or is not exactly what was written
        (a bad file is removed). ``expect_ids``: the prompt prefix it must hold."""
        try:
            with open(entry.path, "rb") as f:
                header = self._read_header(f)
                if header is None or header["key"] != entry.key or header["length"] != entry.length:
                    raise _Corrupt("header does not match the index")
                base = f.tell()
                out: Dict[str, torch.Tensor] = {}
                for spec in header["tensors"]:
                    dtype = _DTYPES.get(spec["dtype"])
                    if dtype is None:
                        raise _Corrupt(f"unknown dtype {spec['dtype']!r}")
                    n = int(spec["nbytes"])
                    buf = torch.empty(n, dtype=torch.uint8)
                    f.seek(base + int(spec["offset"]))
                    if n:
                        mv = memoryview(buf.numpy())
                        got = 0
                        while got < n:
                            r = f.readinto(mv[got : min(n, got + _IO_CHUNK)])
                            if not r:
                                raise _Corrupt(f"{spec['name']}: short read ({got} of {n} bytes)")
                            got += r
                        if zlib.crc32(mv) != spec["crc32"]:
                            raise _Corrupt(f"{spec['name']}: checksum mismatch")
                    shape = tuple(int(s) for s in spec["shape"])
                    out[spec["name"]] = buf.view(dtype).reshape(shape)
                self._advise_dontneed(f.fileno())
            ids = out.get("ids")
            if ids is None or ids.numel() != entry.length:
                raise _Corrupt("token ids missing or of the wrong length")
            if expect_ids is not None:
                want = expect_ids.detach().to("cpu", torch.int32)
                if want.numel() != entry.length or not torch.equal(ids.to(torch.int32), want):
                    # a key collision, not a corrupt file: leave it for the prompt it belongs to
                    return None
            return out
        except FileNotFoundError:
            with self._lock:
                self._drop_locked(entry)
            return None
        except (_Corrupt, OSError, ValueError, KeyError, RuntimeError) as exc:
            if os.path.dirname(entry.path) != self.dir:
                return None  # not ours to delete (another configuration's directory)
            self._log(f"prefix disk cache: dropping unreadable entry {os.path.basename(entry.path)}: {exc}")
            self.stats["removed_corrupt"] += 1
            self.remove(entry)
            return None

    def _read_header(self, f) -> Optional[dict]:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise _Corrupt("bad magic")
        raw = f.read(_HEADER_LEN.size)
        if len(raw) != _HEADER_LEN.size:
            raise _Corrupt("truncated header")
        (n,) = _HEADER_LEN.unpack(raw)
        if n > _MAX_HEADER:
            raise _Corrupt("header too large")
        blob = f.read(n)
        if len(blob) != n:
            raise _Corrupt("truncated header")
        header = json.loads(blob)
        if header.get("version") != FORMAT_VERSION:
            raise _Corrupt(f"format version {header.get('version')} (expected {FORMAT_VERSION})")
        if header.get("fingerprint") != self.digest:
            raise _Corrupt("written by a different configuration")
        return header

    # ------------------------------------------------------------------ startup
    def _write_fingerprint_file(self) -> None:
        path = os.path.join(self.dir, "fingerprint.json")
        blob = json.dumps(self.fingerprint, sort_keys=True, indent=1).encode()
        try:
            with open(path, "rb") as f:
                if f.read() == blob:
                    return
        except OSError:
            pass
        tmp = os.path.join(self.dir, f".fingerprint.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)

    def _scan(self) -> None:
        try:
            subdirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        except OSError:
            subdirs = []
        for sub in subdirs:
            sub_path = os.path.join(self.root, sub)
            own = sub_path == self.dir
            try:
                names = os.listdir(sub_path)
            except OSError:
                continue
            for name in names:
                path = os.path.join(sub_path, name)
                if name.endswith(".tmp"):
                    if _tmp_owner_gone(name):
                        _unlink_quiet(path)
                        self.stats["removed_tmp"] += 1
                    continue
                if not name.endswith(SUFFIX):
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not own:
                    self._foreign[path] = (st.st_mtime, st.st_size)
                    continue
                try:
                    with open(path, "rb") as f:
                        header = self._read_header(f)
                        payload = sum(int(t["nbytes"]) for t in header["tensors"])
                        expected = f.tell() + payload
                    key = header["key"]
                    if name != key + SUFFIX or st.st_size != expected:
                        raise _Corrupt("size or name does not match the header")
                except (_Corrupt, OSError, ValueError, KeyError, TypeError) as exc:
                    self._log(f"prefix disk cache: removing {name}: {exc}")
                    self.stats["removed_corrupt"] += 1
                    _unlink_quiet(path)
                    continue
                self._add_locked(DiskEntry(key, int(header["length"]), st.st_size, path, st.st_mtime))

    def _advise_dontneed(self, fd: int) -> None:
        if self._drop_page_cache and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass


class _Corrupt(Exception):
    pass


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tmp_owner_gone(name: str) -> bool:
    """``.<key>.<pid>.tmp``: a write in progress belongs to a live process; anything whose pid
    is gone (or that does not parse) is the remains of a crash."""
    parts = name.split(".")
    try:
        pid = int(parts[-2])
    except (IndexError, ValueError):
        return True
    if pid == os.getpid():
        return True  # this process has not started writing yet: a leftover of a previous life
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


__all__ = [
    "DiskEntry",
    "PrefixDiskStore",
    "dtype_name",
    "fingerprint_digest",
    "prefix_key",
    "prefix_keys",
]
