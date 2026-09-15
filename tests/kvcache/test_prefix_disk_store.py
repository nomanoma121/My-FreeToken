"""The on-disk store behind --prefix-disk-cache (kvcache/prefix_disk_store.py). CPU only.

What must never happen is an entry being read back that is not exactly what this configuration
wrote for exactly these tokens. The tests pin each wall that stands in the way of that --
fingerprint, key, stored ids, checksum, atomic write -- and the capacity account.
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.kvcache.prefix_disk_store import (
    SUFFIX,
    PrefixDiskStore,
    fingerprint_digest,
    prefix_key,
    prefix_keys,
)

FP = {"model": "a", "kv_cache_dtype": "q4_0", "page_size": 1}


def _ids(n, start=0):
    return torch.arange(start, start + n, dtype=torch.int32)


def _tensors(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        ("ids", _ids(n)),
        ("kv", torch.randn(2, 3, n, 4, generator=g).to(torch.bfloat16)),
        ("kv_scale", torch.randn(2, 3, n, 1, generator=g).to(torch.float16)),
        ("codes", torch.randint(0, 255, (n, 5), generator=g, dtype=torch.uint8)),
        ("recurrent", torch.randn(2, 4, 8, 8, generator=g)),
    ]


def _save(store, n, ids=None, seed=0):
    ids = _ids(n) if ids is None else ids
    tensors = _tensors(n, seed)
    tensors[0] = ("ids", ids)
    return store.save(prefix_key(store.digest, ids), n, tensors), tensors


def test_roundtrip_is_bit_exact_for_every_dtype(tmp_path):
    store = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    entry, tensors = _save(store, 16)
    got = store.load(entry, expect_ids=_ids(16))
    assert got is not None
    for name, t in tensors:
        assert got[name].dtype == t.dtype and got[name].shape == t.shape
        assert torch.equal(got[name].view(torch.uint8), t.view(torch.uint8)), name


def test_keys_are_one_running_hash_and_depend_on_tokens_and_fingerprint():
    d = fingerprint_digest(FP)
    ids = _ids(100)
    many = prefix_keys(d, ids, [10, 64, 100])
    assert many[64] == prefix_key(d, ids[:64])
    assert many[10] != many[64]
    other = ids.clone()
    other[5] = 999
    assert prefix_key(d, other[:64]) != many[64]
    assert prefix_key(fingerprint_digest({**FP, "page_size": 64}), ids[:64]) != many[64]


def test_lookup_returns_the_deepest_prefix_within_the_limit(tmp_path):
    store = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    prompt = _ids(50)
    _save(store, 8, prompt[:8])
    _save(store, 32, prompt[:32])
    _save(store, 20, _ids(20, start=1000))           # someone else's prompt
    assert store.lookup(prompt).length == 32
    assert store.lookup(prompt, max_len=31).length == 8
    assert store.lookup(_ids(50, start=7)) is None


def test_a_restart_sees_the_entries_again(tmp_path):
    store = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    entry, tensors = _save(store, 12)
    again = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    found = again.lookup(_ids(40))
    assert found is not None and found.key == entry.key
    assert torch.equal(again.load(found)["recurrent"], tensors[4][1])


def test_another_configuration_never_reads_them(tmp_path):
    a = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    entry, _ = _save(a, 12)
    b = PrefixDiskStore(str(tmp_path), 1 << 30, {**FP, "kv_cache_dtype": "q8_0"})
    assert len(b) == 0 and b.lookup(_ids(40)) is None
    # even handed the file directly (or the file copied into its directory), b refuses it
    assert b.load(entry) is None
    os.makedirs(b.dir, exist_ok=True)
    stray = os.path.join(b.dir, os.path.basename(entry.path))
    with open(entry.path, "rb") as src, open(stray, "wb") as dst:
        dst.write(src.read())
    c = PrefixDiskStore(str(tmp_path), 1 << 30, {**FP, "kv_cache_dtype": "q8_0"})
    assert len(c) == 0 and not os.path.exists(stray)
    # and a's own entry is untouched by all that
    assert PrefixDiskStore(str(tmp_path), 1 << 30, FP).lookup(_ids(40)) is not None


def test_a_flipped_byte_is_caught_and_the_file_removed(tmp_path):
    store = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    entry, _ = _save(store, 16)
    with open(entry.path, "r+b") as f:
        f.seek(-3, os.SEEK_END)
        b = f.read(1)
        f.seek(-3, os.SEEK_END)
        f.write(bytes([b[0] ^ 0x40]))
    assert store.load(entry) is None
    assert not os.path.exists(entry.path) and len(store) == 0


def test_a_truncated_file_is_removed_at_startup(tmp_path):
    store = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    entry, _ = _save(store, 16)
    size = os.path.getsize(entry.path)
    with open(entry.path, "r+b") as f:
        f.truncate(size - 10)
    again = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    assert len(again) == 0 and not os.path.exists(entry.path)


def test_a_crashed_write_leaves_only_a_tmp_that_the_next_start_removes(tmp_path, monkeypatch):
    store = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    real_replace = os.replace

    def crash(src, dst):
        if dst.endswith(SUFFIX):
            raise KeyboardInterrupt  # the process dies between the write and the rename
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        _save(store, 16)
    monkeypatch.setattr(os, "replace", real_replace)
    # the writer cleans up after itself when it can; a hard kill cannot, so plant the leftover
    dead = os.path.join(store.dir, ".deadbeef.999999999.tmp")
    open(dead, "wb").write(b"half an entry")
    live = os.path.join(store.dir, f".cafe.{os.getppid()}.tmp")  # a live process's write
    open(live, "wb").write(b"in progress")
    again = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    assert len(again) == 0
    assert not os.path.exists(dead) and os.path.exists(live)
    assert not [n for n in os.listdir(store.dir) if n.endswith(SUFFIX)]


def test_a_key_collision_is_not_served_and_not_deleted(tmp_path):
    store = PrefixDiskStore(str(tmp_path), 1 << 30, FP)
    entry, _ = _save(store, 16)
    assert store.load(entry, expect_ids=_ids(16, start=1)) is None
    assert os.path.exists(entry.path)


def test_capacity_evicts_least_recently_used_first(tmp_path):
    one = _save(PrefixDiskStore(str(tmp_path / "probe"), 1 << 30, FP), 64)[0].nbytes
    store = PrefixDiskStore(str(tmp_path / "c"), int(one * 2.5), FP)
    a, _ = _save(store, 64, _ids(64, 0))
    b, _ = _save(store, 64, _ids(64, 100))
    os.utime(a.path, (1, 1)); a.mtime = 1                # a is the older ...
    os.utime(b.path, (2, 2)); b.mtime = 2
    store.touch(a)                                         # ... until it is used
    c, _ = _save(store, 64, _ids(64, 200))
    assert c is not None
    keys = {e.key for e in store.entries()}
    assert keys == {a.key, c.key}
    assert store.total_bytes <= store.capacity
    # an entry that could never fit is refused rather than emptying the directory for nothing
    assert store.save("x" * 64, 4096, _tensors(4096)) is None
    assert {e.key for e in store.entries()} == keys


def test_other_configurations_share_the_capacity(tmp_path):
    one = _save(PrefixDiskStore(str(tmp_path / "probe"), 1 << 30, FP), 64)[0].nbytes
    root = str(tmp_path / "c")
    old = PrefixDiskStore(root, 1 << 30, {**FP, "model": "old"})
    stale, _ = _save(old, 64)
    os.utime(stale.path, (1, 1))
    store = PrefixDiskStore(root, int(one * 1.5), FP)
    assert store.total_bytes == stale.nbytes, "the other model's entry counts"
    fresh, _ = _save(store, 64)
    assert fresh is not None and store.total_bytes <= store.capacity
    assert not os.path.exists(stale.path), "the other model's older entry made room"
