"""moe/bank_reader.py: the parallel direct reads hand back exactly the file's bytes, in order,
whatever the offsets' alignment, and never reuse a buffer set before the caller says so."""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.bank_reader import ALIGN, DirectRangeReader


@pytest.fixture
def blob(tmp_path):
    # not a multiple of the page size, and not a repeating pattern
    data = os.urandom(3 * (1 << 20) + 1234)
    path = tmp_path / "bank.ftmb"
    path.write_bytes(data)
    return str(path), data


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("offset,nbytes", [(0, 1 << 20), (4095, 700_001), (123_457, 2 * (1 << 20)), (8192, 3 * (1 << 20) + 1234 - 8192)])
def test_reads_back_the_file(blob, direct, offset, nbytes):
    path, data = blob
    reader = DirectRangeReader(path, threads=3, piece_bytes=64 * ALIGN, direct=direct)
    try:
        out = bytearray(nbytes)
        order = []
        sets = []

        def sink(p, host, s):
            order.append(p)
            out[p:p + host.numel()] = bytes(host.numpy())

        reader.read(offset, nbytes, sink, before_group=sets.append)
        assert bytes(out) == data[offset:offset + nbytes]
        assert order == sorted(order)
        pieces = -(-nbytes // reader.piece)
        assert len(sets) == -(-pieces // reader.threads)
        assert sets == [i % 2 for i in range(len(sets))]
    finally:
        reader.close()


def test_buffers_are_page_aligned(blob):
    path, _ = blob
    # an allocator that hands back storage starting mid-page
    reader = DirectRangeReader(path, threads=2, piece_bytes=8 * ALIGN,
                               alloc=lambda n: torch.empty(n + 3, dtype=torch.uint8)[3:])
    try:
        for group in reader.buffers:
            for view, _ in group:
                assert view.data_ptr() % ALIGN == 0
                assert view.numel() >= reader.piece + ALIGN
    finally:
        reader.close()


def test_a_read_past_the_end_is_an_error(blob):
    path, data = blob
    reader = DirectRangeReader(path, threads=2, piece_bytes=64 * ALIGN, direct=False)
    try:
        with pytest.raises(OSError, match="short read"):
            reader.read(len(data) - 10, 100, lambda p, h, s: None)
    finally:
        reader.close()


def test_cached_pieces_come_from_the_source(blob):
    path, data = blob
    reader = DirectRangeReader(path, threads=3, piece_bytes=64 * ALIGN, direct=True)
    try:
        offset, nbytes = 4095, 2 * (1 << 20)
        # a source that differs from the file, so the output shows which path each piece took
        source = torch.full((nbytes,), 7, dtype=torch.uint8)
        take = lambda p, n: (p // reader.piece) % 3 == 0  # every third piece "cached"
        out = bytearray(nbytes)

        def sink(p, host, s):
            out[p:p + host.numel()] = bytes(host.numpy())

        copied = reader.read(offset, nbytes, sink, source=source, cached=take)
        expect = bytearray(data[offset:offset + nbytes])
        for p in range(0, nbytes, reader.piece):
            if take(p, 0):
                n = min(reader.piece, nbytes - p)
                expect[p:p + n] = b"\x07" * n
        assert bytes(out) == bytes(expect)
        assert copied == sum(min(reader.piece, nbytes - p) for p in range(0, nbytes, reader.piece) if take(p, 0))
    finally:
        reader.close()


def test_a_reader_built_under_inference_mode_copies_on_its_threads(blob):
    """The engine builds the reader under torch.inference_mode(), which is per thread: the pool's
    copies out of the mapping must not trip "Inplace update to inference tensor" (it killed the
    scheduler on the first prefill on the RTX 2060)."""
    path, data = blob
    with torch.inference_mode():
        reader = DirectRangeReader(path, threads=2, piece_bytes=64 * ALIGN)
        source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    try:
        out = bytearray(1 << 20)

        def sink(p, host, s):
            out[p:p + host.numel()] = bytes(host.numpy())

        with torch.inference_mode():
            copied = reader.read(0, 1 << 20, sink, source=source[: 1 << 20], cached=lambda p, n: True)
        assert copied == 1 << 20 and bytes(out) == data[: 1 << 20]
    finally:
        reader.close()


def test_direct_only_when_the_page_cache_cannot_hold_the_rows():
    from freetoken.moe.bank_reader import choose_direct

    G = 2**30
    # the RTX 2060 host with RAM to spare: 23.5 GiB, 6 resident, 11 non-resident -> buffered
    direct, why = choose_direct(11 * G, 6 * G, 1, mem_total=int(23.5 * G), limit=None)
    assert not direct and "MemTotal" in why
    # the same server held to 13 GiB by its cgroup -> direct
    direct, why = choose_direct(11 * G, 6 * G, 1, mem_total=int(23.5 * G), limit=13 * G)
    assert direct and "cgroup limit" in why
    # the reported host: 64 GB, two ranks of 21 GiB resident and 10.7 non-resident -> direct
    assert choose_direct(int(10.7 * G), 21 * G, 2, mem_total=64 * G, limit=None)[0]


def test_cgroup_limit_takes_the_tightest_level(tmp_path):
    from freetoken.moe.bank_reader import _cgroup_limit

    proc, sys = tmp_path / "proc", tmp_path / "sys"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/user.slice/run.scope\n")
    root = sys / "fs" / "cgroup"
    (root / "user.slice" / "run.scope").mkdir(parents=True)
    (root / "user.slice" / "memory.max").write_text("max\n")
    (root / "user.slice" / "run.scope" / "memory.high").write_text(str(13 * 2**30) + "\n")
    (root / "user.slice" / "run.scope" / "memory.max").write_text("max\n")
    assert _cgroup_limit(str(proc), str(sys)) == 13 * 2**30
    (root / "user.slice" / "memory.max").write_text(str(8 * 2**30) + "\n")
    assert _cgroup_limit(str(proc), str(sys)) == 8 * 2**30
    (proc / "self" / "cgroup").write_text("1:name=systemd:/\n")
    assert _cgroup_limit(str(proc), str(sys)) is None
