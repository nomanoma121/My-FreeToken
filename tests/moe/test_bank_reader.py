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
