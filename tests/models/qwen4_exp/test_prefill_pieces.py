"""--prefill-mixer-pieces on Qwen3.8-Flash-Next: a chunk's GDN / QSA in pieces equals the chunk whole.

The shipping QSA geometry (head_dim 256, index budget 2048, page 64) at toy width, so a
3000-token chunk takes the sparse path, not the dense one below the budget. Three levels,
so a failure says where: the QSA layer, the GDN layer, and whole decoder layers with their
hyper-connections and a real MoE, where only the mixer is pieced.

The oracle is the same code run over the whole chunk. QSA already guarantees that a prefill
split across forwards is bit-identical to one shot (test_qsa_backend); pieces are that split
inside one forward, so exact equality is the bar here too -- and all three levels meet it,
the GDN recurrent / conv state the next chunk resumes from included.
"""

from __future__ import annotations

import pytest
import torch

from .common import Fixture, fill_weights, parsed_config, requires_cuda

QSA_LAYER, GDN_LAYER = 3, 0
LENGTH = 3000  # past the 2048 index budget: sparse selection is exercised


def _prefill_batch(fixture, table_idx, length):
    req = fixture.req(table_idx, 0, length)
    req.linear_slot_idx = None
    req.mamba_ping_pong = None
    req.uid = table_idx
    batch = fixture.batch([req], "prefill")
    batch.input_ids = torch.zeros(length, dtype=torch.int32, device=fixture.device)
    batch.fla_metadata = None
    batch.spec_verify = False
    return batch


def _with_state_pool(fixture, config):
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    group = config.linear_attention_group()
    fixture.ctx.linear_state_pool = LinearStatePool(group, 8, fixture.dtype, fixture.device, tp_size=1)
    return fixture.ctx.linear_state_pool


def _pieced(fixture, batch, n, pool, run_piece):
    from freetoken.models.prefill_pieces import plan_prefill_pieces

    pieces = plan_prefill_pieces(batch, n, fixture.backend, fixture.device, pool)
    assert pieces is not None and len(pieces) == n
    outs = []
    for start, end, piece in pieces:
        with fixture.ctx.piece_batch(piece):
            outs.append(run_piece(start, end, piece))
    return torch.cat(outs)


@requires_cuda
@pytest.mark.parametrize("n", [2, 3])
def test_qsa_layer_in_pieces_equals_whole(n):
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    x = torch.randn(LENGTH, config.hidden_size, device=fixture.device, dtype=fixture.dtype) * 0.5

    whole_batch = _prefill_batch(fixture, 1, LENGTH)
    with fixture.ctx.forward_batch(whole_batch):
        whole = attn.forward(x, whole_batch)

    batch = _prefill_batch(fixture, 2, LENGTH)
    with fixture.ctx.forward_batch(batch):
        got = _pieced(fixture, batch, n, None, lambda s, e, piece: attn.forward(x[s:e], piece))
    assert torch.equal(got, whole)


def _gdn_layer(config, fixture):
    from freetoken.models.qwen4_exp.model import build_linear_mixer
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device(fixture.device), torch_dtype(fixture.dtype):
        gdn = build_linear_mixer(config, GDN_LAYER, "linear_attn")
    fill_weights(gdn, 5, fixture.device)
    with torch.no_grad():  # a zero-mean A_log / dt_bias would hide a state that is not carried
        gdn.A_log.uniform_(0.0, 2.0)
        gdn.dt_bias.uniform_(-1.0, 1.0)
    return gdn


@requires_cuda
@pytest.mark.parametrize("n", [2, 3])
def test_gdn_layer_in_pieces_equals_whole(n):
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    pool = _with_state_pool(fixture, config)
    gdn = _gdn_layer(config, fixture)
    x = torch.randn(LENGTH, config.hidden_size, device=fixture.device, dtype=fixture.dtype) * 0.5

    whole_batch = _prefill_batch(fixture, 1, LENGTH)
    with fixture.ctx.forward_batch(whole_batch):
        whole = gdn.forward(x)

    batch = _prefill_batch(fixture, 2, LENGTH)
    with fixture.ctx.forward_batch(batch):
        got = _pieced(fixture, batch, n, pool, lambda s, e, piece: gdn.forward(x[s:e]))
    assert torch.equal(got, whole)
    # and the state the next chunk resumes from is the same state
    li = pool.local_index(GDN_LAYER)
    assert torch.equal(pool.recurrent_states[li][2], pool.recurrent_states[li][1])
    assert torch.equal(pool.conv_states[li][2], pool.conv_states[li][1])


@requires_cuda
@pytest.mark.parametrize("n", [2, 4])
def test_decoder_layers_in_pieces_equal_whole(n):
    """A GDN layer then the QSA layer, each with hyper-connections and a real (fused) MoE."""
    from freetoken.models.prefill_pieces import plan_prefill_pieces
    from freetoken.models.qwen4_exp.model import Qwen4ExpDecoderLayer
    from freetoken.utils.torch_utils import torch_dtype

    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    pool = _with_state_pool(fixture, config)
    layers = []
    for layer_id, seed in ((GDN_LAYER, 11), (QSA_LAYER, 12)):
        with torch.device(fixture.device), torch_dtype(fixture.dtype):
            layer = Qwen4ExpDecoderLayer(config, layer_id)
        fill_weights(layer, seed, fixture.device)
        if layer._is_linear:
            with torch.no_grad():
                layer.linear_attn.A_log.uniform_(0.0, 2.0)
                layer.linear_attn.dt_bias.uniform_(-1.0, 1.0)
        assert layer.ple is None  # layers 0 and 3 carry no PLE; PLE runs whole in both paths anyway
        layers.append(layer)
    width = config.qwen4_args.hc_count * config.hidden_size
    R0 = torch.randn(LENGTH, width, device=fixture.device, dtype=fixture.dtype) * 0.5

    whole_batch = _prefill_batch(fixture, 1, LENGTH)
    with fixture.ctx.forward_batch(whole_batch):
        R = R0
        for layer in layers:
            R = layer.forward(R, whole_batch)
    whole = R

    batch = _prefill_batch(fixture, 2, LENGTH)
    with fixture.ctx.forward_batch(batch):
        pieces = plan_prefill_pieces(batch, n, fixture.backend, fixture.device, pool)
        assert pieces is not None and batch.prefill_pieces is pieces
        R = R0
        for layer in layers:
            R = layer.forward_pieces(R, batch, pieces, fixture.ctx)
    assert torch.equal(R, whole)
