"""Dump the offload MoE cache's decode instrumentation to JSON (``--moe-stats-out``).

Kept apart from ``engine.py`` because it only duck-types the cache (six attributes and
three methods) and pulls in nothing from the kernel stack, so it is testable without a
built ``flashlib``.

The payload's reason for existing is the per-expert routing histogram: a disk-backed
expert bank keeps the frequently routed experts in RAM and leaves the rest on the SSD,
and that placement table is exactly this histogram, ordered.
"""

from __future__ import annotations

import json
import os

from freetoken.utils import init_logger

logger = init_logger(__name__)


def rank_path(path: str, rank: int, size: int) -> str:
    """``moe.json`` -> ``moe.rank1.json`` under a multi-rank run; unchanged when size == 1.

    Under ``--pp-size`` each rank's cache covers only its own layer window, so the
    histograms are disjoint slices of the model rather than replicas -- writing them to one
    path would keep whichever rank exited last and silently drop the other half.
    """
    if size <= 1:
        return path
    stem, dot, ext = path.rpartition(".")
    return f"{stem}{dot and '.'}rank{rank}{dot and '.'}{ext}" if dot else f"{path}.rank{rank}"


def collect_moe_stats(cache, rank: int, size: int, layer_range=None) -> dict:
    """The JSON payload for one rank's cache. No I/O."""
    return {
        "rank": rank,
        "world_size": size,
        # Global decoder-layer window this rank owns, so per-rank files can be
        # concatenated back into one model-wide table.
        "layer_range": list(layer_range) if layer_range else None,
        "num_layers": int(cache.num_layers),
        "num_experts": int(cache.num_experts),
        "cache_size": int(cache.cache_size),
        "decode_target": cache.decode_target,
        "miss_stats": cache.decode_miss_stats(),
        "per_layer": cache.decode_miss_stats_per_layer().get("per_layer", []),
        "routing": cache.decode_routing_stats(),
        # [num_layers][num_experts] raw decode routing counts -- the placement table is
        # this, sorted. ~12k ints per rank for Flash-Next, so JSON stays small.
        "decode_freq": cache.decode_freq.tolist() if cache.collect_decode_freq else None,
    }


def write_moe_stats(
    cache, path: str | None, rank: int, size: int, layer_range=None, quiet: bool = False
) -> str | None:
    """Write one rank's stats; return the path written, or None when nothing was written.

    Called every time the scheduler goes idle as well as on an orderly stop, so the file
    never depends on how the server is stopped: ``kill``, ``systemctl stop`` and a launcher
    that ignores SIGINT never reach the stop path, and neither does a crash. The file is
    replaced whole (written next to it, then renamed), so a stop in the middle of a write
    leaves the previous one readable.

    Never raises: a bad path, a full disk or a stats method that fails must not take down a
    serving scheduler or turn a clean stop into a traceback. A dense model has no offload
    cache at all, which is also not an error. ``quiet`` logs the success at debug level (the
    idle rewrites); failures are always warnings.
    """
    if not path or cache is None:
        return None
    out = rank_path(path, rank, size)
    tmp = f"{out}.tmp"
    try:
        payload = collect_moe_stats(cache, rank, size, layer_range)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, out)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        logger.warning(f"--moe-stats-out: could not write {out}: {exc}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    routing = payload["routing"] or {}
    cover = routing.get("experts_for_90pct")
    (logger.debug if quiet else logger.info)(
        f"--moe-stats-out: wrote {out} "
        f"(miss_rate={payload['miss_stats'].get('miss_rate', 0.0):.3f}"
        + (f", experts_for_90pct={cover:.1f}/{payload['num_experts']}" if cover else "")
        + ")"
    )
    return out
