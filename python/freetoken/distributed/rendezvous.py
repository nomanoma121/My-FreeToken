"""Let a rank that finishes a startup step early wait for the others, for longer than a step would.

The gloo group's timeout (``distributed_timeout``, 60 s) is also what turns a wedged rank into a
failure while serving, so it stays short. At startup it is wrong: the ranks load different halves
of the model and reach the first collective at different times, and the gap is not bounded by
anything a step is. Measured on two RTX 3060s (Qwen3.8-Flash-Next, ``--pp-size 2``,
``--moe-bank-ram 48G``) right after WSL restarted, with 46 GiB of RAM held by another process: rank
1 was still reading its experts (the published build) or faulting in its resident rows (the one
bank file) when rank 0 reached the KV page count agreement, and rank 0's all_reduce gave up after
60 s -- ``Timed out waiting 60000ms for recv operation`` -- four starts out of four. Warm, the same
start passed, which is why it went unseen.

``wait_for_ranks`` puts a ``monitored_barrier`` with its own, long timeout in front of such a
collective: the early rank waits there, the collective then runs with every rank present, and a
wait long enough to notice is logged with what was being waited for.
"""

from __future__ import annotations

import time
from datetime import timedelta

import torch.distributed as dist

from freetoken.env import ENV
from freetoken.utils import init_logger

logger = init_logger(__name__)

# below this a wait is ordinary startup jitter and not worth a line
_LOG_AFTER_S = 10.0


def wait_for_ranks(group: dist.ProcessGroup | None, what: str) -> float:
    """Block until every rank of ``group`` reaches this call, for up to
    ``FREETOKEN_RANK_JOIN_TIMEOUT_SECONDS``. Returns the seconds waited (0 with one rank).

    Only for points every rank passes exactly once and in the same order (startup), since a
    barrier that one rank skips waits for the whole timeout.
    """
    if group is None or group.size() <= 1:
        return 0.0
    timeout = float(ENV.RANK_JOIN_TIMEOUT_SECONDS.value)
    started = time.monotonic()
    dist.monitored_barrier(group=group, timeout=timedelta(seconds=timeout), wait_all_ranks=True)
    waited = time.monotonic() - started
    if waited >= _LOG_AFTER_S:
        logger.info(
            f"waited {waited:.0f} s for the other ranks before {what} "
            f"(FREETOKEN_RANK_JOIN_TIMEOUT_SECONDS {timeout:g})"
        )
    return waited
