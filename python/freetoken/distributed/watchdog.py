"""Say which blocking wait between ranks has gone on too long.

The pipeline engine and the scheduler's message-count channel block in gloo sends/receives
with no timeout. When one rank stops -- a crash its peers did not see, a relayed request the
PUB socket dropped, a forward that never returns -- every other rank sits in one of those calls
and the server reports a running request at 0 tok/s with nothing in the log (upstream #453).

This adds no timeout: an abandoned gloo/NCCL operation leaves the group unusable, and a long wait
can be legitimate -- the other rank may be in a prefill chunk that takes tens of seconds on a
3060 with CPU MoE. It only logs, once a wait passes ``FREETOKEN_RANK_WAIT_WARN_SECONDS``, what
this rank is blocked on and which peer it is waiting for, and again every interval while the
wait lasts. The blocking call brackets itself with ``begin()``/``end()`` (a clock read and two
attribute writes, no lock: the one-tuple slot is swapped atomically under the GIL) and a single
daemon thread per process looks at the slot.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from freetoken.utils import init_logger

logger = init_logger(__name__)

# floor on the repeat interval, so a low threshold still does not flood the log
_MIN_REPEAT_S = 60.0


class RankWaitWatchdog:
    def __init__(
        self,
        warn_after: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        warn: Callable[[str], None] | None = None,
        info: Callable[[str], None] | None = None,
        start_thread: bool = True,
    ):
        self.warn_after = warn_after
        self.repeat_every = max(warn_after, _MIN_REPEAT_S)
        self._clock = clock
        self._warn = warn or logger.warning
        self._info = info or logger.info
        # (since, what, peer, detail) of the wait in progress; ``what`` is a constant template,
        # formatted only when it is reported
        self._wait: tuple | None = None
        self._reported: tuple | None = None
        self._next_report = 0.0
        if start_thread and warn_after > 0:
            threading.Thread(target=self._run, name="rank-wait-watchdog", daemon=True).start()

    def begin(self, what: str, peer: int | None = None, detail: object = None) -> None:
        self._wait = (self._clock(), what, peer, detail)

    def end(self) -> None:
        self._wait = None
        if self._reported is not None:  # only after a warning: say how it ended
            reported, self._reported = self._reported, None
            self._info(
                f"{_rank_prefix()}the wait for {_describe(reported)} ended after "
                f"{self._clock() - reported[0]:.0f}s"
            )

    def tick(self) -> None:
        """One look at the open wait; the thread calls this, tests call it with a fake clock."""
        wait = self._wait
        if wait is None:
            return
        now = self._clock()
        if wait is not self._reported:
            if now - wait[0] < self.warn_after:
                return
            self._reported = wait
            if self._wait is not wait:  # ended while we looked: nothing to report, or to close
                self._reported = None
                return
        elif now < self._next_report:
            return
        self._next_report = now + self.repeat_every
        self._warn(
            f"{_rank_prefix()}blocked for {now - wait[0]:.0f}s waiting for {_describe(wait)}. "
            "This can be normal while the other rank runs a long prefill chunk; if it keeps "
            "growing, that rank has stopped or is stuck -- its log (or py-spy dump) says where. "
            "Nothing here times out."
        )

    def _run(self) -> None:
        poll = min(max(self.warn_after / 4, 0.01), 5.0)
        while True:
            time.sleep(poll)
            try:
                self.tick()
            except Exception:  # a diagnostic must never take the process down
                pass


def _describe(wait: tuple) -> str:
    _, what, peer, detail = wait
    return what.format(peer="?" if peer is None else peer, detail=detail)


def _rank_prefix() -> str:
    from .info import try_get_world_info

    info = try_get_world_info()
    return f"rank {info.rank}: " if info is not None else ""


_WATCHDOG: RankWaitWatchdog | None = None


def rank_wait_watchdog() -> RankWaitWatchdog:
    """The process's watchdog, started on first use (only multi-rank paths ever ask for it)."""
    global _WATCHDOG
    if _WATCHDOG is None:
        from freetoken.env import ENV

        _WATCHDOG = RankWaitWatchdog(ENV.RANK_WAIT_WARN_SECONDS.value)
    return _WATCHDOG


__all__ = ["RankWaitWatchdog", "rank_wait_watchdog"]
