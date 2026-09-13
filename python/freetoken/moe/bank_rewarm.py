"""Read a ``--moe-bank-ram`` bank's file-backed rows back in while the server is idle.

The resident prefix of a mapped bank is held down; the rest is ordinary page cache, and
anything may take it: another process's memory, or -- on WSL2 with
``autoMemoryReclaim=gradual`` -- the VM handing file cache back to Windows while nothing is
running. The server keeps working, but the next request pays for every page it touches, one
fault at a time.

Measured on the RTX 2060 (Ornith, 16.9 GiB bank, ``--moe-bank-ram 6G``) after pressure took
the cold side from 100% to 36% of the page cache:

    long prompt (~2k tokens)   TTFT 4.0 s -> 13.5 s   (prefill streams layers: sequential reads)
    short prompt (a follow-up) TTFT 1.0 s -> 30.9 s   (CPU prefill + decode: random faults)

Both return to warm on the very next request, because that request has read everything
back. The short case is the one a chat hits after a pause, and it is the worse one: random
4 KiB faults instead of readahead. Reading the same bytes in file order while nobody is
waiting takes about as long as the long prompt's penalty and costs no one anything.

So this waits for the scheduler to go idle, and every ``delay`` seconds after that checks how
much of the cold side the page cache still holds. Below ``threshold`` it walks the cold rows
in file order, a step at a time, and stops between steps the moment a request arrives -- the
request then pays only for what was not read yet.

It never runs while the engine is busy and never holds the GIL across I/O (the walk is a
torch op over the mapping), so the only cost is disk reads during idle time, which is why it
is opt-in (``--moe-bank-rewarm``).
"""

from __future__ import annotations

import threading
import time

from freetoken.utils import init_logger

logger = init_logger(__name__)


class BankRewarm:
    """Idle-time re-read of a mapped bank's cold rows. ``idle()`` / ``busy()`` are called
    from the scheduler thread; the reading happens on a daemon thread of its own."""

    def __init__(self, banks, delay_s: float, threshold: float = 0.95, log=None,
                 max_backoff_s: float = 600.0):
        self.banks = banks
        self.delay_s = float(delay_s)
        self.threshold = float(threshold)
        self.max_backoff_s = float(max_backoff_s)
        self._log = log or logger.info
        self._cancel = threading.Event()
        self._cancel.set()  # nothing running yet
        self._thread: threading.Thread | None = None
        # Visible to tests and to anyone reading a trace: what the last walk did.
        self.last: dict | None = None

    def idle(self) -> None:
        """The scheduler has nothing to do. Start watching (a no-op if already watching)."""
        if self._thread is not None and self._thread.is_alive() and not self._cancel.is_set():
            return
        cancel = threading.Event()
        self._cancel = cancel
        self._thread = threading.Thread(
            target=self._watch, args=(cancel,), name="bank-rewarm", daemon=True
        )
        self._thread.start()

    def busy(self) -> None:
        """A request arrived. The walk stops at its next step boundary."""
        self._cancel.set()

    def _watch(self, cancel: threading.Event) -> None:
        wait = self.delay_s
        while not cancel.wait(wait):
            try:
                share = self.banks.cold_residency()
            except Exception as exc:  # noqa: BLE001 -- a measurement must not kill the thread's owner
                self._log(f"--moe-bank-rewarm: cannot read page-cache residency ({exc}); stopping")
                return
            if share >= self.threshold:
                wait = self.delay_s
                continue
            walked, seconds, finished = self.banks.rewarm(cancel)
            after = self.banks.cold_residency()
            self.last = {"before": share, "after": after, "bytes": walked,
                         "seconds": seconds, "finished": finished}
            self._log(
                f"--moe-bank-rewarm: cold rows were {share:.0%} in page cache; walked "
                f"{walked / 2**30:.1f} GiB of them in {seconds:.1f} s -> {after:.0%}"
                + ("" if finished else " (stopped early: a request arrived)")
            )
            # A full walk that ends with less in the cache than it started with means something
            # is still pressing on memory and taking pages as fast as they come back. Measured:
            # a walk during held pressure took 49 s and went from 77% to 25%. Walking again at
            # once only fights it, so wait longer each time until the cache starts holding.
            if finished and after < share:
                wait = min(max(wait, self.delay_s) * 4, self.max_backoff_s)
                self._log(f"--moe-bank-rewarm: memory is still under pressure; next check in {wait:g} s")
            else:
                wait = self.delay_s

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
