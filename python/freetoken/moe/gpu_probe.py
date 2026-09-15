"""GPU side of ``ft doctor disk``: each GPU's PCIe link and the host -> device rate it achieves.

A prefill chunk of an offloaded MoE streams every layer's whole expert bank to its GPU, so the
link is a per-chunk cost, not only a decode one. The link as the driver reports it at idle is
not the link a copy gets: GPUs drop to a lower PCIe generation to save power and come back up
under load, so the generation is read again while a copy runs.

``nvidia-smi`` rather than sysfs: under WSL2 the GPU is not on the Linux PCI bus at all, and
the driver's view is the same on both. The rate needs torch and a free GPU; the parsing does not.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

_FIELDS = (
    "index", "pci.bus_id", "name",
    "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.gen.hostmax",
    "pcie.link.width.current", "pcie.link.width.max",
)


def _int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


@dataclass
class GpuLink:
    index: int
    bus_id: str
    name: str
    gen: int | None
    gen_max: int | None  # the GPU's own maximum
    gen_host: int | None  # the slot's (host side) maximum
    width: int | None
    width_max: int | None

    def describe(self) -> str:
        text = f"PCIe Gen{self.gen or '?'} x{self.width or '?'}"
        top = min(g for g in (self.gen_max, self.gen_host) if g) if (self.gen_max or self.gen_host) else None
        if (top and self.gen and top > self.gen) or (self.width_max and self.width and self.width_max > self.width):
            text += f" (GPU Gen{self.gen_max or '?'} x{self.width_max or '?'}, slot Gen{self.gen_host or '?'})"
        return text

    @property
    def lane_gbs(self) -> float | None:
        """Nominal one-direction payload rate of the link in GB/s (after line coding)."""
        per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}.get(self.gen or 0)
        return per_lane * self.width if per_lane and self.width else None


def parse_links(csv: str) -> list[GpuLink]:
    out = []
    for line in csv.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(_FIELDS) or _int(cells[0]) is None:
            continue
        out.append(GpuLink(
            index=_int(cells[0]), bus_id=cells[1], name=cells[2],
            gen=_int(cells[3]), gen_max=_int(cells[4]), gen_host=_int(cells[5]),
            width=_int(cells[6]), width_max=_int(cells[7]),
        ))
    return out


def query_links(index: int | None = None, timeout: float = 10.0) -> list[GpuLink] | None:
    """Every GPU's link as the driver reports it now; None without a usable ``nvidia-smi``."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    cmd = [exe, f"--query-gpu={','.join(_FIELDS)}", "--format=csv,noheader,nounits"]
    if index is not None:
        cmd.insert(1, f"--id={index}")
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return parse_links(done.stdout)


def cuda_index_of(bus_id: str) -> int | None:
    """The CUDA ordinal of the GPU at ``bus_id`` (nvidia-smi's form): CUDA numbers the fastest first
    by default, nvidia-smi by PCI address, so the two indices differ on mixed hosts."""
    import torch

    want = bus_id.strip().lower()
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        dom, bus, dev = (getattr(p, k, None) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
        if bus is None:
            return None
        if f"{dom or 0:08x}:{bus:02x}:{dev:02x}.0" == want:
            return i
    return None


def h2d_rate(link: GpuLink, seconds: float = 2.0, mib: int = 256) -> tuple[float, GpuLink | None]:
    """(GB/s, the link read mid-copy) for pinned host -> device copies to the GPU ``link`` names.

    The number is a ceiling for the registered bank rows, which the GPU reads directly; rows
    that go through a staging copy first are bounded by that copy as well.
    """
    import torch

    index = cuda_index_of(link.bus_id)
    if index is None:
        raise RuntimeError(f"no CUDA device at {link.bus_id}")
    n = mib << 20
    device = torch.device("cuda", index)
    src = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    src.fill_(1)
    dst = torch.empty(n, dtype=torch.uint8, device=device)
    dst.copy_(src)
    torch.cuda.synchronize(device)
    seen: list[GpuLink | None] = [None]

    def look() -> None:
        time.sleep(min(0.5, seconds / 2))
        links = query_links(link.index)
        seen[0] = links[0] if links else None

    watcher = threading.Thread(target=look, daemon=True)
    watcher.start()
    copied = 0
    start = time.perf_counter()
    while True:
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(device)
        copied += n
        elapsed = time.perf_counter() - start
        if elapsed >= seconds and not watcher.is_alive():
            break
    del dst, src
    torch.cuda.empty_cache()
    return copied / elapsed / 1e9, seen[0]
