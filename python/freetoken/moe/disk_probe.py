"""What a ``--moe-bank-ram`` bank sits on, and what the host has to give it. Stdlib only.

Everything the mapped bank's speed depends on outside the code is invisible from inside the
process and was found the hard way (docs/bank-ram.md): the device readahead window (worth
2.5x), the filesystem (a WSL2 ``/mnt/c`` path is a 9p file server), the transport and the
slot the drive is in (a SATA SSD adds ~300 ms a step; a chipset M.2 shares its uplink with a
chipset GPU slot), how much RAM is really available and how much of it may be locked.

This module only reads: ``/proc`` and ``/sys`` (both roots are parameters, so the tests
hand it a fake tree), plus an O_DIRECT read benchmark that writes nothing and leaves the page
cache alone. ``ft doctor disk`` (moe/disk_doctor.py) prints it; the server uses the parts
that decide something at startup: ``auto_bank_ram`` for ``--moe-bank-ram auto``,
``recommend_readahead_kb`` / ``set_readahead`` for ``--moe-bank-readahead``, and
``storage_warnings`` for the one-line complaint about where the bank file is.
"""

from __future__ import annotations

import math
import mmap
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

GiB = 2**30

# ---------------------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return int(text, 0) if text.startswith("0x") else int(text)
    except ValueError:
        return None


def nearest_existing(path: str) -> str:
    """``path``, or the closest ancestor that exists -- a bank directory is made on first use,
    and the question "what filesystem will it be on" has an answer before that."""
    path = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def is_wsl(proc: str = "/proc") -> bool:
    release = _read(os.path.join(proc, "sys/kernel/osrelease")) or ""
    return "microsoft" in release.lower() or "wsl" in release.lower()


# ---------------------------------------------------------------------------------------
# mounts and filesystems
# ---------------------------------------------------------------------------------------


@dataclass
class Mount:
    mountpoint: str
    fstype: str
    source: str
    dev: str  # "major:minor" as mountinfo gives it
    super_options: str = ""


def _unescape(text: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), text)


def parse_mountinfo(text: str) -> list[Mount]:
    """``/proc/self/mountinfo`` -> mounts in the order the kernel lists them (later ones
    stack on earlier ones at the same point)."""
    out = []
    for line in text.splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        f, r = left.split(), right.split()
        if len(f) < 5 or len(r) < 2:
            continue
        out.append(Mount(_unescape(f[4]), r[0], _unescape(r[1]), f[2], r[2] if len(r) > 2 else ""))
    return out


def mount_of(path: str, mounts: list[Mount]) -> Mount | None:
    """The mount ``path`` is on: the longest mount point that contains it, the later one on a tie."""
    best = None
    for m in mounts:
        mp = m.mountpoint.rstrip("/") or "/"
        if mp == "/" or path == mp or path.startswith(mp + "/"):
            if best is None or len(mp) >= len(best.mountpoint.rstrip("/") or "/"):
                best = m
    return best


# (severity, why). "bad" is a place --moe-bank-ram should not be run from at all; "warn" is
# one that works but was never measured or costs something specific.
_FS_VERDICTS = {
    "9p": ("bad", "a 9p file server (on WSL2, a Windows drive under /mnt): every page fault on "
                  "the mapping is a round trip to Windows, orders of magnitude slower than a disk"),
    "drvfs": ("bad", "a Windows drive seen through WSL (drvfs): every page fault on the mapping "
                     "is a round trip to Windows, orders of magnitude slower than a disk"),
    "virtiofs": ("bad", "a virtiofs share from the VM host: page faults go through the host's "
                        "file server"),
    "tmpfs": ("bad", "tmpfs, which is RAM: a bank file there costs the RAM --moe-bank-ram exists to save"),
    "ramfs": ("bad", "ramfs, which is RAM: a bank file there costs the RAM --moe-bank-ram exists to save"),
    "overlay": ("warn", "an overlay (container) layer: 30-60 GiB of bank written into the "
                        "container's upper layer; bind-mount a host directory and point "
                        "--moe-bank-dir at it"),
    "zfs": ("warn", "ZFS, where mapped reads are cached by the ARC and the page cache both, and the device "
                    "readahead window does not govern them; not measured"),
    "fuseblk": ("warn", "a FUSE block filesystem (NTFS/exFAT via FUSE): every fault goes through "
                        "a userspace daemon; not measured"),
    "ntfs3": ("warn", "NTFS (ntfs3), not measured"),
    "exfat": ("warn", "exFAT, not measured"),
    "vfat": ("bad", "FAT, which cannot hold a file over 4 GiB"),
}
_NETWORK_FS = {
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "afs", "ceph", "glusterfs", "lustre", "gpfs",
    "beegfs", "davfs", "fuse.sshfs", "sshfs", "fuse.rclone", "fuse.s3fs", "fuse.glusterfs",
}


def filesystem_verdict(mount: Mount) -> tuple[str, str]:
    """``("ok"|"warn"|"bad", reason)`` for the filesystem a bank file would be mapped from."""
    fstype = mount.fstype
    if fstype == "9p" and "drvfs" in mount.super_options:
        return _FS_VERDICTS["drvfs"]
    if fstype in _FS_VERDICTS:
        return _FS_VERDICTS[fstype]
    if fstype in _NETWORK_FS:
        return "bad", f"{fstype}, a network filesystem: every page fault is a network round trip"
    if fstype.startswith("fuse"):
        return "warn", f"{fstype}, where every fault goes through a userspace daemon; not measured"
    return "ok", fstype


# ---------------------------------------------------------------------------------------
# the block device under a filesystem
# ---------------------------------------------------------------------------------------

_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_GT_TO_GEN = {"2.5": 1, "5.0": 2, "5": 2, "8.0": 3, "8": 3, "16.0": 4, "16": 4, "32.0": 5,
              "32": 5, "64.0": 6, "64": 6}


@dataclass
class PciLink:
    bdf: str
    speed: str | None
    width: int | None
    max_speed: str | None
    max_width: int | None

    @staticmethod
    def gen(speed: str | None) -> int | None:
        if not speed:
            return None
        return _GT_TO_GEN.get(speed.split()[0])

    def describe(self) -> str:
        cur = self.gen(self.speed)
        top = self.gen(self.max_speed)
        text = f"PCIe Gen{cur or '?'} x{self.width or '?'}"
        if (top and cur and top > cur) or (self.max_width and self.width and self.max_width > self.width):
            text += f" (device supports Gen{top or '?'} x{self.max_width or '?'})"
        return text


@dataclass
class BlockDevice:
    name: str  # the whole disk the filesystem is on: nvme0n1, sda
    transport: str  # nvme, sata, usb, mmc, virtio, hyperv, xen, scsi, loop, ram, unknown
    virtual: bool
    rotational: bool | None
    model: str | None = None
    stacked_on: list[str] = field(default_factory=list)  # dm-0 -> nvme0n1p2 -> ...
    partition: str | None = None
    pci_chain: list[tuple[str, str]] = field(default_factory=list)  # (bdf, sysfs dir), root first
    link: PciLink | None = None
    backing_file: str | None = None  # loop devices


def _block_link(dev: str, source: str, sys: str) -> str | None:
    link = os.path.join(sys, "dev/block", dev)
    if os.path.exists(link):
        return link
    # btrfs and friends report an anonymous 0:N device; the mount source still names the disk
    if source.startswith("/dev/"):
        cand = os.path.join(sys, "class/block", os.path.basename(source))
        if os.path.exists(cand):
            return cand
    return None


def _pci_chain(real: str) -> list[tuple[str, str]]:
    parts = real.split("/")
    return [(p, "/".join(parts[: i + 1])) for i, p in enumerate(parts) if _BDF.match(p)]


def block_device(dev: str, source: str = "", sys: str = "/sys") -> BlockDevice | None:
    """The physical disk behind ``dev`` ("major:minor"), through dm/md stacking and partitions."""
    link = _block_link(dev, source, sys)
    if link is None:
        return None
    real = os.path.realpath(link)
    stacked = []
    for _ in range(8):  # dm on md on a partition is as deep as anything sane goes
        slaves = os.path.join(real, "slaves")
        members = sorted(os.listdir(slaves)) if os.path.isdir(slaves) else []
        if not members:
            break
        stacked.append(os.path.basename(real))
        real = os.path.realpath(os.path.join(slaves, members[0]))
    partition = None
    if os.path.exists(os.path.join(real, "partition")):
        partition = os.path.basename(real)
        real = os.path.dirname(real)
    name = os.path.basename(real)
    vendor = (_read(os.path.join(real, "device/vendor")) or "").strip()
    model = (_read(os.path.join(real, "device/model")) or "").strip() or None
    rot = _read_int(os.path.join(real, "queue/rotational"))
    backing = None
    virtual = False
    if name.startswith("nvme") or "/nvme/" in real:
        transport = "nvme"
    elif name.startswith("loop"):
        transport, backing = "loop", _read(os.path.join(real, "loop/backing_file"))
    elif name.startswith(("ram", "zram")):
        transport = "ram"
    elif name.startswith("mmcblk"):
        transport = "mmc"
    elif "/usb" in real:
        transport = "usb"
    elif "/virtio" in real or name.startswith("vd"):
        transport, virtual = "virtio", True
    elif vendor == "Msft" and "Virtual" in (model or ""):
        transport, virtual = "hyperv", True
    elif name.startswith("xvd") or "/vbd-" in real:
        transport, virtual = "xen", True
    elif "/ata" in real:
        transport = "sata"
    elif "/host" in real and "/target" in real:
        transport = "scsi"
    else:
        transport = "unknown"
    chain = _pci_chain(real)
    link_info = None
    if chain and transport == "nvme":
        bdf, d = chain[-1]
        link_info = PciLink(
            bdf,
            _read(os.path.join(d, "current_link_speed")),
            _read_int(os.path.join(d, "current_link_width")),
            _read(os.path.join(d, "max_link_speed")),
            _read_int(os.path.join(d, "max_link_width")),
        )
    return BlockDevice(
        name=name, transport=transport, virtual=virtual,
        rotational=None if rot is None else bool(rot), model=model, stacked_on=stacked,
        partition=partition, pci_chain=chain, link=link_info, backing_file=backing,
    )


# AMD chipset (Promontory) bridge device ids seen on AM4/AM5 boards. Not exhaustive; a board
# whose bridge is not listed reads as "unknown", never as CPU-attached.
_AMD_CHIPSET_BRIDGES = {
    0x43B4, 0x43B5, 0x43B6, 0x43B7, 0x43C6, 0x43C7, 0x43E9, 0x43EA, 0x43EB,
    0x57A3, 0x57A4, 0x57AD, 0x43F4, 0x43F5,
}


def upstream_kind(chain: list[tuple[str, str]]) -> str:
    """``"chipset"``, ``"cpu"`` or ``"unknown"``: which side of the chipset uplink a PCI device is.

    An estimate from the topology, for the one question it matters for: a chipset M.2 shares
    one DMI / Promontory uplink with every other chipset device, a chipset x4 GPU slot
    included, so under ``--pp-size 2`` the bank reads and rank 1's residual stream fight
    over it (docs/bank-ram.md, Limits).
    """
    if not chain:
        return "unknown"
    root_bdf, root_dir = chain[0]
    vendor = _read_int(os.path.join(root_dir, "vendor"))
    bus, slot = root_bdf.split(":")[1], int(root_bdf.split(":")[2].split(".")[0], 16)
    if vendor == 0x8086 and bus == "00" and slot in (0x1B, 0x1C, 0x1D):
        return "chipset"  # Intel PCH root ports: behind DMI
    for _, d in chain[:-1]:
        v, dv = _read_int(os.path.join(d, "vendor")), _read_int(os.path.join(d, "device"))
        if (v == 0x1022 and dv in _AMD_CHIPSET_BRIDGES) or v == 0x1B21:
            return "chipset"  # AMD Promontory, or an ASMedia chipset bridge
    if len(chain) == 2:
        return "cpu"  # an endpoint straight under a root port, and not a PCH one
    return "unknown"


def nvidia_gpus(sys: str = "/sys") -> list[tuple[str, list[tuple[str, str]]]]:
    """``[(bdf, chain)]`` for every NVIDIA display controller sysfs lists."""
    base = os.path.join(sys, "bus/pci/devices")
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        d = os.path.join(base, name)
        cls = _read(os.path.join(d, "class")) or ""
        if _read_int(os.path.join(d, "vendor")) == 0x10DE and cls.startswith(("0x0300", "0x0302")):
            out.append((name, _pci_chain(os.path.realpath(d))))
    return out


def shared_upstream(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> str | None:
    """The deepest bridge two chains share, or None when they meet only at the root complex."""
    common = None
    for (x, _), (y, _) in zip(a[:-1], b[:-1]):
        if x != y:
            break
        common = x
    return common


# ---------------------------------------------------------------------------------------
# readahead
# ---------------------------------------------------------------------------------------


def readahead(dev: str, sys: str = "/sys") -> tuple[int, str] | None:
    """``(kB, file)`` of the readahead window that governs faults on a file on device ``dev``.

    The file is named by where it really is. The partition case goes through ``..``, and
    ``os.path.normpath`` resolves that lexically -- ``/sys/dev/block/259:2/../queue`` becomes
    ``/sys/dev/block/queue``, a path that does not exist, which is what the startup line used to
    tell people to ``echo`` into.
    """
    for rel in (f"dev/block/{dev}/queue/read_ahead_kb",
                f"dev/block/{dev}/../queue/read_ahead_kb",
                f"class/bdi/{dev}/read_ahead_kb"):
        path = os.path.join(sys, rel)
        kb = _read_int(path)
        if kb is not None:
            return kb, _pretty_sysfs(os.path.realpath(path), sys)
    return None


def _pretty_sysfs(real: str, sys: str) -> str:
    """``/sys/devices/.../block/sdf/queue/read_ahead_kb`` -> ``/sys/block/sdf/queue/read_ahead_kb``
    when that shorter spelling reaches the same file."""
    m = re.search(r"/([^/]+)/queue/read_ahead_kb$", real)
    if m:
        short = os.path.join(sys, "block", m.group(1), "queue/read_ahead_kb")
        if os.path.exists(short) and os.path.realpath(short) == real:
            return short
    return real


def device_of(path: str) -> str | None:
    if not hasattr(os, "major"):
        return None
    try:
        st = os.stat(nearest_existing(path))
    except OSError:
        return None
    return f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"


def recommend_readahead_kb(widest_row_bytes: int) -> int:
    """The window to suggest for a bank whose widest per-expert block row is this many bytes.

    Measured optimum, two models: a quarter to a sixth of the widest row -- Flash-Next's is
    1600 kB and 256 won (512 and 128 each 3% slower, 2048 31%, 8192 49%, over 110 tokens);
    gpt-oss-120b's is 8100 kB and 2048 won (4096 5% slower, 1024 13%). This takes the geometric
    middle of that band, widest / sqrt(24), to the nearest power of two, which lands on both
    measured optima. Not a measurement for any other model.
    """
    kb = max(1.0, widest_row_bytes / 1024 / math.sqrt(24))
    return max(16, 2 ** round(math.log2(kb)))


def readahead_command(kb: int, where: str) -> str:
    return f"echo {kb} | sudo tee {where}"


def set_readahead(where: str, kb: int) -> tuple[bool, str]:
    """Write ``kb`` to a ``read_ahead_kb`` file. ``(True, "")`` or ``(False, why)``; never raises."""
    try:
        with open(where, "w", encoding="utf-8") as f:
            f.write(f"{int(kb)}\n")
        return True, ""
    except OSError as exc:
        return False, exc.strerror or str(exc)


# ---------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------


def meminfo(proc: str = "/proc") -> dict[str, int]:
    """``/proc/meminfo`` in bytes."""
    out = {}
    for line in (_read(os.path.join(proc, "meminfo")) or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return out


def memlock_limit(proc: str = "/proc") -> tuple[int | None, int | None] | None:
    """``(soft, hard)`` RLIMIT_MEMLOCK in bytes (None = unlimited), from ``/proc/self/limits``."""
    for line in (_read(os.path.join(proc, "self/limits")) or "").splitlines():
        if line.startswith("Max locked memory"):
            fields = line[len("Max locked memory"):].split()
            if len(fields) >= 2:
                vals = [None if v == "unlimited" else int(v) for v in fields[:2]]
                return vals[0], vals[1]
    return None


# Flash-Next under --pp-size 2 held 9 GiB of anonymous memory across both rank processes with
# the banks mapped (docs/bank-ram.md: "the rest of the process wants about 9 GiB"). The one
# host-side measurement there is, so it is taken per rank. A model with a large host-resident
# embedding or PLE table wants more; pass an explicit size then.
NONBANK_PER_RANK_BYTES = int(4.5 * GiB)
# What is left over becomes page cache, and the design leans on it: the non-resident rows a
# session keeps routing to settle there (effective disk share 3% instead of the placement's
# 12.5%). The measured 64 GB configuration (48G on a ~62 GiB MemTotal) left about 5% of
# MemTotal after the 9 GiB above, which is where this number comes from.
HEADROOM_FRACTION = 0.05
HEADROOM_MIN_BYTES = 2 * GiB


@dataclass
class AutoBankRam:
    total_bytes: int
    available: int
    mem_total: int
    nonbank: int
    headroom: int
    ranks: int

    def as_flag(self) -> str:
        """A ``--moe-bank-ram`` value parse_size reads back to (nearly) the same bytes."""
        return f"{self.total_bytes / GiB:.2f}G"

    def reason(self) -> str:
        return (
            f"--moe-bank-ram auto: {self.total_bytes / GiB:.1f} GiB for the banks across "
            f"{self.ranks} rank{'s' if self.ranks != 1 else ''} = MemAvailable "
            f"{self.available / GiB:.1f} GiB - {self.nonbank / GiB:.1f} GiB for the rest of the "
            f"server ({NONBANK_PER_RANK_BYTES / GiB:.1f} per rank) - {self.headroom / GiB:.1f} GiB "
            f"left as page cache for the non-resident rows; pass a size to override"
        )


def auto_bank_ram(mem: dict[str, int], ranks: int) -> AutoBankRam:
    """``--moe-bank-ram auto``: MemAvailable, less the rest of the server, less a page cache margin.

    MemAvailable rather than MemTotal so that whatever else is running right now is left
    alone, and read once in the launcher so every rank splits the same number.
    """
    avail, total = mem.get("MemAvailable"), mem.get("MemTotal")
    if not avail or not total:
        raise ValueError("--moe-bank-ram auto: MemAvailable/MemTotal not readable; pass a size")
    ranks = max(1, int(ranks))
    nonbank = NONBANK_PER_RANK_BYTES * ranks
    headroom = max(HEADROOM_MIN_BYTES, int(total * HEADROOM_FRACTION))
    budget = avail - nonbank - headroom
    if budget < GiB:
        raise ValueError(
            f"--moe-bank-ram auto: MemAvailable is {avail / GiB:.1f} GiB, which leaves "
            f"{budget / GiB:.1f} GiB for the banks after {nonbank / GiB:.1f} GiB for the rest of "
            f"the server and {headroom / GiB:.1f} GiB of page cache margin. Free some memory or "
            f"pass a size"
        )
    return AutoBankRam(budget, avail, total, nonbank, headroom, ranks)


# ---------------------------------------------------------------------------------------
# who has the file open
# ---------------------------------------------------------------------------------------


def mapped_by(path: str, proc: str = "/proc") -> tuple[list[tuple[int, str]], int]:
    """``([(pid, comm)], unreadable)``: processes that map ``path``, and how many could not be
    checked (other users' maps are not readable without privilege)."""
    target = os.path.realpath(path)
    found, unreadable = [], 0
    me = os.getpid() if proc == "/proc" else None
    try:
        pids = [int(p) for p in os.listdir(proc) if p.isdigit()]
    except OSError:
        return found, 0
    for pid in pids:
        if pid == me:
            continue
        try:
            with open(os.path.join(proc, str(pid), "maps"), encoding="utf-8", errors="replace") as f:
                hit = any(line.rstrip("\n").endswith(target) or line.rstrip("\n").endswith(target + " (deleted)")
                          for line in f)
        except PermissionError:
            unreadable += 1
            continue
        except OSError:
            continue  # exited meanwhile
        if hit:
            found.append((pid, _read(os.path.join(proc, str(pid), "comm")) or "?"))
    return found, unreadable


# ---------------------------------------------------------------------------------------
# the read benchmark
# ---------------------------------------------------------------------------------------


def physical_cores(sys: str = "/sys") -> int:
    """Distinct (package, core) pairs, which is what the CPU executor runs one thread per."""
    base = os.path.join(sys, "devices/system/cpu")
    cores = set()
    try:
        names = os.listdir(base)
    except OSError:
        names = []
    for name in names:
        if re.fullmatch(r"cpu\d+", name):
            topo = os.path.join(base, name, "topology")
            pkg, core = _read(os.path.join(topo, "physical_package_id")), _read(os.path.join(topo, "core_id"))
            if pkg is not None and core is not None:
                cores.add((pkg, core))
    return len(cores) or max(1, (os.cpu_count() or 2) // 2)


def random_row_read(path: str, row_bytes: int, threads: int, seconds: float) -> float:
    """GB/s (1e9) reading whole expert rows at random offsets from ``threads`` threads, O_DIRECT.

    The shape of a decode step's cold reads: whole rows, anywhere in the file, from every CPU
    worker at once. O_DIRECT so the page cache can neither answer nor be disturbed -- nothing
    is dropped and nothing is written. Raises OSError where the filesystem refuses O_DIRECT
    (tmpfs, 9p), because a buffered number there would be a measurement of RAM.
    """
    o_direct = getattr(os, "O_DIRECT", None)
    if o_direct is None:
        raise OSError("O_DIRECT is not available on this platform")
    size = os.path.getsize(path)
    row = max(4096, -(-int(row_bytes) // 4096) * 4096)
    if size < 2 * row:
        raise OSError(f"{path} is too small to benchmark ({size} bytes)")
    stop = time.perf_counter() + seconds
    done = [0] * threads
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | o_direct)
        except OSError as exc:
            errors.append(exc)
            return
        buf = mmap.mmap(-1, row)  # page aligned, which O_DIRECT needs
        rnd = random.Random(i)
        try:
            while time.perf_counter() < stop:
                off = rnd.randrange(0, (size - row) // 4096) * 4096
                done[i] += os.preadv(fd, [buf], off)
        except OSError as exc:
            errors.append(exc)
        finally:
            buf.close()
            os.close(fd)

    started = time.perf_counter()
    pool = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if errors and not sum(done):
        raise errors[0]
    return sum(done) / (time.perf_counter() - started) / 1e9


# ---------------------------------------------------------------------------------------
# the startup complaint
# ---------------------------------------------------------------------------------------


@dataclass
class Storage:
    path: str
    mount: Mount | None
    fs_verdict: tuple[str, str]
    device: BlockDevice | None
    dev: str | None
    wsl: bool


def probe_storage(path: str, proc: str = "/proc", sys: str = "/sys") -> Storage:
    real = os.path.realpath(nearest_existing(path))
    mounts = parse_mountinfo(_read(os.path.join(proc, "self/mountinfo")) or "")
    mount = mount_of(real, mounts)
    verdict = filesystem_verdict(mount) if mount else ("ok", "unknown")
    # mountinfo's major:minor is the superblock's device, the same st_dev a stat of any file
    # on it returns -- and it comes from the proc root, so a fake tree can supply it
    dev = mount.dev if mount else device_of(real)
    device = block_device(dev, mount.source if mount else "", sys) if dev else None
    return Storage(real, mount, verdict, device, dev, is_wsl(proc))


def storage_warnings(path: str, proc: str = "/proc", sys: str = "/sys") -> list[str]:
    """One line per thing about where ``path`` is that makes --moe-bank-ram slow. Empty = nothing
    known to be wrong (not the same as measured to be right)."""
    if not os.path.isdir(os.path.join(proc, "self")):
        return []
    s = probe_storage(path, proc, sys)
    out = []
    level, why = s.fs_verdict
    if level != "ok":
        where = os.path.abspath(os.path.expanduser(path))
        out.append(f"--moe-bank-ram: the bank directory {where} is on {why}. Move it with --moe-bank-dir")
        return out  # the device under a 9p or network mount is not the one that matters
    d = s.device
    if d is None or d.virtual:
        return out  # WSL2 / VM disks say nothing true about the drive behind them
    if d.transport == "usb":
        out.append(
            f"--moe-bank-ram: the bank file is on a USB disk ({d.name}); decode reads expert rows "
            f"from it at random every token. Put it on an internal NVMe with --moe-bank-dir"
        )
    elif d.rotational:
        out.append(
            f"--moe-bank-ram: the bank file is on a rotating disk ({d.name}); random row reads "
            f"there cost seconds per token. Put it on an NVMe with --moe-bank-dir"
        )
    elif d.transport == "sata":
        out.append(
            f"--moe-bank-ram: the bank file is on a SATA device ({d.name}); measured, a SATA SSD "
            f"adds about 300 ms per decode step even at 64 GB. An NVMe is what this was built for"
        )
    return out
