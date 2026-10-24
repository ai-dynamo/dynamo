# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start one GMS V1 process per GPU, each able to open only its own GPU.

The NVIDIA driver lets a process use GPU N only while it holds an open
/dev/nvidiaN, and libcuda initializes every GPU whose node it can open. A
process that can open just its own node therefore starts CUDA several times
faster, and cannot reach another rank's GPU.

Run as root, each rank runs as its own user whose group alone may open its GPU
node in this container's /dev, serves from a directory only it can enter, and
cannot regain privileges. Otherwise each rank confines itself with Landlock
before it starts CUDA (``confine_from_env``); its sockets are then not private.

Ranks follow PCI bus order, the order the engine sees with
CUDA_DEVICE_ORDER=PCI_BUS_ID. A rank's GPU is its only CUDA device, ordinal 0.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROC_GPUS = Path("/proc/driver/nvidia/gpus")
_DEV = Path("/dev")
_DOMAINS = ("weights", "kv_cache")
_RANK_UID_BASE = 20000
# Set for a rank that must confine itself: the minor of its GPU node.
CONFINE_ENV = "GMS_V1_CONFINE_GPU_MINOR"


@dataclass(frozen=True)
class Gpu:
    bus_id: str
    minor: int
    uuid: str


def node_gpus(proc_gpus: Path = _PROC_GPUS, dev: Path = _DEV) -> list[Gpu]:
    """GPUs this container can open, in PCI bus order, without starting CUDA."""
    gpus = []
    for entry in proc_gpus.iterdir():
        info = dict(
            line.split(":", 1)
            for line in (entry / "information").read_text().splitlines()
            if ":" in line
        )
        minor = int(info["Device Minor"])
        if os.access(dev / f"nvidia{minor}", os.R_OK | os.W_OK):
            gpus.append(Gpu(entry.name.lower(), minor, info["GPU UUID"].strip()))
    return sorted(gpus, key=lambda gpu: gpu.bus_id)


def start_ranks(
    gpus: list[Gpu],
    socket_dir: str,
    command: Callable[[int], list[str]],
    dev: Path = _DEV,
) -> list[subprocess.Popen]:
    """Start ``command(rank)`` for each GPU, confined to that GPU.

    Rank r serves from ``socket_dir``/rank-r, linked from the names clients
    use in ``socket_dir``.
    """
    as_root = os.geteuid() == 0
    if as_root:
        # Before any rank starts, so none finds a node it may still open.
        for rank, gpu in enumerate(gpus):
            node = dev / f"nvidia{gpu.minor}"
            os.chown(node, 0, _RANK_UID_BASE + rank)
            os.chmod(node, 0o660)
    processes = []
    for rank, gpu in enumerate(gpus):
        rank_dir = _expose_rank_sockets(socket_dir, rank)
        env = {**os.environ, "GMS_SOCKET_DIR": rank_dir}
        if as_root:
            process = _start_as_rank_user(command(rank), env, rank, rank_dir)
        else:
            process = subprocess.Popen(
                command(rank), env={**env, CONFINE_ENV: str(gpu.minor)}
            )
        logger.info(
            "Started GMS V1 rank=%d gpu=%s bus=%s pid=%d",
            rank,
            gpu.uuid,
            gpu.bus_id,
            process.pid,
        )
        processes.append(process)
    if as_root:
        # Only the supervisor may change the links clients follow.
        os.chmod(socket_dir, 0o755)
    return processes


def _expose_rank_sockets(socket_dir: str, rank: int) -> str:
    rank_dir = os.path.join(socket_dir, f"rank-{rank}")
    os.makedirs(rank_dir, exist_ok=True)
    for domain in _DOMAINS:
        name = f"gms_{rank}_{domain}.sock"
        link = os.path.join(socket_dir, name)
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(os.path.join(f"rank-{rank}", name), link)
    return rank_dir


def _start_as_rank_user(
    command: list[str], env: dict[str, str], rank: int, rank_dir: str
) -> subprocess.Popen:
    uid = _RANK_UID_BASE + rank
    home = f"/tmp/gms-rank-{rank}"
    for path in (rank_dir, home):
        os.makedirs(path, exist_ok=True)
        os.chown(path, uid, uid)
        os.chmod(path, 0o700)
    return subprocess.Popen(
        command,
        env={**env, "HOME": home},
        cwd=home,
        user=uid,
        group=uid,
        extra_groups=[],
        # No setuid program (sudo, nvidia-modprobe) may hand the rank root back.
        preexec_fn=_set_no_new_privs,
    )


def confine_from_env() -> None:
    """Confine this process to its GPU if its supervisor asked for it.

    Call before any thread starts: Landlock confines the calling thread and
    the threads and processes it starts afterwards.
    """
    minor = os.environ.get(CONFINE_ENV)
    if minor is not None:
        confine_to_gpu(int(minor))


# Landlock, an unprivileged LSM. x86_64 syscall numbers and uapi constants.
_LANDLOCK_CREATE_RULESET, _LANDLOCK_ADD_RULE, _LANDLOCK_RESTRICT_SELF = 444, 445, 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_ACCESS_FS_WRITE_FILE, _ACCESS_FS_READ_FILE = 1 << 1, 1 << 2
_ACCESS_FS_IOCTL_DEV = 1 << 15  # Landlock ABI 5
_PR_SET_NO_NEW_PRIVS = 38
_GPU_NODE = re.compile(r"nvidia(\d+)$")


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def confine_to_gpu(minor: int, dev: Path = _DEV) -> None:
    """Let this thread, and all it starts, open no GPU node in ``dev`` but its own.

    Every other path stays open for reading and writing.
    """
    libc = ctypes.CDLL(None, use_errno=True)

    def check(result: int, operation: str) -> int:
        if result < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"{operation}: {os.strerror(errno)}")
        return result

    abi = check(
        libc.syscall(
            _LANDLOCK_CREATE_RULESET,
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        ),
        "Landlock",
    )
    handled = _ACCESS_FS_WRITE_FILE | _ACCESS_FS_READ_FILE
    if abi >= 5:
        handled |= _ACCESS_FS_IOCTL_DEV
    attr = ctypes.c_uint64(handled)
    ruleset = check(
        libc.syscall(
            _LANDLOCK_CREATE_RULESET,
            ctypes.byref(attr),
            ctypes.c_size_t(ctypes.sizeof(attr)),
            ctypes.c_uint32(0),
        ),
        "landlock_create_ruleset",
    )

    def allow(path: Path) -> None:
        if path.is_symlink():  # covered through its target, e.g. /dev/stderr
            return
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = _PathBeneathAttr(handled, fd)
            check(
                libc.syscall(
                    _LANDLOCK_ADD_RULE,
                    ruleset,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(rule),
                    ctypes.c_uint32(0),
                ),
                f"landlock_add_rule {path}",
            )
        finally:
            os.close(fd)

    try:
        # Allow everything beside the path from / down to dev, then dev's
        # entries but the other GPUs' nodes.
        dev = dev.resolve()
        for child in (*reversed(dev.parents[:-1]), dev):
            for entry in child.parent.iterdir():
                if entry != child:
                    allow(entry)
        for entry in dev.iterdir():
            match = _GPU_NODE.match(entry.name)
            other_gpu = match is not None and int(match.group(1)) != minor
            if entry.name != "nvidia-caps" and not other_gpu:
                allow(entry)
        check(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
        check(
            libc.syscall(_LANDLOCK_RESTRICT_SELF, ruleset, ctypes.c_uint32(0)),
            "landlock_restrict_self",
        )
    finally:
        os.close(ruleset)


def _set_no_new_privs() -> None:
    if ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
