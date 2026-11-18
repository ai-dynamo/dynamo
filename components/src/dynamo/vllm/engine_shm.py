# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove the POSIX shared memory a dead vLLM engine left in ``/dev/shm``.

vLLM's message queues are anonymous ``SharedMemory`` segments
(``/dev/shm/psm_*``). Their creator unlinks them from a finalizer, and Python's
resource tracker unlinks them if the creator dies first. Neither runs when the
engine's whole process group is killed, tracker included, which is how the
launcher ends what is left of a dead or hung engine. The segments then stay.

``/dev/shm`` is shared by every container of a pod while ``/proc`` is not, and
the names carry no owner: a sweep by name, or of segments no visible process
maps, would also delete the live queues of an engine in another container. So
only the process that started an engine cleans up after it. It records the
segments its own engine's processes map while they run, and once they are gone
it removes the recorded segments that are still the same files. A segment the
engine maps is taken to be the engine's own: vLLM does not attach to another
engine's queues.
"""

import logging
import os
from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

SHM_DIR = "/dev/shm"
# Python's name for anonymous SharedMemory segments, which vLLM's queues use.
SEGMENT_PREFIX = "psm_"


def process_group_members(pgid: int, proc_root: str = "/proc") -> list[int]:
    """Pids of the live processes (zombies included) in process group ``pgid``."""
    members = []
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        try:
            # Bytes: a command name need not be valid UTF-8.
            with open(os.path.join(proc_root, entry, "stat"), "rb") as f:
                stat = f.read()
        except OSError:  # exited since the listing, or not readable
            continue
        # "pid (comm) state ppid pgrp ...": comm may contain spaces and ")".
        fields = stat[stat.rindex(b")") + 2 :].split()
        if int(fields[2]) == pgid:
            members.append(int(entry))
    return members


def mapped_segments(
    pids: Iterable[int], proc_root: str = "/proc", shm_dir: str = SHM_DIR
) -> dict[str, int]:
    """Name and inode of the ``psm_*`` segments in ``shm_dir`` that ``pids`` map."""
    prefix = os.path.join(shm_dir, SEGMENT_PREFIX).encode()
    segments = {}
    for pid in pids:
        try:
            with open(os.path.join(proc_root, str(pid), "maps"), "rb") as f:
                lines = f.readlines()
        except OSError:  # exited, or not ours to read
            continue
        for line in lines:
            # address perms offset dev inode [path]; a path may contain spaces.
            fields = line.rstrip(b"\n").split(maxsplit=5)
            if len(fields) < 6 or not fields[5].startswith(prefix):
                continue
            name = fields[5][len(prefix) - len(SEGMENT_PREFIX) :]
            if b"/" not in name and not name.endswith(b" (deleted)"):
                segments[os.fsdecode(name)] = int(fields[4])
    return segments


def remove_segments(
    segments: Mapping[str, int], proc_root: str = "/proc", shm_dir: str = SHM_DIR
) -> list[str]:
    """Unlink recorded segments that no process visible here still maps.

    Call after the engine's processes are gone. A segment is kept if a live
    process in this container still maps it, or if its name now belongs to
    another file (the engine unlinked its own, and the name was reused).
    Returns the names removed.
    """
    if not segments:
        return []
    pids = [int(entry) for entry in os.listdir(proc_root) if entry.isdigit()]
    in_use = mapped_segments(pids, proc_root, shm_dir)
    removed = []
    for name, inode in sorted(segments.items()):
        path = os.path.join(shm_dir, name)
        try:
            if name in in_use or os.stat(path).st_ino != inode:
                continue
            os.unlink(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            # Best effort: the replacement engine does not depend on it.
            logger.warning("Launcher: cannot remove %s: %s", path, exc)
            continue
        removed.append(name)
    return removed
