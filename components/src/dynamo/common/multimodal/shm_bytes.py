# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-IPC-namespace byte transport shared by processed kwargs and raw inputs.

The producer owns unlink. Readers copy into private writable storage before closing.
Using mmap for readers avoids Python resource_tracker unlinking a producer-owned
segment at consumer exit (including Python versions without SharedMemory.track).
"""

import mmap
import multiprocessing.shared_memory as shm
import os
import re
import time
import uuid


def write_shm_bytes(data: bytes) -> shm.SharedMemory:
    segment = shm.SharedMemory(
        name=f"dynamo_mm_{uuid.uuid4().hex}", create=True, size=max(1, len(data))
    )
    try:
        segment.buf[: len(data)] = data
    except BaseException:
        segment.close()
        segment.unlink()
        raise
    return segment


def read_shm_bytes(name: str, size: int) -> bytearray:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or size < 0:
        raise ValueError("invalid shared-memory byte descriptor")
    # POSIX SharedMemory uses this mount on supported Linux deployments.
    fd = os.open(f"/dev/shm/{name}", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if os.fstat(fd).st_size != max(1, size):
            raise ValueError("shared-memory segment size does not match descriptor")
        with mmap.mmap(fd, max(1, size), access=mmap.ACCESS_READ) as view:
            data = bytearray()
            # Bound transient copies and GIL hold times for async callers using
            # a worker thread. A single large bytes/bytearray copy holds the GIL.
            for offset in range(0, size, 1024 * 1024):
                data.extend(view[offset : min(size, offset + 1024 * 1024)])
                time.sleep(0)
            return data
    finally:
        os.close(fd)
