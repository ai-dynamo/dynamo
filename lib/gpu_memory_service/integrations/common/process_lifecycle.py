# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linux process-lifetime fencing for shared GPU-memory writers."""

from __future__ import annotations

import ctypes
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1


def arm_parent_death_signal(signum: int = signal.SIGKILL) -> None:
    """Terminate this process if the process that created it exits.

    A GMS failover flock lives in the Dynamo leader, while EngineCore and CUDA
    writers are descendants. Without this fence, killing only the leader can
    release ownership while an orphaned child still writes shared HBM.
    """
    if sys.platform != "linux":
        raise RuntimeError("GMS shared-KV failover requires Linux PDEATHSIG support")

    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, int(signum), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

    # Close the race where the parent dies between getppid() and prctl().
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signum)
