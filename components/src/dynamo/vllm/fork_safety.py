# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What threads a process has, for code that is about to fork it.

A forked child keeps only the forking thread, so a lock any other thread holds
stays locked in the child.
"""

import os
import threading
import time

# A vLLM dependency (humming) compiles in the background when it is imported
# and keeps reaper threads for a few seconds; forking waits this long for such
# threads to finish.
THREAD_SETTLE_SECONDS = 30.0


def other_python_threads() -> list[str]:
    """Names of the running Python threads besides the calling one."""
    current = threading.current_thread()
    return [t.name for t in threading.enumerate() if t is not current]


def wait_for_python_threads(timeout: float) -> list[str]:
    """Wait for the other Python threads to finish; return those still running."""
    deadline = time.monotonic() + timeout
    while other_python_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    return other_python_threads()


def os_thread_names() -> list[str]:
    """Names of the process's OS threads, native ones included."""
    names = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as comm:
                names.append(comm.read().strip())
        except OSError:  # the thread exited while listing
            continue
    return names
