#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import ctypes.util
import os
import sys


def _maybe_preload_jemalloc() -> None:
    """Enable jemalloc before importing the frontend runtime.

    Set DYN_FRONTEND_JEMALLOC to 1, true, or yes (case-insensitive).
    LD_PRELOAD takes effect at process start, so re-exec once. Child processes
    inherit the preload. If the library is missing or exec fails, warn and
    continue with the current allocator.
    """
    # Match env_bool without importing Dynamo utilities before the re-exec.
    if os.environ.get("DYN_FRONTEND_JEMALLOC", "").lower() not in ("1", "true", "yes"):
        return
    existing = os.environ.get("LD_PRELOAD", "")
    if any(
        os.path.basename(entry).startswith("libjemalloc")
        for entry in existing.replace(":", " ").split()
    ):
        return  # already configured (or we already re-exec'd)

    lib = ctypes.util.find_library("jemalloc")
    if not lib:
        # Requested but unavailable: warn and stay on glibc rather than fail to
        # start. (Logging isn't configured this early, so write to stderr.)
        print(
            "WARNING: DYN_FRONTEND_JEMALLOC is enabled but libjemalloc was not found "
            "(install libjemalloc2); continuing with the default allocator.",
            file=sys.stderr,
        )
        return

    env = os.environ.copy()
    env["LD_PRELOAD"] = f"{lib}:{existing}" if existing else lib
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execve(sys.executable, sys.orig_argv, env)
    except OSError as exc:
        print(
            f"WARNING: Could not re-exec frontend with jemalloc: {exc}; "
            "continuing with the current allocator.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    _maybe_preload_jemalloc()

    # Import the runtime only after configuring the allocator.
    from dynamo.frontend.main import main

    main()
