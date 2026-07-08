#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os
import sys


def _maybe_preload_jemalloc() -> None:
    """Opt-in: re-exec the frontend under jemalloc.

    The frontend's hot path under load is high-churn per-request allocation
    (Vec::from_iter in the tokenizer id->token and serde_json serialization
    paths); glibc ptmalloc spends a large share of loaded CPU on arena
    management there, which jemalloc's per-thread caches avoid.

    LD_PRELOAD is honored only at process start, so we set it and re-exec this
    module once. Scoped to the frontend process only. Enable with
    DYN_FRONTEND_JEMALLOC=1; requires libjemalloc2 (the runtime container already
    ships it). No-op if jemalloc is already preloaded or the library is absent.
    """
    if os.environ.get("DYN_FRONTEND_JEMALLOC") != "1":
        return
    if "jemalloc" in os.environ.get("LD_PRELOAD", ""):
        return  # already active (or we already re-exec'd)

    import ctypes.util

    lib = ctypes.util.find_library("jemalloc")
    if not lib:
        # Requested but unavailable: warn and stay on glibc rather than fail to
        # start. (Logging isn't configured this early, so write to stderr.)
        print(
            "WARNING: DYN_FRONTEND_JEMALLOC=1 but libjemalloc was not found "
            "(install libjemalloc2); continuing with the default allocator.",
            file=sys.stderr,
        )
        return

    existing = os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = f"{lib}:{existing}" if existing else lib
    os.execv(sys.executable, [sys.executable, "-m", "dynamo.frontend", *sys.argv[1:]])


_maybe_preload_jemalloc()

from dynamo.frontend.main import main  # noqa: E402  (must follow the re-exec guard)

if __name__ == "__main__":
    main()
