# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in jemalloc preload for Dynamo Python entry points.

Entry points call ``maybe_preload_jemalloc`` before importing the Dynamo runtime.
This module uses only the standard library so importing it does not load the
Rust extension.
"""

import ctypes.util
import os
import sys

JEMALLOC_ENV = "DYN_JEMALLOC"


def _enabled(name: str) -> bool:
    # Match env_bool without importing Dynamo utilities before the re-exec.
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def maybe_preload_jemalloc(alias: str | None = None) -> None:
    """Re-exec the current process with jemalloc preloaded when requested.

    Set DYN_JEMALLOC, or the entry point's ``alias`` variable, to 1, true, or yes
    (case-insensitive). LD_PRELOAD takes effect at process start, so re-exec
    once. Child processes inherit the preload. If the library is missing or
    exec fails, warn and continue with the current allocator.
    """
    enabled_by = next(
        (name for name in (JEMALLOC_ENV, alias) if name and _enabled(name)), None
    )
    if enabled_by is None:
        return
    existing = os.environ.get("LD_PRELOAD", "")
    if any(
        os.path.basename(entry).startswith("libjemalloc")
        for entry in existing.replace(":", " ").split()
    ):
        return  # already configured (or we already re-exec'd)

    lib = ctypes.util.find_library("jemalloc")
    if not lib:
        # Logging is not configured yet, so write the warning to stderr.
        print(
            f"WARNING: {enabled_by} is enabled but libjemalloc was not found "
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
            f"WARNING: {enabled_by} is enabled but re-exec with jemalloc failed: "
            f"{exc}; continuing with the current allocator.",
            file=sys.stderr,
        )
