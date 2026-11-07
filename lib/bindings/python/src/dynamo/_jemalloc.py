# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preload the jemalloc build that ships with Dynamo.

Wheel builds put libjemalloc_dynamo.so.2 next to this module. LD_PRELOAD only takes effect when
a process starts, so `maybe_preload()` re-executes the interpreter once with the library
prepended to LD_PRELOAD. Entrypoints call it before importing anything else; this module uses
only the standard library because it runs before dynamo._core is loaded.

Set DYN_DISABLE_JEMALLOC=1 to keep the system allocator.
"""

import os
import sys
from pathlib import Path

DISABLE_ENV = "DYN_DISABLE_JEMALLOC"
LIBRARY = Path(__file__).with_name("libjemalloc_dynamo.so.2")


def maybe_preload() -> None:
    """Re-exec with the bundled jemalloc in LD_PRELOAD, unless disabled or already preloaded.

    Warns on stderr and continues with the current allocator when the library is not bundled
    (for example a `maturin develop` build) or the exec fails.
    """
    if sys.platform != "linux":
        return
    # Match env_bool without importing Dynamo utilities, which load dynamo._core.
    if os.environ.get(DISABLE_ENV, "").lower() in ("1", "true", "yes"):
        return
    existing = os.environ.get("LD_PRELOAD", "")
    if any(
        os.path.basename(entry).startswith("libjemalloc")
        for entry in existing.replace(":", " ").split()
    ):
        return  # preloaded already: by our own re-exec, or a jemalloc the user chose

    if not LIBRARY.is_file():
        # Logging is not configured yet, so write the warning to stderr.
        print(
            f"WARNING: {LIBRARY.name} is not bundled with this Dynamo build; "
            "continuing with the system allocator.",
            file=sys.stderr,
        )
        return

    env = os.environ.copy()
    # Prepend: some images read the last LD_PRELOAD entry (TensorRT-LLM's libnixl).
    env["LD_PRELOAD"] = f"{LIBRARY}:{existing}" if existing else str(LIBRARY)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execve(sys.executable, sys.orig_argv, env)
    except OSError as exc:
        print(
            f"WARNING: could not re-exec with jemalloc: {exc}; "
            "continuing with the system allocator.",
            file=sys.stderr,
        )
