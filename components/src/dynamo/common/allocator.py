#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Default-on jemalloc activation via LD_PRELOAD re-exec.

Issue #9466 indicates that default allocator (glibc) incurs overhead
under long-context streaming load. Using jemalloc with heuristics config
for tail-end performance. In general, "jemalloc" should not be considered
as a silver bullet and one should have proper profiling before adopting this
utility. The Dynamo frontend is a good candidate is it is "highly concurrent,
allocation-heavy server workload".

Usage: a dynamo Python entry point calls `maybe_reexec_with_jemalloc()`
at the top of its `__main__.py`.

Note: If re-execution happens, import before `maybe_reexec_with_jemalloc()`
can be wasted startup overhead. Turn to lazy import if that is critical.

Override via env:

* ``DYN_NO_JEMALLOC`` — if this env var is set to *any* value (including empty),
  jemalloc is disabled and the default (glibc) allocator is used. Unset to enable.
* ``LD_PRELOAD=…libjemalloc.so.2`` — pre-set externally; the helper detects
  this and skips the re-exec. Recommended when profiling to avoid the re-exec:

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
        python -m dynamo.frontend ...
"""

from __future__ import annotations

import logging
import os
import sys

from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging()

# FIXME: may be different paths based on platform.
JEMALLOC_SO = "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
DEFAULT_MALLOC_CONF = (
    "tcache:true,tcache_max:32768,dirty_decay_ms:5000,muzzy_decay_ms:5000"
)
_REEXEC_SENTINEL = "_DYN_JEMALLOC_REEXECED"


def maybe_reexec_with_jemalloc(module_name: str) -> None:
    """Re-exec the current process with LD_PRELOAD=libjemalloc.so.2 set.

    Idempotent and safe to call from any dynamo Python entry point. Does
    nothing (and returns) if any of the following holds:

    * ``DYN_NO_JEMALLOC`` is set to any value (presence-based, including empty).
    * ``LD_PRELOAD`` already contains ``libjemalloc``.
    * The re-exec sentinel env var is set (i.e. we already re-execed once).
    * ``libjemalloc.so.2`` is not installed on the system.

    Otherwise, logs a one-line activation message and re-execs the process
    via :func:`os.execvpe`, preserving ``sys.argv[1:]`` and relaunching as
    ``python -m <module_name>``.

    Args:
        module_name: Dotted module name of the caller's entry point (e.g.
            ``"dynamo.frontend"``). Used so the re-exec re-invokes
            ``python -m <module_name>`` with the same arguments. Pass the
            caller's ``__spec__.parent`` or hard-code the dotted name.
    """
    if "DYN_NO_JEMALLOC" in os.environ:
        logger.info("jemalloc disabled by DYN_NO_JEMALLOC; using default allocator")
        return

    if os.environ.get(_REEXEC_SENTINEL) == "1":
        logger.info("jemalloc active (post-re-exec)")
        return

    if "libjemalloc" in os.environ.get("LD_PRELOAD", ""):
        logger.info("jemalloc already active via pre-set LD_PRELOAD; skipping re-exec")
        return

    if not os.path.exists(JEMALLOC_SO):
        logger.warning(
            f"{JEMALLOC_SO} not found — running with default (glibc) allocator. "
            "Install libjemalloc2 to enable, or set DYN_NO_JEMALLOC=1 to silence."
        )
        return

    logger.info(
        "defaulting to jemalloc via LD_PRELOAD (re-execing process). "
        "Set DYN_NO_JEMALLOC=1 to disable. When profiling, pre-set LD_PRELOAD "
        f"to {JEMALLOC_SO} yourself to skip this re-exec."
    )
    env = dict(os.environ)
    existing = env.get("LD_PRELOAD", "").strip(":")
    env["LD_PRELOAD"] = JEMALLOC_SO + (":" + existing if existing else "")
    env.setdefault("MALLOC_CONF", DEFAULT_MALLOC_CONF)
    env[_REEXEC_SENTINEL] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", module_name, *sys.argv[1:]],
        env,
    )
