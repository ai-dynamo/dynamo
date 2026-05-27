# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler-process Python GC tuning controlled by environment variables.

Under high concurrency (thousands of in-flight streaming coroutines),
GC sweeps can pause the handler process for hundreds of ms to several
seconds while holding the GIL, which can push the system into a
lower-throughput equilibrium it cannot recover from during a load burst.

Controlled by env var ``DYN_WORKER_DISABLE_GC``:

- ``0`` (or unset): no change — stock CPython GC behavior.
- ``1``: ``gc.collect()`` then ``gc.disable()`` at handler startup.
  Reference counting still frees most objects; reference cycles will
  leak (fine for a bounded-lifetime process, may accumulate over days
  in long-running servers).
"""

import gc
import logging
import os

logger = logging.getLogger(__name__)

_ENV_VAR = "DYN_WORKER_DISABLE_GC"


def configure_gc(worker_role: str) -> None:
    """Apply the GC setting selected by ``DYN_WORKER_DISABLE_GC``.

    Call this once at the start of a worker's init function. ``worker_role``
    (e.g., ``"decode"``, ``"prefill"``, ``"encode"``) is included in the log
    message for visibility in multi-process deployments.
    """
    env_value = os.environ.get(_ENV_VAR, "0").lower().strip()
    if env_value == "0":
        return
    if env_value == "1":
        gc.collect()
        gc.disable()
        logger.info("GC disabled for %s handler (%s=1)", worker_role, _ENV_VAR)
        return
    logger.warning(
        "Unrecognized %s=%r (expected 0 or 1); leaving GC unchanged",
        _ENV_VAR,
        env_value,
    )
