# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make gpu_memory_service logs visible inside the vLLM worker subprocess.

vLLM configures logging (``vllm/logger.py``) by attaching a handler only to the
``"vllm"`` logger and leaving the root logger unconfigured. In the vLLM worker
subprocess that is the *only* logging config that runs, so ``gpu_memory_service``
records — which inherit root's default WARNING level and have no handler up their
chain — are dropped at INFO/DEBUG (only WARNING+ escape via Python's last-resort
handler). GMS runs plenty of INFO in the worker (model load, KV scratch
allocation, sleep/wake), so that output silently disappears.

Fix: attach vLLM's own handler to the ``gpu_memory_service`` logger at vLLM's
level, with propagation disabled so it does not double-log in processes where the
root logger is already configured (e.g. the engine main process). Idempotent.
This mirrors ``modelexpress.configure_vllm_logging()``, the sibling integration's
equivalent shim.
"""

from __future__ import annotations

import logging
import sys

_GMS_LOGGING_CONFIGURED = False


def configure_gms_worker_logging() -> None:
    """Route ``gpu_memory_service`` logs through vLLM's handler. Idempotent."""
    global _GMS_LOGGING_CONFIGURED
    if _GMS_LOGGING_CONFIGURED:
        return

    # Importing vllm.logger runs vLLM's logging config, so the "vllm" logger's
    # handler exists regardless of import ordering. Match its level too.
    level: int | str = logging.INFO
    try:
        import vllm.envs as envs
        import vllm.logger  # noqa: F401  (import for side-effect: configures logging)

        level = envs.VLLM_LOGGING_LEVEL
    except Exception:
        pass

    gms_logger = logging.getLogger("gpu_memory_service")
    vllm_handlers = logging.getLogger("vllm").handlers
    if vllm_handlers:
        for handler in vllm_handlers:
            if handler not in gms_logger.handlers:
                gms_logger.addHandler(handler)
    elif not gms_logger.handlers:
        # vLLM logging not configured (e.g. VLLM_CONFIGURE_LOGGING=0) — fall back
        # to a plain stdout handler so GMS logs are not lost.
        gms_logger.addHandler(logging.StreamHandler(sys.stdout))

    gms_logger.setLevel(level)
    gms_logger.propagate = False
    _GMS_LOGGING_CONFIGURED = True
