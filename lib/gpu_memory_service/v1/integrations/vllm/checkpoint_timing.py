# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timing wrappers for the engine-side checkpoint restore path.

Promotion after a failover has a phase nobody can see. Between the standby
taking the failover lock and GMS beginning its wake, a loaded engine spends
around 28 seconds emitting nothing at all; idle it spends about 0.3. Working the
gap out by subtraction points at ``checkpoint_restore()``, because the wake
itself is instrumented and reports ~10.7 s in both cases -- but subtraction is
not measurement, and the path underneath forks several ways:

    handlers.py                 checkpoint_restore()
    parallel_state.py:2072        torch.accelerator.synchronize()      barrier
                                  _apply_to_device_comms(...)
                                  torch.accelerator.synchronize()      barrier
    cuda_communicator.py:596        checkpoint_restore_fi_ar_workspaces()
                                    all2all_manager.checkpoint_restore()

Two engines share the same GPUs here, so any of those could be paying for the
dying engine's in-flight work: the barriers by draining the device, the
workspace calls by contending for MNNVL multicast and IPC setup. They imply
different fixes, so the split has to be measured rather than argued.

These wrappers are installed only when DYN_GMS_CHECKPOINT_TIMING is truthy, and
they add a monotonic clock read and a log line around calls that already take
seconds. Every wrapper delegates unconditionally and swallows nothing -- if
instrumentation fails to install, the engine runs exactly as before.
"""

from __future__ import annotations

import os
from time import monotonic

from vllm.logger import init_logger

# Must be a "vllm.*" logger. vLLM only attaches handlers to that hierarchy, so a
# stdlib logging.getLogger(__name__) under gpu_memory_service.* is created fine,
# wraps fine, and then silently drops every record -- which is exactly what
# happened: the wrappers installed and ran while emitting nothing at all.
logger = init_logger("vllm.gpu_memory_service.v1.timing")

ENV_ENABLED = "DYN_GMS_CHECKPOINT_TIMING"
_installed = False


def _enabled() -> bool:
    return os.environ.get(ENV_ENABLED, "").strip().lower() in ("1", "true", "yes", "on")


def _rank() -> int:
    """Best-effort rank for log correlation; never raises."""
    for var in ("VLLM_DP_RANK", "RANK", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val and val.isdigit():
            return int(val)
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.current_device()
    except Exception:
        pass
    return -1


def _timed(fn, label):
    def wrapper(*args, **kwargs):
        t0 = monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            logger.info(
                "[ckpt-timing] %s rank=%d elapsed=%.3fs", label, _rank(), monotonic() - t0
            )

    wrapper.__name__ = getattr(fn, "__name__", label)
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    wrapper.__wrapped__ = fn
    return wrapper


def install() -> None:
    """Wrap the checkpoint-restore path. Idempotent, and safe to call blind."""
    global _installed
    if _installed or not _enabled():
        return
    _installed = True

    # parallel_state: the two synchronize() barriers plus the per-communicator
    # fan-out live inside this one function, so timing it bounds the whole
    # worker-side restore.
    try:
        from vllm.distributed import parallel_state

        parallel_state.checkpoint_restore_distributed_state = _timed(
            parallel_state.checkpoint_restore_distributed_state,
            "parallel_state.checkpoint_restore_distributed_state",
        )
    except Exception:
        logger.warning("[ckpt-timing] could not wrap parallel_state", exc_info=True)

    # cuda_communicator: splits FlashInfer all-reduce workspaces from the
    # all2all manager. These are the two candidates that would contend with the
    # dying engine for MNNVL setup.
    try:
        from vllm.distributed.device_communicators import cuda_communicator

        cuda_communicator.CudaCommunicator.checkpoint_restore = _timed(
            cuda_communicator.CudaCommunicator.checkpoint_restore,
            "CudaCommunicator.checkpoint_restore",
        )
    except Exception:
        logger.warning("[ckpt-timing] could not wrap CudaCommunicator", exc_info=True)

    try:
        from vllm.distributed.device_communicators import flashinfer_all_reduce

        flashinfer_all_reduce.checkpoint_restore_fi_ar_workspaces = _timed(
            flashinfer_all_reduce.checkpoint_restore_fi_ar_workspaces,
            "flashinfer.checkpoint_restore_fi_ar_workspaces",
        )
        # cuda_communicator imports the symbol inside the function body, so the
        # module-level rebind above is what it will pick up on the next call.
    except Exception:
        logger.warning("[ckpt-timing] could not wrap flashinfer_all_reduce", exc_info=True)

    # all2all: wrap every manager subclass that defines its own restore, since
    # which one is live depends on --all2all-backend.
    try:
        from vllm.distributed.device_communicators import all2all

        for name in dir(all2all):
            cls = getattr(all2all, name, None)
            if not isinstance(cls, type) or not name.endswith("All2AllManager"):
                continue
            fn = cls.__dict__.get("checkpoint_restore")
            if fn is not None:
                setattr(cls, "checkpoint_restore", _timed(fn, f"{name}.checkpoint_restore"))
    except Exception:
        logger.warning("[ckpt-timing] could not wrap all2all managers", exc_info=True)

    logger.info("[ckpt-timing] instrumentation installed (rank=%d)", _rank())
