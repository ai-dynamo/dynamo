# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service integration for TensorRT-LLM.

The supported TRT-LLM path is KVCacheManagerV2. GMS owns model weights and
coordinates V2 KV slot leases from Python; the legacy V1 connector hook that
required a patched TRT-LLM wheel is intentionally not part of this runtime path.
"""

from __future__ import annotations

import logging
from typing import Any

from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    get_gms_lock_mode as _resolve_lock_mode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "setup_gms",
    "get_gms_lock_mode",
]


def get_gms_lock_mode():
    from gpu_memory_service.integrations.trtllm.model_loader import (
        get_gms_lock_mode as _get_gms_lock_mode,
    )

    return _get_gms_lock_mode()


def setup_gms(
    model_loader_extra_config: dict[str, Any] | None = None,
    *,
    _patch_mpi_workers: bool = True,
) -> None:
    """Set up GMS integration for TensorRT-LLM. Call once before creating the engine.

    For TP>1 the engine spawns one MPI worker per rank (MpiPoolSession); those
    workers are fresh processes that do not import this module. ``_patch_mpi_workers``
    (internal) installs a hook so each spawned worker also runs ``setup_gms`` —
    the worker initializer passes ``_patch_mpi_workers=False`` to avoid re-patching.
    """
    extra = model_loader_extra_config or {}
    lock_mode = _resolve_lock_mode(extra)

    from gpu_memory_service.integrations.trtllm.model_loader import (
        patch_model_loader,
        set_gms_enabled,
        set_gms_lock_mode,
    )

    set_gms_enabled(True)
    set_gms_lock_mode(lock_mode)

    patch_empty_cache()
    from gpu_memory_service.integrations.trtllm.remote_code_cache import (
        patch_remote_code_cache,
    )

    patch_remote_code_cache()
    patch_model_loader()

    from gpu_memory_service.integrations.trtllm import install_kv_leases_v2

    install_kv_leases_v2.install()

    # Under GMS the custom IPC-workspace all-reduce strategies (AUTO/ONESHOT/TWOSHOT)
    # share buffers across ranks via CUDA/VMM IPC, which conflicts with GMS-managed
    # GPU memory and faults (CUDA 700) in the spawned MPI rank workers. NCCL all-reduce
    # uses its own buffers (no custom workspace), so force it for every AllReduce.
    _force_nccl_allreduce()

    # Multi-node: AllReduce.__init__ under the AUTO strategy probes multi-node-NVLink
    # support via pynvml.nvmlDeviceGetNvLinkCapability, which raises NVMLError on this
    # hardware and crashes model construction. MNNVL is only ever used on aarch64 + real
    # NVL domains anyway, so neutralize the probe (return False) to make AllReduce robust
    # even if the force-NCCL patch above did not bind (e.g. import-order at sitecustomize).
    _patch_disable_mnnvl()

    # Fast HANG detection (matches vllm/sglang). TRT-LLM uses an MPI/C++ NCCL communicator
    # (not torch.distributed PGs) for the executor, so the torch _set_pg_timeout path is a
    # no-op here — but TRT-LLM ships its own PyExecutor HangDetector. Lower its (fixed 300s)
    # timeout to the serving value; on_detected already does _handle_errors + shutdown which
    # releases the failover flock so the shadow takes over. Keep the torch-PG tighten too for
    # any config that runs the TorchDist (torch.distributed) communicator.
    _patch_hang_detector_timeout()
    _patch_serving_collective_timeout()

    # CUDA graphs + GMS: the V2 KV lease acquire is a cross-rank flock/shared-mmap round-trip
    # that diverges TP ranks during cuda-graph capture warmup -> captured NCCL all-reduce
    # deadlocks. Bypass leases for the (transient) warmup allocations so capture matches the
    # no-GMS path; real per-request leases (failover) are untouched.
    _patch_warmup_lease_bypass()

    # TP>1 spawns rank workers via MpiPoolSession; those fresh processes must also
    # run setup_gms or weights/KV bypass GMS. Opt-in (still being hardened: running
    # the GMS patches inside the spawned ranks currently triggers a CUDA illegal
    # access during V2 KV executor init — see GMS_TRTLLM_MPI_WORKER_SETUP notes).
    import os as _os

    if _patch_mpi_workers and _os.environ.get(
        "GMS_TRTLLM_MPI_WORKER_SETUP", ""
    ).strip().lower() not in ("", "0", "false", "no", "off"):
        _install_mpi_worker_gms(extra)

    logger.info("[GMS] TensorRT-LLM integration enabled (mode=%s)", lock_mode)


def _force_nccl_allreduce() -> None:
    """Force every TRT-LLM AllReduce module to the NCCL strategy under GMS.

    The custom IPC-workspace strategies (AUTO/ONESHOT/TWOSHOT/MNNVL) share a
    workspace buffer across TP ranks via CUDA/VMM IPC; that handle is invalid in a
    Comm-spawned MPI rank whose GPU memory is GMS-managed -> CUDA 700 illegal access
    during the autotuner warmup. NCCL all-reduce uses NCCL's own buffers (no custom
    workspace), so it is safe with GMS. ``self.strategy`` is read per forward, so
    setting it post-init is sufficient.
    """
    try:
        from tensorrt_llm._torch.distributed import ops as _ops
        from tensorrt_llm.functional import AllReduceStrategy
    except Exception:  # pragma: no cover - import shape varies by version
        logger.debug("[GMS] could not import AllReduce to force NCCL", exc_info=True)
        return
    cls = getattr(_ops, "AllReduce", None)
    if cls is None or getattr(cls, "_gms_nccl_forced", False):
        return
    orig_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        # Force NCCL as the strategy ARGUMENT, before the original __init__ runs. This
        # both avoids the custom IPC workspace (which faults against GMS memory) AND
        # skips the AUTO/MNNVL branch in __init__ whose multi-node NVLink probe
        # (MnnvlMemory.supports_mnnvl -> pynvml.nvmlDeviceGetNvLinkCapability) raises
        # NVMLError_InvalidArgument on this hardware and crashes model construction.
        # Signature: __init__(self, mapping, strategy=AUTO, dtype=None).
        if len(args) >= 2:
            args = (args[0], AllReduceStrategy.NCCL) + tuple(args[2:])
        else:
            kwargs["strategy"] = AllReduceStrategy.NCCL
        orig_init(self, *args, **kwargs)
        try:
            self.strategy = AllReduceStrategy.NCCL
        except Exception:  # pragma: no cover
            logger.debug("[GMS] failed to force NCCL on AllReduce", exc_info=True)

    cls.__init__ = patched_init
    cls._gms_nccl_forced = True
    logger.info(
        "[GMS] forced AllReduce strategy=NCCL (custom IPC all-reduce conflicts with GMS memory)"
    )


def _patch_disable_mnnvl() -> None:
    """Make TRT-LLM's multi-node-NVLink (MNNVL) support probe a safe no-op.

    ``AllReduce.__init__`` (AUTO strategy) calls ``MNNVLAllReduce.is_mnnvl`` ->
    ``MnnvlMemory.supports_mnnvl`` -> ``support_nvlink`` ->
    ``pynvml.nvmlDeviceGetNvLinkCapability``, which raises ``NVMLError_InvalidArgument``
    on this driver/GPU and crashes model construction in multi-node runs. MNNVL is only
    actually selected on aarch64 with a real NVLink domain, so forcing the probe to False
    is behaviorally safe and prevents the crash regardless of the all-reduce strategy.
    """
    try:
        from tensorrt_llm import _mnnvl_utils as _mn
    except Exception:  # pragma: no cover - import shape varies by version
        logger.debug("[GMS] could not import _mnnvl_utils to disable MNNVL", exc_info=True)
        return
    cls = getattr(_mn, "MnnvlMemory", None)
    if cls is None or getattr(cls, "_gms_mnnvl_disabled", False):
        return

    def _supports_mnnvl(*_a, **_k):
        return False

    try:
        cls.supports_mnnvl = staticmethod(_supports_mnnvl)
    except Exception:  # pragma: no cover
        logger.debug("[GMS] failed to patch supports_mnnvl", exc_info=True)
        return
    cls._gms_mnnvl_disabled = True
    logger.info("[GMS] disabled MNNVL probe (avoids nvml NvLink-capability crash, multi-node)")


def _patch_warmup_lease_bypass() -> None:
    """Make CUDA-graph capture work under GMS by allocating warmup KV slots locally.

    PyExecutor warmup (which includes CUDA-graph capture) calls
    PyTorchModelEngine.warmup; wrap it to set install_kv_leases_v2's warmup-bypass flag for
    the duration, so the patched V2 SlotAllocator skips the cross-rank GMS lease acquire and
    falls back to the engine's local allocator. That keeps warmup batch construction
    deterministic and identical on every TP rank (like the no-GMS path), preventing the
    rank-divergence -> captured-all-reduce deadlock. Per-request leases (failover) unaffected.
    """
    try:
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
        from gpu_memory_service.integrations.trtllm import install_kv_leases_v2 as _kv
    except Exception:  # pragma: no cover - import shape varies by version
        logger.debug("[GMS] could not import PyTorchModelEngine for warmup bypass", exc_info=True)
        return
    if getattr(PyTorchModelEngine, "_gms_warmup_bypass_patched", False):
        return
    orig_warmup = PyTorchModelEngine.warmup

    def patched_warmup(self, *args, **kwargs):
        _kv.set_warmup_bypass(True)
        try:
            return orig_warmup(self, *args, **kwargs)
        finally:
            _kv.set_warmup_bypass(False)

    PyTorchModelEngine.warmup = patched_warmup
    PyTorchModelEngine._gms_warmup_bypass_patched = True
    logger.info(
        "[GMS] patched PyTorchModelEngine.warmup -> local KV alloc during cuda-graph capture"
    )


def _patch_hang_detector_timeout() -> None:
    """Lower TRT-LLM's native PyExecutor HangDetector timeout to the serving timeout.

    TRT-LLM's executor runs on an MPI/C++ NCCL communicator, not torch.distributed process
    groups, so ``_set_pg_timeout`` (the vllm/sglang serving-timeout mechanism) cannot tighten
    anything here. TRT-LLM instead ships ``HangDetector`` (pyexecutor/hang_detector.py): the
    executor loop ``checkpoint()``s every iteration and, if no checkpoint lands within
    ``timeout`` seconds (a stuck collective on a dead/hung rank), fires ``on_detected`` which
    runs ``_handle_errors`` + sets the shutdown event — the process exits, the failover flock
    releases, and the shadow takes over. The timeout defaults to 300s and is not plumbed from
    any config, so override it to the (low) serving timeout. The detector is constructed AFTER
    warmup in ``PyExecutor.__init__`` and only times the serving loop (idle iterations
    checkpoint too), so a small value is warmup-safe and won't false-positive.
    """
    try:
        from gpu_memory_service.common.serving_timeout import (
            enabled,
            serving_timeout_s,
        )
    except Exception:  # pragma: no cover
        return
    if not enabled():
        return
    try:
        from tensorrt_llm._torch.pyexecutor import hang_detector as _hd
    except Exception:  # pragma: no cover - import shape varies by version
        logger.debug("[GMS serving-timeout] could not import HangDetector", exc_info=True)
        return
    cls = getattr(_hd, "HangDetector", None)
    if cls is None or getattr(cls, "_gms_timeout_patched", False):
        return
    secs = max(1, int(serving_timeout_s()))
    orig_init = cls.__init__

    def patched_init(self, timeout=None, on_detected=None):
        # Force the serving timeout regardless of the caller's default (300s).
        orig_init(self, timeout=secs, on_detected=on_detected)

    cls.__init__ = patched_init
    cls._gms_timeout_patched = True
    logger.info(
        "[GMS] patched TRT-LLM HangDetector timeout -> %ds (fast serving hang detection)",
        secs,
    )


def _patch_serving_collective_timeout() -> None:
    """Lower the collective watchdog to the serving timeout once a rank is past warmup.

    TRT-LLM's PyExecutor runs warmup (model_engine.warmup + CUDA-graph capture) inside
    ``__init__`` and then calls ``start_worker()`` as the final init step to spin up the
    executor loop thread. So wrapping ``start_worker`` gives a precise PER-RANK post-warmup
    hook (mirrors vllm's compile_or_warm_up_model / sglang's run_event_loop): the tight
    serving timeout can never fire during the generous-timeout warmup phase. The tighten
    is a no-op if torch.distributed isn't initialized, so it is safe regardless of backend.
    """
    try:
        from gpu_memory_service.common.serving_timeout import enabled, tighten_now
    except Exception:  # pragma: no cover
        return
    if not enabled():
        return
    try:
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    except Exception:  # pragma: no cover - import shape varies by version
        logger.debug("[GMS serving-timeout] could not import PyExecutor", exc_info=True)
        return
    if getattr(PyExecutor, "_gms_serving_timeout_patched", False):
        return
    orig_start_worker = PyExecutor.start_worker

    def patched_start_worker(self, *args, **kwargs):
        result = orig_start_worker(self, *args, **kwargs)
        try:
            tighten_now()
        except Exception:  # pragma: no cover
            logger.debug("[GMS serving-timeout] trtllm tighten failed", exc_info=True)
        return result

    PyExecutor.start_worker = patched_start_worker
    PyExecutor._gms_serving_timeout_patched = True
    logger.info(
        "[GMS] patched PyExecutor.start_worker to tighten serving NCCL timeout post-warmup"
    )


def _gms_worker_initializer(model_loader_extra_config: dict[str, Any] | None) -> None:
    """Run in each MPI-spawned TRT-LLM rank worker before it builds the engine."""
    # Re-apply the GMS patches in the worker process; do not re-patch the pool
    # (the worker does not spawn sub-workers).
    setup_gms(model_loader_extra_config, _patch_mpi_workers=False)


def _install_mpi_worker_gms(model_loader_extra_config: dict[str, Any] | None) -> None:
    """Make TP>1 MPI workers run the GMS integration.

    TRT-LLM's ``MpiPoolSession._start_mpi_pool`` creates the worker pool with a
    filtered env (only ``TRTLLM*``/``TLLM*``/``CUDA_*``) and no initializer, so the
    spawned rank workers — which actually own the GPUs and load weights/allocate KV —
    never run ``setup_gms``; their weights+KV then bypass the GMS pool. Wrap pool
    creation to (a) also propagate ``GMS*`` env and (b) run ``setup_gms`` in each
    worker via an initializer.
    """
    try:
        from tensorrt_llm.llmapi import mpi_session as _mpi
    except Exception:  # pragma: no cover - TRT-LLM not importable
        return
    cls = getattr(_mpi, "MpiPoolSession", None)
    if cls is None or getattr(cls, "_gms_worker_init_patched", False):
        return

    import os as _os
    import sys as _sys

    from mpi4py.futures import MPIPoolExecutor

    def _start_mpi_pool(self) -> None:
        assert not self.mpi_pool, "MPI session already started"
        # Spawned rank workers must inherit not just GMS/TRTLLM config but the full
        # toolchain env they need to load/JIT (CUDA, driver, the compiler+linker
        # search paths for flashinfer's runtime JIT, HF cache, the failover lock).
        # Without LIBRARY_PATH/PATH the flashinfer JIT link fails (-lcudart/-lcuda);
        # without LD_LIBRARY_PATH the driver/OMPI libs are missing.
        # DYN_GMS_* carries the serving-timeout config; TORCH_NCCL_* carries the NCCL
        # watchdog/teardown knobs — both must reach the spawned ranks where the engine runs.
        _prefixes = (
            "TRTLLM", "TLLM", "GMS", "CUDA", "NCCL", "HF_", "OMPI_", "MPI",
            "DYN_GMS", "TORCH_NCCL",
        )
        _exact = {
            "CUDA_HOME", "CUDA_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "PATH",
            "CPATH", "CC", "CXX", "TRITON_LIBCUDA_PATH", "PYTHONPATH",
            "FAILOVER_LOCK_PATH", "HOME", "TLLM_WORKER_USE_SINGLE_PROCESS",
        }
        env = {
            k: v
            for k, v in _os.environ.items()
            if k.startswith(_prefixes) or k in _exact
        }
        self.mpi_pool = MPIPoolExecutor(
            max_workers=self.n_workers,
            path=_sys.path,
            env=env,
            initializer=_gms_worker_initializer,
            initargs=(model_loader_extra_config or {},),
        )

    cls._start_mpi_pool = _start_mpi_pool
    cls._gms_worker_init_patched = True
    logger.info(
        "[GMS] patched MpiPoolSession: propagate GMS env + init workers (TP>1)"
    )
