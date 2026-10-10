# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific patches for GPU Memory Service integration.

- patch_torch_memory_saver: Routes weights and kv_cache to GMS
- patch_model_runner: Fixes memory accounting with pre-loaded weights
- patch_kv_cache_sizing_for_gms: Makes the XPU KV sizing probe see GMS weights
- patch_static_state_for_gms: No-ops named-buffer export/import (GMS preserves them)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import gpu_memory_service.integrations.sglang as gms_sglang
from gpu_memory_service.integrations.sglang.memory_saver import (
    GMSMemorySaverImpl,
    get_gms_memory_saver_impl,
)

logger = logging.getLogger(__name__)

_torch_memory_saver_patched = False
_model_runner_patched = False
_kv_cache_sizing_patched = False
_static_state_patched = False


def patch_torch_memory_saver() -> None:
    """Patch torch_memory_saver to use GPU Memory Service implementation.

    This function is idempotent - calling it multiple times has no effect.
    This patch is only applied when GMSModelLoader is imported (load_format="gms").
    """
    global _torch_memory_saver_patched
    if _torch_memory_saver_patched:
        return

    try:
        import torch_memory_saver
        import torch_memory_saver.entrypoint as entrypoint_module
    except ImportError:
        logger.debug("[GMS] torch_memory_saver not installed, skipping patch")
        return

    # Store reference to original method
    original_ensure_initialized = entrypoint_module.TorchMemorySaver._ensure_initialized
    original_configure_subprocess = torch_memory_saver.configure_subprocess

    def patched_ensure_initialized(self):
        """Patched _ensure_initialized that uses GPU Memory Service implementation."""
        # Check if already initialized
        if self._impl is not None:
            logger.debug("[GMS] TorchMemorySaver already initialized, skipping")
            return

        # Check hook_mode - use GMS for None or explicit "gms"
        hook_mode = self._impl_ctor_kwargs.get("hook_mode")
        logger.info(f"[GMS] TorchMemorySaver initializing with hook_mode={hook_mode}")

        if hook_mode == "torch":
            # SGLang forces hook_mode="torch" at import time on Intel XPU
            # (srt/utils/torch_memory_saver_adapter.py) because the LD_PRELOAD
            # preload mode is CUDA/HIP-only. That leaves no way to request
            # "gms" on XPU, so claim the saver here rather than falling through
            # to upstream, which would leave GMSModelLoader with no impl and
            # fail the load. On CUDA an explicit "torch" stays a real opt-out.
            from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm_device_type

            use_gms = get_vmm_device_type() == VMMDeviceType.XPU
        else:
            use_gms = hook_mode is None or hook_mode == "gms"

        if use_gms:
            # In GMS mode we install only the strict GMS implementation:
            # weights + kv_cache go through GMS, generic unsupported tags stay
            # no-ops/warnings, and cuda_graph remains unsupported.
            # Get device from the active VMM device module (already set by SGLang)
            from gpu_memory_service.integrations.common.utils import torch_device

            device_index = torch_device().current_device()

            # Read lock mode set by setup_gms() (defaults to RW_OR_RO)
            gms_impl = GMSMemorySaverImpl(
                device_index=device_index,
                mode=gms_sglang._gms_lock_mode,
                ro_connect_timeout_ms=gms_sglang._gms_ro_connect_timeout_ms,
            )

            # Set _impl directly (accessible via gms_impl property)
            self._impl = gms_impl
            logger.info(
                "[GMS] Using GMS mode (device=%d, mode=%s)",
                device_index,
                gms_impl.allocators["weights"].granted_lock_type.name,
            )
            del self._impl_ctor_kwargs
        else:
            # Fall back to original implementation
            logger.info("[GMS] Using default torch_memory_saver hook mode")
            original_ensure_initialized(self)

    entrypoint_module.TorchMemorySaver._ensure_initialized = patched_ensure_initialized

    @contextmanager
    def patched_configure_subprocess():
        """Avoid LD_PRELOAD in GMS mode; keep upstream behavior otherwise."""
        singleton = torch_memory_saver.torch_memory_saver
        ctor_kwargs = getattr(singleton, "_impl_ctor_kwargs", None) or {}
        hook_mode = ctor_kwargs.get("hook_mode")

        if hook_mode is None or hook_mode == "gms":
            logger.info("[GMS] torch_memory_saver.configure_subprocess is a no-op")
            yield
            return

        with original_configure_subprocess():
            yield

    torch_memory_saver.configure_subprocess = patched_configure_subprocess

    # Add property to access GMS impl directly from the singleton
    @property
    def gms_impl(self) -> Optional[GMSMemorySaverImpl]:
        """Get the GMS impl if installed, None otherwise."""
        if isinstance(self._impl, GMSMemorySaverImpl):
            return self._impl
        return None

    entrypoint_module.TorchMemorySaver.gms_impl = gms_impl

    # If the singleton was already initialized before this patch ran (e.g.,
    # due to import ordering in multiprocessing spawn), reset _impl so the
    # next call to _ensure_initialized goes through the patched version and
    # creates GMSMemorySaverImpl instead of the default _TorchMemorySaverImpl.
    import torch_memory_saver

    singleton = torch_memory_saver.torch_memory_saver
    if singleton._impl is not None:
        logger.debug(
            "[GMS] TorchMemorySaver singleton already initialized, "
            "resetting to force GMS re-init on next use"
        )
        singleton._impl = None
        # The original _ensure_initialized deletes _impl_ctor_kwargs after
        # creating _impl.  Restore it so the patched version can read it.
        if not hasattr(singleton, "_impl_ctor_kwargs"):
            singleton._impl_ctor_kwargs = {}

    _torch_memory_saver_patched = True
    logger.debug("[GMS] Patched torch_memory_saver")


def patch_model_runner() -> None:
    """Patch SGLang's ModelRunner to size KV cache with GMS-resident weights.

    SGLang's KV sizing formula reserves dynamic headroom from a free-memory
    snapshot taken before its own model load. In GMS read mode, the committed
    weight handles already exist in the GMS server before that snapshot, so the
    snapshot is lower by those weights. Add just those preloaded weight bytes
    back to the baseline. Do not adjust write mode: weights are loaded after
    the snapshot there, so upstream's formula already subtracts them correctly.

    Only needed on SGLang builds that predate native preloaded-weight
    accounting. v0.5.19 and later ships ModelRunner.account_preloaded_weights(),
    which Scheduler.init_target_memory_pool() calls with the value read from
    GMSModelLoader.preloaded_weights_bytes. Applying both would add the weight
    bytes to the baseline twice and oversize the KV cache, so skip the patch
    whenever upstream owns the accounting.
    """
    global _model_runner_patched

    if _model_runner_patched:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError:
        logger.warning("[GMS] Could not import ModelRunner, skipping patch")
        return

    if hasattr(ModelRunner, "account_preloaded_weights"):
        _model_runner_patched = True
        logger.info(
            "[GMS] SGLang accounts for preloaded weights natively; "
            "skipping ModelRunner.alloc_memory_pool patch"
        )
        return

    if hasattr(ModelRunner, "_gms_patched"):
        return

    original_alloc_memory_pool = ModelRunner.alloc_memory_pool

    def patched_alloc_memory_pool(self, *args, **kwargs):
        impl = get_gms_memory_saver_impl()
        if (
            impl is not None
            and impl.preloaded_weights_bytes > 0
            and not self.__dict__.get("_gms_memory_baseline_adjusted", False)
        ):
            preloaded_weights_gib = impl.preloaded_weights_bytes / (1 << 30)
            old_value = self.pre_model_load_memory
            self.pre_model_load_memory += preloaded_weights_gib
            self._gms_memory_baseline_adjusted = True
            logger.info(
                "[GMS] Adjusted pre_model_load_memory for preloaded weights: "
                "%.2f GiB + %.2f GiB = %.2f GiB",
                old_value,
                preloaded_weights_gib,
                self.pre_model_load_memory,
            )

        return original_alloc_memory_pool(self, *args, **kwargs)

    ModelRunner.alloc_memory_pool = patched_alloc_memory_pool
    ModelRunner._gms_patched = True
    _model_runner_patched = True
    logger.info("[GMS] Patched ModelRunner.alloc_memory_pool")


def patch_kv_cache_sizing_for_gms() -> None:
    """Make the XPU KV sizing probe account for GMS-resident weights.

    On CUDA sglang's ``get_available_gpu_memory()`` calls ``torch.cuda.mem_get_info()``,
    which is device-wide, so GMS-resident weights are visible to writer and reader
    alike and no adjustment is needed.

    On XPU it returns ``total_memory - torch.xpu.memory_allocated(gpu_id)``,
    which only sees allocations made through this process's torch caching
    allocator.  A writer allocates weights through the GMS mempool and is
    therefore charged for them; a reader materialises them as VMM mappings
    outside the allocator and is not.  memory_allocated() also cannot see
    another process's allocations at all, so once the writer exits the GMS
    server still holds W bytes resident on the card.

    GMSModelLoader reports preloaded_weights_bytes = 0 on XPU: sglang's
    add-back exists to undo a depressed *pre-load* reading,
    and on XPU that reading was never depressed in the first place.

    Three modules bind the helper with a from-import, so each name is rebound
    separately: kv_cache_configurator (sizes the pool), model_runner
    (weight_load_mem_usage) and scheduler (startup_available_gpu_memory_gb).
    distributed.bootstrap is left alone -- it samples the pre-load baseline
    before the loader has connected, and that baseline should exclude W.
    """
    global _kv_cache_sizing_patched

    if _kv_cache_sizing_patched:
        return

    from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm_device_type

    if get_vmm_device_type() != VMMDeviceType.XPU:
        # CUDA's probe is device-wide; nothing to correct.
        _kv_cache_sizing_patched = True
        return

    def make_gms_aware_probe(original):
        def patched_get_available_gpu_memory(*args, **kwargs):
            available_gb = original(*args, **kwargs)

            impl = get_gms_memory_saver_impl()
            if impl is None:
                return available_gb

            # Non-zero only on the import path: _finalize_pending_write()
            # resets this to 0 after a writer commits, so a writer (which
            # reconnects as RO and is already charged for its weights) is
            # never adjusted.
            preloaded_bytes = getattr(impl, "preloaded_weights_bytes", 0)
            if preloaded_bytes <= 0:
                return available_gb

            adjusted_gb = available_gb - preloaded_bytes / (1 << 30)
            logger.info(
                "[GMS] Adjusted avail-mem probe for GMS-resident weights: "
                "%.2f GiB - %.2f GiB = %.2f GiB",
                available_gb,
                preloaded_bytes / (1 << 30),
                adjusted_gb,
            )
            return adjusted_gb

        return patched_get_available_gpu_memory

    modules = []

    try:
        from sglang.srt.mem_cache import kv_cache_configurator

        modules.append(kv_cache_configurator)
    except ImportError:
        logger.warning(
            "[GMS] Could not import kv_cache_configurator, KV pool sizing "
            "will not account for GMS-resident weights"
        )

    try:
        from sglang.srt.model_executor import model_runner

        modules.append(model_runner)
    except ImportError:
        logger.warning(
            "[GMS] Could not import model_runner, its avail-mem logs will "
            "not account for GMS-resident weights"
        )

    try:
        from sglang.srt.managers import scheduler

        modules.append(scheduler)
    except ImportError:
        logger.warning(
            "[GMS] Could not import scheduler, startup_available_gpu_memory_gb "
            "will not account for GMS-resident weights"
        )

    if not modules:
        return

    for module in modules:
        original = getattr(module, "get_available_gpu_memory", None)
        if original is None:
            # Partially initialised module (circular import) or an upstream
            # rename; leave it alone rather than guess.
            logger.warning(
                "[GMS] %s has no get_available_gpu_memory, not patching it",
                module.__name__,
            )
            continue

        module.get_available_gpu_memory = make_gms_aware_probe(original)
        logger.info("[GMS] Patched %s.get_available_gpu_memory", module.__name__)

    _kv_cache_sizing_patched = True


def patch_static_state_for_gms() -> None:
    """No-op SGLang's _export/_import_static_state when using GMS.

    SGLang's release_memory_occupation clones every named buffer via
    buffer.detach().clone() through the default CUDA allocator, then restores
    them during resume_memory_occupation.
    This patch must run inside the scheduler child process (which uses
    multiprocessing spawn).  It is triggered by the GMSModelLoader import
    in model_loader.py, which executes at module level in the child.
    """
    import os

    global _static_state_patched
    logger.info(
        "[GMS] patch_static_state_for_gms called (pid=%d, already_patched=%s)",
        os.getpid(),
        _static_state_patched,
    )
    if _static_state_patched:
        return

    try:
        try:
            # SGLang >=0.5.21.
            from sglang.srt.managers.scheduler_components import (
                weight_updater as _mixin,
            )
        except ImportError:
            # SGLang <0.5.21 kept these on the scheduler mixin module.
            from sglang.srt.managers import (
                scheduler_update_weights_mixin as _mixin,  # type: ignore[no-redef]
            )

        def _export_noop(model):
            """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
            return dict(buffers=[])

        def _import_noop(model, static_params):
            """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
            pass

        _mixin._export_static_state = _export_noop
        _mixin._import_static_state = _import_noop
        _static_state_patched = True
        logger.info(
            "[GMS] Patched _export/_import_static_state -> no-op (pid=%d)",
            os.getpid(),
        )
    except Exception:
        logger.warning(
            "[GMS] Could not patch scheduler_update_weights_mixin: ",
            exc_info=True,
        )
