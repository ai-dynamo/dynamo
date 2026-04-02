# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific patches for GPU Memory Service integration.

- patch_torch_memory_saver: Routes to GMS hybrid implementation
- patch_model_runner: Fixes memory accounting with pre-loaded weights
- patch_static_state_for_gms: No-ops named-buffer export/import (GMS preserves them)
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_torch_memory_saver_patched = False
_model_runner_patched = False
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

        if hook_mode is None or hook_mode == "gms":
            # Use our GPU Memory Service implementation
            from gpu_memory_service.integrations.sglang.memory_saver import (
                GMSMemorySaverImpl,
            )
            from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

            # Get device from torch.cuda.current_device() (already set by SGLang)
            device_index = torch.cuda.current_device()

            # Create underlying torch impl for non-GMS tags.
            torch_impl = _TorchMemorySaverImpl(hook_mode="torch")

            # Read lock mode set by setup_gms() (defaults to RW_OR_RO)
            from gpu_memory_service.integrations.sglang import _gms_lock_mode

            gms_impl = GMSMemorySaverImpl(
                torch_impl=torch_impl,
                device_index=device_index,
                mode=_gms_lock_mode,
            )

            # Set _impl directly (accessible via gms_impl property)
            self._impl = gms_impl
            logger.info(
                "[GMS] Using GMS mode (device=%d, mode=%s)",
                device_index,
                gms_impl.get_mode(),
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
    from gpu_memory_service.integrations.sglang.memory_saver import GMSMemorySaverImpl

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
    """Patch SGLang's ModelRunner to fix memory accounting with pre-loaded weights.

    SGLang 0.5.9 passes a startup free-memory snapshot as total_gpu_memory into
    init_memory_pool(). In GMS read mode, imported weights can already occupy GPU
    memory, so that snapshot is lower than physical device capacity and the KV cache
    overhead term is under-reserved.
    """
    global _model_runner_patched

    if _model_runner_patched:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError:
        logger.warning("[GMS] Could not import ModelRunner, skipping patch")
        return

    if hasattr(ModelRunner, "_gms_patched"):
        return

    original_init_memory_pool = ModelRunner.init_memory_pool
    memory_arg_name = next(
        (
            name
            for name in inspect.signature(original_init_memory_pool).parameters
            if name != "self"
        ),
        None,
    )

    def patched_init_memory_pool(self, *args, **kwargs):
        """Patch init_memory_pool for SGLang versions that use total_gpu_memory.

        SGLang's KV cache formula uses total_gpu_memory as the baseline:
        rest_memory = available - total*(1-mem_fraction).
        Replace that baseline with physical device capacity when GMS imported
        weights are already resident. Newer SGLang versions changed this API, so
        only rewrite the old total_gpu_memory parameter shape.
        """
        from gpu_memory_service.integrations.sglang.memory_saver import (
            get_gms_memory_saver_impl,
        )

        impl = get_gms_memory_saver_impl()
        if impl is not None and impl.get_imported_weights_bytes() > 0:
            total_memory_gib = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory / (1 << 30)
            if memory_arg_name == "total_gpu_memory":
                if args:
                    old_value = args[0]
                    args = (total_memory_gib,) + args[1:]
                elif memory_arg_name in kwargs:
                    old_value = kwargs[memory_arg_name]
                    kwargs = dict(kwargs)
                    kwargs[memory_arg_name] = total_memory_gib
                else:
                    old_value = None
                logger.info(
                    "[GMS] Adjusted total_gpu_memory: %s -> %.2f GiB",
                    (
                        f"{old_value:.2f} GiB"
                        if isinstance(old_value, (int, float))
                        else "<missing>"
                    ),
                    total_memory_gib,
                )
            elif memory_arg_name is not None:
                logger.info(
                    "[GMS] Leaving %s unchanged in patched init_memory_pool",
                    memory_arg_name,
                )

        return original_init_memory_pool(self, *args, **kwargs)

    ModelRunner.init_memory_pool = patched_init_memory_pool
    ModelRunner._gms_patched = True
    _model_runner_patched = True
    logger.info("[GMS] Patched ModelRunner.init_memory_pool")


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
        from sglang.srt.managers import scheduler_update_weights_mixin as _mixin

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
