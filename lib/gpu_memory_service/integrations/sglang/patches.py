# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific patches for GPU Memory Service integration.

- patch_torch_memory_saver: Routes weights and kv_cache to GMS
- patch_model_runner: Fixes memory accounting with pre-loaded weights
- patch_static_state_for_gms: No-ops named-buffer export/import (GMS preserves them)
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager
from typing import Optional

import gpu_memory_service.integrations.sglang as gms_sglang
import torch
from gpu_memory_service.integrations.sglang.memory_saver import (
    GMSMemorySaverImpl,
    get_gms_memory_saver_impl,
)

logger = logging.getLogger(__name__)

_torch_memory_saver_patched = False
_model_runner_patched = False
_static_state_patched = False
_kv_pool_geometry_patched = False


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
            # In GMS mode we install only the strict GMS implementation:
            # weights + kv_cache go through GMS, generic unsupported tags stay
            # no-ops/warnings, and cuda_graph remains unsupported.
            # Get device from torch.cuda.current_device() (already set by SGLang)
            device_index = torch.cuda.current_device()

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
        """Patch memory baseline for SGLang old/new init_memory_pool signatures."""
        impl = get_gms_memory_saver_impl()
        preloaded_weights_gib = 0.0
        if impl is not None:
            preloaded_weights_gib = impl.preloaded_weights_bytes / (1 << 30)

        if preloaded_weights_gib > 0 and memory_arg_name in (
            "pre_model_load_memory",
            "total_gpu_memory",
        ):
            if args:
                old_value = args[0]
                new_value = (
                    old_value + preloaded_weights_gib
                    if isinstance(old_value, (int, float))
                    else old_value
                )
                args = (new_value,) + args[1:]
            elif memory_arg_name in kwargs:
                old_value = kwargs[memory_arg_name]
                new_value = (
                    old_value + preloaded_weights_gib
                    if isinstance(old_value, (int, float))
                    else old_value
                )
                kwargs = dict(kwargs)
                kwargs[memory_arg_name] = new_value
            else:
                old_value = None
                new_value = None

            if isinstance(old_value, (int, float)) and isinstance(
                new_value, (int, float)
            ):
                logger.info(
                    "[GMS] Adjusted %s for preloaded weights: "
                    "%.2f GiB + %.2f GiB = %.2f GiB",
                    memory_arg_name,
                    old_value,
                    preloaded_weights_gib,
                    new_value,
                )
            else:
                logger.info(
                    "[GMS] Could not adjust %s for preloaded weights; value=%r",
                    memory_arg_name,
                    old_value,
                )
        elif impl is not None and impl.imported_weights_bytes > 0:
            if preloaded_weights_gib > 0:
                logger.info(
                    "[GMS] Leaving %s unchanged; unsupported SGLang "
                    "init_memory_pool signature for preloaded weights",
                    memory_arg_name,
                )
            else:
                logger.info(
                    "[GMS] Leaving %s unchanged; weights were loaded by this process",
                    memory_arg_name,
                )

        return original_init_memory_pool(self, *args, **kwargs)

    ModelRunner.init_memory_pool = patched_init_memory_pool
    ModelRunner._gms_patched = True
    _model_runner_patched = True
    logger.info("[GMS] Patched ModelRunner.init_memory_pool")


def patch_shared_kv_pool_geometry() -> None:
    """Make shared-GMS SGLang workers agree on KV page geometry.

    SGLang sizes KV from a local free-memory profile. With GMS shared KV,
    the first worker owns the persistent physical pool and later workers
    reattach to it. Later workers must therefore use the first worker's page
    count instead of their smaller post-attach free-memory profile.
    """
    global _kv_pool_geometry_patched
    if _kv_pool_geometry_patched:
        return

    try:
        from gpu_memory_service.integrations.common.kv_lease_client import (
            kv_leases_enabled,
            resolve_kv_lease_namespace_total_blocks,
            resolve_lease_device,
        )
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )
    except ImportError:
        logger.warning("[GMS] Could not import SGLang KV geometry hooks", exc_info=True)
        return

    if hasattr(
        mixin.ModelRunnerKVCacheMixin,
        "_gms_shared_kv_pool_geometry_patched",
    ):
        _kv_pool_geometry_patched = True
        return

    original_resolve = mixin.ModelRunnerKVCacheMixin._resolve_memory_pool_config

    def patched_resolve_memory_pool_config(self, pre_model_load_memory):
        config = original_resolve(self, pre_model_load_memory)

        shared_kv = os.environ.get("GMS_SGLANG_SHARED_KV", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not shared_kv or not kv_leases_enabled("sglang"):
            return config

        page_size = int(self.server_args.page_size)
        proposed_pages = int(config.max_total_num_tokens) // page_size
        if proposed_pages <= 0:
            return config

        device_idx = _resolve_shared_kv_geometry_device(self, resolve_lease_device)
        model_id = (
            getattr(self.server_args, "served_model_name", None)
            or getattr(self.server_args, "model_path", None)
            or "model"
        )
        dynamo_namespace = (
            os.environ.get("DYN_NAMESPACE")
            or os.environ.get("DYN_PARENT_DGD_K8S_NAME")
            or "default"
        )
        suffix = (
            f"shared:{dynamo_namespace}:{self.__class__.__name__}:"
            f"model{model_id}:page{page_size}"
        )
        namespace, total_blocks = resolve_kv_lease_namespace_total_blocks(
            "sglang",
            device_idx,
            total_blocks=proposed_pages + 1,
            namespace_suffix=suffix,
            reserved_blocks=[0],
        )
        target_pages = int(total_blocks) - 1
        if target_pages == proposed_pages:
            return config

        target_tokens = target_pages * page_size
        configurator = create_memory_pool_configurator(self)
        adjusted = configurator.calculate_pool_sizes_from_max_tokens(
            target_tokens,
            page_size,
        )
        adjusted.max_running_requests = self._resolve_max_num_reqs(
            adjusted.max_total_num_tokens
        )
        adjusted.mem_fraction_static = self.server_args.mem_fraction_static
        logger.info(
            "[GMS] Adjusted SGLang shared KV geometry from %d pages to %d "
            "pages (namespace=%s)",
            proposed_pages,
            target_pages,
            namespace,
        )
        return adjusted

    mixin.ModelRunnerKVCacheMixin._resolve_memory_pool_config = (
        patched_resolve_memory_pool_config
    )
    mixin.ModelRunnerKVCacheMixin._gms_shared_kv_pool_geometry_patched = True
    _kv_pool_geometry_patched = True
    logger.info("[GMS] Patched SGLang shared KV pool geometry")


def _resolve_shared_kv_geometry_device(runner, resolve_lease_device_fn) -> int:
    explicit = os.environ.get("GMS_SGLANG_KV_LEASE_DEVICE")
    if explicit is not None:
        try:
            return int(explicit)
        except ValueError:
            logger.warning(
                "Ignoring invalid GMS_SGLANG_KV_LEASE_DEVICE=%r for SGLang KV geometry",
                explicit,
            )

    gpu_id = getattr(runner, "gpu_id", None)
    if gpu_id is not None:
        try:
            return int(gpu_id)
        except (TypeError, ValueError):
            logger.debug("Unable to parse SGLang runner.gpu_id=%r", gpu_id)

    return int(resolve_lease_device_fn("GMS_SGLANG_KV_LEASE_DEVICE"))


_serving_timeout_patched = False


def patch_serving_collective_timeout_for_gms() -> None:
    """Tighten the NCCL collective watchdog to the serving timeout once SGLang is past
    warmup. Model load + CUDA-graph capture (the heavy/slow collectives) happen in
    ``Scheduler.__init__``, BEFORE ``run_event_loop``; wrapping ``run_event_loop`` entry
    therefore applies the low serving timeout only after warmup, in every rank's
    scheduler process — the precise post-warmup hook for SGLang, analogous to vLLM's
    ``GMSWorker.compile_or_warm_up_model``. No grace-delay heuristic, so a tight 2-3s
    serving timeout can never fire during warmup. No-op unless
    DYN_GMS_SERVING_NCCL_TIMEOUT_S>0 (checked inside ``tighten_now``).
    """
    global _serving_timeout_patched
    if _serving_timeout_patched:
        return
    try:
        from sglang.srt.managers.scheduler import Scheduler
    except ImportError:
        logger.debug(
            "[GMS] Could not import SGLang Scheduler, skipping serving-timeout patch"
        )
        return
    if getattr(Scheduler, "_gms_serving_timeout_patched", False):
        _serving_timeout_patched = True
        return

    original_run_event_loop = Scheduler.run_event_loop

    def patched_run_event_loop(self, *args, **kwargs):
        # First (and only) entry == post-warmup, pre-traffic: tighten now.
        try:
            from gpu_memory_service.common.serving_timeout import tighten_now

            tighten_now()
        except Exception:
            logger.debug("[GMS serving-timeout] sglang tighten failed", exc_info=True)
        return original_run_event_loop(self, *args, **kwargs)

    Scheduler.run_event_loop = patched_run_event_loop
    Scheduler._gms_serving_timeout_patched = True
    _serving_timeout_patched = True
    logger.info(
        "[GMS serving-timeout] patched SGLang Scheduler.run_event_loop "
        "(post-warmup collective-timeout tighten)"
    )


def patch_static_state_for_gms() -> None:
    """No-op SGLang's _export/_import_static_state when using GMS.

    SGLang's release_memory_occupation clones every named buffer through the
    default CUDA allocator, then restores them during resume_memory_occupation.
    GMS preserves the same VAs across unmap/remap, so this static-state backup
    is unnecessary and can fail after VMM remap. This patch must run inside the
    scheduler child process, where the weight updater module is imported.
    """
    import importlib

    global _static_state_patched
    logger.info(
        "[GMS] patch_static_state_for_gms called (pid=%d, already_patched=%s)",
        os.getpid(),
        _static_state_patched,
    )
    if _static_state_patched:
        return

    def _export_noop(model):
        """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
        return dict(buffers=[])

    def _import_noop(model, static_params):
        """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
        pass

    module_names = (
        "sglang.srt.managers.scheduler_components.weight_updater",
        "sglang.srt.managers.scheduler_update_weights_mixin",
    )
    patched_modules: list[str] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            logger.debug(
                "[GMS] %s unavailable; static-state patch skipped", module_name
            )
            continue

        if not hasattr(module, "_export_static_state") or not hasattr(
            module, "_import_static_state"
        ):
            logger.debug(
                "[GMS] %s has no static-state helpers; patch skipped",
                module_name,
            )
            continue

        module._export_static_state = _export_noop
        module._import_static_state = _import_noop
        patched_modules.append(module_name)

    if patched_modules:
        _static_state_patched = True
        logger.info(
            "[GMS] Patched SGLang static-state helpers -> no-op modules=%s pid=%d",
            patched_modules,
            os.getpid(),
        )
        return

    logger.info("[GMS] no SGLang static-state helper module available to patch")
