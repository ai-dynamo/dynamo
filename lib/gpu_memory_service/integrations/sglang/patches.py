# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific patches for GPU Memory Service integration.

- patch_torch_memory_saver: Routes weights and kv_cache to GMS
- patch_model_runner: Fixes memory accounting with pre-loaded weights
- patch_static_state_for_gms: No-ops named-buffer export/import (GMS preserves them)
"""

from __future__ import annotations

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
_failover_extend_warmup_patched = False


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

    # New SGLang versions ask the loader for ``preloaded_weights_bytes`` and
    # apply this correction natively. Keep the wrapper only for older versions;
    # applying both would double-count imported GMS weights and oversize KV.
    if not hasattr(ModelRunner, "account_preloaded_weights"):
        ModelRunner.alloc_memory_pool = patched_alloc_memory_pool

    ModelRunner._gms_patched = True
    _model_runner_patched = True
    logger.info("[GMS] Patched ModelRunner KV sizing")


def patch_failover_extend_warmup_for_gms() -> None:
    """Warm SGLang's first page-sized EXTEND before serving failover traffic.

    A sleeping shadow normally warms decode kernels but does not execute an
    eager EXTEND. Its first cache-hit request must still recompute one page,
    so kernel/JIT initialization can otherwise become visible as failover
    downtime. Use a one-request scratch buffer and SGLang's reserved KV page
    zero; the GMS lease allocator deliberately excludes that page from
    both allocation and persistent directory publication.

    This is intentionally bounded to one request and one KV page. SGLang's
    general FlashInfer EXTEND autotuner uses the maximum prefill shape, which
    consumes too much transient memory for a fully resident primary/shadow.
    """
    global _failover_extend_warmup_patched
    if _failover_extend_warmup_patched:
        return

    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner.base_runner import BaseRunner
    except ImportError:
        logger.debug("[GMS] SGLang runner unavailable; EXTEND warmup skipped")
        return

    if getattr(BaseRunner, "_gms_failover_extend_warmup_patched", False):
        _failover_extend_warmup_patched = True
        return

    original_warmup = BaseRunner.warmup

    def patched_warmup(self, *args, **kwargs):
        mr = self.model_runner
        was_warmed = getattr(mr, "_kernel_warmed_up", False)
        result = original_warmup(self, *args, **kwargs)
        if was_warmed or getattr(mr, "_gms_failover_extend_warmed_up", False):
            return result

        from gpu_memory_service.integrations.sglang.kv_identity import shared_kv_enabled

        failover_shadow = any(
            os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
            for name in (
                "DYN_GMS_FAILOVER_SHADOW_MODE",
                "GMS_KV_DIRECTORY_STANDBY",
            )
        )
        if not (
            shared_kv_enabled()
            and failover_shadow
            and getattr(mr, "device", None) == "cuda"
        ):
            return result

        spec_algorithm = getattr(mr, "spec_algorithm", None)
        if spec_algorithm is not None and not spec_algorithm.is_none():
            logger.info(
                "[GMS] Skipping bounded failover EXTEND warmup for speculative model"
            )
            return result

        page_size = int(getattr(mr, "page_size", 0) or 0)
        if page_size <= 0:
            logger.info(
                "[GMS] Skipping bounded failover EXTEND warmup without page size"
            )
            return result

        # These page-sized dummy controls are present in current SGLang but not
        # in the supported N-1 runner. Skip there instead of guessing at private
        # signatures or turning an optimization into a startup regression.
        import inspect

        if (
            "extend_num_tokens_per_req"
            not in inspect.signature(self._dummy_run).parameters
            or "allocate_logits_buffer"
            not in inspect.signature(self._alloc_dummy_decode_buffers).parameters
        ):
            logger.info(
                "[GMS] Skipping bounded failover EXTEND warmup on legacy runner"
            )
            return result

        # CudaGraphBufferRegistry intentionally hides its backing fields. Build
        # the smallest supported dummy buffer instead of allocating at the
        # normal maximum-prefill shape. _dummy_run uses token index zero, which
        # selects the dedicated page reserved by the GMS lease allocator.
        buffers = self._alloc_dummy_decode_buffers(
            1,
            num_tokens_per_req=page_size,
            allocate_logits_buffer=False,
        )
        buffers.out_cache_loc.zero_()
        self._dummy_run(
            batch_size=1,
            forward_mode_override=ForwardMode.EXTEND,
            buffers=buffers,
            extend_num_tokens_per_req=page_size,
        )
        self.device_module.synchronize()
        mr._gms_failover_extend_warmed_up = True
        logger.info("[GMS] Warmed one page-sized SGLang EXTEND (%d tokens)", page_size)
        return result

    BaseRunner.warmup = patched_warmup
    BaseRunner._gms_failover_extend_warmup_patched = True
    _failover_extend_warmup_patched = True
    logger.info("[GMS] Patched bounded SGLang failover EXTEND warmup")


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
            default_kv_lease_namespace_suffix,
            kv_leases_enabled,
            resolve_kv_lease_namespace_total_blocks,
            resolve_lease_device,
        )
        from gpu_memory_service.integrations.sglang.kv_identity import shared_kv_enabled
    except ImportError:
        logger.warning("[GMS] Could not import SGLang KV geometry hooks", exc_info=True)
        return

    # SGLang relocated _resolve_memory_pool_config: newer builds define it on
    # KVCacheConfigurator (sglang.srt.mem_cache.kv_cache_configurator); older
    # builds had it on ModelRunnerKVCacheMixin (sglang.srt.model_executor).
    # Patch whichever exists so shadow reattach geometry is pinned either way.
    target_cls = None
    try:
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        if hasattr(KVCacheConfigurator, "_resolve_memory_pool_config"):
            target_cls = KVCacheConfigurator
    except ImportError:
        pass
    if target_cls is None:
        try:
            from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin

            if hasattr(mixin.ModelRunnerKVCacheMixin, "_resolve_memory_pool_config"):
                target_cls = mixin.ModelRunnerKVCacheMixin
        except ImportError:
            pass
    if target_cls is None:
        logger.warning(
            "[GMS] Could not locate SGLang _resolve_memory_pool_config; shadow "
            "reattach KV geometry will NOT be pinned"
        )
        return

    if hasattr(target_cls, "_gms_shared_kv_pool_geometry_patched"):
        _kv_pool_geometry_patched = True
        return

    original_resolve = target_cls._resolve_memory_pool_config

    def patched_resolve_memory_pool_config(self, pre_model_load_memory):
        config = original_resolve(self, pre_model_load_memory)

        if not shared_kv_enabled() or not kv_leases_enabled("sglang"):
            return config

        page_size = _resolved_sglang_page_size(self)
        proposed_pages = int(config.max_total_num_tokens) // page_size
        if proposed_pages <= 0:
            return config

        device_idx = _resolve_shared_kv_geometry_device(self, resolve_lease_device)
        suffix = default_kv_lease_namespace_suffix("sglang")
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

        # Clamp the KV pool to the shared page count and re-derive every dependent
        # pool size deterministically from the pinned token count. This is
        # critical: the persistent GMS allocation covers both the token_to_kv_pool
        # (sized by max_total_num_tokens) AND the req_to_token_pool (sized by
        # max_running_requests). Both must therefore be pure functions of the
        # shared token count so the primary and every shadow build byte-identical
        # persistent allocations (else claim_persistent fails on a size mismatch).
        # Mirror SGLang's own _resolve_memory_pool_config finalize sequence.
        target_tokens = target_pages * page_size
        config.max_total_num_tokens = target_tokens
        resolve_reqs = getattr(self, "resolve_max_num_reqs", None) or getattr(
            self, "_resolve_max_num_reqs", None
        )
        if resolve_reqs is not None:
            config.max_running_requests = resolve_reqs(target_tokens)
        try:
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            configurator = create_memory_pool_configurator(self)
            finalize = getattr(configurator, "finalize_with_max_running_requests", None)
            if finalize is not None:
                config = finalize(config)
        except Exception:
            logger.debug(
                "[GMS] pool finalize after geometry clamp skipped", exc_info=True
            )
        if hasattr(config, "mem_fraction_static"):
            config.mem_fraction_static = self.server_args.mem_fraction_static
        logger.info(
            "[GMS] Adjusted SGLang shared KV geometry from %d pages to %d "
            "pages (namespace=%s)",
            proposed_pages,
            target_pages,
            namespace,
        )
        return config

    target_cls._resolve_memory_pool_config = patched_resolve_memory_pool_config
    target_cls._gms_shared_kv_pool_geometry_patched = True
    _kv_pool_geometry_patched = True
    logger.info(
        "[GMS] Patched SGLang shared KV pool geometry (%s)", target_cls.__name__
    )


def _resolved_sglang_page_size(configurator) -> int:
    """Return SGLang's effective page size across old and new APIs."""
    value = getattr(configurator, "page_size", None)
    if value is None:
        value = getattr(getattr(configurator, "server_args", None), "page_size", None)
    if value is None or int(value) <= 0:
        raise RuntimeError("SGLang did not expose a valid resolved KV page size")
    return int(value)


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
