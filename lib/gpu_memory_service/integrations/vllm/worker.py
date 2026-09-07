# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service Worker subclass for vLLM integration.

This module provides a custom Worker class that properly integrates with
GPU Memory Service for VA-stable weight sharing and unmap/remap functionality.

Usage:
    Set --worker-cls=gpu_memory_service.integrations.vllm.worker:GMSWorker
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from contextlib import nullcontext
from typing import List, Optional

import torch
from gpu_memory_service.client.memory_manager import StaleMemoryLayoutError
from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
    get_or_create_persistent_allocator,
    gms_use_persistent_pool,
)
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_lock_mode,
    get_gms_persistent_kv_socket,
    get_gms_ro_connect_timeout_ms,
    torch_device,
)
from gpu_memory_service.integrations.vllm.install_kv_leases import (
    install as install_kv_leases,
)
from gpu_memory_service.integrations.vllm.install_kv_leases import (
    install_engine_core_hook,
)
from gpu_memory_service.integrations.vllm.install_vmm_ipc_kv import (
    install as install_vmm_ipc_kv,
)
from gpu_memory_service.integrations.vllm.kv_identity import (
    allocation_engine_id,
    allocation_shared,
    shared_kv_enabled,
    stable_engine_id,
)
from gpu_memory_service.integrations.vllm.model_loader import (
    abort_pending_gms_write,
    get_imported_weights_bytes,
    get_mx_load_context,
    publish_pending_gms_write,
    register_gms_loader,
)
from gpu_memory_service.integrations.vllm.patches import patch_memory_snapshot
from gpu_memory_service.integrations.vllm.utils import configure_gms_worker_logging

logger = logging.getLogger(__name__)

# Keep GMS logs visible after vLLM configures subprocess logging.
configure_gms_worker_logging()

# Trigger model loader registration and utility patches on import
register_gms_loader()

# Apply core utility patches (always needed for GMS)
patch_empty_cache()
patch_memory_snapshot()
install_kv_leases()
install_engine_core_hook()
install_vmm_ipc_kv()

# Register the KV-cache GDS-direct connector under the short name so
# users can wire it via vLLM's standard --kv-transfer-config flag.
# Opt-in: the connector is only constructed if the user names it.

logger.info("[GMS] Worker module loaded - model loader registered, all patches applied")



# MX imports — only when MX_ENABLED=1 (modelexpress is an optional dependency).
# Pause/resume serving lifecycle is implemented in modelexpress.lifecycle, which
# composes publish/unpublish_metadata + register_tensors + MxClient/NIXL
# teardown into a single pause/resume pair.
if os.environ.get("MX_ENABLED", "0") == "1":
    try:
        from modelexpress import configure_vllm_logging
        from modelexpress.lifecycle import pause_serving, resume_serving

        configure_vllm_logging()
    except ImportError as e:
        raise ImportError(
            "MX_ENABLED=1 but modelexpress is not installed. "
            "Install with: pip install modelexpress"
        ) from e


def _install_gms_engine_core_sleep() -> None:
    """Install a GMS-only sleep utility that skips prefix-cache clearing.

    Bulwark startup quiesce has no user traffic to discard, and vLLM can have
    initialized internal KV blocks that make reset_prefix_cache() fail before
    the engine ever serves. This utility keeps the scheduler pause semantics
    but delegates directly to model_executor.sleep(level).
    """
    try:
        from concurrent.futures import Future

        from vllm.v1.engine.core import EngineCore
    except Exception:
        logger.debug("[GMS] EngineCore sleep utility patch skipped", exc_info=True)
        return

    if hasattr(EngineCore, "gms_sleep_no_clear"):
        return

    def gms_sleep_no_clear(self, level: int = 1, mode: str = "abort"):
        pause_future = self.pause_scheduler(mode=mode, clear_cache=False)
        if level < 1:
            return pause_future

        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future = Future()

        def pause_complete(f):
            try:
                f.result()
                future.set_result(model_executor.sleep(level))
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)

        logger.info("[GMS] Waiting for in-flight requests before no-clear sleep")
        pause_future.add_done_callback(pause_complete)
        return future

    EngineCore.gms_sleep_no_clear = gms_sleep_no_clear
    logger.info("[GMS] Installed EngineCore.gms_sleep_no_clear utility")


_install_gms_engine_core_sleep()

# Import Worker after patches are applied
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


def _get_dp_adjusted_local_rank(local_rank: int, parallel_config) -> int:
    """Return the CUDA device index vLLM will use for this worker.

    vLLM adjusts ``self.local_rank`` inside ``Worker.init_device()`` for
    intra-node data parallelism so that every local DP engine lands on a
    different GPU:

        DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK

    GMS intentionally connects before ``super().init_device()`` because the
    initial vLLM ``MemorySnapshot`` needs GMS-aware committed-byte accounting.
    That means GMS cannot observe vLLM's in-place local-rank adjustment yet, so
    duplicate the upstream calculation here and use it only for the early GMS
    socket/device selection.

    TODO: add an upstream vLLM hook/API that exposes the resolved CUDA device
    before the initial MemorySnapshot, then replace this duplicated vLLM logic.
    """
    adjusted_local_rank = local_rank
    if (
        parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
        and parallel_config.data_parallel_backend != "ray"
        and parallel_config.nnodes_within_dp == 1
    ):
        # Use local DP rank if available, otherwise use global DP rank.
        dp_local_rank = parallel_config.data_parallel_rank_local
        if dp_local_rank is None:
            dp_local_rank = parallel_config.data_parallel_index

        tp_pp_world_size = (
            parallel_config.pipeline_parallel_size
            * parallel_config.tensor_parallel_size
        )
        adjusted_local_rank += dp_local_rank * tp_pp_world_size

    return adjusted_local_rank


class GMSWorker(Worker):
    """vLLM Worker subclass with GMS integration."""

    def _determine_available_memory_with_gms_weight_accounting(self) -> int:
        """Avoid double-counting RW GMS weights during vLLM KV profiling.

        RW GMS weight allocations are visible to cudaMemGetInfo, but not to
        PyTorch's reserved-memory counters. vLLM therefore observes them as
        non-torch memory growth during profiling. Passing the same bytes as
        model_memory_usage would count them twice and can produce negative KV
        capacity on large models.
        """
        manager = get_gms_client_memory_manager("weights")
        model_runner = getattr(self, "model_runner", None)
        if (
            manager is None
            or manager.granted_lock_type != GrantedLockType.RW
            or model_runner is None
            or not hasattr(model_runner, "model_memory_usage")
        ):
            return super().determine_available_memory()

        old_usage = int(getattr(model_runner, "model_memory_usage") or 0)
        if old_usage <= 0:
            return super().determine_available_memory()

        logger.info(
            "[GMS] Suppressing %.2f GiB RW GMS weight bytes from vLLM "
            "weights_memory during KV profiling; cudaMemGetInfo already "
            "accounts for them as non-torch memory",
            old_usage / (1 << 30),
        )
        model_runner.model_memory_usage = 0
        try:
            return super().determine_available_memory()
        finally:
            model_runner.model_memory_usage = old_usage

    def init_device(self) -> None:
        """Initialize device with early GMS connection.

        We set CUDA device and establish GMS connection BEFORE calling super()
        so that MemorySnapshot.measure can query committed bytes.
        """
        from vllm.platforms import current_platform

        # Set CUDA device first. Do not mutate self.local_rank here; the parent
        # Worker will apply the same DP adjustment during super().init_device().
        device = _get_dp_adjusted_local_rank(self.local_rank, self.parallel_config)
        current_platform.set_device(torch.device(f"cuda:{device}"))

        # Establish weights GMS connection (so MemorySnapshot can query committed bytes).
        # Lock type is determined by model_loader_extra_config, set upstream by
        # configure_gms_lock_mode() in main.py.
        extra = (
            getattr(self.vllm_config.load_config, "model_loader_extra_config", {}) or {}
        )
        mode = get_gms_lock_mode(extra)
        self.gms_ro_connect_timeout_ms = get_gms_ro_connect_timeout_ms(extra)
        get_or_create_gms_client_memory_manager(
            get_socket_path(device, "weights"),
            device,
            mode=mode,
            tag="weights",
        )

        if env_enabled_by_default("GMS_VLLM_VMM_IPC_KV", default=True):
            socket = get_gms_persistent_kv_socket(device, "GMS_VLLM_VMM_IPC_SOCKET")
            engine_id = allocation_engine_id(device)
            get_or_create_persistent_allocator(
                socket,
                device,
                engine_id,
                tag="kv_pool",
                shared=allocation_shared(),
            )

        # Parent will set device again (harmless) and do memory checks
        super().init_device()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profile memory, then publish a pending first-writer layout.

        Publication is delayed until profiling completes so waiting RO
        consumers cannot attach mid-profile and perturb vLLM's accounting.
        """
        try:
            available = self._determine_available_memory_with_gms_weight_accounting()
        except BaseException:
            try:
                abort_pending_gms_write()
            except BaseException:
                logger.exception("[GMS] Failed to release pending write")
            raise
        publish_pending_gms_write()
        return available

    def _maybe_tighten_serving_collective_timeout(self) -> None:
        """Post-warmup hook: the engine is fully initialized in this rank's worker
        process, so lower the NCCL collective watchdog to the (low) serving timeout
        for fast hang detection. Only ever called after warmup actually ran, so it
        cannot fire during init/warmup. No-op unless DYN_GMS_SERVING_NCCL_TIMEOUT_S>0.
        """
        try:
            from gpu_memory_service.common.serving_timeout import (
                apply_serving_collective_timeout,
            )

            apply_serving_collective_timeout()
        except Exception:
            logger.debug("[GMS serving-timeout] vLLM tighten failed", exc_info=True)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._maybe_tighten_serving_collective_timeout()
        return result

    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate persistent KV backing after publishing pending weights."""
        # EngineCore can skip determine_available_memory for models with no
        # KV cache. Publish before connector setup, allocation, or warm-up.
        publish_pending_gms_write()

        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized

        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        device = self.local_rank
        socket = get_gms_persistent_kv_socket(device, "GMS_VLLM_VMM_IPC_SOCKET")
        engine_id = allocation_engine_id(device)
        self._gms_kv_engine_id = engine_id
        get_or_create_persistent_allocator(
            socket,
            device,
            engine_id,
            tag="kv_pool",
            shared=allocation_shared(),
        )
        self.model_runner.initialize_kv_cache(kv_cache_config)

        if self.model_config.enable_return_routed_experts:
            self.model_runner.init_routed_experts_capturer()

        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()

    def load_model(self, *args, **kwargs) -> None:
        """Load model with corrected memory accounting.

        After the parent loads the model, we correct the model_memory_usage
        to reflect the actual bytes imported from GMS (not the delta measured
        by vLLM's memory tracking).
        """
        super().load_model(*args, **kwargs)

        # Correct memory accounting for GMS-imported weights
        try:
            from gpu_memory_service.integrations.vllm.model_loader import (
                get_imported_weights_bytes,
                get_model_memory_usage_offset_bytes,
            )

            imported_weights_bytes = get_imported_weights_bytes()
            memory_usage_offset_bytes = get_model_memory_usage_offset_bytes()
            # The offset is not committed/restored GMS weight state. It is
            # load-time memory excluded from committed GMS bytes (pruned
            # load-time allocations plus private rebound clones). vLLM uses
            # model_memory_usage for KV sizing, so omitting it can allocate
            # an oversized cache.
            model_memory_usage_bytes = int(
                imported_weights_bytes + memory_usage_offset_bytes
            )
            if model_memory_usage_bytes > 0 and self.model_runner is not None:
                old_usage = getattr(self.model_runner, "model_memory_usage", 0)
                self.model_runner.model_memory_usage = model_memory_usage_bytes
                logger.info(
                    "[GMS] Corrected vLLM model_memory_usage for KV sizing: "
                    "%.2f GiB -> %.2f GiB "
                    "(weights %.2f GiB + offset %.2f GiB)",
                    old_usage / (1 << 30),
                    model_memory_usage_bytes / (1 << 30),
                    imported_weights_bytes / (1 << 30),
                    memory_usage_offset_bytes / (1 << 30),
                )
        except Exception as e:
            logger.debug("[GMS] Could not correct memory accounting: %s", e)

    def sleep(self, level: int = 1) -> None:
        """vLLM sleep implementation with GMS integration.

        Skips super().sleep() (which copies GPU buffers to CPU and segfaults
        on unmapped GMS memory). We unmap weights plus the persistent KV pool;
        GMS keeps the underlying physical KV pages alive for reconnect.
        """
        free_bytes_before = torch_device().mem_get_info()[0]

        # Pause MX serving before GMS unmap
        mx_ctx = get_mx_load_context()
        if mx_ctx is not None:
            pause_serving(mx_ctx)

        for tag in ("weights", "kv_pool"):
            manager = get_gms_client_memory_manager(tag)
            assert manager is not None, f"GMS {tag} client is not initialized"
            assert not manager.is_unmapped, f"GMS {tag} is already unmapped"
            manager.unmap_all_vas()
            manager.abort()

        gc.collect()
        torch_device().empty_cache()

        free_bytes_after, total = torch_device().mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after
        logger.info(
            "Sleep freed %.2f GiB, %.2f GiB still in use.",
            freed_bytes / (1 << 30),
            used_bytes / (1 << 30),
        )

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """vLLM wake implementation with GMS integration."""
        requested_tags = tags
        if tags is None:
            tags = ["weights", "kv_pool"]
        elif "kv_cache" in tags and "kv_pool" not in tags:
            tags = list(tags) + ["kv_pool"]

        if "weights" in tags:
            weights_manager = get_gms_client_memory_manager("weights")
            assert weights_manager is not None, "GMS weights client is not initialized"
            assert weights_manager.is_unmapped, "GMS weights are not unmapped"

            # These errors are fatal and unrecoverable in a worker subprocess:
            # the worker cannot serve requests without weights. sys.exit(1)
            # ensures clean termination so the orchestrator (K8s) can restart.
            try:
                weights_manager.connect(
                    RequestedLockType.RO,
                    timeout_ms=getattr(self, "gms_ro_connect_timeout_ms", None),
                )
                weights_manager.remap_all_vas()
            except TimeoutError:
                logger.error(
                    "Fatal: timed out waiting for GMS RO lock during remap "
                    "(GMS may be down or RW lock held indefinitely)"
                )
                sys.exit(1)
            except StaleMemoryLayoutError as e:
                logger.error(
                    "Fatal: weight layout changed while unmapped, cannot remap: %s", e
                )
                sys.exit(1)
            except ConnectionError as e:
                logger.error("Fatal: cannot connect to GMS during remap: %s", e)
                sys.exit(1)

            # Resume MX serving after GMS remap
            mx_ctx = get_mx_load_context()
            if mx_ctx is not None:
                resume_serving(mx_ctx, self.model_runner.model)

        if "kv_pool" in tags:
            kv_manager = get_gms_client_memory_manager("kv_pool")
            assert kv_manager is not None, "GMS persistent KV client is not initialized"
            assert kv_manager.is_unmapped, "GMS persistent KV is not unmapped"
            engine_id = getattr(
                self,
                "_gms_kv_engine_id",
                stable_engine_id(self.local_rank),
            )
            logger.info(
                "[GMS] vLLM KV wake_up connecting: engine_id=%s shared=%s",
                engine_id,
                shared_kv_enabled(),
            )
            kv_manager.connect(RequestedLockType.RW_PERSISTENT)
            kv_manager.remap_persistent_vas(engine_id, shared=shared_kv_enabled())
            logger.info("[GMS] vLLM KV wake_up remap done")

        if (
            requested_tags is None
            or "kv_cache" in requested_tags
            or "kv_pool" in requested_tags
        ):
            logger.info("[GMS] vLLM post_kv_cache_wake_up begin")
            self.model_runner.post_kv_cache_wake_up()
            logger.info("[GMS] vLLM post_kv_cache_wake_up done")

            # Reinitialize FP8 KV scales if needed for vLLM versions whose
            # post-wake hook does not already do it.
            if self.cache_config.cache_dtype.startswith("fp8") and hasattr(
                self.model_runner, "init_fp8_kv_scales"
            ):
                logger.info("[GMS] vLLM init_fp8_kv_scales begin")
                self.model_runner.init_fp8_kv_scales()
                logger.info("[GMS] vLLM init_fp8_kv_scales done")

    def _maybe_get_memory_pool_context(self, tag: str):
        """Route tag-scoped runtime allocations to the right allocator.

        Weight tensors are allocated explicitly in the GMS model-loader path,
        not through vLLM's tagged runtime allocator hook. For `weights` we
        therefore only suppress CuMemAllocator here so it does not interfere
        with the loader-managed GMS allocations. `kv_cache` is the tag that
        actually allocates through this hook, so it uses the dedicated GMS
        mempool.
        """
        if tag == "weights":
            logger.debug("[GMS] Skipping CuMemAllocator for weights")
            return nullcontext()
        if tag == "kv_cache":
            return gms_use_persistent_pool(
                "kv_pool", torch.device("cuda", self.local_rank)
            )
        return super()._maybe_get_memory_pool_context(tag)
