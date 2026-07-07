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
    get_or_create_scratch_manager,
    gms_use_mem_pool,
    is_scratch,
)
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path, is_scratch_kv_enabled
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    GMS_TAGS,
    get_gms_lock_mode,
    get_gms_ro_connect_timeout_ms,
)
from gpu_memory_service.integrations.vllm.model_loader import (
    get_imported_weights_bytes,
    get_mx_load_context,
    has_pending_gms_write,
    profile_before_gms_write_publication,
    publish_pending_gms_write,
    register_gms_loader,
    run_gms_load_phase,
)
from gpu_memory_service.integrations.vllm.patches import (
    apply_scratch_kv_patches,
    patch_memory_snapshot,
)

logger = logging.getLogger(__name__)

# Trigger model loader registration and utility patches on import
register_gms_loader()

# Apply core utility patches (always needed for GMS)
patch_empty_cache()
patch_memory_snapshot()

# Apply scratch-KV patches when DYN_GMS_SCRATCH_KV_ENABLED is set
apply_scratch_kv_patches()

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
        # Parent will set device again (harmless) and do memory checks
        super().init_device()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """
        Determine actual available memory for the engine.

        During a failover scenario, this function may be called while there is an active engine colocated on the same device.
        We want our assessment to ignore the kv cache allocation of the active engine if there is one.
        """
        # A directly launched snapshot saver blocks on this writer's RW lock.
        # Publish only after profiling, but before KV allocation/final capture.
        return profile_before_gms_write_publication(
            self._determine_available_memory_before_gms_publish
        )

    def _determine_available_memory_before_gms_publish(self) -> int:
        if not is_scratch_kv_enabled():
            return super().determine_available_memory()

        import vllm.envs as envs
        from vllm.config import CUDAGraphMode
        from vllm.platforms import current_platform

        # Capture this before determine_available_memory() publishes the write.
        # A pending first writer's GMS MemPool and private rebound allocations
        # are both visible in PyTorch's absolute peak. An RO import's GMS
        # mappings are not, so only those imported bytes need adding below.
        has_pending_write = has_pending_gms_write()

        torch.cuda.reset_peak_memory_stats()
        self.model_runner.profile_run()
        torch.cuda.synchronize()
        torch_peak = torch.cuda.max_memory_allocated()

        cudagraph_memory_estimate = 0
        if (
            current_platform.is_cuda()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )
        self.cudagraph_memory_estimate = cudagraph_memory_estimate

        invisible_weights_memory = (
            0 if has_pending_write else get_imported_weights_bytes()
        )
        non_kv_cache_memory = torch_peak + invisible_weights_memory

        projected_available = (
            self.requested_memory
            - non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )
        self.available_kv_cache_memory_bytes = int(projected_available)

        msg = (
            "[GMS] projected available memory "
            "%.2f GiB (requested=%.2f GiB, non_kv=%.2f GiB, "
            "torch_peak=%.2f GiB, invisible_weights=%.2f GiB, "
            "cudagraph_estimate=%.2f GiB, cudagraph_applied=%.2f GiB)"
            % (
                projected_available / (1 << 30),
                self.requested_memory / (1 << 30),
                non_kv_cache_memory / (1 << 30),
                torch_peak / (1 << 30),
                invisible_weights_memory / (1 << 30),
                cudagraph_memory_estimate / (1 << 30),
                cudagraph_memory_estimate_applied / (1 << 30),
            )
        )
        logger.info(msg)
        print(msg, flush=True)

        return int(projected_available)

    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate KV cache backing.

        In scratch-KV mode the tensors are allocated over scratch-aliased
        backing client-side; wake_up drops scratch backing and installs fresh
        server backing via the standard reallocate+remap path. With
        enable_sleep_mode the manager connects RW at init and allocates real
        backing immediately.
        """
        # EngineCore can skip determine_available_memory for models with no KV
        # cache. Publish before connector setup, allocation, or warm-up.
        publish_pending_gms_write()

        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized

        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        device = self.local_rank
        socket = get_socket_path(device, "kv_cache")
        if is_scratch_kv_enabled():
            # Client-local scratch only — no GMS server session at init.
            # wake_up will connect RW and allocate fresh server backing.
            get_or_create_scratch_manager(socket, device, tag="kv_cache")
            with gms_use_mem_pool("kv_cache", torch.device(f"cuda:{device}")):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        elif self.vllm_config.model_config.enable_sleep_mode:
            get_or_create_gms_client_memory_manager(
                socket,
                device,
                mode=RequestedLockType.RW,
                tag="kv_cache",
            )
            with gms_use_mem_pool("kv_cache", torch.device(f"cuda:{device}")):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def load_model(self, *args, **kwargs) -> None:
        """Load model with corrected memory accounting.

        After the parent loads the model, we correct the model_memory_usage
        to reflect the actual bytes imported from GMS (not the delta measured
        by vLLM's memory tracking). All ranks exchange completion status before
        returning so a peer failure releases every unpublished writer.
        """
        run_gms_load_phase(
            lambda: self._load_model_before_gms_coordination(*args, **kwargs)
        )

    def _load_model_before_gms_coordination(self, *args, **kwargs) -> None:
        """Run the complete local vLLM load and post-loader accounting."""
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
            # load-time memory excluded from committed GMS bytes. vLLM uses
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
        on unmapped GMS memory). For both managers: unmap_all_vas + abort.
        Symmetric for regular and deferred-KV — unmap_all_vas walks both
        _mappings and _scratch_mappings, releasing physical and preserving VA
        reservations. Wake reconnects and rebuilds via the standard
        prepare_scratch_for_reallocation → reallocate → remap pipeline.
        """
        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Pause MX serving before GMS unmap
        mx_ctx = get_mx_load_context()
        if mx_ctx is not None:
            pause_serving(mx_ctx)

        for tag in ("weights", "kv_cache"):
            manager = get_gms_client_memory_manager(tag)
            assert manager is not None, f"GMS {tag} client is not initialized"
            assert not manager.is_unmapped, f"GMS {tag} is already unmapped"
            manager.unmap_all_vas()
            manager.abort()

        gc.collect()
        torch.cuda.empty_cache()

        free_bytes_after, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after
        logger.info(
            "Sleep freed %.2f GiB, %.2f GiB still in use.",
            freed_bytes / (1 << 30),
            used_bytes / (1 << 30),
        )

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """vLLM wake implementation with GMS integration."""
        if tags is None:
            tags = list(GMS_TAGS)

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

        if "kv_cache" in tags:
            kv_cache_manager = get_gms_client_memory_manager("kv_cache")
            assert kv_cache_manager is not None, (
                "GMS kv_cache client is not initialized"
            )
            # Capture scratch state BEFORE the flip so we know whether to
            # move bookkeeping and whether to replay the deferred NIXL
            # registration.
            was_scratch = is_scratch(kv_cache_manager)
            assert kv_cache_manager.is_unmapped, "GMS kv_cache is not unmapped"
            kv_cache_manager.connect(RequestedLockType.RW)
            if was_scratch:
                # Move scratch bookkeeping from _scratch_mappings into _mappings
                # as preserved-VA records and flip subsequent allocations on
                # this mempool to server-backed create_mapping.
                kv_cache_manager.prepare_scratch_for_reallocation()
            kv_cache_manager.reallocate_all_handles(tag="kv_cache")
            kv_cache_manager.remap_all_vas()
            self.model_runner.post_kv_cache_wake_up()
            if was_scratch:
                self._register_kv_caches_with_nixl()

    def _register_kv_caches_with_nixl(self) -> None:
        """Fire the NixlConnector KV-cache registration after deferred KV swap.

        During scratch phase the patches.patch_register_kv_caches gate intercepts
        register_kv_caches(dict) and stashes the dict on the connector as
        self._scratch_kv_pending. We replay that here, NOT
        self.model_runner.kv_caches — the latter is the list-of-tensors view
        set by vLLM, and NixlConnector.register_kv_caches does kv_caches.values()
        which requires the dict form.

        Imports from the package root (vllm.distributed.kv_transfer) — the
        kv_connector.v1.base re-exports were unreliable across vLLM versions
        and a silent ImportError here would make this a no-op, leaving
        NixlConnectorWorker.kv_topo=None and crashing on the first
        scheduler tick.
        """
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            return
        group = get_kv_transfer_group()
        pending = getattr(group, "_scratch_kv_pending", None)
        if not pending:
            # Nothing was stashed — either no deferred registration, or a
            # non-NixlConnector connector that didn't hit the patched path.
            return
        group.register_kv_caches(pending)
        # Drop the stash so a second call is a no-op.
        try:
            delattr(group, "_scratch_kv_pending")
        except AttributeError:
            pass
        logger.info(
            "[GMS] Registered %d kv_cache tensors with KV transfer group",
            len(pending),
        )

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
            return gms_use_mem_pool("kv_cache", torch.device("cuda", self.local_rank))
        return super()._maybe_get_memory_pool_context(tag)
