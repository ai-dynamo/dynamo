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
import sys
from contextlib import nullcontext
from typing import List, Optional

import torch
from gpu_memory_service.client.memory_manager import StaleMemoryLayoutError
from gpu_memory_service.client.torch.allocator import (
    ensure_scratch_disabled,
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
    get_or_create_scratch_manager,
    gms_use_mem_pool,
    is_scratch,
)
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path, is_scratch_kv_enabled
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import GMS_TAGS, get_gms_lock_mode
from gpu_memory_service.integrations.vllm.model_loader import register_gms_loader
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

# Import Worker after patches are applied
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


class GMSWorker(Worker):
    """vLLM Worker subclass with GMS integration."""

    def init_device(self) -> None:
        """Initialize device with early GMS connection.

        We set CUDA device and establish GMS connection BEFORE calling super()
        so that MemorySnapshot.measure can query committed bytes.
        """
        from vllm.platforms import current_platform

        # Set CUDA device first (vLLM provides self.local_rank)
        device = self.local_rank
        current_platform.set_device(torch.device(f"cuda:{device}"))

        # Establish weights GMS connection (so MemorySnapshot can query committed bytes).
        # Lock type is determined by model_loader_extra_config, set upstream by
        # configure_gms_lock_mode() in main.py.
        extra = (
            getattr(self.vllm_config.load_config, "model_loader_extra_config", {}) or {}
        )
        mode = get_gms_lock_mode(extra)
        get_or_create_gms_client_memory_manager(
            get_socket_path(device, "weights"),
            device,
            mode=mode,
            tag="weights",
        )
        # Parent will set device again (harmless) and do memory checks
        super().init_device()

    def determine_available_memory(self) -> int:
        """
        Determine actual available memory for the engine.

        During a failover scenario, this function may be called while there is an active engine colocated on the same device.
        We want our assessment to ignore the kv cache allocation of the active engine if there is one.
        """
        if not is_scratch_kv_enabled():
            return super().determine_available_memory()

        torch.cuda.reset_peak_memory_stats()
        self.model_runner.profile_run()
        torch.cuda.synchronize()
        torch_peak = torch.cuda.max_memory_allocated()

        # GMS weights mapped via cuMemMap are invisible to PyTorch's memory
        # stats on RO engines. Add them explicitly. On RW engines, torch_peak
        # already includes weights so skip to avoid double-counting.
        weights_memory = int(getattr(self.model_runner, "model_memory_usage", 0))
        if torch_peak < weights_memory:
            non_kv_cache_memory = torch_peak + weights_memory
        else:
            non_kv_cache_memory = torch_peak

        projected_available = self.requested_memory - non_kv_cache_memory

        msg = (
            "[GMS] projected available memory "
            "%.2f GiB (requested=%.2f GiB, non_kv=%.2f GiB, "
            "torch_peak=%.2f GiB, weights=%.2f GiB)"
            % (
                projected_available / (1 << 30),
                self.requested_memory / (1 << 30),
                non_kv_cache_memory / (1 << 30),
                torch_peak / (1 << 30),
                weights_memory / (1 << 30),
            )
        )
        logger.info(msg)
        print(msg, flush=True)

        return int(projected_available)

    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate KV cache backing.

        In scratch-KV mode the tensors are allocated over scratch-aliased
        backing client-side; wake_up promotes to real per-tensor backing via
        the standard reallocate+remap path. With enable_sleep_mode the manager
        connects RW at init and allocates real backing immediately.
        """
        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized

        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        device = self.local_rank
        socket = get_socket_path(device, "kv_cache")
        if is_scratch_kv_enabled():
            # Client-local scratch only — no GMS server session at init.
            # wake_up will connect RW and migrate to real backing.
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
        by vLLM's memory tracking).
        """
        super().load_model(*args, **kwargs)

        # Correct memory accounting for GMS-imported weights
        try:
            from gpu_memory_service.integrations.vllm.model_loader import (
                get_imported_weights_bytes,
            )

            imported_bytes = int(get_imported_weights_bytes())
            if (
                imported_bytes > 0
                and hasattr(self, "model_runner")
                and self.model_runner is not None
            ):
                old_usage = getattr(self.model_runner, "model_memory_usage", 0)
                self.model_runner.model_memory_usage = imported_bytes
                logger.info(
                    "[GMS] Corrected model_memory_usage: %.2f GiB -> %.2f GiB",
                    old_usage / (1 << 30),
                    imported_bytes / (1 << 30),
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
                weights_manager.connect(RequestedLockType.RO, timeout_ms=30_000)
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

        if "kv_cache" in tags:
            kv_cache_manager = get_gms_client_memory_manager("kv_cache")
            assert (
                kv_cache_manager is not None
            ), "GMS kv_cache client is not initialized"
            # Capture scratch state BEFORE the flip so we know whether to
            # migrate and whether to replay the deferred NIXL registration.
            was_scratch = is_scratch(kv_cache_manager)
            assert kv_cache_manager.is_unmapped, "GMS kv_cache is not unmapped"
            kv_cache_manager.connect(RequestedLockType.RW)
            if was_scratch:
                # Move scratch entries from _scratch_mappings into _mappings
                # as preserved-VA records, then flip routing to server-backed
                # so subsequent torch allocations on this mempool go through
                # create_mapping. Order matters: migrate first, flip second.
                kv_cache_manager.prepare_scratch_for_reallocation()
                ensure_scratch_disabled(kv_cache_manager)
            kv_cache_manager.reallocate_all_handles(tag="kv_cache")
            kv_cache_manager.remap_all_vas()
            if was_scratch:
                self._register_kv_caches_with_nixl()

            # Reinitialize FP8 KV scales if needed
            if self.cache_config.cache_dtype.startswith("fp8") and hasattr(
                self.model_runner, "init_fp8_kv_scales"
            ):
                self.model_runner.init_fp8_kv_scales()

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
