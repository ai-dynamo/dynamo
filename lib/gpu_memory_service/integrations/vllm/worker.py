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
    retarget_persistent_allocator,
)
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path, is_scratch_kv_enabled
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    env_enabled_by_default,
    get_gms_lock_mode,
    get_gms_persistent_kv_socket,
    get_gms_ro_connect_timeout_ms,
)
from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
    register_gms_gds_connector,
)
from gpu_memory_service.integrations.vllm.install_kv_leases import (
    install as install_kv_leases,
)
from gpu_memory_service.integrations.vllm.install_kv_leases import (
    install_engine_core_hook,
    install_gms_engine_core_sleep,
)
from gpu_memory_service.integrations.vllm.install_vmm_ipc_kv import (
    install as install_vmm_ipc_kv,
)
from gpu_memory_service.integrations.vllm.kv_identity import (
    allocation_engine_id,
    allocation_shared,
    private_bootstrap_kv_enabled,
    private_bootstrap_scratch_warmup_enabled,
    promotion_engine_id,
    release_private_bootstrap_kv_pool,
    shared_kv_enabled,
    stable_engine_id,
)
from gpu_memory_service.integrations.vllm.model_loader import (
    abort_pending_gms_write,
    get_imported_weights_bytes,
    get_mx_load_context,
    has_pending_gms_write,
    publish_pending_gms_write,
    register_gms_loader,
)
from gpu_memory_service.integrations.vllm.patches import patch_memory_snapshot

logger = logging.getLogger(__name__)


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
register_gms_gds_connector()

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


install_gms_engine_core_sleep()

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


def _bootstrap_memory_utilization_cap() -> float:
    for name in (
        "DYN_VLLM_GMS_BOOTSTRAP_GPU_MEMORY_UTILIZATION",
        "GMS_VLLM_BOOTSTRAP_GPU_MEMORY_UTILIZATION",
    ):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            cap = float(value)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, value)
            continue
        if cap > 0:
            return min(cap, 1.0)
    return 0.05


def _existing_shared_kv_blocks(device: int) -> Optional[int]:
    if os.environ.get("GMS_KV_LEASE_SHM_RESET", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    try:
        from gpu_memory_service.integrations.common.kv_lease_client import (
            kv_leases_enabled,
            read_any_kv_lease_namespace_total_blocks,
            read_kv_lease_namespace_total_blocks,
        )

        if not kv_leases_enabled("vllm"):
            return None
        namespace, total_blocks = read_kv_lease_namespace_total_blocks(
            "vllm",
            device,
            namespace_suffix="block-pool",
        )
        if total_blocks is None:
            namespace, total_blocks = read_any_kv_lease_namespace_total_blocks("vllm")
        if total_blocks is not None:
            logger.debug(
                "[GMS] Found existing vLLM KV lease geometry for startup cap: "
                "namespace_or_path=%s blocks=%d device=%d",
                namespace,
                int(total_blocks),
                device,
            )
    except Exception:
        logger.debug(
            "[GMS] Failed to inspect existing vLLM KV lease geometry",
            exc_info=True,
        )
        return None
    if total_blocks is None:
        return None
    return int(total_blocks)


def _maybe_cap_private_bootstrap_memory(vllm_config, device: int) -> None:
    """Reduce vLLM's startup memory target for a standby attach.

    Large-model Bulwark shadows start while the active engine already owns most
    HBM. vLLM checks ``gpu_memory_utilization`` before it reaches the GMS
    persistent KV allocation path, so a standby can fail even though it will
    later attach to the existing shared KV geometry. Private bootstrap is only
    enabled for non-primary shadows, so cap even if the primary has not yet
    published lease geometry.
    """
    if not private_bootstrap_kv_enabled():
        return
    existing_blocks = _existing_shared_kv_blocks(device)

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return
    current = getattr(cache_config, "gpu_memory_utilization", None)
    if current is None:
        return

    cap = _bootstrap_memory_utilization_cap()
    try:
        current_value = float(current)
    except (TypeError, ValueError):
        return
    if current_value <= cap:
        return

    cache_config.gpu_memory_utilization = cap
    if existing_blocks is None:
        logger.info(
            "[GMS] Capping private-bootstrap gpu_memory_utilization for "
            "cuda:%d from %.4f to %.4f for startup; shared KV geometry "
            "is not published yet",
            device,
            current_value,
            cap,
        )
    else:
        logger.info(
            "[GMS] Existing shared KV geometry detected for cuda:%d "
            "(%d blocks); capping private-bootstrap gpu_memory_utilization "
            "from %.4f to %.4f for startup",
            device,
            existing_blocks,
            current_value,
            cap,
        )


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

    def _private_bootstrap_kv_active(self) -> bool:
        return bool(
            getattr(self, "_gms_kv_private_bootstrap", False)
            or private_bootstrap_kv_enabled()
        )

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
        self._gms_kv_private_bootstrap = private_bootstrap_kv_enabled()
        self._gms_kv_scratch_warmup = private_bootstrap_scratch_warmup_enabled()
        if self._gms_kv_scratch_warmup:
            os.environ["GMS_PERSISTENT_DEFER_PHYSICAL_SCRATCH_BACKED"] = "1"
            logger.info(
                "[GMS] Enabling scratch-backed private-bootstrap KV warmup "
                "for cuda:%d",
                device,
            )
        self._gms_deferred_private_bootstrap_warmup = False
        _maybe_cap_private_bootstrap_memory(self.vllm_config, device)

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
                defer_physical=private_bootstrap_kv_enabled(),
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
            available = self._determine_available_memory_before_gms_publish()
        except BaseException:
            try:
                abort_pending_gms_write()
            except BaseException:
                logger.exception("[GMS] Failed to release pending write")
            raise
        publish_pending_gms_write()
        return available

    def _determine_available_memory_before_gms_publish(self) -> int:
        """Profile without touching reserve-only private-bootstrap KV."""
        if not self._private_bootstrap_kv_active():
            return self._determine_available_memory_with_scratch_accounting()
        if private_bootstrap_scratch_warmup_enabled():
            logger.info(
                "[GMS] Allowing vLLM CUDA graph memory profiling for "
                "scratch-backed private-bootstrap shadow KV"
            )
            return self._determine_available_memory_with_scratch_accounting()

        from vllm.config import CUDAGraphMode

        compilation_config = self.vllm_config.compilation_config
        saved_mode = compilation_config.cudagraph_mode
        if saved_mode == CUDAGraphMode.NONE:
            return self._determine_available_memory_with_scratch_accounting()

        logger.info(
            "[GMS] Deferring vLLM CUDA graph memory profiling for "
            "private-bootstrap shadow until KV promotion"
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        try:
            return self._determine_available_memory_with_scratch_accounting()
        finally:
            compilation_config.cudagraph_mode = saved_mode

    def _determine_available_memory_with_scratch_accounting(self) -> int:
        if not is_scratch_kv_enabled():
            return self._determine_available_memory_with_gms_weight_accounting()

        import vllm.envs as envs
        from vllm.config import CUDAGraphMode
        from vllm.platforms import current_platform

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
        """Defer warmup/cudagraph capture while private-bootstrap KV is VA-only."""
        if not self._private_bootstrap_kv_active():
            result = super().compile_or_warm_up_model()
            self._maybe_tighten_serving_collective_timeout()
            return result
        if private_bootstrap_scratch_warmup_enabled():
            logger.info(
                "[GMS] Running vLLM warmup and CUDA graph capture on "
                "scratch-backed private-bootstrap shadow KV"
            )
            result = super().compile_or_warm_up_model()
            self._maybe_tighten_serving_collective_timeout()
            return result

        from vllm.v1.worker.worker_base import CompilationTimes

        self._gms_deferred_private_bootstrap_warmup = True
        logger.info(
            "[GMS] Deferring vLLM warmup and CUDA graph capture for "
            "private-bootstrap shadow until KV promotion"
        )
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def _run_deferred_private_bootstrap_warmup(self) -> None:
        if not getattr(self, "_gms_deferred_private_bootstrap_warmup", False):
            return

        if not env_enabled_by_default(
            "GMS_VLLM_RUN_DEFERRED_PRIVATE_BOOTSTRAP_WARMUP", default=False
        ):
            logger.info(
                "[GMS] Skipping deferred vLLM warmup and CUDA graph capture "
                "after private-bootstrap KV promotion"
            )
            self._gms_deferred_private_bootstrap_warmup = False
            return

        logger.info(
            "[GMS] Running deferred vLLM warmup and CUDA graph capture after "
            "private-bootstrap KV promotion"
        )
        self._gms_deferred_private_bootstrap_warmup = False
        try:
            super().compile_or_warm_up_model()
        except Exception:
            self._gms_deferred_private_bootstrap_warmup = True
            raise
        # Deferred warmup just completed (shadow promoted + about to serve): tighten.
        self._maybe_tighten_serving_collective_timeout()

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
        self._gms_kv_promote_engine_id = promotion_engine_id(device)
        self._gms_kv_private_bootstrap = private_bootstrap_kv_enabled()
        get_or_create_persistent_allocator(
            socket,
            device,
            engine_id,
            tag="kv_pool",
            shared=allocation_shared(),
            defer_physical=self._gms_kv_private_bootstrap,
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
        free_bytes_before = torch.cuda.mem_get_info()[0]

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
        torch.cuda.empty_cache()

        free_bytes_after, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after
        logger.info(
            "Sleep freed %.2f GiB, %.2f GiB still in use.",
            freed_bytes / (1 << 30),
            used_bytes / (1 << 30),
        )

    def _reset_fp8_kv_scales_without_zeroing(self) -> None:
        """Reset quantized-KV scale tensors without modifying KV contents."""
        cache_config = getattr(self, "cache_config", None)
        cache_dtype = getattr(cache_config, "cache_dtype", "")
        if not str(cache_dtype).startswith("fp8"):
            return

        model_runner = getattr(self, "model_runner", None)
        compilation_config = getattr(model_runner, "compilation_config", None)
        attn_layers = getattr(compilation_config, "static_forward_context", {}) or {}
        reset_count = 0
        for module in attn_layers.values():
            for attr in ("_k_scale", "k_scale", "_v_scale", "v_scale"):
                if not hasattr(module, attr):
                    continue
                param = getattr(module, attr)
                if isinstance(param, torch.Tensor):
                    param.fill_(1.0)
                    reset_count += 1
        logger.info(
            "[GMS] Reset vLLM FP8 KV scale tensors without zeroing KV: count=%d",
            reset_count,
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

        promoted_private_bootstrap = False
        if "kv_pool" in tags:
            kv_manager = get_gms_client_memory_manager("kv_pool")
            assert kv_manager is not None, "GMS persistent KV client is not initialized"
            private_bootstrap = bool(getattr(self, "_gms_kv_private_bootstrap", False))
            logger.info(
                "[GMS] vLLM KV wake_up start: private_bootstrap=%s "
                "is_unmapped=%s mappings=%d scratch_mappings=%d",
                private_bootstrap,
                kv_manager.is_unmapped,
                len(kv_manager.mappings),
                len(getattr(kv_manager, "_scratch_mappings", {})),
            )
            if private_bootstrap and not kv_manager.is_unmapped:
                migrated = kv_manager.prepare_deferred_scratch_for_persistent_remap()
                logger.info(
                    "[GMS] vLLM KV wake_up prepared private-bootstrap VAs: "
                    "migrated=%d is_unmapped=%s mappings=%d scratch_mappings=%d",
                    migrated,
                    kv_manager.is_unmapped,
                    len(kv_manager.mappings),
                    len(getattr(kv_manager, "_scratch_mappings", {})),
                )
            else:
                assert kv_manager.is_unmapped, "GMS persistent KV is not unmapped"
            engine_id = getattr(
                self,
                "_gms_kv_engine_id",
                stable_engine_id(self.local_rank),
            )
            target_engine_id = engine_id
            target_shared = shared_kv_enabled()
            if private_bootstrap:
                target_engine_id = getattr(
                    self, "_gms_kv_promote_engine_id", stable_engine_id(self.local_rank)
                )
                target_shared = True

            logger.info(
                "[GMS] vLLM KV wake_up connecting: engine_id=%s "
                "target_engine_id=%s target_shared=%s",
                engine_id,
                target_engine_id,
                target_shared,
            )
            kv_manager.connect(RequestedLockType.RW_PERSISTENT)
            if (
                private_bootstrap
                and kv_manager.is_unmapped
                and getattr(kv_manager, "_scratch_mappings", {})
            ):
                migrated = kv_manager.prepare_scratch_for_reallocation()
                logger.info(
                    "[GMS] vLLM KV wake_up prepared scratch VAs after connect: "
                    "migrated=%d",
                    migrated,
                )
            logger.info(
                "[GMS] vLLM KV wake_up remap begin: target_engine_id=%s "
                "sync_mode=%s",
                target_engine_id,
                "batched" if private_bootstrap else "per_mapping",
            )
            kv_manager.remap_persistent_vas(
                target_engine_id,
                shared=target_shared,
                synchronize_per_mapping=not private_bootstrap,
                validate_after_remap=not private_bootstrap,
            )
            logger.info("[GMS] vLLM KV wake_up remap done")
            if target_engine_id != engine_id:
                logger.info(
                    "[GMS] vLLM KV wake_up retargeting allocator: %s -> %s",
                    engine_id,
                    target_engine_id,
                )
                retarget_persistent_allocator(
                    "kv_pool", target_engine_id, shared=target_shared
                )
                release_private_bootstrap_kv_pool(kv_manager, engine_id, logger=logger)
                self._gms_kv_engine_id = target_engine_id
                promoted_private_bootstrap = private_bootstrap
                self._gms_kv_private_bootstrap = False
                logger.info(
                    "[GMS] Promoted vLLM KV namespace %s -> %s",
                    engine_id,
                    target_engine_id,
                )

        if (
            requested_tags is None
            or "kv_cache" in requested_tags
            or "kv_pool" in requested_tags
        ):
            if promoted_private_bootstrap:
                logger.info(
                    "[GMS] Skipping vLLM post_kv_cache_wake_up after "
                    "private-bootstrap promotion to preserve shared KV"
                )
                self._reset_fp8_kv_scales_without_zeroing()
            else:
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

            self._run_deferred_private_bootstrap_warmup()

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
