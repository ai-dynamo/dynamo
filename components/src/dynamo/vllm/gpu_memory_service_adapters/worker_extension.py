"""vLLM worker-side helpers for GPU Memory Service integration.

Today this module provides:
1. Registration of GPU Memory Service loader in worker processes
2. Fix for vLLM's model memory accounting when weights are allocated via CUDA VMM
3. Worker-level patches for sleep/wake that coordinate GPU Memory Service and CuMemAllocator

Architecture:
- The GPU Memory Service allocator is registered in the WORKER process (where model loading happens)
- The handler in main.py runs in the MAIN process and triggers sleep/wake via RPC
- Worker.sleep()/wake_up() patches intercept in the worker and call gpu_memory_service_sleep_weights()
- This is necessary because the allocator isn't accessible from the main process

Call `patch_model_runner_for_gpu_memory_service()` before constructing the vLLM engine.
Call `patch_worker_sleep_wake()` after the model is loaded (done by model_loader.py).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dynamo.gpu_memory_service import GMSClientMemoryManager

logger = logging.getLogger(__name__)

# Auto-register the loader when this module is imported in worker processes
# This is triggered by the GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER environment variable
if os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER") == "1":
    try:
        from dynamo.vllm.gpu_memory_service_adapters.model_loader import (
            register_gpu_memory_service_loader,
        )

        register_gpu_memory_service_loader()
        logger.info(
            "[GPU Memory Service] Auto-registered GPU Memory Service loader in worker process"
        )
    except Exception as e:
        logger.warning(
            f"[GPU Memory Service] Failed to auto-register loader in worker: {e}"
        )

# Flag for early patches - will be applied after function definitions
_early_patches_applied = False
# Apply early patches if GPU Memory Service is enabled - they're harmless for write mode
_should_apply_early_patches = (
    os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER") == "1"
)

_worker_patched = False
_original_load_model = None
_original_init_device = None


def get_gpu_memory_service_allocator() -> Optional["GMSClientMemoryManager"]:
    """Get the GPU Memory Service allocator singleton."""
    from dynamo.gpu_memory_service import get_allocator

    return get_allocator()


def gpu_memory_service_sleep_weights() -> bool:
    """Sleep GPU Memory Service weights (VA-stable).

    Unmaps physical memory while preserving VA reservations.
    Tensor pointers remain valid but memory is released.

    Returns True if sleep was performed.
    """
    allocator = get_gpu_memory_service_allocator()
    if allocator is None:
        return False
    if allocator.is_sleeping:
        return False
    try:
        allocator.sleep()
        logger.info("[GPU Memory Service] Slept GPU Memory Service weights (VA-stable)")
        return True
    except Exception as e:
        logger.warning(
            f"[GPU Memory Service] Failed to sleep GPU Memory Service weights: {e}"
        )
        return False


def gpu_memory_service_wake_weights() -> bool:
    """Wake GPU Memory Service weights (VA-stable).

    Remaps physical memory to the same VAs.
    Tensor pointers remain valid after wake.

    Returns True if wake was performed.

    Raises:
        StaleWeightsError: If weights were structurally changed while sleeping.
    """
    import torch

    allocator = get_gpu_memory_service_allocator()
    if allocator is None:
        return False
    if not allocator.is_sleeping:
        return False

    # Note: StaleWeightsError may propagate to caller
    # wake() sets the correct device context internally
    allocator.wake()

    # Synchronize the allocator's device to ensure all remapping operations are visible
    if torch.cuda.is_available():
        device = allocator.device
        torch.cuda.synchronize(device)
        logger.info(
            f"[GPU Memory Service] CUDA synchronized on device {device} after wake"
        )
    logger.info("[GPU Memory Service] Woke GPU Memory Service weights (VA-stable)")
    return True


def _get_gpu_memory_service_committed_bytes() -> int:
    """Query the allocation server for total committed bytes on this device.

    Returns 0 if query fails or no allocations exist.

    This function first checks for a registered allocator (which already holds
    an RO connection) and uses it to avoid creating short-lived connections.
    Only falls back to creating a new connection when no allocator is registered.
    """
    # First, try to use the registered allocator's existing connection
    allocator = get_gpu_memory_service_allocator()
    if allocator is not None and allocator._client is not None:
        try:
            allocations = allocator._client.list_allocations()
            total_bytes = sum(alloc.get("aligned_size", 0) for alloc in allocations)
            if total_bytes > 0:
                logger.debug(
                    "[GPU Memory ServicePatch] Queried committed bytes via registered allocator: "
                    "%.2f GiB (%d allocations)",
                    total_bytes / (1 << 30),
                    len(allocations),
                )
            return total_bytes
        except Exception as e:
            logger.warning(
                "[GPU Memory ServicePatch] Failed to query via registered allocator: %s",
                e,
            )
            # Fall through to create a new connection

    # No registered allocator - use default socket path from model_loader
    # This matches the default used by GPU Memory Service server and model loader
    from dynamo.vllm.gpu_memory_service_adapters.model_loader import (
        DEFAULT_GPU_MEMORY_SERVICE_SOCKET_PATH,
    )

    # Get local rank for socket path
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        import torch

        if torch.cuda.is_initialized():
            local_rank = torch.cuda.current_device()
    except Exception:
        pass

    socket_path = DEFAULT_GPU_MEMORY_SERVICE_SOCKET_PATH.format(device=local_rank)

    try:
        from dynamo.gpu_memory_service import GMSRPCClient

        client = GMSRPCClient(socket_path, lock_type="ro")
        allocations = client.list_allocations()
        total_bytes = sum(alloc.get("aligned_size", 0) for alloc in allocations)
        client.close()
        if total_bytes > 0:
            logger.info(
                "[GPU Memory ServicePatch] Queried committed bytes from server (new connection): "
                "%.2f GiB (%d allocations)",
                total_bytes / (1 << 30),
                len(allocations),
            )
        return total_bytes
    except Exception as e:
        logger.debug("[GPU Memory ServicePatch] Could not query committed bytes: %s", e)
        return 0


def _patch_init_device_for_gpu_memory_service(Worker) -> None:
    """Patch MemorySnapshot to include committed GPU Memory Service bytes in free_memory.

    In GPU Memory Service read mode, weights are already on GPU and will be imported (mapped),
    not newly allocated. The startup memory check should account for this by
    adding committed bytes to the reported free memory.
    """
    global _original_init_device

    if _original_init_device is not None:
        return  # Already patched

    # Patch MemorySnapshot.measure to add committed bytes to free_memory
    from vllm.utils.mem_utils import MemorySnapshot

    _original_measure = MemorySnapshot.measure

    def patched_measure(self):
        _original_measure(self)
        # Add committed GPU Memory Service bytes to free_memory for the startup check
        committed_bytes = _get_gpu_memory_service_committed_bytes()
        if committed_bytes > 0:
            original_free = self.free_memory
            self.free_memory += committed_bytes
            logger.info(
                "[GPU Memory ServicePatch] Adjusted MemorySnapshot.free_memory for GPU Memory Service read mode: "
                "%.2f GiB + %.2f GiB committed = %.2f GiB",
                original_free / (1 << 30),
                committed_bytes / (1 << 30),
                self.free_memory / (1 << 30),
            )

    MemorySnapshot.measure = patched_measure
    _original_init_device = _original_measure  # Store original for unpatch
    logger.info(
        "[GPU Memory ServicePatch] Patched MemorySnapshot.measure for GPU Memory Service read mode memory check"
    )


def patch_model_runner_for_gpu_memory_service() -> None:
    """Monkey-patch vLLM Worker.load_model to correct model_memory_usage.

    This mirrors the prior VMM accounting patch but sources bytes from the
    GPU Memory Service loader.
    """
    global _worker_patched, _original_load_model

    if _worker_patched:
        return

    # vLLM 0.12+ renamed GPUWorker to Worker
    Worker = None
    try:
        from vllm.v1.worker.gpu_worker import Worker  # type: ignore
    except ImportError:
        pass

    if Worker is None:
        try:
            from vllm.v1.worker.gpu_worker import GPUWorker as Worker  # type: ignore
        except ImportError:
            pass

    if Worker is None:
        logger.warning(
            "[GPU Memory ServicePatch] Could not import vLLM Worker; patch not applied"
        )
        return

    _original_load_model = Worker.load_model

    def patched_load_model(self):
        _original_load_model(self)
        try:
            from dynamo.vllm.gpu_memory_service_adapters.model_loader import (
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
                    "[GPU Memory ServicePatch] Corrected model_memory_usage: %.2f GiB -> %.2f GiB",
                    old_usage / (1 << 30),
                    imported_bytes / (1 << 30),
                )
        except Exception:
            pass

    Worker.load_model = patched_load_model
    _worker_patched = True
    logger.info(
        "[GPU Memory ServicePatch] Patched vLLM Worker.load_model for VMM memory accounting"
    )

    # Also patch init_device to fix memory check for GPU Memory Service read mode
    _patch_init_device_for_gpu_memory_service(Worker)

    # Patch _maybe_get_memory_pool_context to skip CuMemAllocator for weights
    # when using GPU Memory Service. GPU Memory Service handles weights; CuMemAllocator should only handle KV cache.
    _original_get_memory_pool_context = Worker._maybe_get_memory_pool_context

    def patched_get_memory_pool_context(self, tag: str):
        from contextlib import nullcontext

        # When using GPU Memory Service, skip CuMemAllocator for weights - GPU Memory Service handles them
        if tag == "weights":
            if os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER") == "1":
                logger.debug(
                    "[GPU Memory ServicePatch] Skipping CuMemAllocator for weights (GPU Memory Service mode)"
                )
                return nullcontext()
        # For KV cache and other tags, use original (CuMemAllocator if enable_sleep_mode)
        return _original_get_memory_pool_context(self, tag)

    Worker._maybe_get_memory_pool_context = patched_get_memory_pool_context
    logger.info(
        "[GPU Memory ServicePatch] Patched _maybe_get_memory_pool_context for GPU Memory Service weights"
    )


def unpatch_model_runner_for_gpu_memory_service() -> None:
    """Remove the Worker.load_model patch."""
    global _worker_patched, _original_load_model
    if not _worker_patched:
        return

    try:
        # vLLM 0.12+ renamed GPUWorker to Worker
        try:
            from vllm.v1.worker.gpu_worker import Worker  # type: ignore
        except ImportError:
            from vllm.v1.worker.gpu_worker import GPUWorker as Worker  # type: ignore

        if _original_load_model is not None:
            Worker.load_model = _original_load_model
    except Exception:
        pass
    finally:
        _original_load_model = None
        _worker_patched = False


def is_worker_patched() -> bool:
    return _worker_patched


_sleep_wake_patched = False
_original_sleep = None
_original_wake_up = None


def patch_worker_sleep_wake() -> None:
    """Patch vLLM Worker.sleep() and Worker.wake_up() for GPU Memory Service integration.

    This patch is applied in the WORKER process (via model_loader.py) because
    the GPU Memory Service allocator is registered there. The handler in main.py triggers
    sleep/wake via RPC, which ends up calling these patched methods.

    IMPORTANT: We do NOT call the original Worker.sleep()/wake_up() at all!
    The original methods try to copy GPU memory to CPU for backup, which causes
    segfaults when GPU Memory Service has already unmapped the weights.

    Instead, we directly use the allocators:
    - GPU Memory Service allocator for weights (VA-stable sleep/wake)
    - CuMemAllocator for KV cache (discard on sleep, recreate on wake)
    """
    global _sleep_wake_patched, _original_sleep, _original_wake_up

    if _sleep_wake_patched:
        return

    # Find Worker class (vLLM 0.12+ uses Worker, older uses GPUWorker)
    Worker = None
    try:
        from vllm.v1.worker.gpu_worker import Worker  # type: ignore
    except ImportError:
        pass

    if Worker is None:
        try:
            from vllm.v1.worker.gpu_worker import GPUWorker as Worker  # type: ignore
        except ImportError:
            pass

    if Worker is None:
        logger.warning(
            "[GPU Memory Service] Could not import vLLM Worker; sleep/wake patch not applied"
        )
        return

    # Check if sleep/wake_up exist on Worker
    if not hasattr(Worker, "sleep") or not hasattr(Worker, "wake_up"):
        logger.debug(
            "[GPU Memory Service] Worker does not have sleep/wake_up methods; patch skipped"
        )
        return

    _original_sleep = Worker.sleep
    _original_wake_up = Worker.wake_up

    def patched_sleep(self, level: int = 1):
        """Patched sleep: Directly use allocators, never call original Worker.sleep().

        With GPU Memory Service, sleep means:
        1. GPU Memory Service allocator.sleep() - unmaps weights (VA preserved)
        2. CuMemAllocator.sleep() - discards KV cache (no CPU backup)

        The 'level' parameter is ignored - we always sleep everything.
        We do NOT call original Worker.sleep() because it tries to copy
        GPU buffers to CPU, which segfaults on already-unmapped GPU Memory Service memory.
        """
        import torch

        free_bytes_before = torch.cuda.mem_get_info()[0]

        # 1. Sleep GPU Memory Service weights (VA-preserving, no CPU copy)
        gpu_memory_service_slept = gpu_memory_service_sleep_weights()
        if gpu_memory_service_slept:
            logger.info(
                "[GPU Memory Service] VA-stable slept GPU Memory Service weights"
            )

        # 2. Sleep KV cache via CuMemAllocator directly
        # DO NOT call original Worker.sleep() - it tries to copy buffers to CPU
        # which segfaults on unmapped GPU Memory Service memory
        try:
            from vllm.device_allocator.cumem import CuMemAllocator

            kv_allocator = CuMemAllocator.get_instance()
            # Empty tuple = discard everything, no CPU backup (like level >= 2)
            kv_allocator.sleep(offload_tags=tuple())
            logger.info(
                "[GPU Memory Service] Slept KV cache via CuMemAllocator (discarded)"
            )
        except Exception as e:
            logger.warning(
                f"[GPU Memory Service] Failed to sleep KV cache via CuMemAllocator: {e}"
            )

        free_bytes_after, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after
        logger.info(
            "[GPU Memory Service] Sleep freed %.2f GiB memory, %.2f GiB still in use",
            freed_bytes / (1 << 30),
            used_bytes / (1 << 30),
        )

    def patched_wake_up(self, tags: Optional[list] = None):
        """Patched wake_up: Directly use allocators, never call original Worker.wake_up().

        With GPU Memory Service, wake means:
        1. GPU Memory Service allocator.wake() - remaps weights to same VAs
        2. CuMemAllocator.wake_up() - reallocates KV cache memory

        The 'tags' parameter is ignored - we always wake everything.
        """
        if tags is None:
            tags = ["weights", "kv_cache"]

        # 1. Wake GPU Memory Service weights (VA-stable, remaps to same addresses)
        if "weights" in tags:
            try:
                gpu_memory_service_woke = gpu_memory_service_wake_weights()
                if gpu_memory_service_woke:
                    logger.info(
                        "[GPU Memory Service] VA-stable woke GPU Memory Service weights"
                    )
            except Exception as e:
                # StaleWeightsError or other - let it propagate
                logger.error(
                    f"[GPU Memory Service] Failed to wake GPU Memory Service weights: {e}"
                )
                raise

        # 2. Wake KV cache via CuMemAllocator directly
        # DO NOT call original Worker.wake_up() - we manage allocators directly
        if "kv_cache" in tags:
            try:
                from vllm.device_allocator.cumem import CuMemAllocator

                kv_allocator = CuMemAllocator.get_instance()
                kv_allocator.wake_up(tags=["kv_cache"])
                logger.info("[GPU Memory Service] Woke KV cache via CuMemAllocator")
            except Exception as e:
                logger.warning(
                    f"[GPU Memory Service] Failed to wake KV cache via CuMemAllocator: {e}"
                )

        # 3. Reinitialize FP8 KV scales if needed (from original Worker.wake_up)
        if "kv_cache" in tags:
            try:
                if (
                    hasattr(self, "cache_config")
                    and hasattr(self.cache_config, "cache_dtype")
                    and self.cache_config.cache_dtype.startswith("fp8")
                    and hasattr(self, "model_runner")
                    and hasattr(self.model_runner, "init_fp8_kv_scales")
                ):
                    self.model_runner.init_fp8_kv_scales()
                    logger.info("[GPU Memory Service] Reinitialized FP8 KV scales")
            except Exception as e:
                logger.debug(f"[GPU Memory Service] FP8 KV scale reinit skipped: {e}")

    Worker.sleep = patched_sleep
    Worker.wake_up = patched_wake_up
    _sleep_wake_patched = True
    logger.info(
        "[GPU Memory Service] Patched Worker.sleep() and Worker.wake_up() for VA-stable GPU Memory Service integration"
    )
    logger.info(
        "[GPU Memory Service] Sleep/wake behavior: "
        "(1) GPU Memory Service weights: VA-stable unmap/remap (physical memory stays in allocation server); "
        "(2) KV cache: discarded on sleep, reallocated on wake via CuMemAllocator; "
        "(3) Original Worker.sleep()/wake_up() are NEVER called"
    )


def unpatch_worker_sleep_wake() -> None:
    """Remove Worker.sleep() and Worker.wake_up() patches."""
    global _sleep_wake_patched, _original_sleep, _original_wake_up

    if not _sleep_wake_patched:
        return

    try:
        try:
            from vllm.v1.worker.gpu_worker import Worker  # type: ignore
        except ImportError:
            from vllm.v1.worker.gpu_worker import GPUWorker as Worker  # type: ignore

        if _original_sleep is not None:
            Worker.sleep = _original_sleep
        if _original_wake_up is not None:
            Worker.wake_up = _original_wake_up
    except Exception:
        pass
    finally:
        _original_sleep = None
        _original_wake_up = None
        _sleep_wake_patched = False


# Apply early memory patches AFTER all functions are defined (avoids circular import)
# This is needed because vLLM's memory check happens in init_device() BEFORE load_model()
# The patch adjusts MemorySnapshot to account for committed GPU Memory Service bytes
if _should_apply_early_patches and not _early_patches_applied:
    try:
        patch_model_runner_for_gpu_memory_service()
        _early_patches_applied = True
        logger.info("[GPU Memory Service] Applied early memory patches for read mode")
    except Exception as e:
        logger.warning(
            f"[GPU Memory Service] Failed to apply early memory patches: {e}"
        )
