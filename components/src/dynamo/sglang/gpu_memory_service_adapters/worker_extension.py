"""SGLang worker-side helpers for GPU Memory Service integration.

This module provides patches for SGLang to work correctly with GPU Memory Service:
1. Memory accounting fixes when weights are pre-loaded via GPU Memory Service

Call `patch_model_runner_for_gpu_memory_service()` before constructing the SGLang runtime.

For multiprocessing with spawn mode (which SGLang uses), set the environment
variable GPU_MEMORY_SERVICE_SGLANG_AUTO_REGISTER=1 to automatically apply patches in child
processes when the dynamo.sglang.gpu_memory_service_adapters module is imported.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_model_runner_patched = False
_original_init_memory_pool = None


def patch_model_runner_for_gpu_memory_service() -> None:
    """Monkey-patch SGLang for GPU Memory Service compatibility.

    This applies multiple patches:
    1. torch.cuda.empty_cache() - prevents segfaults with VMM allocations
    2. ModelRunner.init_memory_pool - fixes memory accounting when weights are pre-loaded

    When weights are pre-loaded via GPU Memory Service (import-only mode), SGLang's
    `min_per_gpu_memory` value captured in `init_torch_distributed()` will be lower
    than the true device total. This causes under-reservation of overhead memory
    in the KV cache calculation:

        rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)

    If `total_gpu_memory` is reduced (e.g., 32 GB instead of 48 GB), the overhead
    reservation `total_gpu_memory * 0.14` is too small, leading to over-allocation
    of KV cache and potential OOM.

    This patch ensures `total_gpu_memory` is always the true device capacity from
    `torch.cuda.mem_get_info()[1]`, regardless of when weights were loaded.
    """
    global _model_runner_patched, _original_init_memory_pool

    # Always patch empty_cache first - this is critical for VMM safety
    from dynamo.sglang.gpu_memory_service_adapters.model_loader import (
        _patch_empty_cache_if_needed,
    )

    _patch_empty_cache_if_needed()

    if _model_runner_patched:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner  # type: ignore
    except Exception:
        logger.warning(
            "[GPU Memory ServicePatch] Could not import SGLang ModelRunner; patch not applied"
        )
        return

    _original_init_memory_pool = ModelRunner.init_memory_pool

    def patched_init_memory_pool(self, total_gpu_memory, *args, **kwargs):
        """Patched init_memory_pool that ensures correct overhead reservation.

        The key fix: Always use the device's total memory capacity for the overhead
        calculation, not the potentially-reduced `min_per_gpu_memory` that was
        captured before/after model loading.
        """
        try:
            device_id = getattr(self, "gpu_id", None)
            if device_id is None:
                device_id = torch.cuda.current_device()

            # Get the true device total memory capacity
            _, total_bytes = torch.cuda.mem_get_info(int(device_id))
            device_total_gb = float(total_bytes) / (1 << 30)

            # Log imported bytes from impl if available
            try:
                from dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_impl import (
                    get_gpu_memory_service_impl,
                )

                impl = get_gpu_memory_service_impl()
                if impl:
                    imported_bytes = impl.get_imported_weights_bytes()
                    if imported_bytes > 0:
                        logger.info(
                            "[GPU Memory ServicePatch] Model weights: %.2f GiB via GPU Memory Service",
                            imported_bytes / (1 << 30),
                        )
            except Exception:
                pass

            # Always use device total for overhead calculation
            # This ensures consistent overhead reservation regardless of when
            # weights were loaded (before or during init_torch_distributed)
            if float(total_gpu_memory) < device_total_gb * 0.95:  # Allow 5% tolerance
                logger.info(
                    "[GPU Memory ServicePatch] Correcting total_gpu_memory for overhead calculation: "
                    "%.2f GB -> %.2f GB (device total)",
                    float(total_gpu_memory),
                    device_total_gb,
                )
                total_gpu_memory = device_total_gb
        except Exception as e:
            logger.warning(
                "[GPU Memory ServicePatch] Could not correct total_gpu_memory: %s", e
            )

        assert _original_init_memory_pool is not None
        return _original_init_memory_pool(self, total_gpu_memory, *args, **kwargs)

    ModelRunner.init_memory_pool = patched_init_memory_pool
    _model_runner_patched = True
    logger.info(
        "[GPU Memory ServicePatch] Patched SGLang ModelRunner.init_memory_pool for memory accounting safety"
    )


def unpatch_model_runner_for_gpu_memory_service() -> None:
    """Remove the ModelRunner.init_memory_pool patch."""
    global _model_runner_patched, _original_init_memory_pool

    if not _model_runner_patched:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner  # type: ignore

        if _original_init_memory_pool is not None:
            ModelRunner.init_memory_pool = _original_init_memory_pool
    except Exception:
        pass
    finally:
        _original_init_memory_pool = None
        _model_runner_patched = False


def is_model_runner_patched() -> bool:
    return _model_runner_patched
