"""vLLM adapter for GPU Memory Service (Allocation Server + embedded registry)."""

import logging
import os

from dynamo.vllm.gpu_memory_service_adapters.import_only_loader import (
    ImportOnlyModelLoader,
    ImportOnlyModelLoaderMeta,
)
from dynamo.vllm.gpu_memory_service_adapters.model_loader import (
    compute_vllm_config_hash,
    get_imported_weights_bytes,
    register_gpu_memory_service_loader,
)
from dynamo.vllm.gpu_memory_service_adapters.worker_extension import (
    is_worker_patched,
    patch_model_runner_for_gpu_memory_service,
    patch_worker_sleep_wake,
    unpatch_model_runner_for_gpu_memory_service,
    unpatch_worker_sleep_wake,
)

logger = logging.getLogger(__name__)

__all__ = [
    "register_gpu_memory_service_loader",
    "ImportOnlyModelLoader",
    "ImportOnlyModelLoaderMeta",
    "compute_vllm_config_hash",
    "get_imported_weights_bytes",
    "patch_model_runner_for_gpu_memory_service",
    "unpatch_model_runner_for_gpu_memory_service",
    "is_worker_patched",
    "patch_worker_sleep_wake",
    "unpatch_worker_sleep_wake",
    "vllm_plugin_init",
]


def vllm_plugin_init():
    """vLLM plugin entry point for GPU Memory Service.

    This function is called by vLLM's plugin system in all processes (main process,
    engine core, and worker processes). It registers the GPU Memory Service loader
    if the GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER environment variable is set.
    """
    if os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        register_gpu_memory_service_loader()
        patch_model_runner_for_gpu_memory_service()
        logger.info(
            "[GPU Memory Service] vLLM plugin initialized - GPU Memory Service loader registered"
        )


# Auto-register the GPU Memory Service loader when environment variable is set.
# This is necessary for vLLM's multiprocessing with spawn mode, where child
# processes start fresh and don't inherit the parent's loader registration.
if os.environ.get("GPU_MEMORY_SERVICE_VLLM_AUTO_REGISTER", "").lower() in (
    "1",
    "true",
    "yes",
):
    register_gpu_memory_service_loader()
    patch_model_runner_for_gpu_memory_service()
