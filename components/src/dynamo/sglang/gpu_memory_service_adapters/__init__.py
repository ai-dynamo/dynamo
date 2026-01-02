"""SGLang adapter for GPU Memory Service (Allocation Server + embedded registry)."""

import logging

# Monkey-patch torch_memory_saver to support GPU Memory Service mode
# This MUST happen before any torch_memory_saver imports
import dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_patch  # noqa: F401
from dynamo.sglang.gpu_memory_service_adapters.import_only_loader import (
    ImportOnlyModelLoader,
    ImportOnlyModelLoaderMeta,
    ImportOnlyModelLoaderWithDevice,
)
from dynamo.sglang.gpu_memory_service_adapters.model_loader import (
    GPUServiceModelLoader,
    compute_sglang_config_hash,
    get_imported_weights_bytes,
)
from dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_impl import (
    GPUMemoryServiceMemorySaverImpl,
    get_gpu_memory_service_impl,
    is_gpu_memory_service_mode,
)
from dynamo.sglang.gpu_memory_service_adapters.worker_extension import (
    is_model_runner_patched,
    patch_model_runner_for_gpu_memory_service,
    unpatch_model_runner_for_gpu_memory_service,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GPUServiceModelLoader",
    "ImportOnlyModelLoader",
    "ImportOnlyModelLoaderMeta",
    "ImportOnlyModelLoaderWithDevice",
    "compute_sglang_config_hash",
    "get_imported_weights_bytes",
    "patch_model_runner_for_gpu_memory_service",
    "unpatch_model_runner_for_gpu_memory_service",
    "is_model_runner_patched",
    "GPUMemoryServiceMemorySaverImpl",
    "get_gpu_memory_service_impl",
    "is_gpu_memory_service_mode",
]
