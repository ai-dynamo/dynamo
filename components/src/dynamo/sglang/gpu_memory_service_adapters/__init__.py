"""SGLang adapter for GPU Memory Service (Allocation Server + embedded metadata store)."""

import logging

# Monkey-patch torch_memory_saver to support GPU Memory Service mode
# This MUST happen before any torch_memory_saver imports
import dynamo.sglang.gpu_memory_service_adapters.torch_memory_saver_patch  # noqa: F401
from dynamo.sglang.gpu_memory_service_adapters.model_loader import GPUServiceModelLoader
from dynamo.sglang.gpu_memory_service_adapters.worker_extension import (
    patch_model_runner_for_gpu_memory_service,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GPUServiceModelLoader",
    "patch_model_runner_for_gpu_memory_service",
]
