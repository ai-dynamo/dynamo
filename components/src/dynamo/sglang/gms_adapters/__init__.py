"""SGLang adapter for GPU Memory Service (Allocation Server + embedded registry)."""

import logging

# Monkey-patch torch_memory_saver to support GMS mode
# This MUST happen before any torch_memory_saver imports
import dynamo.sglang.gms_adapters.torch_memory_saver_patch  # noqa: F401
from dynamo.sglang.gms_adapters.import_only_loader import (
    ImportOnlyModelLoader,
    ImportOnlyModelLoaderMeta,
    ImportOnlyModelLoaderWithDevice,
)
from dynamo.sglang.gms_adapters.model_loader import (
    GPUServiceModelLoader,
    compute_sglang_config_hash,
    get_imported_weights_bytes,
)
from dynamo.sglang.gms_adapters.torch_memory_saver_gms import (
    GMSMemorySaverImpl,
    _get_gms_allocator,
    is_gms_mode,
)
from dynamo.sglang.gms_adapters.worker_extension import (
    is_model_runner_patched,
    patch_model_runner_for_gms,
    unpatch_model_runner_for_gms,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GPUServiceModelLoader",
    "ImportOnlyModelLoader",
    "ImportOnlyModelLoaderMeta",
    "ImportOnlyModelLoaderWithDevice",
    "compute_sglang_config_hash",
    "get_imported_weights_bytes",
    "patch_model_runner_for_gms",
    "unpatch_model_runner_for_gms",
    "is_model_runner_patched",
    "GMSMemorySaverImpl",
    "_get_gms_allocator",
    "is_gms_mode",
]
