"""Import-only model loader for SGLang.

This loader creates the model structure WITHOUT loading weights from disk.
It's designed for the GPU Memory Service read path where weights are
materialized from the registry after model initialization.

IMPORTANT: This loader runs quantization post-processing to ensure the model
has the same parameter structure as the write-mode model (which had post-processing
applied before registry entries were created). Post-processing can create/destroy
parameters, so it must be run before materializing from registry.

Usage:
    from dynamo.sglang.gpu_memory_service_adapters.import_only_loader import ImportOnlyModelLoader

    # In read mode, use this loader to create model with final structure
    loader = ImportOnlyModelLoader(load_config)
    model = loader.load_model(model_config=model_config, device_config=device_config)

    # Then materialize weights from GPU memory service registry
    materialize_module_from_registry(allocator, model, prefix=..., device_index=...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from sglang.srt.configs import DeviceConfig, LoadConfig, ModelConfig

logger = logging.getLogger(__name__)


class ImportOnlyModelLoader:
    """Model loader that creates model structure without loading weights from disk.

    This loader is used for the GPU Memory Service read/import-only path.
    It creates a model on META device with the FINAL parameter structure (after
    quantization post-processing), which can then be populated with weights from
    the GPU memory service registry.

    Key differences from DefaultModelLoader:
    - Does NOT call load_weights() (no disk I/O)
    - DOES call process_weights_after_loading() to get final parameter structure
    - Creates model on META device (no GPU memory allocation)
    - Does NOT download model weights

    The flow is:
    1. Create model on meta device (no GPU memory)
    2. Run quant post-processing on meta tensors (may create/destroy parameters)
    3. Caller materializes weights from registry (replaces meta tensors)
    """

    def __init__(self, load_config: "LoadConfig"):
        self.load_config = load_config

    def download_model(self, model_config: "ModelConfig") -> None:
        """No-op: import-only loader doesn't need to download weights."""
        pass

    def load_model(
        self,
        *,
        model_config: "ModelConfig",
        device_config: "DeviceConfig",
    ) -> nn.Module:
        """Create model structure with final parameter layout (post-processed).

        The model is created on the meta device and quantization post-processing
        is applied to ensure the parameter structure matches what was registered
        during write mode. The caller is responsible for materializing the weights
        (e.g., via materialize_module_from_registry).

        Args:
            model_config: SGLang model configuration
            device_config: SGLang device configuration

        Returns:
            Model with post-processed structure on meta device
        """
        try:
            from sglang.srt.model_loader.loader import _initialize_model
            from sglang.srt.model_loader.utils import set_default_torch_dtype
        except ImportError as e:
            raise RuntimeError(
                f"SGLang not installed or incompatible version: {e}. "
                "ImportOnlyModelLoader requires SGLang's _initialize_model function."
            ) from e

        meta_device = torch.device("meta")

        # Create model on meta device - no GPU memory allocated
        with set_default_torch_dtype(model_config.dtype):
            with meta_device:
                model = _initialize_model(model_config, self.load_config)

        # Run quantization post-processing to get the FINAL parameter structure.
        # This is critical because:
        # 1. Write mode runs post-processing BEFORE registering tensors
        # 2. Post-processing can create/destroy/rename parameters
        # 3. Registry entries correspond to post-processed parameter names
        # 4. Without this step, materialize_module_from_registry would fail
        #    due to parameter name mismatches
        #
        # Note: Post-processing runs on meta tensors. Some quant methods may:
        # - Work correctly (structure changes happen regardless of tensor data)
        # - Be a no-op (nothing to do on meta tensors)
        # - Fail (requires actual tensor data) - we catch and log these
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                try:
                    quant_method.process_weights_after_loading(module)
                except Exception as e:
                    # Log but continue - some quant methods may fail on meta tensors
                    # but should still produce the correct parameter structure
                    logger.debug(
                        "[ImportOnlyModelLoader] Quant post-processing on meta tensor warning for %s: %s",
                        type(module).__name__,
                        e,
                    )

        logger.info(
            "[ImportOnlyModelLoader] Created post-processed meta model for %s",
            getattr(model_config, "model_path", "unknown"),
        )

        return model


class ImportOnlyModelLoaderMeta(ImportOnlyModelLoader):
    """Variant that creates model on meta device (no GPU memory).

    This is a lightweight version that skips post-processing and creates
    meta tensors. Use this ONLY for models without quantization or when
    you know post-processing doesn't change parameter structure.

    WARNING: If the model has quantization that modifies parameter structure,
    use ImportOnlyModelLoader instead (which runs post-processing).
    """

    def load_model(
        self,
        *,
        model_config: "ModelConfig",
        device_config: "DeviceConfig",
    ) -> nn.Module:
        """Create model with meta tensors (no GPU memory, no post-processing).

        Args:
            model_config: SGLang model configuration
            device_config: SGLang device configuration

        Returns:
            Model with meta tensors (no weights loaded, no post-processing)
        """
        try:
            from sglang.srt.model_loader.loader import _initialize_model
            from sglang.srt.model_loader.utils import set_default_torch_dtype
        except ImportError as e:
            raise RuntimeError(
                f"SGLang not installed or incompatible version: {e}. "
                "ImportOnlyModelLoaderMeta requires SGLang's _initialize_model function."
            ) from e

        # Create model on meta device - no GPU memory allocated
        # WARNING: Skips post-processing, only use for non-quantized models
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(model_config, self.load_config)

        logger.info(
            "[ImportOnlyModelLoaderMeta] Created meta model for %s (no post-processing)",
            getattr(model_config, "model_path", "unknown"),
        )

        return model


# Keep the old name as an alias for backwards compatibility
ImportOnlyModelLoaderWithDevice = ImportOnlyModelLoader
