"""Import-only model loader for vLLM.

This loader creates the model structure WITHOUT loading weights from disk.
It's designed for the GPU Memory Service read path where weights are
materialized from the registry after model initialization.

IMPORTANT: This loader runs quantization post-processing to ensure the model
has the same parameter structure as the write-mode model (which had post-processing
applied before registry entries were created). Post-processing can create/destroy
parameters, so it must be run before materializing from registry.

Usage:
    from dynamo.vllm.gpu_memory_service_adapters.import_only_loader import ImportOnlyModelLoader

    # Register the loader (once, at startup)
    ImportOnlyModelLoader.register()

    # Or use directly
    loader = ImportOnlyModelLoader(load_config)
    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    # Then materialize weights from GPU memory service registry
    materialize_module_from_registry(allocator, model, prefix=..., device_index=...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.config import LoadConfig, ModelConfig, VllmConfig

logger = logging.getLogger(__name__)

# Default load format name for registration
DEFAULT_LOAD_FORMAT = "gpu_memory_service_import_only"


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

    _registered: bool = False

    def __init__(self, load_config: "LoadConfig"):
        self.load_config = load_config

    @classmethod
    def register(cls, load_format: str = DEFAULT_LOAD_FORMAT) -> None:
        """Register this loader with vLLM's model loader registry.

        After registration, you can use load_format="gpu_memory_service_import_only" in LoadConfig.
        """
        if cls._registered:
            return

        try:
            from vllm.model_executor.model_loader import register_model_loader
        except ImportError as e:
            raise RuntimeError(
                f"vLLM not installed or incompatible version: {e}. "
                "ImportOnlyModelLoader requires vLLM's register_model_loader."
            ) from e

        register_model_loader(load_format)(cls)
        cls._registered = True
        logger.info("[ImportOnlyModelLoader] Registered as load_format=%r", load_format)

    def download_model(self, model_config: "ModelConfig") -> None:
        """No-op: import-only loader doesn't need to download weights."""
        pass

    def load_weights(self, model: nn.Module, model_config: "ModelConfig") -> None:
        """No-op: weights will be materialized from registry by caller."""
        pass

    def load_model(
        self,
        vllm_config: "VllmConfig",
        model_config: Optional["ModelConfig"] = None,
    ) -> nn.Module:
        """Create model structure with final parameter layout (post-processed).

        The model is created on the meta device and quantization post-processing
        is applied to ensure the parameter structure matches what was registered
        during write mode. The caller is responsible for materializing the weights
        (e.g., via materialize_module_from_registry).

        Args:
            vllm_config: vLLM configuration
            model_config: Optional model config override

        Returns:
            Model with post-processed structure on meta device
        """
        try:
            from vllm.model_executor.model_loader.utils import (
                initialize_model,
                process_weights_after_loading,
            )
            from vllm.utils.torch_utils import set_default_torch_dtype
        except ImportError as e:
            raise RuntimeError(
                f"vLLM not installed or incompatible version: {e}. "
                "ImportOnlyModelLoader requires vLLM's initialize_model function."
            ) from e

        if model_config is None:
            model_config = vllm_config.model_config

        meta_device = torch.device("meta")

        # Enable meta tensor workaround for operations like torch.nonzero()
        # that don't have proper meta implementations. This is needed for
        # models like Qwen3-MoE that use torch.nonzero during initialization.
        try:
            import torch.fx.experimental._config as fx_config

            fx_config.meta_nonzero_assume_all_nonzero = True
            logger.debug(
                "[ImportOnlyModelLoader] Enabled meta_nonzero_assume_all_nonzero"
            )
        except (ImportError, AttributeError) as e:
            logger.debug(
                "[ImportOnlyModelLoader] Could not set meta_nonzero config: %s", e
            )

        # Create model on meta device - no GPU memory allocated
        with set_default_torch_dtype(model_config.dtype):
            with meta_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

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
        # - Fail (requires actual tensor data) - process_weights_after_loading
        #   should handle this gracefully
        try:
            process_weights_after_loading(model, model_config, meta_device)
        except Exception as e:
            # Log but continue - post-processing may fail on meta tensors
            # but should still produce the correct parameter structure
            logger.debug(
                "[ImportOnlyModelLoader] Post-processing on meta tensors warning: %s",
                e,
            )

        # Debug: Log parameters after post-processing
        param_names = sorted([n for n, _ in model.named_parameters()])
        logger.info(
            "[ImportOnlyModelLoader] Read mode params after post-processing (%d), sample: %s",
            len(param_names),
            [n for n in param_names if "weight_scale" in n or "scale_inv" in n][:10],
        )

        logger.info(
            "[ImportOnlyModelLoader] Created post-processed meta model for %s",
            getattr(model_config, "model", "unknown"),
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

    @classmethod
    def register(cls, load_format: str = "gpu_memory_service_import_only_meta") -> None:
        """Register this loader with vLLM's model loader registry."""
        if cls._registered:
            return

        try:
            from vllm.model_executor.model_loader import register_model_loader
        except ImportError as e:
            raise RuntimeError(
                f"vLLM not installed or incompatible version: {e}. "
                "ImportOnlyModelLoaderMeta requires vLLM's register_model_loader."
            ) from e

        register_model_loader(load_format)(cls)
        cls._registered = True
        logger.info(
            "[ImportOnlyModelLoaderMeta] Registered as load_format=%r", load_format
        )

    def load_model(
        self,
        vllm_config: "VllmConfig",
        model_config: Optional["ModelConfig"] = None,
    ) -> nn.Module:
        """Create model with meta tensors (no GPU memory, no post-processing).

        Args:
            vllm_config: vLLM configuration
            model_config: Optional model config override

        Returns:
            Model with meta tensors (no weights loaded, no post-processing)
        """
        try:
            from vllm.model_executor.model_loader.utils import initialize_model
            from vllm.utils.torch_utils import set_default_torch_dtype
        except ImportError as e:
            raise RuntimeError(
                f"vLLM not installed or incompatible version: {e}. "
                "ImportOnlyModelLoaderMeta requires vLLM's initialize_model function."
            ) from e

        if model_config is None:
            model_config = vllm_config.model_config

        # Enable meta tensor workaround for operations like torch.nonzero()
        # that don't have proper meta implementations. This is needed for
        # models like Qwen3-MoE that use torch.nonzero during initialization.
        try:
            import torch.fx.experimental._config as fx_config

            fx_config.meta_nonzero_assume_all_nonzero = True
            logger.debug(
                "[ImportOnlyModelLoaderMeta] Enabled meta_nonzero_assume_all_nonzero"
            )
        except (ImportError, AttributeError) as e:
            logger.debug(
                "[ImportOnlyModelLoaderMeta] Could not set meta_nonzero config: %s", e
            )

        # Create model on meta device - no GPU memory allocated
        # WARNING: Skips post-processing, only use for non-quantized models
        meta_device = torch.device("meta")
        with set_default_torch_dtype(model_config.dtype):
            with meta_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

        logger.info(
            "[ImportOnlyModelLoaderMeta] Created meta model for %s (no post-processing)",
            getattr(model_config, "model", "unknown"),
        )

        return model
