# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import Any

_EXPORTS = {
    # Base handlers
    "BaseGenerativeHandler": ".handler_base",
    "BaseWorkerHandler": ".handler_base",
    "RLMixin": ".handler_base",
    # LLM handlers
    "DecodeWorkerHandler": ".llm",
    "DiffusionWorkerHandler": ".llm",
    "PrefillWorkerHandler": ".llm",
    # Embedding handlers
    "EmbeddingWorkerHandler": ".embedding",
    # Image diffusion handlers
    "ImageDiffusionWorkerHandler": ".image_diffusion",
    # Video generation handlers
    "VideoGenerationWorkerHandler": ".video_generation",
    # Multimodal handlers
    "MultimodalEncodeWorkerHandler": ".multimodal",
    "MultimodalPrefillWorkerHandler": ".multimodal",
    "MultimodalWorkerHandler": ".multimodal",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    # Base handlers
    "BaseGenerativeHandler",
    "BaseWorkerHandler",
    "RLMixin",
    # LLM handlers
    "DecodeWorkerHandler",
    "DiffusionWorkerHandler",
    "PrefillWorkerHandler",
    # Embedding handlers
    "EmbeddingWorkerHandler",
    # Image diffusion handlers
    "ImageDiffusionWorkerHandler",
    # Video generation handlers
    "VideoGenerationWorkerHandler",
    # Multimodal handlers
    "MultimodalEncodeWorkerHandler",
    "MultimodalPrefillWorkerHandler",
    "MultimodalWorkerHandler",
]
