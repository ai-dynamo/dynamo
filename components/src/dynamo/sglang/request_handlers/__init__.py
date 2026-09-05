# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Declared, not imported at runtime. These mirror `_EXPORTS` below; without
    # them the module-level `__getattr__` is the only thing that resolves the
    # handlers, and hiding it from the type checker would make them all look
    # missing.
    from .embedding import EmbeddingWorkerHandler
    from .handler_base import BaseGenerativeHandler, BaseWorkerHandler
    from .image_diffusion import ImageDiffusionWorkerHandler
    from .llm import DecodeWorkerHandler, DiffusionWorkerHandler, PrefillWorkerHandler
    from .multimodal import (
        MultimodalEncodeWorkerHandler,
        MultimodalPrefillWorkerHandler,
        MultimodalWorkerHandler,
    )
    from .video_generation import VideoGenerationWorkerHandler

_EXPORTS = {
    # Base handlers
    "BaseGenerativeHandler": ".handler_base",
    "BaseWorkerHandler": ".handler_base",
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


# Hidden from the type checker on purpose. A module-level `__getattr__` tells
# mypy this package may have any attribute, so a typo such as
# `from ... request_handlers import DecodeWorkerHandlr` would check clean in
# every module that imports from here, and the real handlers resolve as `Any`.
# The `TYPE_CHECKING` block above declares them, so the checker gains their
# actual types and loses nothing.
if not TYPE_CHECKING:

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
