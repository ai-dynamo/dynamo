# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve configured custom vision-encoder backend classes."""

from __future__ import annotations

import importlib
from typing import Any, cast

from dynamo.vllm.multimodal_utils.custom_encoder.backend.base import (
    VisionEncoderBackend,
)


def resolve_vision_encoder_backend_class(
    class_path: str,
) -> type[VisionEncoderBackend[Any, Any, Any]]:
    """Resolve a dotted path to a ``VisionEncoderBackend`` subclass."""

    if not isinstance(class_path, str) or not class_path:
        raise ValueError("custom encoder class must be a non-empty dotted path")
    module_path, separator, class_name = class_path.rpartition(".")
    if not separator or not module_path or not class_name:
        raise ValueError(
            "custom encoder class must use a dotted module.ClassName path; "
            f"got {class_path!r}"
        )
    backend_class = getattr(importlib.import_module(module_path), class_name)
    if not (
        isinstance(backend_class, type)
        and issubclass(backend_class, VisionEncoderBackend)
    ):
        raise TypeError(
            f"{class_path!r} must resolve to a VisionEncoderBackend subclass; "
            f"got {backend_class!r}"
        )
    return cast(type[VisionEncoderBackend[Any, Any, Any]], backend_class)
