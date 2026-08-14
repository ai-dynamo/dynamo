# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration shared by the remote user-ensemble example."""

from __future__ import annotations

import importlib
from typing import Any, cast

from dynamo.vllm.multimodal_utils.custom_encoder import VisionEncoderBackend

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ENCODER_CLASS = (
    "examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder"
)


def load_encoder_backend(
    class_path: str,
) -> type[VisionEncoderBackend[Any, Any, Any]]:
    """Resolve one configured external encoder backend class."""

    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError(
            "--encoder-class must be a dotted module.ClassName path; "
            f"got {class_path!r}"
        )
    backend_type = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(backend_type, type) or not issubclass(
        backend_type, VisionEncoderBackend
    ):
        raise TypeError(f"{class_path} must name a VisionEncoderBackend subclass")
    return cast(type[VisionEncoderBackend[Any, Any, Any]], backend_type)
