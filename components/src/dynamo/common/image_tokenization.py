# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic image-tokenization contracts shared by worker backends.

Workers publish only the stable spec identifier. Image dimensions, config
validation, per-request counting, and metrics remain frontend-owned.
"""

from enum import Enum

IMAGE_TOKENIZATION_SPEC_RUNTIME_KEY = "image_tokenization_spec"


class ImageTokenizationSpec(str, Enum):
    """Versioned algorithms the Rust frontend can reproduce exactly."""

    QWEN2_VL_V1 = "qwen2_vl_v1"
    QWEN3_VL_V1 = "qwen3_vl_v1"
    MOONVIT_V1 = "moonvit_v1"


def type_identity(value: object) -> str:
    """Return the exact concrete Python type's stable qualified name."""

    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"
