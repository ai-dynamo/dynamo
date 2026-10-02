# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for multimodal router integration tests."""

from collections.abc import Mapping
from io import BytesIO
from types import MappingProxyType
from typing import Any

from PIL import Image

from tests.utils.gpu_args import build_vllm_gpu_mem_args

COMMON_PROCESS_KWARGS: Mapping[str, Any] = MappingProxyType(
    {
        "display_output": False,
        "terminate_all_matching_process_names": False,
    }
)


def make_png_bytes(color: tuple[int, int, int], size: int = 256) -> bytes:
    image = Image.new("RGB", (size, size), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


__all__ = ["COMMON_PROCESS_KWARGS", "build_vllm_gpu_mem_args", "make_png_bytes"]
