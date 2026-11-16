# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for multimodal router integration tests."""

from collections.abc import Mapping
from io import BytesIO
from types import MappingProxyType
from typing import Any

from PIL import Image

from tests.utils.gpu_args import build_vllm_gpu_mem_args, pynvvideocodec_reserved_bytes

# Shared GPU budget for the Qwen3-VL-2B MM router workers: a 2x-safety KV cap
# (min 859_537_408) and its measured solo peak, plus any video decoder
# reservation vLLM takes out of the KV cache.
MM_KV_CACHE_BYTES = 1_719_075_000 + pynvvideocodec_reserved_bytes()
MM_PROFILED_VRAM_GIB = round(7.6 + pynvvideocodec_reserved_bytes() / 1024**3, 1)

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


__all__ = [
    "COMMON_PROCESS_KWARGS",
    "MM_KV_CACHE_BYTES",
    "MM_PROFILED_VRAM_GIB",
    "build_vllm_gpu_mem_args",
    "make_png_bytes",
]
