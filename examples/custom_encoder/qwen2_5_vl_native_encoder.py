# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-VL producer for Dynamo's native external-MM adapter."""

from __future__ import annotations

from typing import List, Optional

from dynamo.vllm.multimodal_utils.vision_encoder_backend import Qwen2VLImageEncoding
from examples.custom_encoder.qwen2_5_vl_benchmark_encoder import (
    Qwen2_5VLBenchmarkEncoder,
    Qwen2VLImageInputs,
)


class Qwen2_5VLNativeEncoder(Qwen2_5VLBenchmarkEncoder):
    """Return native Qwen projected rows plus pre-merge ``grid_thw``.

    The inherited implementation owns image processing, Qwen2.5 window-attention
    ordering, eager execution, and optional custom-encoder CUDA graphs. This class
    only declares and packages the producer artifact. Dynamo's resolved decoder
    still selects the engine prompt adapter.
    """

    output_format = "qwen2_vl_projected_grid"

    def forward_batch(
        self,
        items: List[Qwen2VLImageInputs],
        target_bucket: Optional[int] = None,
    ) -> list[Qwen2VLImageEncoding]:
        projected_rows = super().forward_batch(items, target_bucket=target_bucket)
        return [
            Qwen2VLImageEncoding(
                projected=projected,
                grid_thw=self._grid_key(item),
            )
            for item, projected in zip(items, projected_rows, strict=True)
        ]
