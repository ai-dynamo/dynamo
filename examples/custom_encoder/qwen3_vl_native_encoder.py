# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Qwen3-VL producer for Dynamo's native prompt adapter."""

from __future__ import annotations

import gc
import io
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dynamo.vllm.multimodal_utils.custom_encoder import (
    Preprocessed,
    Qwen3VLImageEncoding,
    VisionEncoderBackend,
)


@dataclass(frozen=True)
class Qwen3VLImageInputs:
    """CPU image-processor output for one image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class Qwen3VLNativeEncoder(
    VisionEncoderBackend[str, Qwen3VLImageInputs, Qwen3VLImageEncoding]
):
    """Run the Qwen3-VL vision tower and return native packed features."""

    preprocess_concurrency = 4

    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._processor: Any | None = None
        self._visual: Any | None = None

    def build(self, model_id: str) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self._visual = model.model.visual.eval().to(self._device)
        model.model.visual = None
        del model
        gc.collect()

    def preprocess(self, raw: str) -> Preprocessed[Qwen3VLImageInputs]:
        processor = self._processor
        if processor is None:
            raise RuntimeError("Qwen3VLNativeEncoder is not loaded")
        with urllib.request.urlopen(raw, timeout=30) as response:
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
        inputs = processor.image_processor(images=[image], return_tensors="pt")
        return Preprocessed(
            item=Qwen3VLImageInputs(
                pixel_values=inputs["pixel_values"].contiguous(),
                image_grid_thw=inputs["image_grid_thw"].to(dtype=torch.long),
            )
        )

    def forward_batch(
        self,
        items: list[Qwen3VLImageInputs],
        target_bucket: int | None = None,
    ) -> list[Qwen3VLImageEncoding]:
        del target_bucket
        visual = self._visual
        if visual is None:
            raise RuntimeError("Qwen3VLNativeEncoder is not loaded")

        pixel_values = torch.cat([item.pixel_values for item in items], dim=0).to(
            device=self._device,
            dtype=visual.dtype,
        )
        image_grid_thw_cpu = torch.cat([item.image_grid_thw for item in items], dim=0)
        image_grid_thw = image_grid_thw_cpu.to(self._device)

        with torch.inference_mode():
            output = visual(pixel_values, grid_thw=image_grid_thw)
        embeddings = torch.cat(
            [output.pooler_output, *output.deepstack_features], dim=-1
        ).to(dtype=torch.bfloat16, device="cpu")
        split_sizes = (
            image_grid_thw_cpu.prod(dim=-1) // visual.spatial_merge_size**2
        ).tolist()

        return [
            Qwen3VLImageEncoding(
                embeddings=rows,
                grid_thw=(int(grid[0]), int(grid[1]), int(grid[2])),
            )
            for rows, grid in zip(
                torch.split(embeddings, split_sizes),
                image_grid_thw_cpu.tolist(),
                strict=True,
            )
        ]

    def close(self) -> None:
        self._processor = None
        self._visual = None
