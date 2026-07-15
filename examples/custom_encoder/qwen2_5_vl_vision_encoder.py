# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager Qwen2.5-VL tower for validating native custom-encoder inputs."""

from __future__ import annotations

import base64
import binascii
import gc
import io
import logging
import math
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    BackendEncodingSpecV1,
    EncodedMediaResultV1,
    ForwardItemV1,
    Preprocessed,
    Qwen2VLImageEncodingV1,
    VisionEncoderBackend,
)

logger = logging.getLogger(__name__)

_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


@dataclass(frozen=True)
class Qwen2_5VLImageInputs:
    """CPU processor output for one image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class Qwen2_5VLVisionEncoder(VisionEncoderBackend[str, Qwen2_5VLImageInputs]):
    """Run the public Qwen2.5-VL ViT/projector outside vLLM.

    This correctness backend intentionally remains eager. It accepts base64 image
    data URLs so the parity path does not introduce a second network-fetch policy;
    production backends that fetch HTTP URLs must apply Dynamo's media URL policy.
    """

    encoding_spec = BackendEncodingSpecV1(
        adapter_abi="vllm-qwen2-vl-external-v1",
        producer_fingerprint=(
            f"qwen2.5-vl-3b@{_MODEL_REVISION}:"
            "hf-auto-processor-same-revision:post-merger-canonical-raster-v1"
        ),
        expected_decoder_config_fingerprint=(
            f"model={_MODEL_ID}:revision={_MODEL_REVISION}:"
            "Qwen2_5_VLForConditionalGeneration:hidden=2048:merge=2:"
            "dtype=torch.bfloat16"
        ),
        output_dtype="bfloat16",
        hidden_size=2048,
        spatial_merge_size=2,
    )
    preprocess_concurrency = int(
        os.environ.get("DYN_QWEN2_5_VL_PREPROCESS_CONCURRENCY", "4")
    )
    max_batch_cost = int(os.environ.get("DYN_QWEN2_5_VL_MAX_BATCH_TOKENS", "2048"))

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Any | None = None
        self._visual: Any | None = None

    def build(self, model_id: str) -> None:
        if model_id != _MODEL_ID:
            raise ValueError(
                f"{type(self).__name__} is validated only for {_MODEL_ID}; "
                f"got {model_id}"
            )
        if self.preprocess_concurrency < 1:
            raise ValueError("DYN_QWEN2_5_VL_PREPROCESS_CONCURRENCY must be positive")
        if self.max_batch_cost is None or self.max_batch_cost < 1:
            raise ValueError("DYN_QWEN2_5_VL_MAX_BATCH_TOKENS must be positive")

        self._processor = AutoProcessor.from_pretrained(
            model_id, revision=_MODEL_REVISION
        )
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=_MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        model = getattr(full_model, "model", full_model)
        visual = getattr(model, "visual", None)
        if visual is None:
            visual = getattr(full_model, "visual", None)
        if visual is None:
            raise RuntimeError("Qwen2.5-VL checkpoint does not expose a vision tower")
        self._visual = visual.eval().to(self._device)

        if hasattr(model, "visual"):
            model.visual = None
        elif hasattr(full_model, "visual"):
            full_model.visual = None
        del model, full_model
        gc.collect()
        self._run_startup_canary()
        logger.info(
            "%s loaded %s on %s",
            type(self).__name__,
            model_id,
            self._device,
        )

    def preprocess(self, raw: str) -> Preprocessed[Qwen2_5VLImageInputs]:
        image = self._decode_data_url(raw)
        return self._process_image(image)

    def _process_image(self, image: Image.Image) -> Preprocessed[Qwen2_5VLImageInputs]:
        processor = self._require_processor()
        inputs = processor.image_processor(images=[image], return_tensors="pt")
        grid = inputs["image_grid_thw"].to(dtype=torch.int64, device="cpu")
        if grid.shape != (1, 3):
            raise ValueError(
                "Qwen2.5-VL processor must return one image_grid_thw row; "
                f"got {tuple(grid.shape)}"
            )
        merge_size = self.encoding_spec.spatial_merge_size
        if merge_size is None:
            raise RuntimeError("Qwen2.5-VL spatial_merge_size is not configured")
        cost = int(grid.prod().item() // (merge_size**2))
        return Preprocessed(
            item=Qwen2_5VLImageInputs(
                pixel_values=inputs["pixel_values"].contiguous(),
                image_grid_thw=grid,
            ),
            cost=cost,
        )

    def _run_startup_canary(self) -> None:
        """Exercise a near-maximum square image with both GPU tenants resident."""
        visual = self._require_visual()
        processor = self._require_processor()
        image_processor = processor.image_processor
        patch_size = getattr(image_processor, "patch_size", None)
        merge_size = self.encoding_spec.spatial_merge_size
        max_cost = self.max_batch_cost
        if (
            not isinstance(patch_size, int)
            or isinstance(patch_size, bool)
            or patch_size < 1
            or merge_size is None
            or max_cost is None
        ):
            raise RuntimeError("Qwen2.5-VL startup canary configuration is invalid")
        grid_side = math.isqrt(max_cost * (merge_size**2))
        grid_side -= grid_side % merge_size
        if grid_side < merge_size:
            raise RuntimeError("Qwen2.5-VL batch budget is too small for one image")
        image_side = grid_side * patch_size
        prepared = self._process_image(
            Image.new("RGB", (image_side, image_side), color=(127, 127, 127))
        )
        if prepared.cost > max_cost:
            raise RuntimeError(
                "Qwen2.5-VL startup canary exceeded max_batch_cost: "
                f"cost={prepared.cost}, max={max_cost}"
            )
        result = self.forward_batch(
            [ForwardItemV1(correlation_id=b"startup-canary", item=prepared.item)]
        )[0]
        if result.media.projected.shape[0] != prepared.cost:
            raise RuntimeError("Qwen2.5-VL startup canary returned the wrong row count")
        del result, prepared
        if visual.dtype != torch.bfloat16:
            raise RuntimeError(
                f"Qwen2.5-VL visual tower must use bfloat16, got {visual.dtype}"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        logger.info(
            "%s startup co-residency canary passed at %dx%d",
            type(self).__name__,
            image_side,
            image_side,
        )

    def forward_batch(
        self,
        items: list[ForwardItemV1[Qwen2_5VLImageInputs]],
        target_bucket: int | None = None,
    ) -> list[EncodedMediaResultV1]:
        if target_bucket is not None:
            raise ValueError(
                "Qwen2.5-VL correctness backend does not use graph buckets"
            )
        if not items:
            raise ValueError("forward_batch requires at least one image")
        visual = self._require_visual()
        raw_items = [item.item for item in items]
        grid_cpu = torch.cat(
            [item.image_grid_thw for item in raw_items], dim=0
        ).contiguous()
        pixel_values = torch.cat([item.pixel_values for item in raw_items], dim=0).to(
            device=self._device, dtype=visual.dtype
        )
        grid_device = grid_cpu.to(self._device)

        with torch.inference_mode():
            image_embeds = visual(pixel_values, grid_thw=grid_device)
            image_embeds = getattr(image_embeds, "pooler_output", image_embeds)
        merge_size = self.encoding_spec.spatial_merge_size
        if merge_size is None:
            raise RuntimeError("Qwen2.5-VL spatial_merge_size is not configured")
        split_sizes = (grid_cpu.prod(dim=-1) // (merge_size**2)).tolist()
        host_embeds = image_embeds.to(dtype=torch.bfloat16, device="cpu").contiguous()
        split_embeds = torch.split(host_embeds, split_sizes)

        return [
            EncodedMediaResultV1(
                correlation_id=tagged.correlation_id,
                media=Qwen2VLImageEncodingV1(
                    projected=projected,
                    grid_thw=tuple(int(value) for value in raw.image_grid_thw[0]),
                ),
            )
            for tagged, raw, projected in zip(items, raw_items, split_embeds)
        ]

    def close(self) -> None:
        self._visual = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError("Qwen2.5-VL processor is not loaded")
        return self._processor

    def _require_visual(self) -> Any:
        if self._visual is None:
            raise RuntimeError("Qwen2.5-VL vision tower is not loaded")
        return self._visual

    @staticmethod
    def _decode_data_url(source: str) -> Image.Image:
        parsed = urlparse(source)
        if parsed.scheme != "data" or not parsed.path.startswith("image/"):
            raise ValueError(
                "Qwen2.5-VL correctness backend accepts base64 image data URLs only"
            )
        try:
            media_type, payload = parsed.path.split(",", 1)
        except ValueError as exc:
            raise ValueError("invalid image data URL: missing comma") from exc
        if ";base64" not in media_type:
            raise ValueError("image data URL must be base64 encoded")
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise ValueError("image data URL contains invalid base64") from exc
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
