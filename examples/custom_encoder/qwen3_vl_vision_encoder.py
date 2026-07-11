# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Qwen3-VL vision tower for the CustomEncoder path.

This backend is intended to exercise a real, public vision encoder through
``AsyncVisionEncoder`` and ``ThreadedMicroBatcher``. Qwen3-VL also produces
DeepStack features that its native language model injects at intermediate
decoder layers. The current CustomEncoder contract carries only the primary
image embeddings, so this backend is suitable for integration and performance
testing but does not claim numerical parity with native Qwen3-VL inference.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
from examples.custom_encoder.qwen_vision_encoder import QwenVisionEncoderBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qwen3VLImageInputs:
    """CPU tensors produced by the Qwen3-VL image processor for one image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class Qwen3VLVisionEncoder(QwenVisionEncoderBackend):
    """Real Qwen3-VL ViT/projector behind Dynamo's custom encoder contract."""

    preprocess_concurrency = 4
    max_batch_cost = 8
    buckets = None

    def build(self, model_id: str) -> None:
        """Load the public checkpoint, retain only its vision module, and warm it."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self._processor.tokenizer

        logger.info(
            "[Qwen3VLVisionEncoder] loading %s vision tower on %s",
            model_id,
            self._device,
        )
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self._visual = full_model.model.visual.eval().to(self._device)

        # Detach the retained module before dropping the full checkpoint. This
        # releases the duplicate language-model weights before vLLM starts.
        full_model.model.visual = None
        del full_model
        gc.collect()

        warmup_image = Image.new("RGB", (500, 500), color=(127, 127, 127))
        warmup_item = self._process_image(warmup_image)
        outputs = self.forward_batch([warmup_item] * self.max_batch_cost)
        del outputs, warmup_item
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        logger.info(
            "[Qwen3VLVisionEncoder] warmup complete: max_batch_cost=%d",
            self.max_batch_cost,
        )

    def preprocess(self, raw: str) -> Preprocessed[Qwen3VLImageInputs]:
        """Fetch/decode one image and run the CPU Qwen3-VL image processor."""
        return Preprocessed(item=self._process_image(self._load_image(raw)), cost=1)

    def _process_image(self, image: Image.Image) -> Qwen3VLImageInputs:
        inputs = self._processor.image_processor(images=[image], return_tensors="pt")
        return Qwen3VLImageInputs(
            pixel_values=inputs["pixel_values"].contiguous(),
            image_grid_thw=inputs["image_grid_thw"].to(dtype=torch.long),
        )

    def forward_batch(
        self,
        items: List[Qwen3VLImageInputs],
        target_bucket: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Pack variable-resolution patches and run one eager vision forward."""
        if not items:
            raise ValueError("forward_batch requires at least one image")
        if len(items) > self.max_batch_cost:
            raise ValueError(
                f"batch size {len(items)} exceeds max_batch_cost={self.max_batch_cost}"
            )
        if target_bucket is not None:
            raise ValueError(
                "Qwen3VLVisionEncoder is eager; target_bucket must be None"
            )
        if getattr(self, "_visual", None) is None:
            raise RuntimeError(
                "Qwen3VLVisionEncoder.forward_batch() called before build()"
            )

        pixel_values = torch.cat([item.pixel_values for item in items], dim=0).to(
            device=self._device,
            dtype=self._visual.dtype,
            non_blocking=True,
        )
        image_grid_thw_cpu = torch.cat([item.image_grid_thw for item in items], dim=0)
        image_grid_thw = image_grid_thw_cpu.to(self._device, non_blocking=True)
        split_sizes = (
            image_grid_thw_cpu.prod(dim=-1) // self._visual.spatial_merge_size**2
        ).tolist()

        with torch.inference_mode():
            visual_output = self._visual(pixel_values, grid_thw=image_grid_thw)
        # Transformers 4.x returns ``(merged, deepstack)``. Transformers 5.x
        # returns BaseModelOutputWithDeepstackFeatures, where the merged image
        # embeddings live in pooler_output (last_hidden_state is pre-merger).
        image_embeds = (
            visual_output.pooler_output
            if hasattr(visual_output, "pooler_output")
            else visual_output[0]
        )

        outputs = [
            embed.to(dtype=torch.bfloat16).cpu().clone()
            for embed in torch.split(image_embeds, split_sizes)
        ]
        logger.debug(
            "[Qwen3VLVisionEncoder] forward_batch n=%d tokens=%s",
            len(items),
            split_sizes,
        )
        return outputs

    def close(self) -> None:
        """Release model and processor references on the actor thread."""
        self._visual = None
        self._processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _load_image(source: str) -> Image.Image:
        """Load an RGB image from a data URL, HTTP(S) URL, or local path."""
        if source.startswith("data:"):
            try:
                header, payload = source.split(",", 1)
            except ValueError as exc:
                raise ValueError("invalid data URL: missing comma") from exc
            if ";base64" not in header:
                raise ValueError("only base64-encoded image data URLs are supported")
            raw = base64.b64decode(payload, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        if source.startswith(("http://", "https://")):
            with urllib.request.urlopen(source, timeout=15) as response:  # nosec B310
                raw = response.read()
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        with Image.open(source) as image:
            return image.convert("RGB")
