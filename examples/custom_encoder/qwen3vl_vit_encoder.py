# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Qwen3-VL vision-tower ``VisionEncoderBackend`` (in-process, CUDA-graphed).

Loads the **actual** Qwen3-VL vision tower (ViT patch-embed + transformer blocks +
spatial merger) and runs it on the ``AsyncVisionEncoder`` dedicated actor thread.
It demonstrates the **bucket-ladder** graph scheme (design theme D2, the vLLM
``EncoderCudaGraphManager`` pattern) end to end:

- ``buckets`` exposes a sorted ladder of **merged-visual-token** rungs.
- The ``ThreadedMicroBatcher`` packs images by ``cost`` (merged tokens) and rounds
  the packed ``sum(cost)`` **up to the nearest rung**, passing it as
  ``target_bucket``.
- ``forward_batch`` **pads** the packed batch up to ``target_bucket`` (appends
  dummy images), replays the graph captured for that rung's shape, and slices the
  real images' embeds back out.

Padding the input to a rung quantises the forward's shape to the (bounded) ladder,
so ``torch.compile(mode="reduce-overhead")`` captures **one CUDA graph per rung**
(not one per arbitrary batch size — that would be SGLang's unbounded per-exact-S
scheme, which the design rejects). Capture + replay both happen on the actor
thread, the affinity the batcher guarantees.

Stable per-image shape: every image is resized to one fixed square, so each image
is exactly ``TOKENS_PER_SIDE**2`` merged tokens — every rung is a whole number of
images and padding is always whole dummy images. ``forward_batch`` copies its
output to CPU, which also detaches it from the reused CUDA-graph output buffer.

Limitation (intentional for this harness): Qwen3-VL's vision tower also emits
``deepstack_features`` that the LM injects at specific layers. The contract
returns one embed tensor per image (the merged ``pooler_output``); deepstack
features are not plumbed through the mixed-embeds splice path, so image grounding
is approximate. This exercises the in-process encoder + batcher + CUDA-graph
bucket-ladder path, **not** Qwen3-VL accuracy.

Usage (via agg_custom.sh):
    DYN_MODEL=Qwen/Qwen3-VL-2B-Instruct
    DYN_ENCODER_CLASS=examples.custom_encoder.qwen3vl_vit_encoder.Qwen3VLViTEncoder
    DYN_WORKER_GPU=2 ./agg_custom.sh

Env knobs:
    DYN_VIT_COMPILE          1 (default) → torch.compile + bucket ladder; 0 → eager (no graphs)
    DYN_VIT_TOKENS_PER_SIDE  merged visual tokens per image side (default 16 → 256/img)
    DYN_VIT_MAX_IMAGES       max images per forward → top rung (default 16)
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
from examples.custom_encoder.qwen_vision_encoder import QwenVisionEncoderBackend

logger = logging.getLogger(__name__)

_COMPILE = os.environ.get("DYN_VIT_COMPILE", "1") == "1"
_TOKENS_PER_SIDE = int(os.environ.get("DYN_VIT_TOKENS_PER_SIDE", "16"))
_MAX_IMAGES = int(os.environ.get("DYN_VIT_MAX_IMAGES", "16"))


def _load_image(image_url: str) -> Image.Image:
    """Load a PIL RGB image from a data: URI, http(s) URL, or local path."""
    if image_url.startswith("data:"):
        b64 = image_url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if image_url.startswith(("http://", "https://")):
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(image_url).convert("RGB")


def _power_of_two_ladder(tokens_per_img: int, max_images: int) -> List[int]:
    """Merged-token rungs at 1, 2, 4, ... images up to ``max_images``.

    Power-of-two image counts (not every count) so the ladder — and thus the
    captured-graph count — stays bounded; sub-rung batches pad up to the next."""
    mults: List[int] = []
    m = 1
    while m < max_images:
        mults.append(m)
        m *= 2
    mults.append(max_images)
    return sorted({tokens_per_img * x for x in mults})


class Qwen3VLViTEncoder(QwenVisionEncoderBackend):
    """In-process Qwen3-VL vision-tower backend with a CUDA-graphed bucket ladder."""

    def __init__(self) -> None:
        self._tokens_per_img = _TOKENS_PER_SIDE**2
        if _COMPILE:
            # Graphed: expose the ladder; the top rung is the dispatch ceiling.
            self.buckets = _power_of_two_ladder(self._tokens_per_img, _MAX_IMAGES)
            self.max_batch_cost = self.buckets[-1]
        else:
            # Eager: no ladder; pack up to max_images worth of tokens, no padding.
            self.buckets = None
            self.max_batch_cost = self._tokens_per_img * _MAX_IMAGES

    def build(self, model_id: str, device: str) -> None:
        """Load tokenizer (Qwen base) + processor + ViT; compile and warm up so one
        CUDA graph per rung is captured on this (the actor) thread."""
        super().build(model_id, device)  # self.tokenizer
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16
        )
        # Qwen3VLForConditionalGeneration.model is Qwen3VLModel(.visual, .language_model).
        inner = getattr(model, "model", model)
        visual = getattr(inner, "visual", None) or getattr(model, "visual")
        self.visual = visual.to(device).eval()
        del model  # drop the LM half; the vLLM worker owns the LM
        torch.cuda.empty_cache()

        vc = self.visual.config
        self.merge = int(vc.spatial_merge_size)
        patch = int(vc.patch_size)
        # Fixed square so grid_thw (→ cost/bucket_key) is constant: every image is
        # exactly _tokens_per_img merged tokens.
        self.side = patch * self.merge * _TOKENS_PER_SIDE
        self._fixed_hw = (self.side, self.side)

        # One dummy (gray) image's processed tensors, reused for padding to a rung.
        self._dummy = self._process(Image.new("RGB", self._fixed_hw, (127, 127, 127)))

        self._eager_visual = self.visual
        self._compiled = False
        if _COMPILE:
            self.visual = torch.compile(self.visual, mode="reduce-overhead")
            self._compiled = True

        self._warmup()

    def _warmup(self) -> None:
        """Forward once at **each rung** so torch.compile captures that rung's CUDA
        graph here; fall back to eager if compile/capture fails."""
        try:
            rungs = self.buckets if self.buckets else [self._tokens_per_img]
            for _ in range(2 if self._compiled else 1):
                for rung in rungs:
                    # One real image padded up to the rung (the forward_batch path).
                    self.forward_batch(
                        [self._dummy], target_bucket=rung if self._compiled else None
                    )
            torch.cuda.synchronize()
            logger.info(
                "[Qwen3VLViTEncoder] ready: side=%d merge=%d compile=%s "
                "tokens/img=%d buckets=%s max_batch_cost=%d",
                self.side,
                self.merge,
                self._compiled,
                self._tokens_per_img,
                list(self.buckets) if self.buckets else None,
                self.max_batch_cost,
            )
        except Exception as exc:  # noqa: BLE001 — compile/capture is best-effort
            if self._compiled:
                logger.warning(
                    "[Qwen3VLViTEncoder] compile/warmup failed (%s); using eager", exc
                )
                self.visual = self._eager_visual
                self._compiled = False
                self.buckets = None
                self.forward_batch([self._dummy])
                torch.cuda.synchronize()
            else:
                raise

    # ---- preprocess (off the actor thread) ---------------------------------

    def preprocess(self, image_url: str) -> Preprocessed[Dict[str, Any]]:
        """Off-thread: fetch + resize to the fixed square + HF patchify.

        ``cost`` = merged visual tokens for this image; ``bucket_key`` = its
        ``grid_thw`` (constant here, so all images share one bucket)."""
        img = _load_image(image_url).resize(self._fixed_hw)
        item = self._process(img)
        t, h, w = item["grid_thw"][0].tolist()
        cost = (t * h * w) // (self.merge**2)
        return Preprocessed(item=item, cost=cost, bucket_key=(t, h, w))

    def _process(self, img: Image.Image) -> Dict[str, Any]:
        out = self.processor.image_processor(images=[img], return_tensors="pt")
        return {"pixel_values": out["pixel_values"], "grid_thw": out["image_grid_thw"]}

    # ---- forward (on the actor thread) -------------------------------------

    @torch.inference_mode()
    def forward_batch(
        self, items: List[Dict[str, Any]], target_bucket: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Pad the packed batch up to ``target_bucket``, replay that rung's graph,
        and slice the real images' embeds back out (one tensor per item, in order)."""
        if os.environ.get("DYN_VIT_LOG_BATCH") == "1":
            logger.info(
                "[Qwen3VLViTEncoder] forward_batch images=%d target_bucket=%s",
                len(items),
                target_bucket,
            )
        n_real = len(items)
        pix_parts = [it["pixel_values"] for it in items]
        grid_parts = [it["grid_thw"] for it in items]

        # Pad up to the rung (whole dummy images) so the forward shape is one of the
        # captured rungs. target_bucket is None in eager mode → no padding.
        if target_bucket is not None:
            real_tokens = sum(
                (t * h * w) // (self.merge**2)
                for t, h, w in torch.cat(grid_parts, dim=0).tolist()
            )
            n_dummy = (target_bucket - real_tokens) // self._tokens_per_img
            for _ in range(max(0, n_dummy)):
                pix_parts.append(self._dummy["pixel_values"])
                grid_parts.append(self._dummy["grid_thw"])

        pix = torch.cat(pix_parts, dim=0).to(self.device, dtype=torch.bfloat16)
        grid = torch.cat(grid_parts, dim=0).to(self.device)
        embeds = self.visual(pix, grid).pooler_output  # (total_merged_tokens, hidden)
        sizes = [(t * h * w) // (self.merge**2) for t, h, w in grid.tolist()]
        parts = torch.split(embeds, sizes, dim=0)
        # Copy to CPU: detaches from any reused CUDA-graph output buffer and matches
        # the assembler's CPU prompt_embeds layout (it preserves dtype). Drop the
        # padding dummies — return only the real images, in input order.
        return [parts[i].detach().to("cpu", copy=True) for i in range(n_real)]

    def close(self) -> None:
        """Drop the ViT + compiled graphs on the actor thread."""
        self.visual = None
        self._eager_visual = None
        torch.cuda.empty_cache()
