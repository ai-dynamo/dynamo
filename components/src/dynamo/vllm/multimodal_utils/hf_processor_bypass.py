# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bypass vLLM's multimodal input processing for Qwen VL models.

When Dynamo's Rust preprocessor has already tokenized the prompt and decoded
images from base64, vLLM's ``_process_multimodal`` re-runs the HF processor
to resize/normalize images and expand placeholder tokens.  This module
performs those steps directly, building a ``MultiModalInput`` dict that vLLM
accepts without re-processing.

Savings: eliminates ~700ms of vLLM wrapper overhead per request (cache
serialisation, UUID processing, dummy-text tokenization, prompt-update
matching, etc.).
"""

import logging
import time
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoTokenizer

from vllm.inputs.engine import MultiModalInput
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)

from .hash_utils import compute_mm_uuids_from_images
from .models.qwen import QwenGridParams

logger = logging.getLogger(__name__)


class QwenHFProcessorBypass:
    """Pre-process images for Qwen VL, building a vLLM-ready MultiModalInput.

    Instantiated once at handler init; thread-safe for concurrent requests
    (the HF image processor is stateless).
    """

    def __init__(
        self,
        model_name: str,
        grid_params: QwenGridParams,
        target_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._model_name = model_name
        self._grid_params = grid_params
        self._merge_size = grid_params.merge_size
        self._target_dtype = target_dtype

        # Load the HF image processor for resize + normalize.
        # We only need the image_processor, not the full processor.
        try:
            full_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._image_processor = full_processor.image_processor
            self._image_token_id: int = full_processor.image_token_id
        except Exception:
            logger.warning(
                "Failed to load HF processor for bypass; falling back to AutoImageProcessor"
            )
            self._image_processor = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            # Fallback: compute from tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # Qwen special token IDs
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self._vision_start_id: int = hf_config.vision_start_token_id
        self._vision_end_id: int = hf_config.vision_end_token_id

        logger.info(
            "QwenHFProcessorBypass initialized for %s "
            "(image_token_id=%d, vision_start=%d, vision_end=%d)",
            model_name,
            self._image_token_id,
            self._vision_start_id,
            self._vision_end_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_mm_input(
        self,
        token_ids: List[int],
        images: List[Image.Image],
        mm_uuids: Optional[List[str]] = None,
    ) -> MultiModalInput:
        """Build a vLLM ``MultiModalInput`` from token IDs + PIL images.

        This does everything ``_process_multimodal`` would do, but without
        the vLLM wrapper overhead.

        Args:
            token_ids: Prompt token IDs (with unexpanded placeholders from Rust).
            images: PIL images in order of appearance.
            mm_uuids: Pre-computed blake3 hex hashes per image; computed here
                      if not provided.

        Returns:
            A ``MultiModalInput`` dict with ``type="multimodal"`` that vLLM
            accepts directly via the fast-path in ``process_inputs``.
        """
        t0 = time.perf_counter()
        num_images = len(images)

        # 1. Run HF image processor (resize + normalize → pixel_values + grid_thw)
        hf_out = self._run_image_processor(images)
        pixel_values: torch.Tensor = hf_out["pixel_values"]  # (total_patches, C, H, W)
        image_grid_thw: torch.Tensor = hf_out["image_grid_thw"]  # (N, 3)
        t_hf = time.perf_counter()

        # 2. Compute per-image token counts and build mm_kwargs
        merge_sq = self._merge_size**2
        per_image_tokens: List[int] = []
        per_image_pixel_sizes: List[int] = []
        for i in range(num_images):
            grid = image_grid_thw[i]
            n_tokens = int(grid.prod().item()) // merge_sq
            n_pixels = int(grid.prod().item())
            per_image_tokens.append(n_tokens)
            per_image_pixel_sizes.append(n_pixels)

        mm_kwargs = self._build_mm_kwargs(
            pixel_values, image_grid_thw, per_image_pixel_sizes
        )

        # 3. Expand placeholder tokens and record placeholder ranges
        expanded_ids, placeholders = self._expand_placeholders(
            token_ids, per_image_tokens
        )

        # 4. Compute hashes
        if mm_uuids is None:
            mm_uuids = compute_mm_uuids_from_images(images)
        mm_hashes: Dict[str, List[str]] = {"image": mm_uuids}

        # 5. Build the final MultiModalInput
        mm_placeholders: Dict[str, List[PlaceholderRange]] = {"image": placeholders}

        result = MultiModalInput(
            type="multimodal",
            prompt_token_ids=expanded_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        hf_ms = (t_hf - t0) * 1000
        logger.info(
            "[PERF] hf_processor_bypass total_ms=%.2f hf_image_proc_ms=%.2f "
            "images=%d tokens=%s",
            elapsed,
            hf_ms,
            num_images,
            per_image_tokens,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_image_processor(
        self, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        """Run the HF image processor on a batch of PIL images.

        Returns dict with 'pixel_values' and 'image_grid_thw' tensors.
        """
        # The image processor handles resize, rescale, normalize, and
        # computes image_grid_thw.  We call it with return_tensors="pt".
        result = self._image_processor(images=images, return_tensors="pt")
        pixel_values = result["pixel_values"]
        # Match vLLM's dtype postprocessing: convert float tensors to model dtype.
        # The vision encoder will do this anyway, but matching vLLM's behaviour
        # avoids surprises with IPC caching and hash-based dedup.
        if pixel_values.is_floating_point() and self._target_dtype is not None:
            pixel_values = pixel_values.to(dtype=self._target_dtype)
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": result["image_grid_thw"],
        }

    def _build_mm_kwargs(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        per_image_pixel_sizes: List[int],
    ) -> MultiModalKwargsItems:
        """Build ``MultiModalKwargsItems`` from processed image data.

        Mirrors what ``MultiModalKwargsItems.from_hf_inputs`` does for
        Qwen VL, but without the overhead of BatchFeature creation and
        the generic field-config dispatch.
        """
        # Build per-image slices for MultiModalFlatField (needed for reduce)
        # Each image contributes per_image_pixel_sizes[i] rows to pixel_values.
        offsets: List[int] = [0]
        for sz in per_image_pixel_sizes:
            offsets.append(offsets[-1] + sz)
        per_image_slices = [
            slice(offsets[i], offsets[i + 1]) for i in range(len(per_image_pixel_sizes))
        ]

        # Shared field objects — all items reference the same field so
        # reduce_data works correctly when batching across items.
        pv_field = MultiModalFlatField(slices=per_image_slices)
        grid_field = MultiModalBatchedField(keep_on_cpu=True)

        # Split pixel_values by image
        pixel_splits = pixel_values.split(per_image_pixel_sizes, dim=0)

        items: List[MultiModalKwargsItem] = []
        for i, pv in enumerate(pixel_splits):
            grid = image_grid_thw[i : i + 1]  # keep as (1, 3) tensor

            pv_elem = MultiModalFieldElem(data=pv, field=pv_field)
            grid_elem = MultiModalFieldElem(data=grid, field=grid_field)

            item = MultiModalKwargsItem(
                {
                    "pixel_values": pv_elem,
                    "image_grid_thw": grid_elem,
                }
            )
            items.append(item)

        return MultiModalKwargsItems({"image": items})

    def _expand_placeholders(
        self,
        token_ids: List[int],
        per_image_tokens: List[int],
    ) -> tuple[List[int], List[PlaceholderRange]]:
        """Replace ``<|vision_start|><|image_pad|><|vision_end|>`` sequences
        in *token_ids* with the correct number of image-pad tokens, and return
        the expanded IDs plus placeholder ranges.

        The Rust preprocessor emits exactly one ``<|image_pad|>`` token between
        ``<|vision_start|>`` and ``<|vision_end|>`` for each image.  vLLM's
        ``_apply_prompt_updates`` would normally expand these; we do the same
        thing here.
        """
        vs = self._vision_start_id
        ip = self._image_token_id
        ve = self._vision_end_id

        expanded: List[int] = []
        placeholders: List[PlaceholderRange] = []
        img_idx = 0
        i = 0
        n = len(token_ids)

        while i < n:
            # Look for the 3-token pattern: vision_start, image_pad, vision_end
            if (
                i + 2 < n
                and token_ids[i] == vs
                and token_ids[i + 1] == ip
                and token_ids[i + 2] == ve
                and img_idx < len(per_image_tokens)
            ):
                n_tokens = per_image_tokens[img_idx]
                # Keep vision_start; Qwen2-VL needs boundary tokens for mRoPE
                # position computation and attention masking.
                expanded.append(vs)
                # Record placeholder offset AFTER vision_start
                offset = len(expanded)
                expanded.extend([ip] * n_tokens)
                placeholders.append(PlaceholderRange(offset=offset, length=n_tokens))
                # Keep vision_end
                expanded.append(ve)
                img_idx += 1
                i += 3
            else:
                expanded.append(token_ids[i])
                i += 1

        if img_idx != len(per_image_tokens):
            logger.warning(
                "Expected %d image placeholders but found %d in token_ids",
                len(per_image_tokens),
                img_idx,
            )

        return expanded, placeholders
