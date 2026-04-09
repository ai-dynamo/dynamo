# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QwenGridParams:
    """Cached Qwen VL image processor parameters for grid_thw computation."""

    patch_size: int
    merge_size: int
    factor: int
    min_pixels: int
    max_pixels: int
    vision_hidden_dim: int


def load_qwen_grid_params(model_name: str) -> QwenGridParams | None:
    """Load Qwen VL grid parameters from model config.

    Reads AutoImageProcessor and vision_config at init time so that
    grid_thw can be computed from image dimensions alone (no GPU needed).

    Returns None if loading fails (e.g. model not cached locally).
    """
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        vision_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        ).vision_config

        patch_size: int = processor.patch_size
        merge_size: int = processor.merge_size
        factor = patch_size * merge_size

        # Qwen2/2.5-VL use min_pixels/max_pixels directly.
        # Qwen3-VL sets them to None and uses size.shortest_edge/longest_edge.
        min_pixels: int = (
            processor.min_pixels
            if processor.min_pixels is not None
            else processor.size.get("shortest_edge", factor)
        )
        max_pixels: int = (
            processor.max_pixels
            if processor.max_pixels is not None
            else processor.size.get("longest_edge", factor * factor * 1280)
        )
        vision_hidden_dim: int = getattr(
            vision_config, "out_hidden_size", vision_config.hidden_size
        )

        return QwenGridParams(
            patch_size=patch_size,
            merge_size=merge_size,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            vision_hidden_dim=vision_hidden_dim,
        )
    except (OSError, ValueError) as exc:
        logger.warning(
            "Failed to load Qwen VL image processor for %s: %s. "
            "P/D disaggregation without encode worker will not "
            "produce embedding_params for decode.",
            model_name,
            exc,
            exc_info=True,
        )
        return None


def _compute_qwen_grid_thw(
    image_data: Any,
    params: QwenGridParams,
) -> tuple[list[list[int]], list[int]] | tuple[None, None]:
    """Compute image_grid_thw and embeddings_shape from PIL images.

    Uses smart_resize with cached processor parameters — the same
    resize logic that vLLM/transformers uses internally, but without
    the overhead of full image preprocessing (~0.4us vs ~5ms).

    Args:
        image_data: Single PIL.Image.Image or list of them.
        params: Cached grid parameters from load_qwen_grid_params().

    Returns:
        (grid_thw, embeddings_shape) or (None, None) on failure.
        grid_thw: list of [grid_t, grid_h, grid_w] per image.
        embeddings_shape: [total_tokens, vision_hidden_dim].
    """
    if isinstance(image_data, Image.Image):
        images = [image_data]
    elif isinstance(image_data, list):
        images = [img for img in image_data if isinstance(img, Image.Image)]
    else:
        return None, None

    if not images:
        return None, None

    grid_thw: list[list[int]] = []
    total_tokens = 0
    merge_sq = params.merge_size**2

    for img in images:
        w, h = img.size  # PIL is (width, height)
        rh, rw = smart_resize(
            h,
            w,
            factor=params.factor,
            min_pixels=params.min_pixels,
            max_pixels=params.max_pixels,
        )
        grid_t = 1  # single image, temporal dim always 1
        grid_h = rh // params.patch_size
        grid_w = rw // params.patch_size
        grid_thw.append([grid_t, grid_h, grid_w])
        total_tokens += (grid_t * grid_h * grid_w) // merge_sq

    return grid_thw, [total_tokens, params.vision_hidden_dim]


def build_qwen_embedding_params(
    multi_modal_data: Dict[str, Any],
    grid_params: QwenGridParams | None,
) -> Dict[str, Any] | None:
    """Build embedding parameters for Qwen VL decode.

    Qwen VL's processor expands image tokens using image_grid_thw for mRoPE
    position initialization. The decode worker needs this metadata even though
    it doesn't re-encode images — the KV cache has the vision context but the
    processor still needs grid dimensions to compute positions.

    Two input paths depending on how prefill processed images:

    1. **Encode worker path** (dict): The encode worker produced embeddings
       via the embedding loader. ``multi_modal_data["image"]`` is a dict with
       ``image_embeds`` (tensor) and ``image_grid_thw`` (tensor/list).
       We extract and serialize them for transfer to decode.

    2. **PIL path** (no encode worker): Prefill loaded images directly as
       PIL.Image objects. We compute grid_thw from image dimensions using
       smart_resize with cached processor parameters (~0.4us per image).

    Args:
        multi_modal_data: The multimodal data dict from prefill processing.
        grid_params: Cached Qwen VL processor parameters, or None if
            loading failed at init time.

    Returns:
        Dict with ``image_grid_thw`` and ``embeddings_shape``, or None if
        no image data or parameters are unavailable.
    """
    embedding_params: Dict[str, Any] = {}
    image_data = multi_modal_data.get("image")
    if isinstance(image_data, dict):
        # Path 1: encode worker produced embeddings as a dict
        image_grid_thw = image_data.get("image_grid_thw")
        image_embeds = image_data.get("image_embeds")
        if image_grid_thw is not None:
            embedding_params["image_grid_thw"] = (
                image_grid_thw.tolist()
                if isinstance(image_grid_thw, torch.Tensor)
                else image_grid_thw
            )
        if image_embeds is not None:
            embedding_params["embeddings_shape"] = list(image_embeds.shape)
    elif image_data is not None and grid_params is not None:
        # Path 2: PIL images — compute grid_thw from image dimensions
        grid_thw, embeddings_shape = _compute_qwen_grid_thw(image_data, grid_params)
        if grid_thw is not None:
            embedding_params["image_grid_thw"] = grid_thw
            embedding_params["embeddings_shape"] = embeddings_shape
    # TODO(DIS-1679): handle np.ndarray from --frontend-decoding NIXL path
    return embedding_params if embedding_params else None
