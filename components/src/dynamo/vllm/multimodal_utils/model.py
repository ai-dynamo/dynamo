# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from vllm import LLM
from vllm.utils.system_utils import update_environment_variables

logger = logging.getLogger(__name__)

# [gluo NOTE] Debug flag to compare vLLM encoder vs transformers encoder,
# should be removed once there is proper way to extract vLLM encoder.
VLLM_ENCODER = int(os.getenv("VLLM_ENCODER", 1))


class SupportedModels:
    """Supported multimodal model identifiers"""

    # TODO: Replace this explicit model list with dynamic detection using
    # HF config `architectures` field or vLLM's model registry, so any
    # vLLM-supported VLM works without maintaining entries here.

    LLAVA_1_5_7B = "llava-hf/llava-1.5-7b-hf"
    QWEN_2_VL_2B = "Qwen/Qwen2-VL-2B-Instruct"
    QWEN_2_5_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN_2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN_2_5_VL_32B = "Qwen/Qwen2.5-VL-32B-Instruct"
    QWEN_3_VL_2B = "Qwen/Qwen3-VL-2B-Instruct"
    QWEN_3_VL_30B_A3B = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    QWEN_3_VL_30B_A3B_FP8 = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    QWEN_3_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
    QWEN_3_VL_8B_FP8 = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    QWEN_3_VL_4B = "Qwen/Qwen3-VL-4B-Instruct"
    QWEN_3_VL_4B_FP8 = "Qwen/Qwen3-VL-4B-Instruct-FP8"
    QWEN_3_VL_32B = "Qwen/Qwen3-VL-32B-Instruct"
    QWEN_3_VL_32B_FP8 = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    LLAVA_NEXT_VIDEO_7B = "llava-hf/LLaVA-NeXT-Video-7B-hf"


def normalize_model_name(model_name: str) -> str:
    """
    Extract and normalize model name from various formats including HuggingFace cache paths.

    Args:
        model_name: Model identifier which can be:
            - A simple model name: "Qwen/Qwen2.5-VL-7B-Instruct"
            - A HuggingFace cache path: "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/..."
            - A local path to a model directory

    Returns:
        Normalized model name in the format "organization/model-name"

    Examples:
        >>> normalize_model_name("Qwen/Qwen2.5-VL-7B-Instruct")
        "Qwen/Qwen2.5-VL-7B-Instruct"
        >>> normalize_model_name("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/...")
        "Qwen/Qwen2.5-VL-7B-Instruct"
    """
    # If it's already a simple model name (org/model format), return as-is
    if "/" in model_name and not model_name.startswith("/"):
        return model_name

    # Handle HuggingFace cache paths
    if "models--" in model_name:
        # Extract from cache path format: models--ORG--MODEL-NAME
        # Split on "models--" then on "--" to handle dashes in org/model names
        parts_after_models = model_name.split("models--", 1)
        if len(parts_after_models) > 1:
            # Split the remaining part on "--" and take the last two segments
            segments = parts_after_models[1].split("--")
            if len(segments) >= 2:
                # Take all segments except the last as org (rejoined with dashes)
                # and the last segment (before any slash) as model name
                org_segments = segments[:-1]
                model_segment = segments[-1].split("/")[
                    0
                ]  # Remove any path after model name

                org = "--".join(org_segments)  # Rejoin org parts with dashes
                model = model_segment
                return f"{org}/{model}"

    # Handle local directory paths - extract the last directory name
    path = Path(model_name)
    if path.exists() and path.is_dir():
        return path.name

    # If no pattern matches, return the original name
    return model_name


def is_model_supported(model_name: str, supported_model: str) -> bool:
    """
    Check if a model name matches a supported model, handling various naming formats.

    Args:
        model_name: The model name to check (may be path, cache name, etc.)
        supported_model: The supported model identifier

    Returns:
        True if the model is supported, False otherwise
    """
    normalized_name = normalize_model_name(model_name).lower()
    normalized_supported = normalize_model_name(supported_model).lower()

    return normalized_name == normalized_supported


# List of all Qwen VL model variants for easy extension
QWEN_VL_MODELS = [
    SupportedModels.QWEN_2_VL_2B,
    SupportedModels.QWEN_2_5_VL_3B,
    SupportedModels.QWEN_2_5_VL_7B,
    SupportedModels.QWEN_2_5_VL_32B,
    SupportedModels.QWEN_3_VL_2B,
    SupportedModels.QWEN_3_VL_30B_A3B,
    SupportedModels.QWEN_3_VL_30B_A3B_FP8,
    SupportedModels.QWEN_3_VL_8B,
    SupportedModels.QWEN_3_VL_8B_FP8,
    SupportedModels.QWEN_3_VL_4B,
    SupportedModels.QWEN_3_VL_4B_FP8,
    SupportedModels.QWEN_3_VL_32B,
    SupportedModels.QWEN_3_VL_32B_FP8,
]


def is_qwen_vl_model(model_name: str) -> bool:
    """
    Check if a model is any Qwen VL variant.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is a Qwen VL variant, False otherwise
    """
    return any(
        is_model_supported(model_name, qwen_model) for qwen_model in QWEN_VL_MODELS
    )


def load_vision_model(model_id: str, enforce_eager: bool = False) -> torch.nn.Module:
    """
    Load a vision model from a HuggingFace model ID.
    """
    if VLLM_ENCODER and is_qwen_vl_model(model_id):
        # Disable to get ViT from the same process
        update_environment_variables(
            {
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            }
        )

        # Load only the vision model via vLLM on encoder workers to avoid loading the full LLM weights, significantly reducing memory usage.
        # Uses native vLLM encoder only model loading added in https://github.com/vllm-project/vllm/pull/32605.
        # Load only the vision model via vLLM
        vllm_model = LLM(
            model=model_id,
            enforce_eager=enforce_eager,
            kv_cache_memory_bytes=1024
            * 1024
            * 64,  # 64MB KV cache for vLLM to complete the init lifecycle, encoder-only doesn't require KV cache.
            max_model_len=1,
            mm_encoder_only=True,
            enable_prefix_caching=False,
        )
        return (
            vllm_model.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model.visual
        )
    return AutoModel.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )


def construct_mm_data(
    model: str,
    embeddings_dtype: torch.dtype,
    image_embeds: Optional[torch.Tensor] = None,
    video_numpy: Optional[Any] = None,
    image_grid_thw: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Construct multimodal data for a vLLM request for models that require additional parameters alongside the embeddings"""

    # Handle video models
    if is_model_supported(model, SupportedModels.LLAVA_NEXT_VIDEO_7B):
        if video_numpy is None:
            raise ValueError("No video frames provided.")
        return {"video": video_numpy}

    # Handle image models - validate image embeddings first
    if image_embeds is None:
        raise ValueError("No image embeddings provided.")

    image_embeds = image_embeds.to(embeddings_dtype)

    # Model-specific image handling
    if is_qwen_vl_model(model):
        return _construct_qwen_image_data(image_embeds, image_grid_thw)
    else:
        # Default image handling for other models (e.g., LLAVA_1_5_7B)
        return {"image": image_embeds}


def _construct_qwen_image_data(
    image_embeds: torch.Tensor, image_grid_thw: Optional[List[Any]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct image data specifically for Qwen models."""
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise ValueError("No image grid provided for Qwen model.")

    grid_thw_tensor = torch.tensor(image_grid_thw)

    return {
        "image": {
            "image_embeds": image_embeds.squeeze(0),
            "image_grid_thw": grid_thw_tensor,
        }
    }


def construct_qwen_decode_mm_data(
    image_grid_thw: Optional[List[Any]],
    embeddings_shape: Optional[Any],
    request_id: str,
    *,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct schema-valid Qwen multimodal data for vLLM v1 disagg decode.

    This is a WORKAROUND (WAR) for vLLM's disaggregated multimodal decode limitations.

    Notes:
    - vLLM parses multimodal inputs and builds `mm_features` from `multi_modal_data`.
    - For Qwen VL models, the parser enforces that image data contains BOTH
      `image_embeds` and `image_grid_thw` keys.
    - In disaggregated decode, the KV cache already includes the vision context
      from prefill; decode still needs `mm_features` for mRoPE initialization.

    WAR Details:
    - We generate unique placeholder embeddings based on request_id to prevent
      incorrect prefix cache matches between different images with same dimensions.
    - Without this, zero embeddings + same image_grid_thw would create identical
      cache signatures, causing decode to incorrectly reuse cached KV from
      different images.

    Caching Caveat:
    - This WAR disables prefix cache reuse on the DECODE worker (each request
      has unique placeholder embeddings).
    - Prefix caching still works correctly on the PREFILL worker, which uses
      actual image embeddings. This is where the caching benefit matters since
      prefill does the heavy computation.
    - Decode receives KV blocks from prefill via NIXL transfer anyway, so
      decode-side prefix caching provides minimal benefit in disaggregated setup.
    """
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise ValueError("No image grid provided for Qwen model.")
    if embeddings_shape is None:
        raise ValueError("embeddings_shape is required for Qwen decode mm data.")

    # WAR: Use request_id hash as seed for unique placeholder values.
    # This prevents prefix cache from incorrectly matching different images
    # that happen to have the same dimensions (same image_grid_thw).
    # bit ops to convert request ID to somewhat unique value that fits in the dtype range
    if not hasattr(construct_qwen_decode_mm_data, "_counter"):
        construct_qwen_decode_mm_data._counter = 0  # type: ignore[attr-defined]
    fill_value = construct_qwen_decode_mm_data._counter  # type: ignore[attr-defined]
    construct_qwen_decode_mm_data._counter += 1  # type: ignore[attr-defined]
    max_val = (
        torch.finfo(dtype).max if dtype.is_floating_point else torch.iinfo(dtype).max
    )
    if construct_qwen_decode_mm_data._counter > max_val:  # type: ignore[attr-defined]
        construct_qwen_decode_mm_data._counter = 0  # type: ignore[attr-defined]
    image_embeds = torch.full(
        embeddings_shape, fill_value=fill_value, dtype=dtype, device="cpu"
    )
    if image_embeds.ndim == 3:
        image_embeds = image_embeds.squeeze(0)

    return {
        "image": {
            "image_embeds": image_embeds,
            "image_grid_thw": torch.tensor(image_grid_thw),
        }
    }


# ── Qwen VL grid_thw computation for P/D disagg (no encoder) ──────


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
        processor = AutoImageProcessor.from_pretrained(model_name)
        vision_config = AutoConfig.from_pretrained(model_name).vision_config

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


def compute_qwen_grid_thw(
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
