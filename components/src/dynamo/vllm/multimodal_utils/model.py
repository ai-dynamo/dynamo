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
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

IMAGE_EMBEDS_KEY = "image_embeds"
IMAGE_GRID_THW_KEY = "image_grid_thw"


def is_grid_thw_image_data(image_data: Any) -> bool:
    """Return True when the image payload carries explicit embed + grid metadata."""
    return (
        isinstance(image_data, dict)
        and IMAGE_EMBEDS_KEY in image_data
        and IMAGE_GRID_THW_KEY in image_data
    )


def _load_vllm_encoder_only_model(model_id: str, enforce_eager: bool) -> Any:
    from vllm import LLM
    from vllm.utils.system_utils import update_environment_variables

    update_environment_variables(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
    )

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
        vllm_model.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
    )


def load_vision_model(model_id: str, enforce_eager: bool = False) -> torch.nn.Module:
    """
    Load a vision-capable model and return the encoder-only object used by the
    encode worker.
    """
    try:
        return _load_vllm_encoder_only_model(model_id, enforce_eager)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load a dedicated vision encoder via "
            f"vLLM `mm_encoder_only` for model '{model_id}'. "
            "This backend requires the vLLM encoder-only path and no longer "
            "falls back to transformers.AutoModel because that path is "
            "materially slower. Use a model that supports vLLM "
            "`mm_encoder_only` loading. "
            f"Original error: {exc}"
        ) from exc


def construct_mm_data(
    model: str,
    embeddings_dtype: torch.dtype,
    image_embeds: Optional[torch.Tensor] = None,
    video_numpy: Optional[Any] = None,
    image_grid_thw: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Construct multimodal data for a vLLM request."""
    del model

    if video_numpy is not None:
        return {"video": video_numpy}

    # Handle image models - validate image embeddings first
    if image_embeds is None:
        raise ValueError("No image embeddings provided.")

    image_embeds = image_embeds.to(embeddings_dtype)

    if image_grid_thw is not None and len(image_grid_thw) > 0:
        return _construct_grid_thw_image_data(image_embeds, image_grid_thw)

    return {"image": image_embeds}


def _construct_grid_thw_image_data(
    image_embeds: torch.Tensor, image_grid_thw: Optional[List[Any]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct image data for models that require `image_grid_thw`."""
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise ValueError("No image grid provided for grid-aware multimodal payload.")

    grid_thw_tensor = torch.tensor(image_grid_thw)

    return {
        "image": {
            IMAGE_EMBEDS_KEY: image_embeds.squeeze(0),
            IMAGE_GRID_THW_KEY: grid_thw_tensor,
        }
    }


def construct_grid_thw_decode_mm_data(
    image_grid_thw: Optional[List[Any]],
    embeddings_shape: Optional[Any],
    request_id: str,
    *,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct schema-valid multimodal data for grid-aware vLLM v1 decode.

    This is a WORKAROUND (WAR) for vLLM's disaggregated multimodal decode limitations.

    Notes:
    - vLLM parses multimodal inputs and builds `mm_features` from `multi_modal_data`.
    - For grid-aware models, the parser enforces that image data contains BOTH
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
        raise ValueError(
            "No image grid provided for grid-aware decode multimodal payload."
        )
    if embeddings_shape is None:
        raise ValueError(
            "embeddings_shape is required for grid-aware decode multimodal payload."
        )

    # WAR: Use request_id hash as seed for unique placeholder values.
    # This prevents prefix cache from incorrectly matching different images
    # that happen to have the same dimensions (same image_grid_thw).
    # bit ops to convert request ID to somewhat unique value that fits in the dtype range
    if not hasattr(construct_grid_thw_decode_mm_data, "_counter"):
        construct_grid_thw_decode_mm_data._counter = 0  # type: ignore[attr-defined]
    fill_value = construct_grid_thw_decode_mm_data._counter  # type: ignore[attr-defined]
    construct_grid_thw_decode_mm_data._counter += 1  # type: ignore[attr-defined]
    max_val = (
        torch.finfo(dtype).max if dtype.is_floating_point else torch.iinfo(dtype).max
    )
    if construct_grid_thw_decode_mm_data._counter > max_val:  # type: ignore[attr-defined]
        construct_grid_thw_decode_mm_data._counter = 0  # type: ignore[attr-defined]
    image_embeds = torch.full(
        embeddings_shape, fill_value=fill_value, dtype=dtype, device="cpu"
    )
    if image_embeds.ndim == 3:
        image_embeds = image_embeds.squeeze(0)

    return {
        "image": {
            IMAGE_EMBEDS_KEY: image_embeds,
            IMAGE_GRID_THW_KEY: torch.tensor(image_grid_thw),
        }
    }
