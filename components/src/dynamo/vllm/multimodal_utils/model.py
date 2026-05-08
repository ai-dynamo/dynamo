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

import functools
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModel
from vllm import LLM
from vllm.utils.system_utils import update_environment_variables

logger = logging.getLogger(__name__)

# [gluo NOTE] Debug flag to compare vLLM encoder vs transformers encoder,
# should be removed once there is proper way to extract vLLM encoder.
VLLM_ENCODER = int(os.getenv("VLLM_ENCODER", 1))


class ModelFamily(str, Enum):
    """Multimodal model families dynamo's encoder pipeline knows how to dispatch."""

    QWEN_VL = "qwen-vl"
    LLAVA = "llava"


# Per-family registries used by `resolve_model_family`. The encoder reaches
# into vLLM internals (`model.visual` for Qwen, `vision_tower` +
# `multi_modal_projector` for LLaVA) whose attribute paths vary per family —
# entries here must correspond to extractor logic in `encode_utils.py`.
#
# Architectures are matched verbatim against `config.json` `architectures`.
_FAMILY_ARCHITECTURES: Dict[ModelFamily, frozenset[str]] = {
    ModelFamily.QWEN_VL: frozenset(
        {
            "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            # Qwen3.5 subclasses Qwen3VL in vLLM
            # (`Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`)
            # with the same `self.visual = Qwen3_VisionTransformer(...)` and an
            # identical preprocessor (`Qwen2VLImageProcessorFast`); the encoder
            # pipeline is inherited unchanged. Inclusion based on source +
            # config inspection — pending empirical verification against a
            # deployed Qwen3.5 model.
            "Qwen3_5ForConditionalGeneration",
        }
    ),
    ModelFamily.LLAVA: frozenset({"LlavaForConditionalGeneration"}),
}

# Name-stage substring patterns (lowercase). A new size / quantization /
# MoE variant of an existing family doesn't require a registry change as
# long as it shares the family-level prefix; only a genuinely new family
# (e.g. Qwen4-VL) needs a pattern added.
_FAMILY_NAME_PATTERNS: Dict[ModelFamily, frozenset[str]] = {
    ModelFamily.QWEN_VL: frozenset(
        {
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen3-vl",
            # Qwen3.5 ids omit "-vl" because it's a unified multimodal
            # architecture; bare "qwen3.5" matches all variants. Routes to
            # QWEN_VL via the subclass relationship documented in
            # _FAMILY_ARCHITECTURES.
            "qwen3.5",
        }
    ),
    ModelFamily.LLAVA: frozenset({"llava-1.5-7b-hf"}),
}


def _load_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Read `config.json` from a local model directory if available."""
    try:
        path = Path(model_name)
        if not path.is_dir():
            return None
        config_path = path / "config.json"
        if not config_path.is_file():
            return None
        return json.loads(config_path.read_text())
    except (OSError, ValueError):
        return None


@functools.lru_cache(maxsize=8)
def resolve_model_family(model_name: str) -> Optional[ModelFamily]:
    """Return the multimodal model family for the given identifier, or `None`
    if no registered family matches.

    Resolution proceeds in two stages: a `config.json` `architectures`
    lookup when `model_name` is a local directory with config (authoritative
    when present), falling through to a lowercased substring scan against
    per-family name patterns (`qwen2-vl`, `qwen3-vl`, `llava-1.5-7b-hf`).
    Patterns are family-level prefixes, so HF ids, HF cache layouts, and
    hand-rolled local paths are all handled by the same scan, and new
    sizes / quantizations of an existing family resolve without a registry
    change.

    Args:
        model_name: A model identifier. Accepts an HF id, an HF cache
            snapshot path, or a local directory path (with or without
            `config.json`, with or without an HF-cache-style parent
            segment).

    Returns:
        The matching `ModelFamily`, or `None` if no registered family
        matches the input.

    Examples:
        All of the following resolve to `ModelFamily.QWEN_VL`:

        >>> resolve_model_family("Qwen/Qwen2-VL-2B-Instruct")
        >>> resolve_model_family(
        ...     "/root/.cache/huggingface/hub/"
        ...     "models--Qwen--Qwen2-VL-2B-Instruct/snapshots/abc123"
        ... )
        >>> resolve_model_family("/local_store/Qwen--Qwen2-VL-2B-Instruct/v2")
        >>> resolve_model_family("/runs/qwen2.5-vl-7b-instruct/v3")

        A local directory containing `config.json` with
        `{"architectures": ["Qwen2VLForConditionalGeneration"]}` resolves
        via the metadata stage regardless of the directory's name.
    """
    config = _load_model_config(model_name)
    if config is not None:
        for arch in config.get("architectures") or []:
            for family, arch_set in _FAMILY_ARCHITECTURES.items():
                if arch in arch_set:
                    return family

    lowered = model_name.lower()
    for family, patterns in _FAMILY_NAME_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return family

    return None


def load_vision_model(model_id: str, enforce_eager: bool = False) -> torch.nn.Module:
    """
    Load a vision model from a HuggingFace model ID.
    """
    if VLLM_ENCODER and resolve_model_family(model_id) is ModelFamily.QWEN_VL:
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
            # vLLM's free-memory precheck runs before kv_cache_memory_bytes applies;
            # default 0.9 fails on <=24 GiB GPUs when another worker shares the device.
            gpu_memory_utilization=0.2,
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

    if video_numpy is not None:
        return {"video": video_numpy}

    # Handle image models - validate image embeddings first
    if image_embeds is None:
        raise ValueError("No image embeddings provided.")

    image_embeds = image_embeds.to(embeddings_dtype)

    # Model-specific image handling
    if resolve_model_family(model) is ModelFamily.QWEN_VL:
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
