# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Qwen video processor contract used by vLLM workers."""

import json
import logging
from collections.abc import Callable
from typing import Any, Optional

import transformers
from vllm.config import VllmConfig

from dynamo.llm import ModelRuntimeConfig

qwen3_smart_resize: Optional[Callable[..., Any]]
try:
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        smart_resize as _qwen3_smart_resize,
    )
except ImportError:
    qwen3_smart_resize = None
else:
    qwen3_smart_resize = _qwen3_smart_resize

logger = logging.getLogger(__name__)

VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY = "vllm_qwen_video_processor_contract"
QWEN_VIDEO_TARGET_BARE = "bare_video_token"
QWEN_VIDEO_TARGET_WRAPPED = "vision_wrapped_video_token"
QWEN_VIDEO_RESIZE_LEGACY_CEIL = "legacy_ceil"
QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN = "round_ties_even"
QWEN_VIDEO_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}
QWEN_VIDEO_ARCHITECTURES = {
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


def _resolve_qwen_video_resize_mode() -> Optional[str]:
    """Identify the installed Qwen video processor's temporal resize rule."""
    if qwen3_smart_resize is None:
        logger.warning(
            "Exact video-aware KV routing disabled because the installed "
            "Transformers package has no Qwen3 video processor"
        )
        return None
    try:
        result = qwen3_smart_resize(
            num_frames=5,
            height=1120,
            width=3760,
            temporal_factor=2,
            factor=32,
            min_pixels=4096,
            max_pixels=25165824,
        )
    except (TypeError, ValueError) as error:
        logger.warning(
            "Exact video-aware KV routing disabled because the installed Qwen "
            "smart_resize API is unsupported: %s",
            error,
        )
        return None
    if result == (1216, 4096):
        return QWEN_VIDEO_RESIZE_LEGACY_CEIL
    if result == (1120, 3776):
        return QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN
    logger.warning(
        "Exact video-aware KV routing disabled because the installed Qwen "
        "smart_resize behavior is unsupported: %s",
        result,
    )
    return None


def _resolve_qwen_video_processor_contract(
    vllm_config: VllmConfig,
) -> Optional[dict[str, str]]:
    """Match the prompt expansion selected by vLLM's Qwen3 processor."""
    hf_config = vllm_config.model_config.hf_config
    if getattr(hf_config, "model_type", None) not in QWEN_VIDEO_MODEL_TYPES:
        return None
    architectures = getattr(hf_config, "architectures", None) or []
    if not QWEN_VIDEO_ARCHITECTURES.intersection(architectures):
        return None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is None:
        return None
    if multimodal_config.mm_processor_kwargs:
        logger.warning(
            "Exact video-aware KV routing disabled because engine-level "
            "mm_processor_kwargs can change the Qwen video token layout"
        )
        return None

    # Some supported vLLM images predate this optional capability query.
    get_video_pruning_spec = getattr(multimodal_config, "get_video_pruning_spec", None)
    if not callable(get_video_pruning_spec):
        logger.warning(
            "Exact video-aware KV routing disabled because the installed vLLM "
            "cannot report video pruning configuration"
        )
        return None
    if get_video_pruning_spec() is not None:
        logger.warning(
            "Exact video-aware KV routing disabled because video pruning "
            "changes the Qwen video token layout"
        )
        return None

    qwen_processor = getattr(transformers, "Qwen3VLProcessor", None)
    if qwen_processor is None:
        logger.warning(
            "Exact video-aware KV routing disabled because the installed "
            "Transformers package has no Qwen3 processor"
        )
        return None
    mixin_impl = getattr(transformers.ProcessorMixin, "replace_video_token", None)
    processor_impl = getattr(qwen_processor, "replace_video_token", None)
    placeholder_target = QWEN_VIDEO_TARGET_WRAPPED
    if processor_impl is not None and processor_impl is not mixin_impl:
        placeholder_target = QWEN_VIDEO_TARGET_BARE
    resize_mode = _resolve_qwen_video_resize_mode()
    if resize_mode is None:
        return None
    return {
        "placeholder_target": placeholder_target,
        "resize_mode": resize_mode,
    }


def publish_vllm_qwen_video_processor_contract(
    runtime_config: ModelRuntimeConfig, vllm_config: VllmConfig
) -> None:
    """Publish exact Qwen video preprocessing behavior for the frontend."""
    contract = _resolve_qwen_video_processor_contract(vllm_config)
    if contract is not None:
        runtime_config.set_engine_specific(
            VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
            json.dumps(contract),
        )
