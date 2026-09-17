# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish SGLang's effective Qwen video preprocessing contract."""

import json
import logging
from typing import Any, Optional

import transformers

from dynamo.llm import ModelRuntimeConfig

try:
    from sglang.srt.multimodal.processors import qwen_vl as sglang_qwen_vl
except ImportError:
    sglang_qwen_vl = None

try:
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        smart_resize as qwen3_smart_resize,
    )
except ImportError:
    qwen3_smart_resize = None

logger = logging.getLogger(__name__)

SGLANG_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY = (
    "sglang_qwen_video_processor_contract"
)
QWEN_VIDEO_TARGET_BARE = "bare_video_token"
QWEN_VIDEO_TARGET_WRAPPED = "vision_wrapped_video_token"
QWEN_VIDEO_RESIZE_LEGACY_CEIL = "legacy_ceil"
QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN = "round_ties_even"
QWEN_VIDEO_RUNLESS_BOUNDARY_TOKENS_ONLY = "tokens_only"
QWEN_VIDEO_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}
QWEN_VIDEO_ARCHITECTURES = {
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


def _resolve_qwen_video_resize_mode() -> Optional[str]:
    """Identify the installed Transformers Qwen video resize rule."""
    if qwen3_smart_resize is None:
        logger.warning(
            "Exact SGLang video-aware KV routing disabled because the installed "
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
            "Exact SGLang video-aware KV routing disabled because the installed "
            "Qwen smart_resize API is unsupported: %s",
            error,
        )
        return None
    if result == (1216, 4096):
        return QWEN_VIDEO_RESIZE_LEGACY_CEIL
    if result == (1120, 3776):
        return QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN
    logger.warning(
        "Exact SGLang video-aware KV routing disabled because the installed "
        "Qwen smart_resize behavior is unsupported: %s",
        result,
    )
    return None


def _resolve_qwen_video_processor_contract(engine: Any) -> Optional[dict[str, Any]]:
    """Resolve the two-stage video preprocessing used by this SGLang worker."""
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    mm_processor = getattr(tokenizer_manager, "mm_processor", None)
    hf_config = getattr(mm_processor, "hf_config", None)
    if hf_config is None:
        model_config = getattr(tokenizer_manager, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)

    model_type = getattr(hf_config, "model_type", None)
    if model_type not in QWEN_VIDEO_MODEL_TYPES:
        return None
    architectures = getattr(hf_config, "architectures", None) or []
    if not QWEN_VIDEO_ARCHITECTURES.intersection(architectures):
        return None
    if mm_processor is None:
        logger.warning(
            "Exact SGLang video-aware KV routing disabled because the Qwen "
            "multimodal processor is unavailable"
        )
        return None

    video_config = getattr(mm_processor, "video_config", None) or {}
    if video_config:
        logger.warning(
            "Exact SGLang video-aware KV routing disabled because engine-level "
            "mm_process_config.video can change frame sampling or resizing"
        )
        return None
    if sglang_qwen_vl is None:
        logger.warning(
            "Exact SGLang video-aware KV routing disabled because the installed "
            "SGLang package has no Qwen video preprocessor"
        )
        return None

    processor = getattr(mm_processor, "_processor", None)
    processor_impl = getattr(type(processor), "replace_video_token", None)
    mixin_impl = getattr(transformers.ProcessorMixin, "replace_video_token", None)
    placeholder_target = QWEN_VIDEO_TARGET_WRAPPED
    if processor_impl is not None and processor_impl is not mixin_impl:
        placeholder_target = QWEN_VIDEO_TARGET_BARE

    resize_mode = _resolve_qwen_video_resize_mode()
    if resize_mode is None:
        return None

    return {
        "placeholder_target": placeholder_target,
        "resize_mode": resize_mode,
        # SGLang KV events do not attach MM metadata to delimiter/timestamp
        # boundary blocks that contain no video placeholder run.
        "runless_boundary_hash": QWEN_VIDEO_RUNLESS_BOUNDARY_TOKENS_ONLY,
        "sglang_preprocess": {
            "image_factor": int(sglang_qwen_vl.IMAGE_FACTOR),
            "video_min_pixels": int(sglang_qwen_vl.VIDEO_MIN_PIXELS),
            "video_max_pixels": int(sglang_qwen_vl.VIDEO_MAX_PIXELS),
            "video_total_pixels": int(sglang_qwen_vl.VIDEO_TOTAL_PIXELS),
            "frame_factor": int(sglang_qwen_vl.FRAME_FACTOR),
            "fps": float(sglang_qwen_vl.FPS),
            "min_frames": int(sglang_qwen_vl.FPS_MIN_FRAMES),
            "max_frames": int(sglang_qwen_vl.FPS_MAX_FRAMES),
        },
    }


def publish_sglang_qwen_video_processor_contract(
    runtime_config: ModelRuntimeConfig, engine: Any
) -> None:
    """Publish exact Qwen video preprocessing behavior for the frontend."""
    contract = _resolve_qwen_video_processor_contract(engine)
    if contract is not None:
        runtime_config.set_engine_specific(
            SGLANG_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
            json.dumps(contract),
        )
