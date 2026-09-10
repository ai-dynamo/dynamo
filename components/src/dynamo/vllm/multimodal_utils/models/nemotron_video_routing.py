# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the installed vLLM Nemotron video processor contract."""

import json
import logging
import math
from typing import Any, Optional

from vllm.config import VllmConfig

from dynamo.llm import ModelRuntimeConfig

logger = logging.getLogger(__name__)

VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY = (
    "vllm_nemotron_video_processor_contract"
)
NEMOTRON_MODEL_TYPE = "NemotronH_Nano_Omni_Reasoning_V3"


class _ProbeTokenizer:
    """Minimal callable used to pin vLLM's frame-separator layout."""

    expected = [
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.04 seconds: ",
        "\nFrame 3 sampled at 0.08 seconds: ",
    ]

    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        if texts != self.expected:
            raise ValueError(f"unexpected Nemotron video separators: {texts!r}")
        return {"input_ids": [[101], [102]]}


def _load_nemotron_processor_contract() -> tuple[Any, Any, Any]:
    # Keep the model import lazy: most vLLM workers do not serve Nemotron, and
    # importing its full model module during generic worker startup is costly.
    from vllm.model_executor.models.nano_nemotron_vl import (
        NanoNemotronVLProcessor,
        get_video_target_size_and_feature_size,
    )
    from vllm.multimodal.video_prune.evs import compute_retained_tokens_count

    return (
        NanoNemotronVLProcessor,
        get_video_target_size_and_feature_size,
        compute_retained_tokens_count,
    )


def _installed_processor_matches_contract() -> bool:
    """Probe the vLLM helpers whose behavior the Rust adapter mirrors."""
    try:
        processor, geometry_helper, retention_helper = (
            _load_nemotron_processor_contract()
        )
        geometry = geometry_helper(
            orig_w=640,
            orig_h=360,
            target_patches=1024,
            maintain_aspect_ratio=True,
            patch_size=16,
            downsample_ratio=0.5,
        )
        replacement = processor.get_video_repl(
            tokens_per_frame=[2, 0],
            frames_indices=[0, 1, 2],
            frame_duration_ms=40,
            tokenizer=_ProbeTokenizer(),
            img_start_token_ids=[19],
            img_end_token_ids=[20],
            img_context_token_ids=[18],
            video_temporal_patch_size=2,
        )
        replacement_tokens = list(replacement.full)
        truncated_retention = retention_helper(tokens_per_frame=7, num_frames=3, q=0.5)
        first_frame_retention = retention_helper(
            tokens_per_frame=256, num_frames=4, q=0.9
        )
    except Exception as error:
        logger.warning(
            "Exact Nemotron video-aware KV routing disabled because the installed "
            "vLLM processor API is unsupported: %s",
            error,
        )
        return False

    if (
        geometry != (672, 384, 252)
        or replacement_tokens != [101, 19, 18, 18, 20, 102, 19, 20]
        or truncated_retention != 10
        or first_frame_retention != 256
    ):
        logger.warning(
            "Exact Nemotron video-aware KV routing disabled because the installed "
            "vLLM processor behavior is unsupported"
        )
        return False
    return True


def _resolve_nemotron_video_processor_contract(
    vllm_config: VllmConfig,
) -> Optional[dict[str, float]]:
    hf_config = vllm_config.model_config.hf_config
    if getattr(hf_config, "model_type", None) != NEMOTRON_MODEL_TYPE:
        return None
    if NEMOTRON_MODEL_TYPE not in (getattr(hf_config, "architectures", None) or []):
        return None

    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is None:
        return None
    if multimodal_config.mm_processor_kwargs:
        logger.warning(
            "Exact Nemotron video-aware KV routing disabled because engine-level "
            "mm_processor_kwargs can change the video token layout"
        )
        return None
    if not _installed_processor_matches_contract():
        return None

    get_video_pruning_spec = getattr(multimodal_config, "get_video_pruning_spec", None)
    if not callable(get_video_pruning_spec):
        logger.warning(
            "Exact Nemotron video-aware KV routing disabled because the installed "
            "vLLM cannot report video pruning configuration"
        )
        return None
    try:
        pruning_spec = get_video_pruning_spec()
    except Exception as error:
        logger.warning(
            "Exact Nemotron video-aware KV routing disabled because the installed "
            "vLLM cannot resolve video pruning configuration: %s",
            error,
        )
        return None
    video_pruning_rate = 0.0
    if pruning_spec is not None:
        try:
            method, rate = pruning_spec
            video_pruning_rate = float(rate)
        except (TypeError, ValueError):
            logger.warning(
                "Exact Nemotron video-aware KV routing disabled for invalid "
                "video pruning configuration %r",
                pruning_spec,
            )
            return None
        if method != "evs":
            logger.warning(
                "Exact Nemotron video-aware KV routing disabled for unsupported "
                "video pruning method %r",
                method,
            )
            return None
    if not math.isfinite(video_pruning_rate) or not 0.0 <= video_pruning_rate < 1.0:
        logger.warning(
            "Exact Nemotron video-aware KV routing disabled for invalid pruning rate %r",
            video_pruning_rate,
        )
        return None
    return {"video_pruning_rate": video_pruning_rate}


def publish_vllm_nemotron_video_processor_contract(
    runtime_config: ModelRuntimeConfig, vllm_config: VllmConfig
) -> None:
    """Publish exact Nemotron video preprocessing behavior for the frontend."""
    contract = _resolve_nemotron_video_processor_contract(vllm_config)
    if contract is not None:
        runtime_config.set_engine_specific(
            VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
            json.dumps(contract),
        )
