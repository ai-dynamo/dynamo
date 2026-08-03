# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM KV-event routing helpers."""

import logging
import os
from typing import Optional

from huggingface_hub import try_to_load_from_cache
from huggingface_hub.utils import HFValidationError
from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


def resolve_image_token_id(model: str, vllm_config: VllmConfig) -> Optional[int]:
    """Resolve the placeholder token normalized in vLLM KV events.

    This deliberately uses the same Rust resolver as the frontend MM
    preprocessor. Returning ``None`` disables worker-side token substitution.
    """
    try:
        from dynamo._core import resolve_routing_image_token_id
    except ImportError:
        return None

    model_dir = None
    try:
        revision = vllm_config.model_config.revision
        config_path = try_to_load_from_cache(
            repo_id=model, filename="config.json", revision=revision
        )
        if config_path and isinstance(config_path, str):
            model_dir = os.path.dirname(config_path)
    except (HFValidationError, OSError) as exc:
        logger.debug(
            "HF cache lookup for %s failed (%s); falling back to raw model arg",
            model,
            exc,
        )
    if model_dir is None:
        model_dir = vllm_config.model_config.model
        logger.debug("Resolved model_dir via raw arg fallback: %s", model_dir)

    return resolve_routing_image_token_id(model, model_dir)
