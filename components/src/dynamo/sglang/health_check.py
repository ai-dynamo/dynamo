#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
sglang-specific health check configuration.

This module defines the default health check payload for sglang backends.
"""

import logging
from typing import Optional

import sglang as sgl

from dynamo.health_check import HealthCheckPayload

logger = logging.getLogger(__name__)


def _get_bos_token_id_from_engine(engine: Optional[sgl.Engine]) -> int:
    """Extract BOS token ID from the SGLang engine's tokenizer.

    Args:
        engine: SGLang Engine instance.

    Returns:
        BOS token ID from the model's tokenizer, or 1 as fallback.
    """
    if engine is None:
        return 1

    try:
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        if tokenizer_manager:
            tokenizer = getattr(tokenizer_manager, "tokenizer", None)
            if tokenizer:
                bos_token_id = getattr(tokenizer, "bos_token_id", None)
                if bos_token_id is not None:
                    logger.info(
                        f"Using model's BOS token ID for health check: {bos_token_id}"
                    )
                    return int(bos_token_id)
    except Exception as e:
        logger.debug(f"Failed to get BOS token from engine: {e}")

    logger.debug("Using default BOS token ID (1) for health check")
    return 1


class SglangHealthCheckPayload(HealthCheckPayload):
    """SGLang-specific health check payload for decode workers.

    Provides SGLang defaults and inherits environment override support from base class.
    """

    def __init__(
        self, engine: Optional[sgl.Engine] = None, use_text_input: bool = False
    ) -> None:
        """Initialize SGLang health check payload with model-specific BOS token.

        Args:
            engine: Optional SGLang Engine instance to extract BOS token from.
        """
        bos_token_id = _get_bos_token_id_from_engine(engine)

        self.default_payload = {
            "stop_conditions": {
                "max_tokens": 1,  # Generate only 1 token
                "ignore_eos": False,
            },
            "sampling_options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            },
            "eos_token_ids": [],
            "annotations": [],
        }

        if use_text_input:
            self.default_payload["prompt"] = "Test"
        else:
            self.default_payload["token_ids"] = [bos_token_id]

        super().__init__()


class SglangDisaggHealthCheckPayload(HealthCheckPayload):
    """SGLang-specific health check payload for PD-disaggregated mode (prefill or decode).

    Both prefill and decode handlers support flat format with bootstrap_info.
    Uses FAKE_BOOTSTRAP_HOST to enable fake-transfer mode, so health checks
    don't require real KV-transfer between prefill/decode workers.

    Uses bootstrap_room=0 (same as SGLang). This means health checks always go to
    DP rank 0. For proper DP coverage, runtime would need to support dynamic payload
    generation per health check request.
    """

    def __init__(
        self,
        engine: Optional[sgl.Engine] = None,
        use_text_input: bool = False,
    ) -> None:
        """Initialize SGLang disaggregated health check payload.

        Args:
            engine: SGLang Engine instance to extract BOS token and bootstrap port from.
            use_text_input: Whether to use text prompt instead of token IDs.
        """
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        bos_token_id = _get_bos_token_id_from_engine(engine)

        # Get bootstrap port from engine
        bootstrap_port = 0
        if engine is not None:
            try:
                inner_tm = engine.tokenizer_manager
                bootstrap_port = getattr(
                    inner_tm.server_args, "disaggregation_bootstrap_port", 0
                )
            except Exception as e:
                logger.warning(f"Failed to get bootstrap port from engine: {e}")

        # Create bootstrap_info for fake-transfer mode
        # FAKE_BOOTSTRAP_HOST tells SGLang to skip real KV-transfer
        # bootstrap_room=0 matches SGLang behavior (always routes to DP rank 0)
        # TODO: For proper DP coverage, runtime needs to support dynamic payload generation
        bootstrap_info = {
            "bootstrap_host": FAKE_BOOTSTRAP_HOST,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": 0,
        }

        self.default_payload = {
            "bootstrap_info": bootstrap_info,
            "stop_conditions": {
                "max_tokens": 1,  # Generate only 1 token
                "ignore_eos": False,
            },
            "sampling_options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            },
            "eos_token_ids": [],
            "annotations": [],
        }

        if use_text_input:
            self.default_payload["prompt"] = "Test"
        else:
            self.default_payload["token_ids"] = [bos_token_id]

        logger.info(
            f"Disagg health check configured: "
            f"bootstrap_host={FAKE_BOOTSTRAP_HOST}, "
            f"bootstrap_port={bootstrap_port}, "
            f"bootstrap_room=0"
        )

        super().__init__()


class ImageDiffusionHealthCheckPayload(HealthCheckPayload):
    """Image diffusion-specific health check payload for image generation workers.

    Sends a minimal image generation request to verify the diffusion worker
    is responding and the model is loaded. Uses minimal resources for fast checks.
    """

    def __init__(self, model_path: str):
        """Initialize diffusion health check payload with minimal generation request.

        Args:
            model_path: The diffusion model being served.
        """
        self.default_payload = {
            "prompt": "test",  # Minimal prompt
            "model": model_path,
            "n": 1,  # Generate 1 image
            "size": "512x512",  # Small size for fast health check
            "num_inference_steps": 1,  # Just 1 step (fast but low quality)
            "guidance_scale": 7.5,  # Standard guidance scale
            "response_format": "b64_json",  # Don't require S3 for health check
        }

        super().__init__()


class VideoGenerationHealthCheckPayload(HealthCheckPayload):
    """Video generation-specific health check payload for video generation workers.

    Sends a minimal video generation request to verify the video worker
    is responding and the model is loaded. Uses minimal resources for fast checks.
    """

    def __init__(self, model_path: str):
        """Initialize video health check payload with minimal generation request.

        Args:
            model_path: The video generation model being served.
        """
        self.default_payload = {
            "prompt": "test",  # Minimal prompt
            "model": model_path,
            "seconds": 1,
            "size": "256x256",  # Small size for fast health check
            "response_format": "b64_json",  # Don't require filesystem for health check
            "nvext": {
                "fps": 8,
                "num_frames": 8,  # Minimal frames for fast health check
                "num_inference_steps": 1,  # Just 1 step (fast but low quality)
                "guidance_scale": 5.0,  # Standard guidance scale for video
            },
        }

        super().__init__()
