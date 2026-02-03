# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for TensorRT-LLM backend.

This module defines enums and constants used throughout the trtllm module.
"""

from enum import Enum


class DisaggregationMode(Enum):
    """Disaggregation mode for LLM workers."""

    AGGREGATED = "prefill_and_decode"
    PREFILL = "prefill"
    DECODE = "decode"
    ENCODE = "encode"


class Modality(Enum):
    """Modality types for different generative models.

    This enum determines which type of model and handler to use:
    - TEXT: Text-only LLM (generates text tokens)
    - MULTIMODAL: Vision-language LLM (understands images, generates text)
    - VIDEO_DIFFUSION: Video generation from text (generates video files)
    - IMAGE_DIFFUSION: Image generation from text (generates image files)
    """

    TEXT = "text"
    MULTIMODAL = "multimodal"
    VIDEO_DIFFUSION = "video_diffusion"
    IMAGE_DIFFUSION = "image_diffusion"

    @classmethod
    def is_diffusion(cls, modality: "Modality") -> bool:
        """Check if a modality is a diffusion modality.

        Args:
            modality: The modality to check.

        Returns:
            True if the modality is VIDEO_DIFFUSION or IMAGE_DIFFUSION.
        """
        return modality in (cls.VIDEO_DIFFUSION, cls.IMAGE_DIFFUSION)

    @classmethod
    def is_llm(cls, modality: "Modality") -> bool:
        """Check if a modality is an LLM modality.

        Args:
            modality: The modality to check.

        Returns:
            True if the modality is TEXT or MULTIMODAL.
        """
        return modality in (cls.TEXT, cls.MULTIMODAL)
