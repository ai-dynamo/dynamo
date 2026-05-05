# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import BaseModel

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.protocols.image_protocol import NvCreateImageRequest
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest
from dynamo.llm import ModelType


class OutputModality(Enum):
    """Maps CLI modality names to their corresponding ModelType flags."""

    TEXT = (ModelType.Chat, ModelType.Completions)
    IMAGE = (ModelType.Images, ModelType.Chat)
    VIDEO = (ModelType.Videos,)
    AUDIO = (ModelType.Audios,)

    @classmethod
    def from_name(cls, name: str) -> "OutputModality":
        """Look up a modality by its CLI name (case-insensitive)."""
        try:
            return cls[name.upper()]
        except KeyError:
            valid = ", ".join(m.name.lower() for m in cls)
            raise ValueError(
                f"Unknown output modality: {name!r}. Valid options: {valid}"
            )

    @classmethod
    def valid_names(cls) -> set:
        """Return the set of valid CLI modality names (lowercase)."""
        return {m.name.lower() for m in cls}


class RequestType(Enum):
    """Identifies the parsed request type returned by parse_request_type."""

    CHAT_COMPLETION = "chat_completion"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"


def get_output_modalities(cli_input: List[str], model_repo: str) -> Optional[ModelType]:
    """
    Get the combined ModelType flags for omni models based on CLI input.

    Args:
        cli_input: List of modality name strings (e.g. ["text", "image"]).
        model_repo: Model repo string (reserved for future per-model logic).

    Returns:
        Combined ModelType flags, or None if no recognized modalities are present.
    """
    # For now, we ignore model repo and just use cli input to determine output modalities.
    output_modalities = None
    for name in normalize_output_modalities(cli_input):
        modality = OutputModality.from_name(name)
        for flag in modality.value:
            output_modalities = (
                flag if output_modalities is None else output_modalities | flag
            )
    return output_modalities


def normalize_output_modalities(cli_input: Iterable[str] | None) -> List[str]:
    """Normalize modality tokens from CLI/env input.

    ``--output-modalities`` uses ``nargs="*"`` for shell-friendly invocations
    such as ``--output-modalities text audio``. The environment variable and
    older launchers can still arrive as a single comma-separated token. Split
    both forms into the canonical lowercase list while preserving user order.
    """
    if not cli_input:
        return []

    normalized: list[str] = []
    for raw in cli_input:
        for part in str(raw).split(","):
            token = part.strip().lower()
            if token:
                normalized.append(token)
    return normalized


def parse_request_type(
    raw_request: Dict[str, Any],
    output_modalities: List[str],
) -> Tuple[Union[BaseModel, Dict[str, Any]], RequestType]:
    """
    Classify an OpenAI request by payload shape and enabled output modalities.

    Omni workers can enable multiple modalities behind one Dynamo endpoint.
    In that mode the endpoint payload, not the first configured modality, is
    the only reliable discriminator: chat requests contain ``messages``, TTS
    requests contain ``input``, and image/video generation requests contain a
    top-level ``prompt`` with modality-specific fields.
    """
    modality_names = normalize_output_modalities(output_modalities)
    if not modality_names:
        raise ValueError("output_modalities must not be empty")

    modalities = {OutputModality.from_name(name) for name in modality_names}

    if "messages" in raw_request:
        return raw_request, RequestType.CHAT_COMPLETION

    if OutputModality.AUDIO in modalities and "input" in raw_request:
        return NvCreateAudioSpeechRequest(**raw_request), RequestType.AUDIO_GENERATION

    if OutputModality.VIDEO in modalities and (
        OutputModality.IMAGE not in modalities or _looks_like_video_request(raw_request)
    ):
        return NvCreateVideoRequest(**raw_request), RequestType.VIDEO_GENERATION

    if OutputModality.IMAGE in modalities:
        return NvCreateImageRequest(**raw_request), RequestType.IMAGE_GENERATION

    return raw_request, RequestType.CHAT_COMPLETION


def _looks_like_video_request(raw_request: Dict[str, Any]) -> bool:
    """Return True for fields that only video generation requests use."""
    if any(key in raw_request for key in ("seconds", "output_format", "stream")):
        return True

    nvext = raw_request.get("nvext") or {}
    return isinstance(nvext, dict) and any(
        key in nvext
        for key in ("fps", "num_frames", "boundary_ratio", "guidance_scale_2")
    )
