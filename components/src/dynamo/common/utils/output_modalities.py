# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

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
    for name in cli_input:
        modality = OutputModality.from_name(name)
        for flag in modality.value:
            output_modalities = (
                flag if output_modalities is None else output_modalities | flag
            )
    return output_modalities


def parse_request_type(
    raw_request: Dict[str, Any],
    output_modalities: List[str],
) -> Tuple[Union[BaseModel, Dict[str, Any]], RequestType]:
    """
    Classify the endpoint based on the output modality and serialize the request if necessary.

    Assumption: Right now we only consider user passes only one modality at a time.
    """
    # Fetch the first output modality from the list.
    if not output_modalities:
        raise ValueError("output_modalities must not be empty")
    output_modality = output_modalities[0]
    modality = OutputModality.from_name(output_modality)

    if modality is OutputModality.IMAGE:
        if "messages" in raw_request:
            return raw_request, RequestType.CHAT_COMPLETION
        return NvCreateImageRequest(**raw_request), RequestType.IMAGE_GENERATION

    if modality is OutputModality.VIDEO:
        return NvCreateVideoRequest(**raw_request), RequestType.VIDEO_GENERATION

    if modality is OutputModality.AUDIO:
        # Audio protocol types are not yet defined; pass through the raw dict.
        return raw_request, RequestType.AUDIO_GENERATION

    # Text Modality
    return raw_request, RequestType.CHAT_COMPLETION
