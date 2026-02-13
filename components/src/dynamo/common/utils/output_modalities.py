# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import List, Optional

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
            output_modalities = flag if output_modalities is None else output_modalities | flag
    return output_modalities

