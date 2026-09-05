# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned generation artifacts and provider-neutral delivery."""

from .format_v1 import (
    GenerationArtifactChoice,
    GenerationArtifactView,
    encode_generation_artifact,
)
from .storage import put_artifact

__all__ = [
    "GenerationArtifactChoice",
    "GenerationArtifactView",
    "encode_generation_artifact",
    "put_artifact",
]
