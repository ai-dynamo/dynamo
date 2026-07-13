# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral in-memory representation of frontend-processed media."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProcessedField:
    value: np.ndarray
    layout: dict[str, Any]
    keep_on_host: bool
    forward: bool


@dataclass(frozen=True)
class ProcessedMedia:
    modality: str
    fields: dict[str, ProcessedField]
    feature_token_counts: list[int]
    original_sizes: list[tuple[int, int]]
    content_hashes: list[str]
