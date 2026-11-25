# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score validity and deterministic priority ordering."""

from math import isfinite


def valid_score(value):
    return isinstance(value, (int, float)) and isfinite(value) and value >= 0


def priority_key(score, arrival, fallback_index):
    """Descending valid scores, then arrival; invalid scores keep input order."""
    if not valid_score(score):
        return (1, 0.0, 0.0, fallback_index)
    return (0, -score, arrival, fallback_index)
