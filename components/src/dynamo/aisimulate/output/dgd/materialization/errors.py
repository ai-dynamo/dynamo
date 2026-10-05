# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors raised while materializing Sweeper Candidates as DGDs."""

from __future__ import annotations


class CandidateMaterializationError(ValueError):
    """A Sweeper result cannot be represented faithfully as a DGD."""


__all__ = ["CandidateMaterializationError"]
