# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated import aliases for AISimulate configuration."""

from .ais_perf_args import AisPerfArgGroup as AicPerfArgGroup
from .ais_perf_args import AisPerfConfigBase as AicPerfConfigBase

__all__ = ["AicPerfArgGroup", "AicPerfConfigBase"]
