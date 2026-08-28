# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical marker vocabulary allowed in CODEOWNERS pytest policy."""

FRAMEWORK_MARKERS = frozenset({"vllm", "sglang", "trtllm"})
SELECTIVE_FEATURE_MARKERS = frozenset(
    {
        "core",
        "fault_tolerance",
        "kvbm",
        "lmcache",
        "multimodal",
        "planner",
        "router",
    }
)
ALLOWED_AREA_MARKERS = FRAMEWORK_MARKERS | SELECTIVE_FEATURE_MARKERS
