# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve model names to inline Custom case profiles."""

from __future__ import annotations

INLINE_CASE_PROFILES = (
    "generic",
    "deepseek_v4",
    "gemma4",
    "qwen3_coder",
    "glm5",
    "glm47",
    "minimax_m2",
    "gpt_oss",
    "kimi_k2",
)


def model_case_profile(model: str) -> str:
    normalized = model.lower().replace("_", "-")
    if "kimi-k2" in normalized or "kimi-k25" in normalized or "kimi-k26" in normalized:
        return "kimi_k2"
    if "deepseek-v4" in normalized:
        return "deepseek_v4"
    if "gemma-4" in normalized or "gemma4" in normalized:
        return "gemma4"
    if (
        "qwen3.6" in normalized
        or "qwen3-6" in normalized
        or "qwen3-coder" in normalized
    ):
        return "qwen3_coder"
    if "glm5" in normalized or "glm-5" in normalized or "glm-51" in normalized:
        return "glm5"
    if "glm47" in normalized or "glm-4.7" in normalized or "glm-47" in normalized:
        return "glm47"
    if "minimax" in normalized or "mini-max" in normalized or "m2.7" in normalized:
        return "minimax_m2"
    if "gpt-oss" in normalized or "gptoss" in normalized:
        return "gpt_oss"
    return "generic"
