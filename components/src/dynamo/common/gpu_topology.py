# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared physical GPU accounting for model-parallel worker topologies."""


def physical_gpu_count(
    *,
    tensor: int,
    pipeline: int,
    attention_data: int,
    moe_tensor: int,
    moe_expert: int,
) -> int:
    """Return physical GPUs after accounting for overlapping parallel axes."""
    stage_width = max(tensor, attention_data, moe_tensor * moe_expert)
    return pipeline * stage_width
