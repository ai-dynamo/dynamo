# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU count filtering utilities for MoE model profiling.

These filters ensure that GPU configurations are compatible with model constraints:
- Expert parallelism requires num_experts % ep_size == 0
- FP8 quantization requires partition_size % block_size == 0
"""

from typing import List, Optional, Tuple

# SGLang FP8 quantization block size
FP8_BLOCK_SIZE = 128


def filter_gpu_counts_by_expert_divisibility(
    gpu_counts: List[int],
    num_experts: int,
    min_gpus: int,
    max_gpus: int,
) -> Tuple[List[int], Optional[str], Optional[str]]:
    """
    Filter GPU counts to only include divisors of num_experts.
    SGLang requires num_physical_experts % ep_size == 0 for proper expert distribution.
    """
    original_counts = gpu_counts.copy()
    filtered = [count for count in gpu_counts if num_experts % count == 0]

    if not filtered:
        # No valid counts
        valid_divisors = [
            d for d in range(min_gpus, max_gpus + 1) if num_experts % d == 0
        ]
        error_msg = (
            f"No valid GPU counts found that divide evenly into num_experts={num_experts}. "
            f"Original candidates were {original_counts}. "
            f"Valid divisors in range would be: {valid_divisors}"
        )
        return [], None, error_msg

    if len(filtered) < len(original_counts):
        # Some counts filtered
        info_msg = (
            f"Filtered GPU counts from {original_counts} to {filtered} "
            f"(only divisors of num_experts={num_experts})"
        )
        return filtered, info_msg, None

    # No filtering needed
    return filtered, None, None


def filter_gpu_counts_by_fp8_alignment(
    gpu_counts: List[int],
    intermediate_size: int,
    block_size: int = FP8_BLOCK_SIZE,
) -> Tuple[List[int], Optional[str], Optional[str]]:
    """
    Filter GPU counts for FP8 quantization alignment.

    FP8 quantization requires partition sizes to be divisible by block_n.
    When a model layer with intermediate_size is partitioned across tp_size GPUs,
    the partition size (intermediate_size / tp_size) must be divisible by block_size.
    """
    original_counts = gpu_counts.copy()
    filtered = [
        count for count in gpu_counts if (intermediate_size // count) % block_size == 0
    ]

    if not filtered:
        # No valid counts
        warning_msg = (
            f"No GPU counts satisfy FP8 alignment constraint "
            f"(intermediate_size={intermediate_size}, block_size={block_size}). "
            f"Keeping original counts {original_counts} - profiling may fail for some configurations."
        )
        return original_counts, None, warning_msg

    if len(filtered) < len(original_counts):
        # Some counts filtered
        removed = set(original_counts) - set(filtered)
        info_msg = (
            f"Filtered GPU counts from {original_counts} to {filtered} for FP8 alignment "
            f"(removed {removed}: intermediate_size={intermediate_size} not divisible by "
            f"{block_size} when partitioned)"
        )
        return filtered, info_msg, None

    # No filtering needed
    return filtered, None, None
