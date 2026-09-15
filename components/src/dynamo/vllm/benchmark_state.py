# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic recurrent-state protocol shared by the scheduler and GPU worker."""

RANDOM_KDA_REQUEST_PREFIX = "__bench_random_kda_"
RANDOM_KDA_WORKER = "dynamo.vllm.benchmark_worker.BenchmarkWorker"
RANDOM_KDA_BOUND = 0.01
RANDOM_KDA_POLICY = "uniform-request-layer-rank-v1"


def recurrent_shadow_range(
    context: int, headroom: int, block_size: int
) -> tuple[int, int]:
    """State-table positions read/written by admission and its steady steps.

    Keep the state preceding the first query as well as every write position.
    Earlier entries are null placeholders, not allocated token-history pages.
    """
    first = max(0, (context - 1) // block_size)
    end = (context + 1 + headroom + block_size - 1) // block_size
    return first, end
