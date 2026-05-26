# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def per_rank_kv_blocks(
    total_kv_blocks: int | None, data_parallel_size: int
) -> int | None:
    if total_kv_blocks is None:
        return None

    if data_parallel_size <= 1 or total_kv_blocks <= 0:
        return total_kv_blocks

    per_rank = total_kv_blocks // data_parallel_size
    if per_rank == 0:
        logger.warning(
            "vLLM reported fewer total KV blocks (%s) than DP ranks (%s); "
            "publishing 1 block per rank",
            total_kv_blocks,
            data_parallel_size,
        )
        return 1

    remainder = total_kv_blocks % data_parallel_size
    if remainder:
        logger.warning(
            "vLLM reported aggregate KV blocks (%s) not divisible by DP ranks (%s); "
            "publishing floor per-rank capacity %s",
            total_kv_blocks,
            data_parallel_size,
            per_rank,
        )

    return per_rank
