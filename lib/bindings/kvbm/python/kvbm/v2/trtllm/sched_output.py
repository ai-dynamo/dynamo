# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate TRT-LLM's SchedulerOutput into the KVBM Rust SchedulerOutput."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from kvbm._core import v2 as _v2

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
        SchedulerOutput as TrtllmSchedulerOutput,
    )

KvbmSchedulerOutput: TypeAlias = _v2.SchedulerOutput


def process_trtllm_scheduler_output(
    iteration: int,
    scheduler_output: "TrtllmSchedulerOutput",
) -> KvbmSchedulerOutput:
    """
    Convert TRT-LLM's SchedulerOutput to KVBM's SchedulerOutput format.

    Key differences from the vLLM adapter:
    - ``request_id`` is ``int`` in TRT-LLM -> ``str()`` for Rust.
    - ``resumed`` is always ``False`` (TRT-LLM re-admits preempted
      requests as new context requests, not resumed cached requests).
    - ``new_token_ids`` is always empty (decode offload disabled in stage 1).
    - Block IDs are flat ``List[int]`` (no hybrid KV cache nesting).

    Args:
        iteration: The current iteration number.
        scheduler_output: TRT-LLM's SchedulerOutput containing
            ``new_requests`` and ``cached_requests``.

    Returns:
        KVBM SchedulerOutput ready for ``build_connector_metadata``.
    """
    output = KvbmSchedulerOutput(iteration)

    for req in scheduler_output.new_requests:
        output.add_new_request(
            str(req.request_id),
            prompt_token_ids=req.new_tokens,
            block_ids=req.new_block_ids,
            num_computed_tokens=req.computed_position,
        )

    for req in scheduler_output.cached_requests:
        output.add_cached_request(
            str(req.request_id),
            False,  # resumed: always False for TRT-LLM
            [],  # new_token_ids: empty (decode offload disabled)
            all_token_ids=None,
            new_block_ids=req.new_block_ids,
            num_computed_tokens=req.computed_position,
            num_output_tokens=req.num_scheduled_tokens,
        )

    counts = {
        str(req.request_id): req.num_scheduled_tokens
        for req in scheduler_output.new_requests + scheduler_output.cached_requests
    }
    output.set_num_scheduled_tokens(counts)

    return output
