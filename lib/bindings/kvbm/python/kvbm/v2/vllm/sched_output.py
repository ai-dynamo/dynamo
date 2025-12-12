# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process vLLM SchedulerOutput into KVBM SchedulerOutput format."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from kvbm._core import v2 as _v2

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

KvbmSchedulerOutput: TypeAlias = _v2.SchedulerOutput


def process_scheduler_output(
    iteration: int,
    scheduler_output: "VllmSchedulerOutput",
) -> KvbmSchedulerOutput:
    """
    Convert vLLM's SchedulerOutput to KVBM's SchedulerOutput format.

    This function processes vLLM's scheduler output, which uses the new API
    with `resumed_req_ids` (set) and `all_token_ids` (dict) instead of the
    deprecated per-item fields.

    Args:
        scheduler_output: vLLM's SchedulerOutput object

    Returns:
        KVBM SchedulerOutput object ready for connector metadata building
    """
    output = KvbmSchedulerOutput(iteration)

    # Process new requests
    for req in scheduler_output.scheduled_new_reqs:
        prompt_ids = [int(token) for token in req.prompt_token_ids]
        # Extract block IDs from the first (and typically only) sequence
        # - todo: add support for hybrid kv caching which will have an outer tuple > 1
        block_ids = (
            [int(block_id) for block_id in req.block_ids[0]]
            if req.block_ids and len(req.block_ids) > 0
            else []
        )
        output.add_new_request(
            req.req_id,
            prompt_token_ids=prompt_ids,
            block_ids=block_ids,
            num_computed_tokens=int(req.num_computed_tokens),
        )

    # Process cached requests using the new API
    cached = scheduler_output.scheduled_cached_reqs
    if cached is not None:
        # Use the new API: resumed_req_ids is a set, all_token_ids is a dict
        resumed_req_ids = cached.resumed_req_ids

        for (
            req_id,
            new_token_ids,
            new_block_ids,
            num_computed_tokens,
            num_output_tokens,
        ) in zip(
            cached.req_ids,
            cached.new_token_ids,
            cached.new_block_ids,
            cached.num_computed_tokens,
            cached.num_output_tokens,
        ):
            # Determine if this request resumed from preemption
            resumed = req_id in resumed_req_ids

            # Get all token IDs if this request resumed
            all_token_ids = None
            if resumed and cached.all_token_ids:
                all_token_ids = cached.all_token_ids.get(req_id)
                if all_token_ids is not None:
                    all_token_ids = [int(token) for token in all_token_ids]

            # Convert new token IDs
            tokens = [int(token) for token in new_token_ids] if new_token_ids else []

            # Extract block IDs from the first sequence
            block_ids = (
                [int(block_id) for block_id in new_block_ids[0]]
                if new_block_ids is not None and len(new_block_ids) > 0
                else []
            )

            output.add_cached_request(
                req_id,
                resumed,
                tokens,
                all_token_ids=all_token_ids,
                new_block_ids=block_ids,
                num_computed_tokens=int(num_computed_tokens),
                num_output_tokens=int(num_output_tokens),
            )

    # Set scheduled token counts
    counts_source = getattr(scheduler_output, "num_scheduled_tokens", None)
    if counts_source:
        counts = {str(req_id): int(value) for req_id, value in counts_source.items()}
        output.set_num_scheduled_tokens(counts)

    return output
