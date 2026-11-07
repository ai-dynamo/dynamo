# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

from dynamo.planner.core.types import WorkerCounts
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.worker_info import WorkerInfo

_MDC_REFRESH_FIELDS = (
    "total_kv_blocks",
    "kv_cache_block_size",
    "max_num_seqs",
    "max_num_batched_tokens",
    "context_length",
    "speculative_nextn",
)


def worker_info_changed(
    old_info: Optional[WorkerInfo], new_info: Optional[WorkerInfo]
) -> bool:
    if old_info is None or new_info is None:
        return old_info is not new_info
    for field_name in _MDC_REFRESH_FIELDS:
        new_val = getattr(new_info, field_name)
        old_val = getattr(old_info, field_name)
        if new_val != old_val:
            return True
    return False


def deployment_state_changed(
    old_state: DeploymentState,
    new_state: DeploymentState,
    check_prefill: bool,
    check_decode: bool,
) -> bool:
    if check_prefill and worker_info_changed(
        old_state.prefill.info, new_state.prefill.info
    ):
        return True
    if check_decode and worker_info_changed(
        old_state.decode.info, new_state.decode.info
    ):
        return True
    if check_prefill and old_state.prefill.num_gpus != new_state.prefill.num_gpus:
        return True
    if check_decode and old_state.decode.num_gpus != new_state.decode.num_gpus:
        return True
    if (
        check_prefill
        and old_state.prefill.gpus_per_replica != new_state.prefill.gpus_per_replica
    ):
        return True
    if (
        check_decode
        and old_state.decode.gpus_per_replica != new_state.decode.gpus_per_replica
    ):
        return True
    return False


def worker_counts_from_deployment_state(
    state: DeploymentState,
    *,
    require_prefill: bool,
    require_decode: bool,
) -> WorkerCounts:
    """Convert a deployment snapshot into the worker counts a tick consumes.

    Ready, expected, and scaling values are reported only for required roles.
    Pending-startup counts are copied for both roles regardless.
    """
    return WorkerCounts(
        ready_num_prefill=(state.prefill.replicas.active if require_prefill else None),
        ready_num_decode=(state.decode.replicas.active if require_decode else None),
        expected_num_prefill=(
            state.prefill.replicas.expected if require_prefill else None
        ),
        expected_num_decode=(
            state.decode.replicas.expected if require_decode else None
        ),
        prefill_scaling_in_progress=(
            require_prefill and state.prefill.replicas.scaling
        ),
        decode_scaling_in_progress=(require_decode and state.decode.replicas.scaling),
        pending_num_prefill=state.prefill.replicas.pending_startup,
        pending_num_decode=state.decode.replicas.pending_startup,
    )
