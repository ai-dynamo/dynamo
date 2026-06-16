from __future__ import annotations

from typing import Optional

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
    return False
