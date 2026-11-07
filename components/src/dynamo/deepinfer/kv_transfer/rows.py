# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row pairing shared by the transfer backends: which local rows receive which remote rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .base import KVLayout

if TYPE_CHECKING:
    from dynamo.deepinfer.disagg import DeepInferHandoff

BLOCKS = "gpu_blocks"
STATES = "gpu_states"

# (region name, local rows, remote rows)
ReadPlan = list[tuple[str, list[int], list[int]]]


def plan_read(
    local: KVLayout,
    remote_shapes: Mapping[str, tuple[int, int]],
    handoff: DeepInferHandoff,
    local_block_ids: Sequence[int],
    local_state_id: int,
) -> ReadPlan:
    """Pair the handoff's block rows (and state slot) with the local ones.

    ``remote_shapes`` maps the peer's region names to ``(num_items, item_bytes)``.
    Raises ``ValueError`` when the layouts differ or an id is out of range: a
    row-for-row copy is only meaningful between identical layouts.
    """
    local_names = {r.name for r in local.regions}
    if local_names != set(remote_shapes):
        raise ValueError(
            f"KV regions differ: local {sorted(local_names)}, peer {sorted(remote_shapes)}"
        )
    if len(local_block_ids) != len(handoff.block_ids):
        raise ValueError(
            f"{len(local_block_ids)} local blocks for {len(handoff.block_ids)} remote blocks"
        )
    plan: ReadPlan = [(BLOCKS, list(local_block_ids), list(handoff.block_ids))]
    if STATES in local_names:
        plan.append((STATES, [local_state_id], [handoff.state_id]))
    for name, dst, src in plan:
        region = local.region(name)
        remote_items, remote_bytes = remote_shapes[name]
        if remote_bytes != region.item_bytes:
            raise ValueError(
                f"{name} rows are {region.item_bytes} B here and {remote_bytes} B on the peer"
            )
        _check_ids(name, "local", dst, region.num_items)
        _check_ids(name, "remote", src, remote_items)
    return plan


def _check_ids(region: str, side: str, ids: Sequence[int], num_items: int) -> None:
    bad = [i for i in ids if not 0 <= i < num_items]
    if bad:
        raise ValueError(f"{side} {region} ids {bad[:4]} outside [0, {num_items})")
