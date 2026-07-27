# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None,
        reason="sglang not installed in this container",
    ),
]


def test_model_card_registration_keeps_global_dp_range():
    from dynamo.sglang.capacity import model_card_dp_rank_bounds

    server_args = SimpleNamespace(
        dp_size=16,
        enable_dp_attention=True,
        nnodes=4,
        node_rank=0,
    )

    assert model_card_dp_rank_bounds(server_args) == (0, 16)


def _args(**kw) -> SimpleNamespace:
    base = dict(dp_size=1, enable_dp_attention=False, nnodes=1, node_rank=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_single_node_publishes_kv_events():
    from dynamo.sglang.capacity import publishes_kv_events

    assert publishes_kv_events(_args()) is True


def test_multinode_without_dp_attention_publishes_only_from_leader():
    """The whole point: TP-only multinode must advertise ONE source per worker.

    Without DP attention every node's local rank slice is [0, 1) and non-leaders
    publish under the leader's worker id, so letting them all advertise makes the
    router see duplicate (worker_id, dp_rank) keys, mark them Ambiguous, and drop
    every KV event.
    """
    from dynamo.sglang.capacity import local_dp_rank_bounds, publishes_kv_events

    leader = _args(nnodes=2, node_rank=0)
    follower = _args(nnodes=2, node_rank=1)

    # Precondition for the collision this guards against.
    assert local_dp_rank_bounds(leader) == local_dp_rank_bounds(follower) == (0, 1)

    assert publishes_kv_events(leader) is True
    assert publishes_kv_events(follower) is False


def test_dp_attention_publishes_from_every_node():
    """With DP attention each node owns a distinct slice, so no key collides."""
    from dynamo.sglang.capacity import local_dp_rank_bounds, publishes_kv_events

    nodes = [
        _args(dp_size=4, enable_dp_attention=True, nnodes=2, node_rank=rank)
        for rank in (0, 1)
    ]
    assert local_dp_rank_bounds(nodes[0]) != local_dp_rank_bounds(nodes[1])
    assert all(publishes_kv_events(n) is True for n in nodes)


def test_dp_size_one_with_dp_attention_still_leader_only():
    """dp_size=1 keeps the shared [0, 1) slice even if the flag is set."""
    from dynamo.sglang.capacity import publishes_kv_events

    assert (
        publishes_kv_events(_args(enable_dp_attention=True, nnodes=2, node_rank=1))
        is False
    )
