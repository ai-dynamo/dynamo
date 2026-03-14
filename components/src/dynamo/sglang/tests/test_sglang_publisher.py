# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from dynamo.sglang.publisher import (
    get_local_dp_rank_range,
    set_forward_pass_metrics_worker_id,
)


def test_get_local_dp_rank_range_defaults_to_rank_zero():
    server_args = SimpleNamespace(
        dp_size=1,
        enable_dp_attention=False,
        nnodes=1,
        node_rank=0,
    )

    assert list(get_local_dp_rank_range(server_args)) == [0]


def test_get_local_dp_rank_range_respects_multinode_dp_attention():
    server_args = SimpleNamespace(
        dp_size=8,
        enable_dp_attention=True,
        nnodes=2,
        node_rank=1,
    )

    assert list(get_local_dp_rank_range(server_args)) == [4, 5, 6, 7]


def test_set_forward_pass_metrics_worker_id_uses_endpoint_identity():
    server_args = SimpleNamespace(forward_pass_metrics_port=20380)
    endpoint = SimpleNamespace(connection_id=lambda: "endpoint-9")

    set_forward_pass_metrics_worker_id(server_args, endpoint)

    assert server_args.forward_pass_metrics_worker_id == "endpoint-9"


def test_set_forward_pass_metrics_worker_id_is_noop_when_disabled():
    server_args = SimpleNamespace(forward_pass_metrics_port=None)
    endpoint = SimpleNamespace(connection_id=lambda: "endpoint-9")

    set_forward_pass_metrics_worker_id(server_args, endpoint)

    assert not hasattr(server_args, "forward_pass_metrics_worker_id")
