# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for declared GPU costs.

The point of this table is that a GPU-free deployment -- above all a mocker
topology -- can take part in the GPU budget it exists to help test.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dynamo.global_planner.gpu_cost import GpuCostConfig, GpuCostResolver
from dynamo.global_planner.kubernetes_capacity_manager import KubernetesCapacityManager

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _resolver(**kwargs) -> GpuCostResolver:
    return GpuCostResolver(GpuCostConfig(**kwargs))


def _component(name, replicas, gpu, ctype):
    """A v1beta1 component; ``gpu=None`` declares no nvidia.com/gpu (a mocker)."""
    resources = {"limits": {"nvidia.com/gpu": gpu}} if gpu is not None else {}
    return {
        "name": name,
        "type": ctype,
        "replicas": replicas,
        "podTemplate": {
            "spec": {"containers": [{"name": "main", "resources": resources}]}
        },
    }


def _manager(components, gpu_cost=None, dgd="dsv4-flash"):
    manager = KubernetesCapacityManager(
        "sachalm", gpu_cost=GpuCostResolver(gpu_cost)
    )
    connector = MagicMock()
    connector.parent_dgd_name = dgd
    connector.kube_api.get_graph_deployment.return_value = {
        "spec": {"components": components}
    }
    manager.connectors[f"sachalm/{dgd}"] = connector
    return manager


# ---------------------------------------------------------------------------- #
# Resolution                                                                   #
# ---------------------------------------------------------------------------- #


def test_undeclared_pool_resolves_to_none():
    # None means "no opinion", not "free" -- pricing an unknown pool at zero
    # would silently shrink the budget.
    assert _resolver().resolve("ns/a", "prefill") is None


def test_most_specific_selector_wins():
    resolver = _resolver(
        pools=[
            {"selector": "sachalm/**", "gpu_per_replica": 1},
            {"selector": "sachalm/dsv4-flash", "gpu_per_replica": 8},
            {"selector": "sachalm/dsv4-flash/prefill", "gpu_per_replica": 4},
        ]
    )
    assert resolver.resolve("sachalm/dsv4-flash", "prefill") == 4
    assert resolver.resolve("sachalm/dsv4-flash", "decode") == 8
    assert resolver.resolve("sachalm/gpt-oss", "decode") == 1


def test_selector_semantics_match_the_priority_table():
    # Both tables share pool_selectors.py; this guards against them drifting apart.
    resolver = _resolver(pools=[{"selector": "a/*/prefill", "gpu_per_replica": 2}])
    assert resolver.resolve("a/b", "prefill") == 2
    assert resolver.resolve("a/b/c", "prefill") is None
    assert resolver.resolve("a/b", "decode") is None


def test_zero_cost_is_allowed():
    # An explicit 0 keeps a pool out of budget totals, which is different from
    # not declaring it at all.
    assert _resolver(pools=[{"selector": "ns/a", "gpu_per_replica": 0}]).resolve(
        "ns/a", "decode"
    ) == 0


@pytest.mark.parametrize(
    "entry",
    [
        {"selector": "ns/a", "gpu_per_replica": -1},
        {"selector": "a//b", "gpu_per_replica": 1},
        {"selector": "ns/a"},
    ],
)
def test_invalid_entries_are_rejected(entry):
    with pytest.raises(ValidationError):
        GpuCostConfig(pools=[entry])


def test_duplicate_selectors_are_rejected():
    with pytest.raises(ValidationError, match="duplicate selector"):
        GpuCostConfig(
            pools=[
                {"selector": "ns/a", "gpu_per_replica": 1},
                {"selector": "ns/a", "gpu_per_replica": 2},
            ]
        )


# ---------------------------------------------------------------------------- #
# Effect on reading pool state                                                 #
# ---------------------------------------------------------------------------- #


def test_mocker_pools_are_unreadable_without_a_declared_cost():
    """The gap this table closes: with a budget enabled, an unpriced pool fails
    the whole snapshot, so every scale request errors out."""
    manager = _manager(
        [
            _component("prefill-svc", 2, None, "prefill"),
            _component("decode-svc", 3, None, "decode"),
        ]
    )
    with pytest.raises(RuntimeError, match="Failed to read deployment"):
        manager.observe(require_complete=True)


def test_declared_cost_lets_mocker_pools_join_the_budget():
    manager = _manager(
        [
            _component("prefill-svc", 2, None, "prefill"),
            _component("decode-svc", 3, None, "decode"),
        ],
        gpu_cost=GpuCostConfig(
            pools=[{"selector": "sachalm/dsv4-**", "gpu_per_replica": 8}]
        ),
    )
    pools = manager.observe(require_complete=True)["sachalm/dsv4-flash"]
    assert (pools["prefill"].current_replicas, pools["prefill"].gpu_per_replica) == (2, 8)
    assert (pools["decode"].current_replicas, pools["decode"].gpu_per_replica) == (3, 8)


def test_the_dgd_wins_where_it_declares_gpus():
    """A declared cost is a fallback, never an override, so a stale config can
    never make the planner under-count real hardware."""
    manager = _manager(
        [
            _component("prefill-svc", 1, 4, "prefill"),  # DGD says 4
            _component("decode-svc", 1, None, "decode"),  # DGD silent
        ],
        gpu_cost=GpuCostConfig(
            pools=[{"selector": "sachalm/**", "gpu_per_replica": 1}]
        ),
    )
    pools = manager.observe(require_complete=True)["sachalm/dsv4-flash"]
    assert pools["prefill"].gpu_per_replica == 4  # DGD, not the declared 1
    assert pools["decode"].gpu_per_replica == 1  # fallback applies


def test_unpriced_pool_error_names_both_remedies():
    manager = _manager([_component("prefill-svc", 1, None, "prefill")])
    with pytest.raises(RuntimeError) as excinfo:
        manager.observe(require_complete=True)
    message = str(excinfo.value)
    assert "resources.limits.nvidia.com/gpu" in message
    assert "gpu_cost" in message
