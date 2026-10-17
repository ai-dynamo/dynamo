# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker GPU resources and per-instance multinode placement."""

import pytest

from dynamo.profiler.utils.config import (
    Component,
    get_main_container,
    setup_worker_component_resources,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


@pytest.mark.parametrize(
    "args",
    [
        ["--tp-size", "16", "--data-parallel-size", "16", "--enable-dp-attention"],
        ["--tp-size=16", "--dp=16", "--enable-dp-attention"],
        ["--tp", "4", "--pp-size", "4"],
        ["--tp-size=4", "--pp=4"],
        ["--tp-size=4", "--pp-size=4"],
    ],
)
def test_parallelism_spellings_preserve_multinode_placement(args: list[str]) -> None:
    """A 16-GPU instance needs two nodes even when CLI aliases are used."""
    component = Component.model_validate(
        {
            "name": "decode",
            "type": "decode",
            "multinode": {"nodeCount": 2},
            "podTemplate": {"spec": {"containers": [{"name": "main", "args": args}]}},
        }
    )

    # The fallback GPU count would require four nodes, not the two needed by TP x PP.
    setup_worker_component_resources(component, gpu_count=32, num_gpus_per_node=8)

    assert component.multinode is not None
    assert component.multinode.nodeCount == 2
    assert get_main_container(component).resources["limits"]["nvidia.com/gpu"] == "8"


@pytest.mark.parametrize("args", [["--dp-size", "16"], ["--dp-size=16"]])
def test_data_parallel_alias_does_not_require_multinode_placement(
    args: list[str],
) -> None:
    """Independent DP replicas do not increase the per-instance GPU count."""
    component = Component.model_validate(
        {
            "name": "decode",
            "type": "decode",
            "multinode": {"nodeCount": 2},
            "podTemplate": {"spec": {"containers": [{"name": "main", "args": args}]}},
        }
    )

    setup_worker_component_resources(component, gpu_count=16, num_gpus_per_node=8)

    assert component.multinode is None
