# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multinode placement derived from worker CLI arguments."""

import pytest

from dynamo.profiler.utils.config import Component, set_multinode_config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _component_with_args(args: list[str], node_count: int = 2) -> Component:
    return Component.model_validate(
        {
            "name": "decode",
            "type": "decode",
            "multinode": {"nodeCount": node_count},
            "podTemplate": {"spec": {"containers": [{"name": "main", "args": args}]}},
        }
    )


def _node_count(component: Component) -> int | None:
    if component.multinode is None:
        return None
    return component.multinode.nodeCount


@pytest.mark.parametrize(
    "args",
    [
        # The spelling combination reported in #15862.
        ["--tp-size", "16", "--data-parallel-size", "16", "--enable-dp-attention"],
        ["--tp-size=16"],
        ["--tp", "4", "--pp-size", "4"],
        ["--tp-size=4", "--pp-size=4"],
        ["--tensor-parallel-size", "4", "--pp-size", "4"],
        ["--tp-size", "4", "--pipeline-parallel-size", "4"],
    ],
)
def test_size_spellings_keep_multinode_placement(args: list[str]) -> None:
    """TP x PP of 16 GPUs needs two 8-GPU nodes, whatever spelling is used."""
    component = _component_with_args(args)

    # The fallback GPU count would need four nodes, so it cannot mask a
    # spelling the parser fails to recognize.
    set_multinode_config(component, gpu_count=32, num_gpus_per_node=8)

    assert _node_count(component) == 2


@pytest.mark.parametrize(
    "args",
    [
        ["--tp-size", "8"],
        ["--tp-size=8", "--pp-size", "1"],
        # Data-parallel replicas are independent and stay on one node.
        ["--dp-size", "16"],
        ["--data-parallel-size", "16"],
    ],
)
def test_single_node_spellings_drop_multinode_placement(args: list[str]) -> None:
    """A per-instance GPU count that fits on one node removes multinode."""
    component = _component_with_args(args)

    set_multinode_config(component, gpu_count=16, num_gpus_per_node=8)

    assert _node_count(component) is None


def test_tp_size_recomputes_an_existing_node_count() -> None:
    """A larger TP spelled with --tp-size grows the recorded node count."""
    component = _component_with_args(["--tp-size", "32"], node_count=2)

    # The fallback GPU count would keep two nodes; TP 32 needs four.
    set_multinode_config(component, gpu_count=16, num_gpus_per_node=8)

    assert _node_count(component) == 4


def test_no_parallelism_args_falls_back_to_supplied_gpu_count() -> None:
    """Without parallelism args, placement follows the supplied GPU count."""
    component = _component_with_args(["--enable-dp-attention"])

    set_multinode_config(component, gpu_count=16, num_gpus_per_node=8)

    assert _node_count(component) == 2
