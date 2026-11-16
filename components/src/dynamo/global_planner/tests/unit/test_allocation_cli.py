# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise policy files through the offline allocation command."""

import json
from pathlib import Path

import pytest

from dynamo.global_planner.plan_allocation import main

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def allocation_files(tmp_path):
    policy = {
        "min_total_gpus": 16,
        "max_total_gpus": 16,
        "pools": [
            {
                "pool_id": name,
                "participant_id": f"prod/{name}",
                "component_name": "DecodeWorker",
                "role": "decode",
                "gpu_per_replica": 4,
                "min_gpus": 4,
                "max_gpus": 16,
                "priority": priority,
            }
            for name, priority in [("interactive", 100), ("batch", 10)]
        ],
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    allocations_path = tmp_path / "allocations.json"
    allocations_path.write_text(json.dumps({"interactive": 8, "batch": 8}))
    return [
        "--policy",
        str(policy_path),
        "--allocations",
        str(allocations_path),
        "--pool",
        "interactive",
        "--replicas",
        "3",
    ]


def test_command_returns_complete_reclamation_plan(allocation_files, capsys):
    assert main(allocation_files) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["desired_replicas"] == output["effective_replicas"] == 3
    assert output["reason"] == "admitted"
    assert output["plan"]["allocations_gpus"] == {"interactive": 12, "batch": 4}
    assert output["plan"]["target_replicas"] == {"interactive": 3, "batch": 1}
    assert output["plan"]["reclaimed_gpus"] == {"batch": 4}


def test_command_reports_pending_without_a_partial_plan(allocation_files, capsys):
    assert main(allocation_files + ["--unavailable-pool", "batch"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["plan"] is None
    assert output["reason"] == "insufficient_capacity"


@pytest.mark.parametrize(
    "argument, contents",
    [
        ("--policy", "not JSON"),
        ("--allocations", '{"interactive": 8}'),
        ("--allocations", "[]"),
        ("--allocations", '{"interactive": true, "batch": 8}'),
    ],
)
def test_command_rejects_invalid_input(allocation_files, capsys, argument, contents):
    Path(allocation_files[allocation_files.index(argument) + 1]).write_text(contents)
    with pytest.raises(SystemExit) as exc:
        main(allocation_files)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "error:" in captured.err


def test_command_rejects_missing_file(allocation_files, capsys):
    allocation_files[1] += ".missing"
    with pytest.raises(SystemExit) as exc:
        main(allocation_files)
    assert exc.value.code == 2
    assert "No such file" in capsys.readouterr().err
