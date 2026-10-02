# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint configuration and failure reporting without a live deployment."""

import json
from pathlib import Path

import pytest
from autoscaling_arena.runners.real import (
    EndpointMatchResult,
    format_endpoint_leaderboard,
    load_endpoints,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_load_endpoints_accepts_json_and_yaml(tmp_path: Path, suffix: str):
    path = tmp_path / f"endpoints{suffix}"
    path.write_text(
        json.dumps(
            {
                "endpoints": [
                    {
                        "name": "static",
                        "url": "http://localhost:8000",
                        "model": "example",
                    }
                ]
            }
        )
        if suffix == ".json"
        else "endpoints:\n  - name: static\n    url: http://localhost:8000\n    model: example\n"
    )
    endpoints = load_endpoints(path)
    assert len(endpoints) == 1
    assert endpoints[0].name == "static"
    assert endpoints[0].model == "example"


def test_committed_endpoint_example_loads():
    path = Path(__file__).resolve().parents[1] / "configs/endpoints.example.yaml"
    assert len(load_endpoints(path)) == 2


def test_failed_endpoint_without_summary_still_renders():
    failure = EndpointMatchResult(
        endpoint="failed",
        workload="flat",
        profile="interactive",
        metrics={},
        returncode=1,
        artifact_dir="artifacts",
    )
    completed = EndpointMatchResult(
        endpoint="completed",
        workload="flat",
        profile="interactive",
        metrics={"goodput_rps": 0.0, "mean_ttft_ms": 100.0},
        returncode=0,
        artifact_dir="artifacts",
    )
    text = format_endpoint_leaderboard([failure, completed])
    assert "[FAILED rc=1]" in text
    assert "n/a" in text
    assert text.index("completed") < text.index("failed")
