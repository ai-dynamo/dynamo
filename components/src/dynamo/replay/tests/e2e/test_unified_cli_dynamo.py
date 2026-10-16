# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""CPU-only end-to-end coverage for the public Dynamo-stack CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip(
    "aisimulate.config.cli",
    reason="AISimulate is an optional Dynamo simulation dependency",
)

from aisimulate.config.cli import CorePredictionConfig
from aisimulate.config.common import split_config_sections

pytestmark = [
    # Release-gating CPU smoke for plugin discovery, replay composition, and
    # recommendation-to-prediction round trips in the shipped Planner image.
    pytest.mark.e2e,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.timeout(240),
]

_REPO_ROOT = Path(__file__).resolve().parents[6]
_CONFIG_ROOT = Path("components/src/dynamo/replay/tests/e2e/configs/unified_cli")
_PREDICT_CASES = tuple(
    sorted((_REPO_ROOT / _CONFIG_ROOT / "predict/dynamo").glob("*.yaml"))
)
_RECOMMEND_CASES = tuple(
    sorted((_REPO_ROOT / _CONFIG_ROOT / "recommend/dynamo").glob("*.yaml"))
)


def _run_cli(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *args],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"aisimulate {' '.join(args)} failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _assert_concrete(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        assert "choices" not in value, f"{path} still contains a choices domain"
        assert "range" not in value, f"{path} still contains a range domain"
        assert "preset" not in value, f"{path} still contains a preset"
        for key, child in value.items():
            _assert_concrete(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_concrete(child, path=f"{path}[{index}]")


@pytest.mark.parametrize("config_path", _PREDICT_CASES, ids=lambda path: path.stem)
def test_dynamo_predict_cli_cases(config_path: Path, tmp_path: Path) -> None:
    output = tmp_path / config_path.stem
    result = _run_cli(
        "predict",
        "--stack",
        "dynamo",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    summary = json.loads(result.stdout)
    report = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
    assert summary["completed_requests"] > 0
    assert report["summary"]["completed_requests"] == summary["completed_requests"]

    expected_trace_requests = {
        "05-trace-standard-round-robin.yaml": 2,
        "09-trace-mooncake-speedup.yaml": 2,
        "10-trace-mooncake-delta-concurrency.yaml": 2,
        "11-trace-agentic-mooncake.yaml": 3,
        "12-trace-applied-compute-agentic.yaml": 3,
        "13-trace-dynamo-agentic.yaml": 4,
    }
    if config_path.name in expected_trace_requests:
        assert (
            summary["completed_requests"] == expected_trace_requests[config_path.name]
        )

    if config_path.name == "08-synthetic-throughput-planner.yaml":
        assert report["planner"]["metadata"]["bootstrap"]["status"] == "installed"
        assert "falling back to load-based scaling only" not in result.stderr


@pytest.mark.parametrize("config_path", _RECOMMEND_CASES, ids=lambda path: path.stem)
def test_dynamo_recommend_cli_cases_round_trip(
    config_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / config_path.stem
    result = _run_cli(
        "recommend",
        "--stack",
        "dynamo",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    rows = json.loads(result.stdout)
    recommendation_paths = sorted((output / "recommendations").glob("*.yaml"))
    assert rows
    assert len(recommendation_paths) == len(rows)
    assert len({path.read_bytes() for path in recommendation_paths}) == len(
        recommendation_paths
    )
    if config_path.name != "05-router-planner-pareto.yaml":
        assert [row["score"] for row in rows] == sorted(
            (row["score"] for row in rows), reverse=True
        )

    generated = []
    for index, recommendation_path in enumerate(recommendation_paths):
        raw = yaml.safe_load(recommendation_path.read_text(encoding="utf-8"))
        generated.append(raw)
        _assert_concrete(raw)
        core, _ = split_config_sections(raw, command="predict")
        CorePredictionConfig.model_validate(core)

        prediction_output = tmp_path / f"{config_path.stem}-predict-{index}"
        prediction = _run_cli(
            "predict",
            "--stack",
            "dynamo",
            "--config",
            str(recommendation_path),
            "--output-dir",
            str(prediction_output),
            "--format",
            "json",
        )
        assert json.loads(prediction.stdout)["completed_requests"] > 0
        assert "falling back to load-based scaling only" not in prediction.stderr

    disabled_planners = [
        config
        for config in generated
        if config.get("planner", {}).get("policy") == "disabled"
    ]
    assert len(disabled_planners) <= 1
    assert "router AIC payload requires" not in result.stderr
