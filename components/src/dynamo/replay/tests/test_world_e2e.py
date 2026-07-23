# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-API smoke tests for the PyO3 multi-deployment Replay world."""

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.planner,
]


def _load_planner_config(
    report_dir: Path,
    *,
    scale_up_percent: float,
    scale_down_percent: float,
) -> dict:
    return {
        "mode": "agg",
        "backend": "mocker",
        "environment": "virtual",
        "optimization_target": "load",
        "enable_throughput_scaling": False,
        "decode_scale_up_kv_rate": scale_up_percent,
        "decode_scale_down_kv_rate": scale_down_percent,
        "load_adjustment_interval_seconds": 1,
        "scheduling": {"scale_interval_seconds": 0.25},
        "report_interval_hours": None,
        "report_output_dir": str(report_dir),
        "live_dashboard_port": 0,
    }


@pytest.mark.parametrize(
    "model_names",
    [
        pytest.param(("model-alpha", "model-beta"), id="different-models"),
        pytest.param(("shared-model", "shared-model"), id="same-model"),
    ],
)
def test_public_api_runs_two_deployments_with_shared_global_planner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_names: tuple[str, str],
) -> None:
    """Run independent deployment workloads through local planners and shared GP."""
    pytest.importorskip("dynamo._core")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    from dynamo.mocker import MockEngineArgs
    from dynamo.replay import (
        ReplayDeploymentConfig,
        ReplayGlobalPlannerConfig,
        ReplaySyntheticWorkload,
        run_replay_world,
    )

    engine = MockEngineArgs(
        num_gpu_blocks=64,
        block_size=16,
        max_num_seqs=32,
        max_num_batched_tokens=512,
        startup_time=0.0,
    )
    report_dir = tmp_path / "planner-reports"

    report = run_replay_world(
        [
            ReplayDeploymentConfig(
                deployment_id="alpha",
                model_name=model_names[0],
                planner_config=_load_planner_config(
                    report_dir,
                    scale_up_percent=100.0,
                    scale_down_percent=99.0,
                ),
                workload=ReplaySyntheticWorkload(
                    input_tokens=16,
                    output_tokens=128,
                    request_count=1,
                    arrival_interval_ms=0.0,
                ),
                extra_engine_args=engine,
                num_workers=2,
            ),
            ReplayDeploymentConfig(
                deployment_id="beta",
                model_name=model_names[1],
                planner_config=_load_planner_config(
                    report_dir,
                    scale_up_percent=0.01,
                    scale_down_percent=0.0,
                ),
                workload=ReplaySyntheticWorkload(
                    input_tokens=16,
                    output_tokens=128,
                    request_count=4,
                    arrival_interval_ms=0.0,
                ),
                extra_engine_args=engine,
                num_workers=1,
            ),
        ],
        global_planner=ReplayGlobalPlannerConfig(
            min_total_gpus=3,
            max_total_gpus=3,
        ),
    )

    assert list(report.deployments) == ["alpha", "beta"]
    assert report.deployments["alpha"].trace_report["completed_requests"] == 1
    assert report.deployments["beta"].trace_report["completed_requests"] == 4

    # Alpha's standalone scale-down cannot cross the fixed budget, but its
    # cached intent pairs with beta's scale-up in the same timestamp barrier.
    assert [
        (
            event.at_s,
            event.participant_id,
            event.status,
            event.target_decode,
        )
        for event in report.global_planner_events[:2]
    ] == [
        (1.0, "alpha", "rejected", 1),
        (1.0, "beta", "success", 2),
    ]

    alpha_scales = report.deployments["alpha"].scaling_events
    beta_scales = report.deployments["beta"].scaling_events
    assert [
        (event.at_s, event.from_count, event.to_count, event.reason)
        for event in alpha_scales
    ] == [(1.0, 2, 1, "global_planner")]
    assert [
        (event.at_s, event.from_count, event.to_count, event.reason)
        for event in beta_scales
    ] == [(1.0, 1, 2, "global_planner")]
