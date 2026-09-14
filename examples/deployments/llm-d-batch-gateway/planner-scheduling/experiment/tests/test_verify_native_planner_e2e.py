# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import verify_native_planner_e2e

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_missing_transitions_fixture(root: Path) -> tuple[Path, Path]:
    run_dir = root / "run"
    evidence_dir = root / "evidence"
    metrics_dir = run_dir / "metrics" / "async"
    metrics_dir.mkdir(parents=True)
    evidence_dir.mkdir()

    _write_json(
        run_dir / "terminal-batch.json",
        {
            "id": "batch-test",
            "status": "completed",
            "request_counts": {"completed": 100, "failed": 0, "total": 100},
        },
    )
    _write_json(
        run_dir / "result-validation.json",
        {
            "valid": True,
            "downloaded_output_lines": 100,
            "unique_custom_ids": 100,
        },
    )
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (metrics_dir / "sample.prom").write_text(
        'llm_d_async_async_dispatched_requests_total{pool_name="dynamo-batch"} 403\n',
        encoding="utf-8",
    )
    _write_json(
        metrics_dir / "sample.json",
        {"observed_at": "2026-08-28T21:38:17.330Z"},
    )

    state = {
        "observed_at": "2026-08-28T21:35:35Z",
        "adapter_spec": 0,
        "ready_worker_pods": 0,
        "worker_ready_replicas": 0,
        "dgd_ready": "False",
        "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","1","decision"',
    }
    (evidence_dir / "state.jsonl").write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )
    _write_json(
        evidence_dir / "dgdsa.before.json",
        {
            "metadata": {"generation": 1, "resourceVersion": "1"},
            "spec": {"replicas": 0},
            "status": {"replicas": 0},
        },
    )
    _write_json(
        evidence_dir / "dgdsa.after.json",
        {
            "metadata": {"generation": 1, "resourceVersion": "1"},
            "spec": {"replicas": 0},
            "status": {"replicas": 0},
        },
    )
    (evidence_dir / "dgdsa.watch.jsonstream").write_text(
        json.dumps({"object": {"spec": {"replicas": 0}}}) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "planner.log").write_text("", encoding="utf-8")
    (evidence_dir / "redis.after.txt").write_text(
        "llm-d.ai/v1alpha1\ndynamo-batch\n0\n1\ndecision\n",
        encoding="utf-8",
    )
    (evidence_dir / "redis.after.pttl-ms.txt").write_text("1000\n", encoding="utf-8")
    (evidence_dir / "async-metrics.before.txt").write_text(
        "llm_d_async_async_dispatched_requests_total 400\n"
        "llm_d_async_async_successful_requests_total 400\n",
        encoding="utf-8",
    )
    (evidence_dir / "async-metrics.after.txt").write_text(
        "llm_d_async_async_dispatched_requests_total 500\n"
        "llm_d_async_async_successful_requests_total 500\n"
        "llm_d_async_async_broker_backlog 0\n"
        "llm_d_async_async_inflight_requests 0\n"
        "llm_d_async_async_queue_depth 0\n"
        "llm_d_async_async_drain_limit_rps 0\n",
        encoding="utf-8",
    )
    return run_dir, evidence_dir


def _write_scale_to_zero_fixture(root: Path) -> tuple[Path, Path]:
    run_dir, evidence_dir = _write_missing_transitions_fixture(root)
    states = [
        {
            "observed_at": "2026-08-28T21:35:35Z",
            "adapter_spec": 0,
            "ready_worker_pods": 0,
            "worker_ready_replicas": 0,
            "dgd_ready": "False",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","1","d-0"',
        },
        {
            "observed_at": "2026-08-28T21:36:16Z",
            "adapter_spec": 1,
            "ready_worker_pods": 0,
            "worker_ready_replicas": 0,
            "dgd_ready": "False",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","2","d-1"',
        },
        {
            "observed_at": "2026-08-28T21:38:11Z",
            "adapter_spec": 1,
            "ready_worker_pods": 1,
            "worker_ready_replicas": 1,
            "dgd_ready": "True",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","3","d-2"',
        },
        {
            "observed_at": "2026-08-28T21:38:17Z",
            "adapter_spec": 1,
            "ready_worker_pods": 1,
            "worker_ready_replicas": 1,
            "dgd_ready": "True",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","5","4","d-3"',
        },
        {
            "observed_at": "2026-08-28T21:38:42Z",
            "adapter_spec": 0,
            "ready_worker_pods": 0,
            "worker_ready_replicas": 0,
            "dgd_ready": "False",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","5","d-4"',
        },
    ]
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )
    _write_json(
        evidence_dir / "dgdsa.after.json",
        {
            "metadata": {"generation": 3, "resourceVersion": "3"},
            "spec": {"replicas": 0},
            "status": {"replicas": 0},
        },
    )
    (evidence_dir / "dgdsa.watch.jsonstream").write_text(
        "".join(
            json.dumps({"object": {"spec": {"replicas": replicas}}}) + "\n"
            for replicas in (0, 1, 0)
        ),
        encoding="utf-8",
    )
    (evidence_dir / "planner.log").write_text(
        "2026-08-28T21:35:35.100Z INFO Batch scheduling decision: "
        "pipeline_action=apply pool_id=dynamo-batch replica_floor=None "
        "max_admission_rps=0.0 decision_id=d-idle-before-job valid_until_s=5.0 "
        "idle_replica_target=0\n"
        "2026-08-28T21:36:15.637Z INFO Batch scheduling decision: "
        "pipeline_action=apply pool_id=dynamo-batch replica_floor=1 "
        "max_admission_rps=0.0 decision_id=d-1 valid_until_s=10.0 "
        "idle_replica_target=None\n"
        "2026-08-28T21:36:15.695Z INFO Updating decode component "
        "VllmDecodeWorker from 0 to desired replica count 1\n"
        "2026-08-28T21:36:15.707Z INFO Scaled DGDSA "
        "qwen3-0-6b-batch-vllmdecodeworker to 1 replicas\n"
        "2026-08-28T21:38:16.891Z INFO Batch scheduling decision: "
        "pipeline_action=apply pool_id=dynamo-batch replica_floor=1 "
        "max_admission_rps=5.0 decision_id=d-2 valid_until_s=20.0 "
        "idle_replica_target=None\n"
        "2026-08-28T21:38:42.162Z INFO Batch scheduling decision: "
        "pipeline_action=apply pool_id=dynamo-batch replica_floor=None "
        "max_admission_rps=0.0 decision_id=d-3 valid_until_s=30.0 "
        "idle_replica_target=0\n"
        "2026-08-28T21:38:42.200Z INFO Updating decode component "
        "VllmDecodeWorker from 1 to desired replica count 0\n"
        "2026-08-28T21:38:42.250Z INFO Scaled DGDSA "
        "qwen3-0-6b-batch-vllmdecodeworker to 0 replicas\n",
        encoding="utf-8",
    )
    return run_dir, evidence_dir


def _write_warm_one_fixture(root: Path) -> tuple[Path, Path]:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(root)
    states = [
        json.loads(line)
        for line in (evidence_dir / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    states[-1].update(
        adapter_spec=1,
        ready_worker_pods=1,
        worker_ready_replicas=1,
        dgd_ready="True",
    )
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )
    _write_json(
        evidence_dir / "dgdsa.after.json",
        {
            "metadata": {"generation": 2, "resourceVersion": "2"},
            "spec": {"replicas": 1},
            "status": {"replicas": 1},
        },
    )
    (evidence_dir / "dgdsa.watch.jsonstream").write_text(
        "".join(
            json.dumps({"object": {"spec": {"replicas": replicas}}}) + "\n"
            for replicas in (0, 1)
        ),
        encoding="utf-8",
    )
    planner_lines = (
        (evidence_dir / "planner.log").read_text(encoding="utf-8").splitlines()
    )
    (evidence_dir / "planner.log").write_text(
        "\n".join(planner_lines[:-2]).replace(
            "idle_replica_target=0", "idle_replica_target=1"
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, evidence_dir


def test_missing_transitions_produce_a_normal_failure_report(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_missing_transitions_fixture(tmp_path)

    result = verify_native_planner_e2e.verify(
        run_dir, evidence_dir, expected_lease_duration_seconds=60
    )

    assert result["all_passed"] is False
    assert result["assertions"]["scale_preceded_worker_readiness"] is False
    assert result["assertions"]["lease_remained_closed_until_ready"] is False
    assert result["assertions"]["dispatch_started_after_positive_lease"] is False
    assert result["assertions"]["terminal_zero_observed_after_positive"] is False
    assert result["timeline"]["adapter_one"] is None
    assert result["timeline"]["worker_ready"] is None
    assert result["timeline"]["positive_lease"] is None
    assert result["timeline"]["terminal_zero_lease"] is None


def test_configured_zero_idle_target_is_proven_end_to_end(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is True
    assert result["assertions"]["adapter_finished_at_expected_idle_target"] is True
    assert result["assertions"]["planner_logged_configured_idle_target"] is True
    assert result["assertions"]["planner_logged_idle_scale"] is True
    assert result["assertions"]["planner_policy_log_order"] is True
    assert result["observed"]["expected_idle_replicas"] == 0
    assert result["observed"]["planner_idle_replica_targets"] == [0, None, None, 0]


def test_requested_idle_target_must_match_evidence(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=1,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["adapter_finished_at_expected_idle_target"] is False
    assert result["assertions"]["planner_logged_configured_idle_target"] is False


def test_warm_one_idle_target_is_a_verified_scaling_noop(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_warm_one_fixture(tmp_path)

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=1,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is True
    assert result["assertions"]["adapter_finished_at_expected_idle_target"] is True
    assert result["assertions"]["planner_logged_configured_idle_target"] is True
    assert "planner_logged_idle_scale" not in result["assertions"]


def test_terminal_lease_freshness_uses_configured_duration(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    (evidence_dir / "redis.after.pttl-ms.txt").write_text("85700\n", encoding="utf-8")

    configured_duration = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=90,
    )

    assert configured_duration["all_passed"] is True
    assert configured_duration["observed"]["redis_terminal_pttl_ms"] == 85_700
    assert configured_duration["observed"]["expected_lease_duration_seconds"] == 90
    assert configured_duration["observed"]["lease_pttl_tolerance_seconds"] == 5
    assert configured_duration["observed"]["maximum_expected_lease_pttl_ms"] == 95_000


@pytest.mark.parametrize(
    ("pttl_ms", "expected_fresh"), [(95_000, True), (95_001, False)]
)
def test_terminal_lease_freshness_enforces_clock_skew_boundary(
    tmp_path: Path, pttl_ms: int, expected_fresh: bool
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    (evidence_dir / "redis.after.pttl-ms.txt").write_text(
        f"{pttl_ms}\n", encoding="utf-8"
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=90,
    )

    assert (
        result["assertions"]["authoritative_terminal_lease_zero_and_fresh"]
        is expected_fresh
    )


@pytest.mark.parametrize("duration", [0, -1, True, float("inf"), float("nan")])
def test_expected_lease_duration_must_be_finite_and_positive(
    tmp_path: Path, duration: float
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)

    with pytest.raises(ValueError, match="expected_lease_duration_seconds"):
        verify_native_planner_e2e.verify(
            run_dir,
            evidence_dir,
            expected_lease_duration_seconds=duration,
        )
