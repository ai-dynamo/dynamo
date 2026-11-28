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


def _write_dispatch_sample(
    run_dir: Path, name: str, value: int, observed_at: str
) -> None:
    metrics_dir = run_dir / "metrics" / "async"
    (metrics_dir / f"{name}.prom").write_text(
        "llm_d_async_async_dispatched_requests_total"
        f'{{pool_name="dynamo-batch"}} {value}\n',
        encoding="utf-8",
    )
    _write_json(metrics_dir / f"{name}.json", {"observed_at": observed_at})


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
    _write_dispatch_sample(
        run_dir,
        "000-unchanged-after-positive-lease",
        400,
        "2026-08-28T21:38:17.100Z",
    )
    (evidence_dir / "capture-valid.txt").write_text("1\n", encoding="utf-8")
    (run_dir / "progress.jsonl").write_text(
        "".join(
            json.dumps(observation) + "\n"
            for observation in (
                {
                    "observed_at": "2026-08-28T21:38:39.500Z",
                    "status": "in_progress",
                },
                {
                    "observed_at": "2026-08-28T21:38:40.100Z",
                    "status": "completed",
                },
                {
                    "observed_at": "2026-08-28T21:38:41.900Z",
                    "status": "completed",
                },
            )
        ),
        encoding="utf-8",
    )
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
            "sample_started_at": "2026-08-28T21:38:42Z",
            "observed_at": "2026-08-28T21:38:42.100Z",
            "adapter_spec": 1,
            "ready_worker_pods": 1,
            "worker_ready_replicas": 1,
            "dgd_ready": "True",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","5","4","d-3"',
        },
        {
            "sample_started_at": "2026-08-28T21:38:43Z",
            "observed_at": "2026-08-28T21:38:43Z",
            "adapter_spec": 0,
            "ready_worker_pods": 0,
            "worker_ready_replicas": 0,
            "dgd_ready": "False",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","5","d-3"',
        },
    ]
    for state in states:
        state.setdefault("sample_started_at", state["observed_at"])
        state.setdefault("adapter_status", state["adapter_spec"])
        state.setdefault("worker_pod_count", state["ready_worker_pods"])
        state.setdefault("nonterminating_worker_pods", state["ready_worker_pods"])
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
    (evidence_dir / "redis.after.txt").write_text(
        "llm-d.ai/v1alpha1\ndynamo-batch\n0\n1\nd-3\n",
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
        adapter_status=1,
        worker_pod_count=1,
        nonterminating_worker_pods=1,
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
    assert result["assertions"]["gateway_completed_observation_present"] is True
    assert result["assertions"]["planner_logged_configured_idle_target"] is True
    assert (
        result["assertions"][
            "adapter_target_observed_after_completion_and_idle_decision"
        ]
        is True
    )
    assert (
        result["assertions"]["terminal_redis_decision_matches_post_completion_idle"]
        is True
    )
    assert (
        result["assertions"][
            "adapter_transition_observed_after_completion_and_idle_decision"
        ]
        is True
    )
    assert result["assertions"]["planner_logged_idle_scale"] is True
    assert result["assertions"]["planner_policy_log_order"] is True
    assert result["timeline"]["gateway_completed_first"] == (
        "2026-08-28T21:38:40.100000Z"
    )
    assert result["timeline"]["gateway_completed_last"] == (
        "2026-08-28T21:38:41.900000Z"
    )
    assert result["timeline"]["idle_decision"] == "2026-08-28T21:38:42.162000Z"
    assert result["observed"]["planner_selected_idle_decision_id"] == "d-3"
    assert result["observed"]["redis_terminal_decision_id"] == "d-3"
    assert result["observed"]["expected_idle_replicas"] == 0
    assert result["observed"]["planner_idle_replica_targets"] == [0, None, None, 0]
    assert result["timeline"]["positive_lease"] == "2026-08-28T21:38:17Z"
    assert result["timeline"]["last_unchanged_dispatch_counter"] == (
        "2026-08-28T21:38:17.100Z"
    )
    assert result["timeline"]["first_dispatch_counter_increase"] == (
        "2026-08-28T21:38:17.330Z"
    )
    assert (
        result["observed"]["dispatch_counter"]["continuous_through_first_increase"]
        is True
    )


def test_dispatch_before_positive_lease_is_not_certified(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    # The counter can increment after this unchanged scrape but before the
    # positive lease at 21:38:17, then remain unseen until the 21:38:17.330
    # scrape. That overlapping interval is intentionally inconclusive.
    _write_dispatch_sample(
        run_dir,
        "000-unchanged-after-positive-lease",
        400,
        "2026-08-28T21:38:16.900Z",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["assertions"]["dispatch_started_after_positive_lease"] is False
    assert result["all_passed"] is False
    assert result["timeline"]["last_unchanged_dispatch_counter"] == (
        "2026-08-28T21:38:16.900Z"
    )


def test_dispatch_counter_reset_is_not_certified(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    _write_dispatch_sample(
        run_dir,
        "001-counter-reset",
        0,
        "2026-08-28T21:38:17.200Z",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["assertions"]["dispatch_started_after_positive_lease"] is False
    assert result["all_passed"] is False
    assert (
        result["observed"]["dispatch_counter"]["continuous_through_first_increase"]
        is False
    )


def test_fresh_async_pod_treats_missing_before_counters_as_zero(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    _write_dispatch_sample(
        run_dir,
        "000-unchanged-after-positive-lease",
        0,
        "2026-08-28T21:38:17.100Z",
    )
    _write_dispatch_sample(
        run_dir,
        "sample",
        3,
        "2026-08-28T21:38:17.330Z",
    )
    (evidence_dir / "async-metrics.before.txt").write_text(
        'llm_d_async_async_broker_backlog{pool_name="dynamo-batch"} 0\n'
        'llm_d_async_async_broker_backlog_source_available{pool_name="dynamo-batch"} 1\n'
        'llm_d_async_async_pool_worker_limit{pool_name="dynamo-batch"} 8\n',
        encoding="utf-8",
    )
    metrics_after = evidence_dir / "async-metrics.after.txt"
    metrics_after.write_text(
        metrics_after.read_text(encoding="utf-8")
        .replace(
            f"{verify_native_planner_e2e.DISPATCHED} 500",
            f"{verify_native_planner_e2e.DISPATCHED} 100",
        )
        .replace(
            f"{verify_native_planner_e2e.SUCCESSFUL} 500",
            f"{verify_native_planner_e2e.SUCCESSFUL} 100",
        ),
        encoding="utf-8",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is True
    assert result["observed"]["dispatch_counter"]["before"] == 0
    assert result["observed"]["successful_counter"]["before"] == 0


def test_missing_after_counter_remains_an_error(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    metrics_after = evidence_dir / "async-metrics.after.txt"
    metrics_after.write_text(
        "\n".join(
            line
            for line in metrics_after.read_text(encoding="utf-8").splitlines()
            if not line.startswith(verify_native_planner_e2e.SUCCESSFUL)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="successful_requests_total"):
        verify_native_planner_e2e.verify(
            run_dir,
            evidence_dir,
            expected_lease_duration_seconds=60,
        )


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
    assert (
        result["assertions"][
            "adapter_target_observed_after_completion_and_idle_decision"
        ]
        is True
    )
    assert "planner_logged_idle_scale" not in result["assertions"]


def test_warm_one_watch_bounce_fails_scaling_noop_claim(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_warm_one_fixture(tmp_path)
    (evidence_dir / "dgdsa.watch.jsonstream").write_text(
        "".join(
            json.dumps({"object": {"spec": {"replicas": replicas}}}) + "\n"
            for replicas in (0, 1, 0, 1)
        ),
        encoding="utf-8",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=1,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["watch_remained_at_expected_idle_target"] is False


def test_warm_one_terminal_dgd_must_be_ready(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_warm_one_fixture(tmp_path)
    states = [
        json.loads(line)
        for line in (evidence_dir / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    states[-1]["dgd_ready"] = "False"
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )

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


def test_terminal_state_must_follow_last_gateway_completed_observation(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    (run_dir / "progress.jsonl").write_text(
        json.dumps(
            {
                "observed_at": "2026-08-28T21:38:42.500Z",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["planner_logged_configured_idle_target"] is True
    assert result["assertions"]["post_completion_active_barrier_observed"] is False
    assert result["timeline"]["idle_decision"] == "2026-08-28T21:38:42.162000Z"


def test_planner_clock_skew_does_not_change_verdict(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    planner_log_path = evidence_dir / "planner.log"
    planner_log_path.write_text(
        planner_log_path.read_text(encoding="utf-8").replace(
            "2026-08-28", "2099-01-01"
        ),
        encoding="utf-8",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is True
    assert result["observed"]["planner_selected_idle_decision_id"] == "d-3"


def test_missing_post_completion_active_barrier_fails(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    states = [
        json.loads(line)
        for line in (evidence_dir / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    states = [
        state
        for state in states
        if state["sample_started_at"] != "2026-08-28T21:38:42Z"
    ]
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["post_completion_active_barrier_observed"] is False


def test_same_sample_scale_readiness_and_positive_lease_are_valid(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    states = [
        json.loads(line)
        for line in (evidence_dir / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    states[1].update(
        adapter_spec=0,
        adapter_status=0,
        worker_pod_count=0,
        nonterminating_worker_pods=0,
        ready_worker_pods=0,
        worker_ready_replicas=0,
        dgd_ready="False",
    )
    states[2].update(
        lease_csv='"llm-d.ai/v1alpha1","dynamo-batch","5","3","d-2"',
    )
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is True
    assert result["assertions"]["scale_preceded_worker_readiness"] is True
    assert result["assertions"]["lease_remained_closed_until_ready"] is True


def test_early_idle_action_is_not_masked_by_later_valid_renewal(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    states = [
        json.loads(line)
        for line in (evidence_dir / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    states[4:4] = [
        {
            "sample_started_at": "2026-08-28T21:38:30Z",
            "observed_at": "2026-08-28T21:38:30Z",
            "adapter_spec": 0,
            "adapter_status": 0,
            "worker_pod_count": 0,
            "nonterminating_worker_pods": 0,
            "ready_worker_pods": 0,
            "worker_ready_replicas": 0,
            "dgd_ready": "False",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","5","d-early"',
        },
        {
            "sample_started_at": "2026-08-28T21:38:35Z",
            "observed_at": "2026-08-28T21:38:35Z",
            "adapter_spec": 1,
            "adapter_status": 1,
            "worker_pod_count": 1,
            "nonterminating_worker_pods": 1,
            "ready_worker_pods": 1,
            "worker_ready_replicas": 1,
            "dgd_ready": "True",
            "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","5","6","d-recover"',
        },
    ]
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )
    (evidence_dir / "dgdsa.watch.jsonstream").write_text(
        "".join(
            json.dumps({"object": {"spec": {"replicas": replicas}}}) + "\n"
            for replicas in (0, 1, 0, 1, 0)
        ),
        encoding="utf-8",
    )
    planner_log = (evidence_dir / "planner.log").read_text(encoding="utf-8")
    early_idle_log = (
        "2026-08-28T21:38:30.100Z INFO Batch scheduling decision: "
        "pipeline_action=apply pool_id=dynamo-batch replica_floor=None "
        "max_admission_rps=0.0 decision_id=d-early valid_until_s=25.0 "
        "idle_replica_target=0\n"
        "2026-08-28T21:38:30.200Z INFO Updating decode component "
        "VllmDecodeWorker from 1 to desired replica count 0\n"
        "2026-08-28T21:38:30.250Z INFO Scaled DGDSA "
        "qwen3-0-6b-batch-vllmdecodeworker to 0 replicas\n"
    )
    later_idle_offset = planner_log.index(
        "2026-08-28T21:38:42.162Z INFO Batch scheduling decision:"
    )
    (evidence_dir / "planner.log").write_text(
        planner_log[:later_idle_offset]
        + early_idle_log
        + planner_log[later_idle_offset:],
        encoding="utf-8",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["planner_logged_configured_idle_target"] is True
    assert (
        result["assertions"][
            "adapter_transition_observed_after_completion_and_idle_decision"
        ]
        is False
    )
    assert (
        result["assertions"]["terminal_redis_decision_matches_post_completion_idle"]
        is True
    )
    assert result["observed"]["planner_selected_idle_decision_id"] == "d-early"


def test_adapter_target_must_follow_completion_and_idle_decision(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    states = [
        json.loads(line)
        for line in (evidence_dir / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    # This target observation is after Gateway completion (21:38:41.900Z),
    # by its end timestamp, but its sample began before completion.
    states[-1]["sample_started_at"] = "2026-08-28T21:38:41.800Z"
    states[-1]["observed_at"] = "2026-08-28T21:38:42Z"
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["planner_logged_configured_idle_target"] is True
    assert (
        result["assertions"][
            "adapter_target_observed_after_completion_and_idle_decision"
        ]
        is False
    )
    assert (
        result["assertions"][
            "adapter_transition_observed_after_completion_and_idle_decision"
        ]
        is False
    )


def test_terminal_redis_decision_must_match_post_completion_idle_decision(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    (evidence_dir / "redis.after.txt").write_text(
        "llm-d.ai/v1alpha1\ndynamo-batch\n0\n1\nunrelated-decision\n",
        encoding="utf-8",
    )

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["planner_logged_configured_idle_target"] is True
    assert (
        result["assertions"]["terminal_redis_decision_matches_post_completion_idle"]
        is False
    )


def _append_zero_state_samples(
    evidence_dir: Path, timestamps: list[str]
) -> list[dict[str, object]]:
    state_path = evidence_dir / "state.jsonl"
    states = [json.loads(line) for line in state_path.read_text().splitlines()]
    zero_template = states[-1]
    for timestamp in timestamps:
        state = dict(zero_template)
        state["sample_started_at"] = timestamp
        state["observed_at"] = timestamp
        states.append(state)
    state_path.write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )
    return states


def _verify_with_stability(run_dir: Path, evidence_dir: Path) -> dict[str, object]:
    return verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
        expected_idle_stability_seconds=12,
    )


def test_valid_twelve_second_terminal_idle_suffix_passes(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    _append_zero_state_samples(
        evidence_dir,
        [
            "2026-08-28T21:38:47Z",
            "2026-08-28T21:38:51Z",
            "2026-08-28T21:38:55Z",
        ],
    )
    (evidence_dir / "T1.txt").write_text("2026-08-28T21:38:56Z\n", encoding="utf-8")

    result = _verify_with_stability(run_dir, evidence_dir)

    assert result["all_passed"] is True
    assert result["assertions"]["terminal_idle_stability_observed"] is True
    assert result["observed"]["idle_stability_observed_seconds"] == 12


def test_terminal_bounce_resets_idle_stability_suffix(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    states = _append_zero_state_samples(
        evidence_dir,
        ["2026-08-28T21:38:47Z", "2026-08-28T21:38:51Z"],
    )
    active = dict(states[-1])
    active.update(
        sample_started_at="2026-08-28T21:38:52Z",
        observed_at="2026-08-28T21:38:52Z",
        adapter_spec=1,
        adapter_status=1,
        worker_pod_count=1,
        nonterminating_worker_pods=1,
        ready_worker_pods=1,
        worker_ready_replicas=1,
        dgd_ready="True",
        lease_csv='"llm-d.ai/v1alpha1","dynamo-batch","5","6","d-bounce"',
    )
    states.append(active)
    zero_template = dict(states[0])
    zero_template.update(
        adapter_spec=0,
        adapter_status=0,
        worker_pod_count=0,
        nonterminating_worker_pods=0,
        ready_worker_pods=0,
        worker_ready_replicas=0,
        dgd_ready="False",
        lease_csv='"llm-d.ai/v1alpha1","dynamo-batch","0","7","d-3"',
    )
    for timestamp in (
        "2026-08-28T21:38:53Z",
        "2026-08-28T21:38:57Z",
        "2026-08-28T21:39:01Z",
    ):
        state = dict(zero_template)
        state.update(sample_started_at=timestamp, observed_at=timestamp)
        states.append(state)
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )
    (evidence_dir / "T1.txt").write_text("2026-08-28T21:39:02Z\n", encoding="utf-8")

    result = _verify_with_stability(run_dir, evidence_dir)

    assert result["all_passed"] is False
    assert result["assertions"]["terminal_idle_stability_observed"] is False
    assert result["observed"]["idle_stability_observed_seconds"] == 8


def test_stale_terminal_idle_sample_fails(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    _append_zero_state_samples(
        evidence_dir,
        [
            "2026-08-28T21:38:47Z",
            "2026-08-28T21:38:51Z",
            "2026-08-28T21:38:55Z",
        ],
    )
    (evidence_dir / "T1.txt").write_text("2026-08-28T21:39:01Z\n", encoding="utf-8")

    result = _verify_with_stability(run_dir, evidence_dir)

    assert result["all_passed"] is False
    assert result["assertions"]["terminal_idle_stability_observed"] is False


def test_terminal_idle_suffix_rejects_sample_gap_over_five_seconds(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    _append_zero_state_samples(
        evidence_dir,
        [
            "2026-08-28T21:38:49Z",
            "2026-08-28T21:38:53Z",
            "2026-08-28T21:38:57Z",
        ],
    )
    (evidence_dir / "T1.txt").write_text("2026-08-28T21:38:58Z\n", encoding="utf-8")

    result = _verify_with_stability(run_dir, evidence_dir)

    assert result["all_passed"] is False
    assert result["assertions"]["terminal_idle_stability_observed"] is False


def test_terminal_idle_suffix_rejects_sample_span_over_five_seconds(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    states = _append_zero_state_samples(
        evidence_dir,
        [
            "2026-08-28T21:38:47Z",
            "2026-08-28T21:38:51Z",
            "2026-08-28T21:38:55Z",
        ],
    )
    states[-1]["observed_at"] = "2026-08-28T21:39:01Z"
    (evidence_dir / "state.jsonl").write_text(
        "".join(json.dumps(state) + "\n" for state in states), encoding="utf-8"
    )
    (evidence_dir / "T1.txt").write_text("2026-08-28T21:39:02Z\n", encoding="utf-8")

    result = _verify_with_stability(run_dir, evidence_dir)

    assert result["all_passed"] is False
    assert result["assertions"]["terminal_idle_stability_observed"] is False


def test_invalid_capture_integrity_marker_fails_strict_mode(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    (evidence_dir / "capture-valid.txt").write_text("0\n", encoding="utf-8")

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["capture_integrity_valid"] is False


def test_present_invalid_capture_integrity_marker_fails_legacy_mode(
    tmp_path: Path,
) -> None:
    run_dir, evidence_dir = _write_missing_transitions_fixture(tmp_path)
    (evidence_dir / "capture-valid.txt").write_text("0\n", encoding="utf-8")

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["capture_integrity_valid"] is False


def test_missing_capture_integrity_marker_fails_strict_mode(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    (evidence_dir / "capture-valid.txt").unlink()

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["capture_integrity_valid"] is False


def test_watch_bounce_after_terminal_idle_transition_fails(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)
    watch_path = evidence_dir / "dgdsa.watch.jsonstream"
    with watch_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"object": {"spec": {"replicas": 1}}}) + "\n")

    result = verify_native_planner_e2e.verify(
        run_dir,
        evidence_dir,
        worker_component="VllmDecodeWorker",
        adapter_name="qwen3-0-6b-batch-vllmdecodeworker",
        expected_idle_replicas=0,
        expected_lease_duration_seconds=60,
    )

    assert result["all_passed"] is False
    assert result["assertions"]["watch_remained_at_expected_idle_target"] is False


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


@pytest.mark.parametrize("duration", [-1, True, float("inf"), float("nan")])
def test_expected_idle_stability_must_be_finite_and_non_negative(
    tmp_path: Path, duration: float
) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)

    with pytest.raises(ValueError, match="expected_idle_stability_seconds"):
        verify_native_planner_e2e.verify(
            run_dir,
            evidence_dir,
            expected_idle_replicas=0,
            expected_lease_duration_seconds=60,
            expected_idle_stability_seconds=duration,
        )


def test_idle_stability_requires_an_expected_idle_target(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_scale_to_zero_fixture(tmp_path)

    with pytest.raises(ValueError, match="expected_idle_replicas"):
        verify_native_planner_e2e.verify(
            run_dir,
            evidence_dir,
            expected_lease_duration_seconds=60,
            expected_idle_stability_seconds=12,
        )
