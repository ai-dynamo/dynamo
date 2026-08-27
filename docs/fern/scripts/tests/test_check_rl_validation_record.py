# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RL runtime validation-record publication gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import check_rl_validation_record
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

TEMPLATE = Path(__file__).resolve().parents[1] / "rl_validation_record.template.json"


def _template() -> dict:
    return check_rl_validation_record.load_record(TEMPLATE)


def _passing_record() -> dict:
    record = _template()
    record["record_state"] = "passed"
    record["record_id"] = "verl-dynamo-2026-08-27-run-1"
    record["framework"].update(
        {
            "integration_artifact": "https://github.com/example/recipe/tree/"
            + "a" * 40,
            "recipe_commit": "a" * 40,
            "core_commit": "b" * 40,
        }
    )
    record["environment"].update(
        {
            "dynamo_commit": "c" * 40,
            "container_image": "registry.example/dynamo:v1",
            "container_image_digest": "sha256:" + "d" * 64,
            "cuda_version": "13.0",
            "driver_version": "590.00",
        }
    )
    record["environment"]["backend"]["version"] = "0.27.1"
    record["environment"]["artifacts"] = [
        "artifact://verl-dynamo-run-1/environment-preflight.json"
    ]
    record["environment"]["model"].update(
        {
            "name": "example/model",
            "revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
        }
    )
    record["hardware"].update(
        {
            "nodes": 1,
            "gpu_model": "H100 SXM",
            "gpus_per_node": 8,
            "interconnect": "NVLink",
            "network": "InfiniBand",
            "artifacts": [
                "artifact://verl-dynamo-run-1/environment-preflight.json",
                "artifact://verl-dynamo-run-1/scheduler-allocation.json",
            ],
        }
    )
    for group in ("trainer_parallelism", "rollout_parallelism"):
        record["topology"][group].update({"tp": 1, "pp": 1, "dp": 1, "ep": 1})
    record["owners"].update(
        {
            "framework": "framework-owner",
            "dynamo": "dynamo-owner",
            "clean_room_reviewer": "independent-reviewer",
        }
    )
    record["run"].update(
        {
            "started_at": "2026-08-27T10:00:00Z",
            "completed_at": "2026-08-27T11:00:00Z",
            "commands": ["python3 -m recipe.dynamo.main_dynamo ..."],
            "artifact_root": "artifact://verl-dynamo-run-1/",
        }
    )
    for gate_name, gate in record["gates"].items():
        gate["status"] = "passed"
        gate["artifacts"] = [f"artifact://verl-dynamo-run-1/{gate_name}.json"]
    record["gates"]["token_logprob"].update(
        {
            "exact_completion_token_ids": True,
            "completion_logprobs_aligned": True,
            "prompt_logprobs": "verified",
            "terminal_reasons_verified": True,
        }
    )
    record["gates"]["training_iteration"].update(
        {
            "optimizer_steps": 1,
            "rollout_phase_completed": True,
            "reward_or_advantage_completed": True,
            "actor_update_completed": True,
            "weight_sync_completed": True,
            "post_update_rollout_completed": True,
        }
    )
    record["gates"]["policy_update"].update(
        {
            "target_version": "trainer-step-1",
            "workers_targeted": 2,
            "workers_verified": 2,
            "cache_handling": "invalidated",
            "post_update_generation": True,
        }
    )
    record["gates"]["retry_and_cancellation"].update(
        {
            "duplicate_suppression_verified": True,
            "canceled_incomplete_sample_verified": True,
        }
    )
    record["gates"]["failure_recovery"].update(
        {
            "request_failure_recovered": True,
            "worker_failure_recovered": True,
            "weight_update_failure_recovered": True,
        }
    )
    record["gates"]["trace_correlation"].update(
        {
            "framework_attempts": 10,
            "joined_payloads": 10,
            "expected_terminals": 9,
            "joined_terminals": 9,
            "trace_overhead_percent": 1.5,
        }
    )
    record["last_validated"] = "2026-08-27T11:30:00Z"
    return record


def test_planned_template_has_valid_structure_but_cannot_publish() -> None:
    record = _template()
    assert check_rl_validation_record.validate_structure(record) == []
    findings = check_rl_validation_record.publication_findings(record)
    assert "record_state must be passed for publication" in findings
    assert "gates.training_iteration.status must be passed" in findings
    assert any("dynamo_commit" in finding for finding in findings)


def test_complete_record_passes_the_publication_gate() -> None:
    assert check_rl_validation_record.publication_findings(_passing_record()) == []


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            lambda record: record["gates"]["policy_update"].update(
                {"workers_verified": 1}
            ),
            "workers_verified must equal workers_targeted",
        ),
        (
            lambda record: record["gates"]["trace_correlation"].update(
                {"joined_payloads": 9}
            ),
            "joined_payloads must equal framework_attempts",
        ),
        (
            lambda record: record["owners"].update(
                {"clean_room_reviewer": "dynamo-owner"}
            ),
            "clean_room_reviewer must be independent",
        ),
        (
            lambda record: record["gates"]["failure_recovery"].update(
                {"weight_update_failure_recovered": False}
            ),
            "weight_update_failure_recovered must be true",
        ),
        (
            lambda record: record["run"].update({"started_at": "2026-08-27T10:00:00"}),
            "run.started_at must be ISO-8601 with a UTC offset",
        ),
        (
            lambda record: record["environment"].update({"artifacts": []}),
            "environment.artifacts must contain at least one artifact URI",
        ),
        (
            lambda record: record["hardware"].update({"artifacts": []}),
            "hardware.artifacts must contain at least one artifact URI",
        ),
    ],
)
def test_publication_gate_rejects_incomplete_evidence(mutation, expected: str) -> None:
    record = copy.deepcopy(_passing_record())
    mutation(record)
    findings = check_rl_validation_record.publication_findings(record)
    assert any(expected in finding for finding in findings)


def test_cli_distinguishes_structure_from_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    planned = tmp_path / "planned.json"
    planned.write_text(json.dumps(_template()), encoding="utf-8")
    assert check_rl_validation_record.main([str(planned)]) == 0
    assert check_rl_validation_record.main([str(planned), "--publication-gate"]) == 1
    captured = capsys.readouterr()
    assert "RL validation record passed (structure" in captured.out
    assert "record_state must be passed for publication" in captured.err
