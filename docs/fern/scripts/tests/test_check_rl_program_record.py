# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cross-cutting RL program-evidence publication gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import check_rl_program_record
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

TEMPLATE = Path(__file__).resolve().parents[1] / "rl_program_record.template.json"


def _template() -> dict:
    return check_rl_program_record.load_record(TEMPLATE)


def _routing_variant(name: str, baseline: bool, hit_rate: float) -> dict:
    return {
        "name": name,
        "baseline": baseline,
        "router_config": {
            "router_mode": "round-robin" if baseline else "kv-aware",
            "router_temperature": 0.0,
        },
        "repetitions": 3,
        "metrics": {
            "prefix_cache_hit_rate": hit_rate,
            "p99_inter_token_latency_ms": 19.0,
        },
        "artifacts": [f"artifact://rl-program/routing/{name}.json"],
    }


def _weight_path(name: str, placement: str, serving_mode: str) -> dict:
    return {
        "name": name,
        "status": "passed",
        "placement": placement,
        "serving_mode": serving_mode,
        "framework_name": "verl",
        "framework_commit": "d" * 40,
        "backend": "vllm",
        "backend_version": "0.10.0",
        "container_image": "registry.example/dynamo-rl:v1",
        "container_image_digest": "sha256:" + "e" * 64,
        "model_name": "example/model",
        "model_revision": "model-revision",
        "transport": "nixl",
        "model_class": "llama-causal-lm",
        "source_parallelism": {"tp": 1, "pp": 1, "dp": 1, "ep": 1},
        "target_parallelism": {"tp": 2, "pp": 1, "dp": 1, "ep": 1},
        "workers_targeted": 2,
        "workers_verified": 2,
        "cache_handling": "invalidated",
        "version_verified": True,
        "output_mutation_or_numerical_validation": True,
        "partial_failure_recovered": True,
        "post_update_generation": True,
        "artifacts": [f"artifact://rl-program/weights/{name}.json"],
    }


def _passing_record() -> dict:
    record = _template()
    record.update(
        {
            "record_state": "passed",
            "record_id": "rl-program-2026-08-27-run-1",
            "last_validated": "2026-08-27T14:30:00Z",
        }
    )
    record["pins"].update(
        {
            "dynamo_commit": "a" * 40,
            "framework_name": "verl",
            "framework_commit": "b" * 40,
            "backend_name": "vllm",
            "backend_version": "0.10.0",
            "container_image": "registry.example/dynamo-rl:v1",
            "container_image_digest": "sha256:" + "c" * 64,
            "model_name": "example/model",
            "model_revision": "model-revision",
        }
    )
    record["owners"].update(
        {
            "routing": "routing-owner",
            "weight_updates": "weight-owner",
            "observability": "observability-owner",
            "replay_simulation": "simulation-owner",
            "clean_room_reviewer": "independent-reviewer",
        }
    )
    record["run_window"].update(
        {
            "started_at": "2026-08-27T12:00:00Z",
            "completed_at": "2026-08-27T14:00:00Z",
            "artifact_root": "artifact://rl-program/",
        }
    )
    record["routing"].update(
        {
            "status": "passed",
            "fixed_controls": [
                "Dynamo and framework commits",
                "model and tokenizer revision",
                "hardware and serving topology",
            ],
            "variants": [
                _routing_variant("round-robin", True, 0.31),
                _routing_variant("kv-aware", False, 0.72),
            ],
            "mechanism_evidence": "artifact://rl-program/routing/mechanism.json",
            "claim_boundary": "live_measurement",
        }
    )
    record["routing"]["headline_metric"].update(
        {
            "name": "prefix_cache_hit_rate",
            "numerator": "cached prefix tokens",
            "denominator": "eligible prefix tokens",
            "freshness_rule": "ignore samples generated before the measured policy version",
        }
    )
    record["routing"]["workload"].update(
        {
            "name": "grouped-policy-sampling",
            "request_count": 96,
            "unique_prompts": 24,
            "samples_per_prompt": 4,
            "schedule": "four samples for each prompt submitted as one group",
            "prompt_length_distribution": "p50=512, p95=1024 tokens",
            "output_length_distribution": "p50=128, p95=256 tokens",
            "prefix_sharing_shape": "four samples share each complete prompt prefix",
            "session_shape": "stateless single turn",
            "concurrency": 32,
            "seed": 7,
        }
    )
    record["weight_paths"].update(
        {
            "status": "passed",
            "paths": [
                _weight_path("colocated-aggregated", "colocated", "aggregated"),
                _weight_path("external-disaggregated", "external", "disaggregated"),
            ],
        }
    )
    record["observability"].update(
        {
            "status": "passed",
            "clock_synchronization": "chrony offset below 1 ms on all nodes",
        }
    )
    record["observability"]["trace_overhead"].update(
        {
            "baseline_repetitions": 3,
            "traced_repetitions": 3,
            "percent": 1.7,
            "artifact": "artifact://rl-program/observability/overhead.json",
        }
    )
    for name, diagnosis in record["observability"]["diagnoses"].items():
        diagnosis.update(
            {
                "status": "passed",
                "conclusion": f"controlled {name} condition was localized",
                "artifacts": [f"artifact://rl-program/observability/{name}.json"],
            }
        )
    record["replay_simulation"].update({"status": "passed"})
    record["replay_simulation"]["capture"].update(
        {
            "framework_attempts": 96,
            "expected_replay_requests": 92,
            "trace_requests": 92,
            "input_tokens": 49152,
            "output_tokens": 11776,
            "sessions": 0,
            "trace_block_size": 16,
            "artifact": "artifact://rl-program/replay/capture-summary.json",
        }
    )
    record["replay_simulation"]["live_replay"].update(
        {
            "status": "passed",
            "repetitions": 3,
            "artifact": "artifact://rl-program/replay/live-summary.json",
        }
    )
    record["replay_simulation"]["dynosim"].update(
        {
            "status": "passed",
            "repetitions": 3,
            "artifact": "artifact://rl-program/replay/dynosim-summary.json",
        }
    )
    record["replay_simulation"]["calibration"].update(
        {
            "metrics": [
                {
                    "name": "p99_inter_token_latency_ms",
                    "unit": "ms",
                    "live": 20.0,
                    "simulated": 21.0,
                    "absolute_error": 1.0,
                    "relative_error_percent": 5.0,
                }
            ],
            "material_error_threshold_percent": 10.0,
            "material_error_disclosure": "No metric crossed the prespecified threshold.",
            "conclusion": "DynoSim reproduced the selected latency metric within threshold.",
            "artifact": "artifact://rl-program/replay/calibration.json",
        }
    )
    return record


def test_planned_template_has_valid_structure_but_cannot_publish() -> None:
    record = _template()
    assert check_rl_program_record.validate_structure(record) == []
    findings = check_rl_program_record.publication_findings(record)
    assert "record_state must be passed for publication" in findings
    assert "routing.status must be passed" in findings
    assert "weight_paths.paths[0].status must be passed" in findings


def test_complete_record_passes_the_publication_gate() -> None:
    assert check_rl_program_record.publication_findings(_passing_record()) == []


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            lambda record: record["routing"].update(
                {"variants": record["routing"]["variants"][:1]}
            ),
            "baseline and at least one variant",
        ),
        (
            lambda record: record["routing"]["variants"][1].update({"repetitions": 2}),
            "repetitions must be at least 3",
        ),
        (
            lambda record: record["routing"]["variants"][1]["metrics"].pop(
                "prefix_cache_hit_rate"
            ),
            "metrics must contain numeric headline metric",
        ),
        (
            lambda record: record["weight_paths"].update(
                {
                    "paths": [
                        _weight_path("colocated-one", "colocated", "aggregated"),
                        _weight_path("colocated-two", "colocated", "aggregated"),
                    ]
                }
            ),
            "must include a disaggregated serving path",
        ),
        (
            lambda record: record["weight_paths"].update(
                {
                    "paths": [
                        _weight_path("hybrid", "colocated", "disaggregated"),
                        _weight_path("external", "external", "aggregated"),
                    ]
                }
            ),
            "must use distinct colocated and disaggregated paths",
        ),
        (
            lambda record: record["weight_paths"]["paths"][0].update(
                {"framework_commit": "unpinned"}
            ),
            "framework_commit must be a full lowercase commit SHA",
        ),
        (
            lambda record: record["weight_paths"]["paths"][0].update(
                {"workers_verified": 1}
            ),
            "workers_verified must equal workers_targeted",
        ),
        (
            lambda record: record["weight_paths"]["paths"][0].update(
                {"partial_failure_recovered": False}
            ),
            "partial_failure_recovered must be true",
        ),
        (
            lambda record: record["observability"]["diagnoses"][
                "blocked_or_failed_weight_refresh"
            ].update({"status": "failed"}),
            "blocked_or_failed_weight_refresh.status must be passed",
        ),
        (
            lambda record: record["observability"]["trace_overhead"].update(
                {"traced_repetitions": 2}
            ),
            "traced_repetitions must be at least 3",
        ),
        (
            lambda record: record["replay_simulation"]["capture"].update(
                {"trace_requests": 91}
            ),
            "trace_requests must equal expected_replay_requests",
        ),
        (
            lambda record: record["replay_simulation"]["dynosim"].update(
                {"repetitions": 2}
            ),
            "dynosim.repetitions must be at least 3",
        ),
        (
            lambda record: record["replay_simulation"]["calibration"]["metrics"][
                0
            ].update({"absolute_error": 2.0}),
            "absolute_error does not match live versus simulated",
        ),
        (
            lambda record: record["owners"].update(
                {"clean_room_reviewer": "routing-owner"}
            ),
            "clean_room_reviewer must be independent",
        ),
    ],
)
def test_publication_gate_rejects_weak_evidence(mutation, expected: str) -> None:
    record = copy.deepcopy(_passing_record())
    mutation(record)
    findings = check_rl_program_record.publication_findings(record)
    assert any(expected in finding for finding in findings)


def test_cli_distinguishes_structure_from_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    planned = tmp_path / "planned.json"
    planned.write_text(json.dumps(_template()), encoding="utf-8")
    assert check_rl_program_record.main([str(planned)]) == 0
    assert check_rl_program_record.main([str(planned), "--publication-gate"]) == 1
    captured = capsys.readouterr()
    assert "RL program evidence passed (structure" in captured.out
    assert "record_state must be passed for publication" in captured.err
