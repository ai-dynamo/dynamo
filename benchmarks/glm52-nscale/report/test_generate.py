#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import subprocess
from pathlib import Path

from generate import (
    DEFAULT_INPUT,
    SchemaError,
    assert_pinned_report_sources,
    render,
    validate,
    validate_sidecars,
)


def load_summary() -> dict:
    data = json.loads(Path(DEFAULT_INPUT).read_text())
    data["campaign"]["source_commit"] = "1" * 40
    data["results"] = []
    data["paired_disagreements"] = []
    return data


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def captured_at(variant: str, phase: str, suite_id: str) -> str:
    suite_rank = {
        "bfcl-v4": 0,
        "swebench-verified": 1,
        "swebench-pro": 2,
        "swebench-multilingual": 3,
        "terminal-bench-2.1": 4,
    }[suite_id]
    if "vllm" in variant:
        order = {
            ("ab", True): 0,
            ("ab", False): 1,
            ("ba", False): 2,
            ("ba", True): 3,
        }[(phase, variant.startswith("dynamo-"))]
    else:
        order = (
            4
            + {
                ("ab", True): 0,
                ("ab", False): 1,
                ("ba", False): 2,
                ("ba", True): 3,
            }[(phase, variant.startswith("dynamo-"))]
        )
    return f"2026-07-05T03:{suite_rank * 10 + order:02d}:00Z"


def campaign_source() -> dict:
    return {
        "schema_version": 1,
        "source_commit": "1" * 40,
        "source_clean": True,
        "source_changed_path_count": 0,
        "bundle_sha256": "9" * 64,
        "source_tree_sha256": "a" * 64,
        "eval_tree_sha256": "b" * 64,
        "campaign_env_sha256": "c" * 64,
        "source_file_count": 101,
        "eval_file_count": 100,
    }


def harbor_environment() -> dict:
    packages = [["harbor", "1.0.0"], ["litellm", "1.80.0"]]
    payload = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    return {
        "uv_sync_check": "passed",
        "python": "3.12.11",
        "package_count": len(packages),
        "packages_sha256": hashlib.sha256(payload).hexdigest(),
        "packages": packages,
    }


def terminal_task_images() -> dict:
    return {
        "task_count": 89,
        "trial_count": 445,
        "task_image_map_sha256": digest("terminal-bench-2.1:task-image-map"),
    }


def runtime_identity(variant: str, phase: str, suite_id: str) -> dict:
    dynamo = variant.startswith("dynamo-")
    roles = ("frontend", "worker") if dynamo else ("worker",)
    return {
        "deployment_sha256": digest(f"{variant}:{suite_id}:{phase}:deployment"),
        "content_sha256": digest(f"{variant}:{phase}:{suite_id}:content"),
        "captured_at": captured_at(variant, phase, suite_id),
        "controller_uid_sha256": digest(f"{variant}:{suite_id}:{phase}:controller"),
        "pod_uid_sha256_by_role": {
            role: digest(f"{variant}:{suite_id}:{phase}:{role}") for role in roles
        },
        "capture_sha256": digest(f"{variant}:{suite_id}:{phase}:capture"),
        "recipe": {
            "source_commit": "1" * 40,
            "template_sha256": digest(f"{variant}:template"),
            "rendered_manifest_sha256": digest(f"{variant}:manifest"),
        },
        "hardware": {
            "gpu_count": 4,
            "gpu_model": "NVIDIA B200",
            "gpu_uuid_set_sha256": "2" * 64,
            "driver_version": "595.58.03",
            "gpu_memory_total_mib": [183359],
            "kernel_version": "6.8.0",
            "kubelet_version": "v1.33.1",
            "container_runtime_version": "containerd://2.1.4",
        },
        "control_plane": (
            {
                "dynamo_operator_image_digests": ["sha256:" + "3" * 64],
                "grove_operator_image_digests": ["sha256:" + "4" * 64],
            }
            if dynamo
            else None
        ),
    }


def complete_result(
    data: dict,
    *,
    variant: str,
    suite_id: str,
    metrics: dict[str, int | float],
    phase: str = "ab",
) -> dict:
    suite = next(suite for suite in data["suites"] if suite["id"] == suite_id)
    row = {
        "variant": variant,
        "suite": suite_id,
        "phase": phase,
        "run_type": "full",
        "status": "complete",
        "completeness": {
            "generated_units": suite.get("generation_units", suite["units"]),
            "evaluated_units": suite["units"],
            "completed_trials": suite["units"] * suite["attempts"],
        },
        "metrics": metrics,
        "evidence": {
            "importer": "glm52-result-import/v1",
            "sources": [
                {
                    "role": "test-summary",
                    "path": f"artifact://{suite_id}/{phase}/{variant}/test-summary/summary.json",
                    "sha256": "0" * 64,
                }
            ],
        },
        "runtime_identity": runtime_identity(variant, phase, suite_id),
        "campaign_source": campaign_source(),
        "task_level": {
            "path": f"results/task-level/{suite_id}/{phase}/{variant}.jsonl",
            "sha256": digest(f"{variant}:{suite_id}:{phase}:tasks"),
            "records": (
                suite["units"] * suite["attempts"]
                if suite["kind"] == "terminalbench"
                else suite["units"]
            ),
        },
        "wall_time_seconds": 123.4,
    }
    if suite["kind"] == "bfcl":
        row["suite_identity"] = {
            "python_environment": {
                "constraints_sha256": "7" * 64,
                "freeze_sha256": "8" * 64,
                "package_count": 141,
                "python": "3.12.11",
                "schema_version": 1,
            },
            "campaign_source": campaign_source(),
        }
    elif suite["kind"] == "swebench":
        row["suite_identity"] = {
            "python_environment": {"lock": "same"},
            "effective_config_file_sha256": digest(f"{variant}:config-file"),
            "effective_config_content_sha256": digest(f"{variant}:config-content"),
            "fairness_config_sha256": digest(f"{suite_id}:fairness"),
            "task_image_evidence_sha256": digest(f"{suite_id}:task-evidence"),
            "task_image_map_sha256": digest(f"{suite_id}:task-map"),
            "generation": {"workers": 16, "batch_size": 8},
            "evaluation": {"workers": 8, "timeout_seconds": 3600},
            "runtime_source_revision": ("5" * 40 if "vllm" in variant else "6" * 40),
            "runtime_family": "vllm" if "vllm" in variant else "sglang",
            "runtime_deployment_sha256": digest(
                f"{variant}:{suite_id}:{phase}:deployment"
            ),
            "runtime_content_sha256": digest(
                f"{variant}:{suite_id}:{phase}:runtime-content"
            ),
        }
    elif suite["kind"] == "terminalbench":
        row["suite_identity"] = {
            "harbor_environment": harbor_environment(),
            "task_images": terminal_task_images(),
        }
    return row


def valid_metrics(data: dict, suite_id: str) -> dict[str, int | float]:
    suite = next(suite for suite in data["suites"] if suite["id"] == suite_id)
    if suite["kind"] == "bfcl":
        return {
            "overall_accuracy": 0.72,
            "correct_cases": 3676,
            "failed_cases": 1430,
            "inference_errors": 0,
        }
    if suite["kind"] == "swebench":
        passed = suite["units"] // 2
        score = passed / suite["units"]
        return {
            "benchmark_score": score,
            "score_on_submitted": score,
            "passed_instances": passed,
            "failed_instances": suite["units"] - passed,
            "missing_instances": 0,
        }
    if suite["kind"] == "terminalbench":
        completed_trials = suite["units"] * suite["attempts"]
        passed = suite["units"]
        return {
            "pass_at_1": passed / completed_trials,
            "pass_at_2": 0.30,
            "pass_at_3": 0.40,
            "pass_at_4": 0.50,
            "pass_at_5": 0.60,
            "passed_attempts": passed,
            "failed_attempts": completed_trials - passed,
            "errored_attempts": 0,
            "no_reward_attempts": 0,
        }
    raise AssertionError(f"unknown suite kind: {suite['kind']}")


def mark_campaign_complete(data: dict) -> None:
    data["campaign"]["status"] = "complete"
    data["campaign"]["completed_at"] = "2026-07-06T03:00:00Z"
    for variant in data["variants"]:
        variant["status"] = "complete"
    data["results"] = [
        complete_result(
            data,
            variant=variant["id"],
            suite_id=suite["id"],
            metrics=valid_metrics(data, suite["id"]),
            phase=phase,
        )
        for variant in data["variants"]
        for suite in data["suites"]
        for phase in ("ab", "ba")
    ]
    data["paired_disagreements"] = [
        {
            "suite": suite["id"],
            "phase": phase,
            "pair": pair["id"],
            "path": (
                f"results/paired-disagreements/{suite['id']}/{phase}/{pair['id']}.jsonl"
            ),
            "sha256": digest(f"{suite['id']}:{phase}:{pair['id']}:disagreements"),
            "compared_records": (
                suite["units"] * suite["attempts"]
                if suite["kind"] == "terminalbench"
                else suite["units"]
            ),
            "disagreement_records": 0,
        }
        for suite in data["suites"]
        for phase in ("ab", "ba")
        for pair in data["pairs"]
    ]


def canonical_jsonl(records: list[dict]) -> bytes:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode()


def paired_sidecar_fixture(root: Path) -> dict:
    data = load_summary()
    suite = next(suite for suite in data["suites"] if suite["id"] == "bfcl-v4")
    suite["units"] = 2
    suite["generation_units"] = 2
    dynamo = complete_result(
        data,
        variant="dynamo-vllm",
        suite_id="bfcl-v4",
        phase="ab",
        metrics={
            "overall_accuracy": 0.5,
            "correct_cases": 1,
            "failed_cases": 1,
            "inference_errors": 0,
        },
    )
    native = complete_result(
        data,
        variant="vllm-serve",
        suite_id="bfcl-v4",
        phase="ab",
        metrics={
            "overall_accuracy": 0.0,
            "correct_cases": 0,
            "failed_cases": 2,
            "inference_errors": 0,
        },
    )
    task_records = {
        "dynamo-vllm": [
            {"id": "case-a", "category": "simple", "outcome": "passed"},
            {
                "id": "case-b",
                "category": "simple",
                "error_types": ["incorrect"],
                "outcome": "failed",
            },
        ],
        "vllm-serve": [
            {
                "id": "case-a",
                "category": "simple",
                "error_types": ["incorrect"],
                "outcome": "failed",
            },
            {
                "id": "case-b",
                "category": "simple",
                "error_types": ["incorrect"],
                "outcome": "failed",
            },
        ],
    }
    for row in (dynamo, native):
        payload = canonical_jsonl(task_records[row["variant"]])
        row["task_level"]["sha256"] = hashlib.sha256(payload).hexdigest()
        row["task_level"]["records"] = 2
        path = root.joinpath(*Path(row["task_level"]["path"]).parts[1:])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    disagreement_records = [
        {
            "dynamo_outcome": "passed",
            "id": "case-a",
            "native_outcome": "failed",
        }
    ]
    disagreement_payload = canonical_jsonl(disagreement_records)
    disagreement_path = "results/paired-disagreements/bfcl-v4/ab/vllm.jsonl"
    physical = root.joinpath(*Path(disagreement_path).parts[1:])
    physical.parent.mkdir(parents=True, exist_ok=True)
    physical.write_bytes(disagreement_payload)
    data["results"] = [dynamo, native]
    data["paired_disagreements"] = [
        {
            "suite": "bfcl-v4",
            "phase": "ab",
            "pair": "vllm",
            "path": disagreement_path,
            "sha256": hashlib.sha256(disagreement_payload).hexdigest(),
            "compared_records": 2,
            "disagreement_records": 1,
        }
    ]
    validate(data)
    return data


class SummarySchemaTests(unittest.TestCase):
    def test_in_progress_campaign_is_valid(self) -> None:
        data = load_summary()
        validate(data)
        report = render(data)
        self.assertIn("0/40 full stack/suite results are complete", report)
        self.assertIn("Dynamo versus native deltas", report)
        self.assertIn("pass@5", report)

    def test_duplicate_variant_suite_result_is_rejected(self) -> None:
        data = load_summary()
        row = {
            "variant": "dynamo-vllm",
            "suite": "bfcl-v4",
            "phase": "ab",
            "status": "running",
            "completeness": {
                "generated_units": 1,
                "evaluated_units": 0,
                "completed_trials": 1,
            },
            "metrics": {},
        }
        data["results"] = [row, copy.deepcopy(row)]
        with self.assertRaisesRegex(SchemaError, "duplicate result row"):
            validate(data)

    def test_pairs_must_cover_every_variant(self) -> None:
        data = load_summary()
        data["pairs"].pop()
        with self.assertRaisesRegex(SchemaError, "pairs must cover every variant"):
            validate(data)

    def test_campaign_status_and_timestamps_are_validated(self) -> None:
        data = load_summary()
        data["campaign"]["status"] = "done"
        with self.assertRaisesRegex(SchemaError, "campaign.status must be one of"):
            validate(data)

        data = load_summary()
        data["campaign"]["started_at"] = "2026-07-05"
        with self.assertRaisesRegex(SchemaError, "must include a UTC offset"):
            validate(data)

        data = load_summary()
        data["campaign"]["completed_at"] = "2026-07-04T03:00:00Z"
        with self.assertRaisesRegex(SchemaError, "must not set campaign.completed_at"):
            validate(data)

        data = load_summary()
        data["campaign"]["status"] = "failed"
        data["campaign"]["completed_at"] = "2026-07-04T03:00:00Z"
        with self.assertRaisesRegex(
            SchemaError, "must not precede campaign.started_at"
        ):
            validate(data)

        data = load_summary()
        data["variants"][0]["status"] = "done"
        with self.assertRaisesRegex(SchemaError, r"variants\['dynamo-vllm'\]\.status"):
            validate(data)

    def test_campaign_model_identity_and_contexts_are_validated(self) -> None:
        data = load_summary()
        data["campaign"]["model_revision"] = "latest"
        with self.assertRaisesRegex(SchemaError, "model_revision"):
            validate(data)

        data = load_summary()
        data["campaign"]["serving_context_tokens"] = 0
        with self.assertRaisesRegex(SchemaError, "serving_context_tokens"):
            validate(data)

        data = load_summary()
        data["campaign"]["terminalbench_context_tokens"] = True
        with self.assertRaisesRegex(SchemaError, "terminalbench_context_tokens"):
            validate(data)

    def test_complete_campaign_requires_exact_complete_matrix(self) -> None:
        data = load_summary()
        mark_campaign_complete(data)
        validate(data)
        self.assertIn("40/40 full stack/suite results are complete", render(data))

        missing_timestamp = copy.deepcopy(data)
        missing_timestamp["campaign"]["completed_at"] = None
        with self.assertRaisesRegex(SchemaError, "requires campaign.completed_at"):
            validate(missing_timestamp)

        missing_result = copy.deepcopy(data)
        removed = missing_result["results"].pop()
        pair_id = next(
            pair["id"]
            for pair in missing_result["pairs"]
            if removed["variant"] in {pair["dynamo_variant"], pair["native_variant"]}
        )
        missing_result["paired_disagreements"] = [
            entry
            for entry in missing_result["paired_disagreements"]
            if (entry["suite"], entry["phase"], entry["pair"])
            != (removed["suite"], removed["phase"], pair_id)
        ]
        with self.assertRaisesRegex(SchemaError, "missing variant/suite/phase results"):
            validate(missing_result)

        noncomplete_result = copy.deepcopy(data)
        noncomplete_result["results"][0]["status"] = "failed"
        with self.assertRaisesRegex(SchemaError, "non-complete results"):
            validate(noncomplete_result)

        noncomplete_variant = copy.deepcopy(data)
        noncomplete_variant["variants"][0]["status"] = "ready"
        with self.assertRaisesRegex(SchemaError, "non-complete variants"):
            validate(noncomplete_variant)

    def test_complete_campaign_requires_forty_combinations(self) -> None:
        data = load_summary()
        mark_campaign_complete(data)
        removed_suite = data["suites"].pop()["id"]
        data["results"] = [
            result for result in data["results"] if result["suite"] != removed_suite
        ]
        data["paired_disagreements"] = [
            entry
            for entry in data["paired_disagreements"]
            if entry["suite"] != removed_suite
        ]
        with self.assertRaisesRegex(
            SchemaError, "exactly 40 variant/suite combinations"
        ):
            validate(data)

    def test_complete_result_requires_exact_completeness(self) -> None:
        data = load_summary()
        row = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="bfcl-v4",
            metrics={"overall_accuracy": 0.5},
        )
        row["completeness"]["evaluated_units"] -= 1
        data["results"] = [row]
        with self.assertRaisesRegex(SchemaError, "complete but completeness"):
            validate(data)

    def test_complete_result_requires_evidence_lineage(self) -> None:
        data = load_summary()
        row = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="bfcl-v4",
            metrics=valid_metrics(data, "bfcl-v4"),
        )
        del row["evidence"]
        data["results"] = [row]
        with self.assertRaisesRegex(SchemaError, "no validated evidence lineage"):
            validate(data)

        row["evidence"] = {
            "importer": "manual",
            "sources": [
                {
                    "role": "summary",
                    "path": "/tmp/summary.json",
                    "sha256": "0" * 64,
                }
            ],
        }
        with self.assertRaisesRegex(SchemaError, "importer must be"):
            validate(data)

        row["evidence"] = {
            "importer": "glm52-result-import/v1",
            "sources": [
                {
                    "role": "summary",
                    "path": "artifact://bfcl-v4/ab/dynamo-vllm/summary/summary.json",
                    "sha256": "bad",
                }
            ],
        }
        with self.assertRaisesRegex(SchemaError, "lowercase SHA-256 digest"):
            validate(data)

    def test_partial_metrics_are_withheld_from_report(self) -> None:
        data = load_summary()
        data["results"] = [
            {
                "variant": "dynamo-vllm",
                "suite": "bfcl-v4",
                "phase": "ab",
                "status": "running",
                "completeness": {
                    "generated_units": 1,
                    "evaluated_units": 1,
                    "completed_trials": 1,
                },
                "metrics": {"overall_accuracy": 0.123456},
            }
        ]
        report = render(data)
        self.assertNotIn("12.35%", report)
        self.assertIn("generated 1/5,217 cases", report)

    def test_bfcl_complete_requires_memory_prerequisite_generations(self) -> None:
        data = load_summary()
        row = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="bfcl-v4",
            metrics={
                "overall_accuracy": 0.72,
                "correct_cases": 3676,
                "failed_cases": 1430,
                "inference_errors": 0,
            },
        )
        row["completeness"]["generated_units"] = 5106
        data["results"] = [row]
        with self.assertRaisesRegex(SchemaError, "5217 generated units"):
            validate(data)

    def test_bfcl_complete_rejects_inference_errors(self) -> None:
        data = load_summary()
        metrics = valid_metrics(data, "bfcl-v4")
        metrics["inference_errors"] = 1
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-vllm",
                suite_id="bfcl-v4",
                metrics=metrics,
            )
        ]
        with self.assertRaisesRegex(SchemaError, "zero inference_errors"):
            validate(data)

    def test_swe_complete_scores_must_reconcile(self) -> None:
        data = load_summary()
        metrics = valid_metrics(data, "swebench-verified")
        metrics["benchmark_score"] = 0.99
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-vllm",
                suite_id="swebench-verified",
                metrics=metrics,
            )
        ]
        with self.assertRaisesRegex(SchemaError, "benchmark_score must equal"):
            validate(data)

        data = load_summary()
        metrics = valid_metrics(data, "swebench-verified")
        metrics["score_on_submitted"] = 0.99
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-vllm",
                suite_id="swebench-verified",
                metrics=metrics,
            )
        ]
        with self.assertRaisesRegex(SchemaError, "score_on_submitted must equal"):
            validate(data)

    def test_bfcl_pair_delta_is_percentage_points(self) -> None:
        data = load_summary()
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-vllm",
                suite_id="bfcl-v4",
                metrics={
                    "overall_accuracy": 0.72,
                    "correct_cases": 3676,
                    "failed_cases": 1430,
                    "inference_errors": 0,
                },
            ),
            complete_result(
                data,
                variant="vllm-serve",
                suite_id="bfcl-v4",
                metrics={
                    "overall_accuracy": 0.70,
                    "correct_cases": 3574,
                    "failed_cases": 1532,
                    "inference_errors": 0,
                },
            ),
        ]
        report = render(data)
        self.assertIn("72.00%", report)
        self.assertIn("70.00%", report)
        self.assertIn("+2.00 pp", report)
        self.assertIn("generated 5,217/5,217 cases", report)
        self.assertIn("evaluated 5,106/5,106 cases", report)

    def test_terminal_pair_renders_every_pass_at_k_delta(self) -> None:
        data = load_summary()
        dynamo_outcomes = {
            "passed_attempts": 89,
            "failed_attempts": 356,
            "errored_attempts": 0,
            "no_reward_attempts": 0,
        }
        native_outcomes = {
            "passed_attempts": 44,
            "failed_attempts": 401,
            "errored_attempts": 0,
            "no_reward_attempts": 0,
        }
        dynamo_metrics = {
            **{f"pass_at_{k}": 0.10 * (k + 1) for k in range(1, 6)},
            **dynamo_outcomes,
        }
        native_metrics = {
            "pass_at_1": 44 / 445,
            **{f"pass_at_{k}": 0.05 + 0.10 * k for k in range(2, 6)},
            **native_outcomes,
        }
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-sglang",
                suite_id="terminal-bench-2.1",
                metrics=dynamo_metrics,
            ),
            complete_result(
                data,
                variant="sglang-serve",
                suite_id="terminal-bench-2.1",
                metrics=native_metrics,
            ),
        ]
        report = render(data)
        for k in range(1, 6):
            self.assertIn(f"pass@{k}", report)
        self.assertEqual(report.count("+5.00 pp"), 4)
        self.assertIn("+10.11 pp", report)
        self.assertIn("trials 445/445", report)

    def test_terminal_complete_rejects_errors_and_missing_rewards(self) -> None:
        for metric, message in (
            ("errored_attempts", "zero errored_attempts"),
            ("no_reward_attempts", "zero no_reward_attempts"),
        ):
            with self.subTest(metric=metric):
                data = load_summary()
                metrics = valid_metrics(data, "terminal-bench-2.1")
                metrics[metric] = 1
                metrics["failed_attempts"] -= 1
                data["results"] = [
                    complete_result(
                        data,
                        variant="dynamo-vllm",
                        suite_id="terminal-bench-2.1",
                        metrics=metrics,
                    )
                ]
                with self.assertRaisesRegex(SchemaError, message):
                    validate(data)

    def test_terminal_pass_at_one_must_reconcile(self) -> None:
        data = load_summary()
        metrics = valid_metrics(data, "terminal-bench-2.1")
        metrics["pass_at_1"] = 0.99
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-vllm",
                suite_id="terminal-bench-2.1",
                metrics=metrics,
            )
        ]
        with self.assertRaisesRegex(SchemaError, "pass_at_1 must equal"):
            validate(data)

    def test_terminal_task_image_identity_is_exact_and_shared(self) -> None:
        data = load_summary()
        row = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="terminal-bench-2.1",
            metrics=valid_metrics(data, "terminal-bench-2.1"),
        )
        row["suite_identity"]["task_images"]["task_count"] = 88
        data["results"] = [row]
        with self.assertRaisesRegex(SchemaError, "task_count must be 89"):
            validate(data)

        data = load_summary()
        data["results"] = [
            complete_result(
                data,
                variant=variant,
                suite_id="terminal-bench-2.1",
                phase=phase,
                metrics=valid_metrics(data, "terminal-bench-2.1"),
            )
            for variant, phase in (
                ("dynamo-vllm", "ab"),
                ("vllm-serve", "ab"),
                ("vllm-serve", "ba"),
                ("dynamo-vllm", "ba"),
            )
        ]
        data["results"][-1]["suite_identity"]["task_images"][
            "task_image_map_sha256"
        ] = digest("different-task-image-map")
        with self.assertRaisesRegex(SchemaError, "one exact task-image map"):
            validate(data)

    def test_percent_metric_must_be_fraction(self) -> None:
        data = load_summary()
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-vllm",
                suite_id="bfcl-v4",
                metrics={"overall_accuracy": 72.0},
            )
        ]
        with self.assertRaisesRegex(SchemaError, r"fraction in \[0, 1\]"):
            validate(data)

    def test_complete_outcome_counts_must_reconcile(self) -> None:
        data = load_summary()
        data["results"] = [
            complete_result(
                data,
                variant="dynamo-vllm",
                suite_id="bfcl-v4",
                metrics={
                    "overall_accuracy": 0.72,
                    "correct_cases": 3676,
                    "failed_cases": 1429,
                    "inference_errors": 0,
                },
            )
        ]
        with self.assertRaisesRegex(SchemaError, r"correct_cases \+ failed_cases"):
            validate(data)

    def test_every_cell_requires_fresh_controller_and_pods(self) -> None:
        data = load_summary()
        metrics = valid_metrics(data, "bfcl-v4")
        ab = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="bfcl-v4",
            phase="ab",
            metrics=metrics,
        )
        ba = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="bfcl-v4",
            phase="ba",
            metrics=metrics,
        )
        ba["runtime_identity"]["controller_uid_sha256"] = ab["runtime_identity"][
            "controller_uid_sha256"
        ]
        data["results"] = [ab, ba]
        with self.assertRaisesRegex(SchemaError, "fresh controller"):
            validate(data)

        data = load_summary()
        first = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="bfcl-v4",
            phase="ab",
            metrics=valid_metrics(data, "bfcl-v4"),
        )
        second = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="swebench-verified",
            phase="ab",
            metrics=valid_metrics(data, "swebench-verified"),
        )
        second["runtime_identity"]["deployment_sha256"] = first["runtime_identity"][
            "deployment_sha256"
        ]
        data["results"] = [first, second]
        with self.assertRaisesRegex(SchemaError, "fresh deployment"):
            validate(data)

    def test_phase_order_must_match_actual_capture_order(self) -> None:
        data = load_summary()
        metrics = valid_metrics(data, "bfcl-v4")
        dynamo = complete_result(
            data,
            variant="dynamo-vllm",
            suite_id="bfcl-v4",
            phase="ab",
            metrics=metrics,
        )
        native = complete_result(
            data,
            variant="vllm-serve",
            suite_id="bfcl-v4",
            phase="ab",
            metrics=metrics,
        )
        dynamo["runtime_identity"]["captured_at"] = "2026-07-05T04:00:00Z"
        native["runtime_identity"]["captured_at"] = "2026-07-05T03:00:00Z"
        data["results"] = [dynamo, native]
        with self.assertRaisesRegex(SchemaError, "ab must deploy Dynamo before native"):
            validate(data)

    def test_sidecars_are_physically_verified_and_orphans_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = paired_sidecar_fixture(root)
            validate_sidecars(data, root)

            task_path = root / "task-level/bfcl-v4/ab/dynamo-vllm.jsonl"
            original = task_path.read_bytes()
            task_path.unlink()
            with self.assertRaisesRegex(SchemaError, "missing or non-regular"):
                validate_sidecars(data, root)

            task_path.write_bytes(original + b"{}\n")
            with self.assertRaisesRegex(SchemaError, "digest mismatch"):
                validate_sidecars(data, root)
            task_path.write_bytes(original)

            orphan = root / "task-level/bfcl-v4/ab/orphan.jsonl"
            orphan.write_bytes(b"")
            with self.assertRaisesRegex(SchemaError, "orphaned"):
                validate_sidecars(data, root)
            orphan.unlink()

            disagreement = root / "paired-disagreements/bfcl-v4/ab/vllm.jsonl"
            wrong = canonical_jsonl(
                [
                    {
                        "dynamo_outcome": "failed",
                        "id": "case-b",
                        "native_outcome": "passed",
                    }
                ]
            )
            disagreement.write_bytes(wrong)
            data["paired_disagreements"][0]["sha256"] = hashlib.sha256(
                wrong
            ).hexdigest()
            with self.assertRaisesRegex(SchemaError, "content mismatch"):
                validate_sidecars(data, root)

    def test_report_source_guard_covers_all_behavior_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            campaign = repository / "benchmarks/glm52-nscale"
            (campaign / "eval/subdir").mkdir(parents=True)
            (campaign / "report").mkdir()
            (campaign / "campaign.env").write_text("MAX_MODEL_LEN=409600\n")
            (campaign / "eval/subdir/runner.py").write_text("VALUE = 1\n")
            (campaign / "report/import_result.py").write_text("IMPORTER = 1\n")
            (campaign / "report/generate.py").write_text("GENERATOR = 1\n")
            (campaign / "README.md").write_text("documentation\n")
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "test@example.com",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            data = {"campaign": {"source_commit": commit}, "results": []}
            assert_pinned_report_sources(data, campaign_root=campaign)

            (campaign / "README.md").write_text("updated documentation\n")
            assert_pinned_report_sources(data, campaign_root=campaign)

            guarded = campaign / "eval/subdir/runner.py"
            original = guarded.read_text()
            guarded.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(SchemaError, "sources differ"):
                assert_pinned_report_sources(data, campaign_root=campaign)
            guarded.write_text(original)

            shadow = campaign / "eval/shadow.py"
            shadow.write_text("raise SystemExit\n")
            with self.assertRaisesRegex(SchemaError, "shadow.py"):
                assert_pinned_report_sources(data, campaign_root=campaign)
            shadow.unlink()

            scaffold = {"campaign": {"source_commit": None}, "results": []}
            assert_pinned_report_sources(
                scaffold,
                campaign_root=campaign,
                allow_unpinned_scaffold=True,
            )
            scaffold["results"] = [{}]
            with self.assertRaisesRegex(SchemaError, "result-free scaffold"):
                assert_pinned_report_sources(
                    scaffold,
                    campaign_root=campaign,
                    allow_unpinned_scaffold=True,
                )


if __name__ == "__main__":
    unittest.main()
