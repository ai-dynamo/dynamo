#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import import_result as importer
from generate import DEFAULT_INPUT, validate
from import_result import EvidenceError, build_result, update_summary
from runtime_binding import VARIANTS, make_wrapper


TEST_SWE_IDS = [f"instance-{index}" for index in range(500)]
TEST_SWE_SCOPE = {
    "schema_version": 1,
    "scope": "full",
    "full_run": True,
    "dataset_instances": 500,
    "target_instances": 500,
    "instance_filter": None,
    "instance_slice": None,
    "target_ids": TEST_SWE_IDS,
}
TEST_SWE_SCOPE_SHA256 = hashlib.sha256(
    (json.dumps(TEST_SWE_SCOPE, indent=2) + "\n").encode()
).hexdigest()
TEST_TERMINAL_TASK_REFS = [
    {
        "name": f"task-{index:03d}",
        "org": "terminal-bench",
        "ref": f"sha256:{index + 1:064x}",
    }
    for index in range(89)
]
TEST_TERMINAL_TASK_REFS_SHA256 = importer.canonical_sha256(TEST_TERMINAL_TASK_REFS)
TEST_BFCL_IDS_BY_CATEGORY = {
    category: [f"{category}-{index:04d}" for index in range(count)]
    for category, count in importer.BFCL_SCORED_CATEGORY_COUNTS.items()
}
TEST_BFCL_SCORED_IDS = {
    case_id for values in TEST_BFCL_IDS_BY_CATEGORY.values() for case_id in values
}
TEST_BFCL_GENERATED_IDS = TEST_BFCL_SCORED_IDS | {
    f"memory-prerequisite-{index:03d}" for index in range(111)
}
TEST_BFCL_SCORED_IDS_SHA256 = importer.ids_sha256(TEST_BFCL_SCORED_IDS)
TEST_BFCL_GENERATED_IDS_SHA256 = importer.ids_sha256(TEST_BFCL_GENERATED_IDS)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def summary_template() -> dict:
    summary = json.loads(DEFAULT_INPUT.read_text())
    summary["campaign"]["source_commit"] = "1" * 40
    return summary


def campaign_source_identity() -> dict:
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


def harbor_environment_identity() -> dict:
    packages = [["harbor", "1.0.0"], ["litellm", "1.80.0"]]
    payload = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    return {
        "uv_sync_check": "passed",
        "python": "3.12.11",
        "package_count": len(packages),
        "packages_sha256": hashlib.sha256(payload).hexdigest(),
        "packages": packages,
    }


def runtime_binding(
    variant: str = "dynamo-vllm",
    phase: str = "ab",
    *,
    evaluator: dict | None = None,
) -> dict:
    fixture = json.loads(
        (importer.EVAL_DIR / "fixtures" / "runtime-binding.json").read_text()
    )
    deployment = copy.deepcopy(fixture)
    definition = VARIANTS[variant]
    deployment.update(
        {
            "variant": variant,
            "campaign_phase": phase,
            "image": definition["image"],
            "endpoint": definition["endpoint"],
            "service_name": definition["service_name"],
        }
    )
    deployment["controller"].update(
        {"kind": definition["controller_kind"], "name": definition["controller_name"]}
    )
    roles = definition["roles"]
    template_pods = fixture["pods"]
    deployment["pods"] = {
        role: copy.deepcopy(template_pods.get(role, template_pods["worker"]))
        for role in roles
    }
    deployment["recipe"]["source_commit"] = "1" * 40
    deployment["recipe"]["template_sha256"] = importer.sha256_file(
        importer.ROOT / "deploy" / "templates" / f"{variant}.yaml"
    )
    deployment["control_plane"] = (
        fixture["control_plane"] if definition["dynamo"] else None
    )
    campaign_source = campaign_source_identity()
    evaluator = (
        {"campaign_source": campaign_source}
        if evaluator is None
        else {**evaluator, "campaign_source": campaign_source}
    )
    return make_wrapper(
        deployment,
        evaluator=evaluator,
        variant=variant,
        campaign_phase=phase,
        endpoint=definition["endpoint"],
    )


def resign_runtime_binding(wrapper: dict) -> None:
    wrapper["deployment_sha256"] = importer.runtime_canonical_sha256(
        wrapper["content"]["deployment"]
    )
    wrapper["content_sha256"] = importer.runtime_canonical_sha256(wrapper["content"])


def write_runtime_continuity(run: Path, wrapper: dict) -> None:
    deployment = wrapper["content"]["deployment"]
    snapshot = {
        "controller": copy.deepcopy(deployment["controller"]),
        "pods": {
            role: {
                **{
                    field: pod[field]
                    for field in (
                        "name_sha256",
                        "uid_sha256",
                        "node_name_sha256",
                        "image_id",
                    )
                },
                "container_id_sha256": hashlib.sha256(
                    f"{deployment['variant']}:{role}:container".encode()
                ).hexdigest(),
                "restart_count": 0,
            }
            for role, pod in deployment["pods"].items()
        },
    }
    write_json(
        run / "runtime-continuity.json",
        {
            "schema_version": 1,
            "variant": deployment["variant"],
            "campaign_phase": deployment["campaign_phase"],
            "deployment_sha256": wrapper["deployment_sha256"],
            "command_exit_code": 0,
            "stable": True,
            "pre_captured_at": "2026-07-05T01:00:00Z",
            "post_captured_at": "2026-07-05T02:00:00Z",
            "pre": snapshot,
            "post": snapshot,
        },
    )


def bfcl_evidence(
    root: Path,
    *,
    complete: bool = True,
    phase: str = "ab",
    endpoint_context_fields: dict | None = None,
) -> Path:
    run = root / f"bfcl-full-{phase}-run"
    commit = importer.load_assignments(importer.EVAL_PINS)["BFCL_COMMIT"]
    source_identity = {
        "head": commit,
        "status": importer.BFCL_SOURCE_STATUS,
        "tracked_diff_sha256": importer.BFCL_TRACKED_DIFF_SHA256,
        "new_handler_sha256": importer.BFCL_NEW_HANDLER_SHA256,
    }
    endpoint_model_response = {
        "id": "zai-org/GLM-5.2",
        "object": "model",
        "owned_by": "nvidia",
    }
    endpoint_model_response.update(
        {"context_window": 409600}
        if endpoint_context_fields is None
        else endpoint_context_fields
    )
    endpoint_models = {
        "object": "list",
        "data": [endpoint_model_response],
    }
    write_json(run / "endpoint-models.json", endpoint_models)
    constraints_lines = sorted(
        (
            line.strip()
            for line in (importer.ROOT / "eval/bfcl/constraints.lock")
            .read_text()
            .splitlines()
            if line.strip() and not line.startswith("#")
        ),
        key=str.casefold,
    )
    environment_freeze = "\n".join(constraints_lines) + "\n"
    (run / "environment.freeze.txt").write_text(environment_freeze)
    environment_lock = {
        "schema_version": 1,
        "constraints_sha256": importer.BFCL_CONSTRAINTS_SHA256,
        "freeze_sha256": hashlib.sha256(environment_freeze.encode()).hexdigest(),
        "package_count": importer.BFCL_PACKAGE_COUNT,
        "python": "3.12.11",
    }
    write_json(run / "environment-lock.json", environment_lock)
    endpoint_model = {
        "id": "zai-org/GLM-5.2",
        "object": "model",
        "owned_by": "nvidia",
        "context_window": 409600,
    }
    binding = runtime_binding(
        phase=phase,
        evaluator={
            "harness": "bfcl-v4",
            "model_registry_name": "zai-org/GLM-5.2-FC",
            "categories": ["all_scoring"],
            "temperature": 0.0,
            "max_tokens": 64000,
            "num_threads": 16,
        },
    )
    metadata = {
        "schema_version": 6,
        "variant": "dynamo-vllm",
        "mode": "full",
        "campaign_phase": phase,
        "run_name": run.name,
        "categories": ["all_scoring"],
        "model_registry_name": "zai-org/GLM-5.2-FC",
        "served_model_name": "zai-org/GLM-5.2",
        "endpoint": "http://glm52-dynamo-vllm-frontend:8000/v1",
        "bfcl_gorilla_commit": commit,
        "bfcl_patch_sha256": importer.sha256_file(importer.BFCL_PATCH),
        "bfcl_source_identity": source_identity,
        "endpoint_models_sha256": importer.sha256_file(run / "endpoint-models.json"),
        "endpoint_model": endpoint_model,
        "temperature": 0.0,
        "max_tokens": 64000,
        "num_threads": 16,
        "include_input_log": True,
        "glm52_openai_extra_body": None,
        "glm52_openai_default_headers_sha256": None,
        "runtime_binding": binding,
        "campaign_source": campaign_source_identity(),
        "python_environment": environment_lock,
    }
    category_stats = {}
    remaining_correct = 4000
    for category, count in importer.BFCL_SCORED_CATEGORY_COUNTS.items():
        correct = min(count, remaining_correct)
        category_stats[category] = {
            "expected_count": count,
            "total_count": count,
            "correct_count": correct,
            "failure_count": count - correct,
        }
        remaining_correct -= correct
    assert remaining_correct == 0
    failure_rows = []
    remaining_correct = 4000
    for category, values in TEST_BFCL_IDS_BY_CATEGORY.items():
        correct = min(len(values), remaining_correct)
        failure_rows.extend(
            {"id": case_id, "category": category, "error_type": "incorrect"}
            for case_id in values[correct:]
        )
        remaining_correct -= correct
    write_json(
        run / "expected-ids.json",
        {
            "schema_version": 1,
            "bfcl_gorilla_commit": commit,
            "requested_categories": ["all_scoring"],
            "expanded_categories": sorted(TEST_BFCL_IDS_BY_CATEGORY),
            "generated": {
                "count": len(TEST_BFCL_GENERATED_IDS),
                "ids_sha256": TEST_BFCL_GENERATED_IDS_SHA256,
                "ids_by_result_category": {},
            },
            "scored": {
                "count": len(TEST_BFCL_SCORED_IDS),
                "ids_sha256": TEST_BFCL_SCORED_IDS_SHA256,
                "ids_by_category": TEST_BFCL_IDS_BY_CATEGORY,
            },
        },
    )
    (run / "failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failure_rows)
    )
    write_runtime_continuity(run, binding)
    write_json(run / "metadata.json", metadata)
    write_json(
        run / "summary.json",
        {
            "schema_version": 1,
            "variant": "dynamo-vllm",
            "mode": "full",
            "campaign_phase": phase,
            "run_name": run.name,
            "bfcl_gorilla_commit": commit,
            "categories": {},
            "totals": {
                "generated_count": 5217,
                "inference_error_count": 0,
                "correct_count": 4000,
                "failure_count": 1106,
                "scored_count": 5106,
            },
            "official_overall_csv_row": {
                "Model": importer.BFCL_OFFICIAL_MODEL,
                "Overall Acc": "73.05%",
            },
        },
    )
    write_json(
        run / "complete-validation.json",
        {
            "schema_version": 2,
            "phase": "complete",
            "campaign_phase": phase,
            "status": "pass" if complete else "fail",
            "errors": [] if complete else ["missing score"],
            "generation": {
                "status": "pass",
                "errors": [],
                "expected_count": 5217,
                "actual_count": 5217,
                "actual_entry_count": 5217,
                "expected_ids_sha256": importer.BFCL_GENERATED_IDS_SHA256,
                "actual_ids_sha256": importer.BFCL_GENERATED_IDS_SHA256,
                "missing_ids": [],
                "extra_ids": [],
                "duplicate_ids": {},
                "missing_categories": [],
                "extra_categories": [],
                "inference_error_ids": [],
            },
            "scores": {
                "status": "pass",
                "errors": [],
                "expected_count": 5106,
                "expected_ids_sha256": importer.BFCL_SCORED_IDS_SHA256,
                "scored_count": 5106,
                "missing_categories": [],
                "extra_categories": [],
                "categories": category_stats,
            },
            "run_identity": {
                "metadata_sha256": importer.sha256_file(run / "metadata.json"),
                "immutable": metadata,
            },
        },
    )
    return run / "summary.json"


def swe_evidence(root: Path, *, phase: str = "ab") -> Path:
    run = root / f"dynamo-vllm-{phase}-full-20260705" / "verified"
    run.mkdir(parents=True, exist_ok=True)
    ids = TEST_SWE_IDS
    effective_config = {
        "agent": {"model": "zai-org/GLM-5.2"},
        "environment": {"image": "swebench/swebase:latest"},
        "model": {"temperature": 0.0},
    }
    base_binding = runtime_binding(phase=phase)
    endpoint_content = {
        "schema_version": 1,
        "requested_model": "zai-org/GLM-5.2",
        "expected_context_window": 409600,
        "selected_model_response": {
            "id": "zai-org/GLM-5.2",
            "context_window": 409600,
        },
        "full_response": {"object": "list", "data": []},
    }
    evaluator = {
        "deployment_source_sha256": base_binding["deployment_sha256"],
        "runtime_family": "vllm",
        "runtime_source_revision": "8" * 40,
        "dynamo_enabled": True,
        "tensor_parallel_size": 4,
        "effective_config_sha256": "5" * 64,
        "effective_config_content_sha256": importer.canonical_sha256(effective_config),
        "effective_config": effective_config,
        "endpoint_evidence": {
            "file_sha256": "6" * 64,
            "content_sha256": importer.canonical_sha256(endpoint_content),
            "content": endpoint_content,
        },
        "generation": {"workers": 16, "batch_size": 8},
        "evaluation": {
            "workers": 8,
            "timeout_seconds": 3600,
            "backend": "official-swebench-docker",
            "docker_platform": None,
        },
        "campaign_source": campaign_source_identity(),
    }
    binding = runtime_binding(phase=phase, evaluator=evaluator)
    python_environment = {
        "constraints_lock_sha256": importer.SWE_CONSTRAINTS_LOCK_SHA256,
        "freeze_sha256": "7" * 64,
        "normalized_freeze_sha256": importer.SWE_NORMALIZED_FREEZE_SHA256,
        "normalized_requirement_count": 101,
    }
    lock_lines = sorted(
        line.strip()
        for line in (importer.ROOT / "eval/swebench/constraints.lock")
        .read_text()
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    normalized_environment = "\n".join(lock_lines) + "\n"
    raw_environment = (
        normalized_environment
        + "-e file:///artifacts/mini-swe-agent\n"
        + "-e file:///artifacts/SWE-bench\n"
    )
    (run / "environment.freeze.txt").write_text(raw_environment)
    (run / "environment.normalized.freeze.txt").write_text(normalized_environment)
    python_environment["freeze_sha256"] = hashlib.sha256(
        raw_environment.encode()
    ).hexdigest()
    task_images = {
        instance_id: {
            "requested_ref": f"swebench/{instance_id}:latest",
            "image_id": "sha256:" + hashlib.sha256(instance_id.encode()).hexdigest(),
            "repo_digests": [
                f"swebench/{instance_id}@sha256:"
                + hashlib.sha256((instance_id + ":repo").encode()).hexdigest()
            ],
        }
        for instance_id in ids
    }
    for identity in task_images.values():
        identity["content_identity_sha256"] = importer.canonical_sha256(
            {"image_id": identity["image_id"], "repo_digests": identity["repo_digests"]}
        )
    task_image_document = {
        "schema_version": 1,
        "suite": "verified",
        "dataset_sha256": importer.SWE_DATASETS["verified"]["jsonl_sha256"],
        "scope_sha256": importer.SWE_DATASETS["verified"]["scope_sha256"],
        "images": task_images,
    }
    write_json(run / "run-scope.json", TEST_SWE_SCOPE)
    write_json(
        run / "score.json",
        {
            "suite": "verified",
            "scope": "full",
            "full_run": True,
            "dataset_instances": 500,
            "expected_instances": 500,
            "target_instances": 500,
            "excluded_dataset_instances": 0,
            "submitted_instances": 500,
            "completed_instances": 500,
            "passed_instances": 125,
            "failed_instances": 375,
            "missing_instances": 0,
            "missing_ids": [],
            "unexpected_ids": [],
            "incomplete_evaluation_ids": [],
            "missing_evaluation_ids": [],
            "unexpected_evaluation_ids": [],
            "evaluation_error_ids": [],
            "unresolved_ids": ids[125:],
            "empty_patch_ids": [],
            "score_on_submitted": 0.25,
            "score_on_scope": 0.25,
            "benchmark_score": 0.25,
            "complete": True,
            "gate_failures": [],
            "passed_ids": ids[:125],
            "failed_ids": ids[125:],
            "raw_result": "/artifacts/raw-score.json",
        },
    )
    write_json(
        run / "generation-summary.json",
        {
            "scope": "full",
            "full_run": True,
            "dataset_instances": 500,
            "expected_instances": 500,
            "target_instances": 500,
            "excluded_dataset_instances": 0,
            "predictions": 500,
            "missing_predictions": 0,
            "missing_prediction_ids": [],
            "unexpected_prediction_ids": [],
            "infrastructure_error_ids": [],
            "exit_statuses": {"Submitted": 500},
            "complete": True,
            "gate_failures": [],
            "instances": [
                {
                    "instance_id": instance_id,
                    "exit_status": "Submitted",
                    "api_calls": 2,
                    "patch_bytes": 10,
                    "trajectory": f"/artifacts/{instance_id}.traj.json",
                    "valid_model_result": True,
                    "validation_failures": [],
                    "effective_config_sha256": importer.canonical_sha256(
                        effective_config
                    ),
                    "task_image": task_images[instance_id],
                    "runtime_deployment_sha256": binding["deployment_sha256"],
                    "runtime_content_sha256": binding["content_sha256"],
                }
                for instance_id in ids
            ],
            "runtime_binding": binding,
            "python_environment": python_environment,
            "effective_config_sha256": importer.canonical_sha256(effective_config),
            "trajectory_effective_config_sha256s": [
                importer.canonical_sha256(effective_config)
            ],
            "task_image_evidence": {
                "sha256": importer.canonical_sha256(task_image_document),
                "images": task_images,
            },
            "missing_task_image_ids": [],
            "unexpected_task_image_ids": [],
            "invalid_task_image_ids": [],
        },
    )
    write_json(
        run / "run-metadata.json",
        {
            "schema_version": 3,
            "run_name": f"dynamo-vllm-{phase}-full-20260705",
            "campaign_phase": phase,
            "suite": "verified",
            "endpoint": "http://glm52-dynamo-vllm-frontend:8000/v1",
            "model": "zai-org/GLM-5.2",
            "scope_sha256": importer.SWE_DATASETS["verified"]["scope_sha256"],
            "configuration": {
                "files": [
                    {
                        "name": "upstream-swebench",
                        "sha256": importer.SWE_UPSTREAM_CONFIG_SHA256,
                    },
                    {
                        "name": "glm52",
                        "sha256": importer.sha256_file(importer.SWE_CONFIG),
                    },
                ],
                "sha256": importer.canonical_sha256(
                    [
                        {
                            "name": "upstream-swebench",
                            "sha256": importer.SWE_UPSTREAM_CONFIG_SHA256,
                        },
                        {
                            "name": "glm52",
                            "sha256": importer.sha256_file(importer.SWE_CONFIG),
                        },
                    ]
                ),
            },
            "dataset": {
                "evaluator_jsonl_sha256": importer.SWE_DATASETS["verified"][
                    "jsonl_sha256"
                ],
                "provenance_sha256": importer.SWE_PROVENANCE_SHA256,
                "provenance": {
                    "agent_dataset": "/artifacts/datasets/agent/verified",
                    "evaluator_dataset": "/artifacts/datasets/evaluator/verified.jsonl",
                    "repo": importer.SWE_DATASETS["verified"]["repo"],
                    "revision": importer.expected_swe_pins()[
                        "SWEBENCH_VERIFIED_REVISION"
                    ],
                    "expected": 500,
                    "rows": 500,
                    "jsonl_sha256": importer.SWE_DATASETS["verified"]["jsonl_sha256"],
                    "parquet_sha256": importer.SWE_DATASETS["verified"][
                        "parquet_sha256"
                    ],
                },
            },
            "pins": {
                "sha256": importer.canonical_sha256(importer.expected_swe_pins()),
                "values": importer.expected_swe_pins(),
            },
            "source": {
                "lock_sha256": importer.serialized_source_lock_sha256(
                    importer.expected_swe_source_lock(importer.expected_swe_pins())
                ),
                "lock": importer.expected_swe_source_lock(importer.expected_swe_pins()),
                "repositories": {
                    "mini_swe_agent": {
                        "commit": importer.expected_swe_pins()["MINI_SWE_AGENT_COMMIT"]
                    },
                    "swebench": {
                        "commit": importer.expected_swe_pins()["SWEBENCH_COMMIT"]
                    },
                    "swebench_pro": {
                        "commit": importer.expected_swe_pins()["SWEBENCH_PRO_COMMIT"]
                    },
                },
            },
            "python_environment": python_environment,
            "campaign_source": campaign_source_identity(),
            "runtime_binding": binding,
        },
    )
    write_runtime_continuity(run, binding)
    return run / "score.json"


def terminal_evidence(root: Path, *, phase: str = "ab") -> Path:
    tasks = []
    for index in range(89):
        passed = 5 if index < 10 else 0
        value = 1.0 if passed else 0.0
        tasks.append(
            {
                "task_name": f"terminal-bench/task-{index:03d}",
                "attempts": 5,
                "passed_attempts": passed,
                "failed_attempts": 5 - passed,
                "errored_attempts": 0,
                "no_reward_attempts": 0,
                "mean_reward_scored": value,
                "mean_reward_all_attempts": value,
                **{f"pass_at_{k}": value for k in range(1, 6)},
            }
        )
    pass_value = round(10 / 89, 8)
    pins = importer.load_assignments(importer.TERMINAL_PINS)
    endpoint = "http://glm52-dynamo-vllm-frontend:8000/v1"
    max_context = int(pins["TERMINALBENCH_MAX_CONTEXT_TOKENS"])
    max_output = int(pins["TERMINALBENCH_MAX_OUTPUT_TOKENS"])
    model_info = json.dumps(
        {
            "max_tokens": max_context,
            "max_input_tokens": max_context - max_output,
            "max_output_tokens": max_output,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        separators=(",", ":"),
    )
    call_kwargs = json.dumps(
        {"max_tokens": max_output, "top_p": float(pins["TERMINALBENCH_TOP_P"])},
        separators=(",", ":"),
    )
    command = [
        "/artifacts/harbor/bin/harbor",
        "run",
        "--dataset",
        pins["TERMINALBENCH_DATASET"],
        "--agent",
        pins["TERMINUS_AGENT"],
        "--model",
        "openai/zai-org/GLM-5.2",
        "--agent-kwarg",
        f"api_base={endpoint}",
        "--agent-kwarg",
        "parser_name=json",
        "--agent-kwarg",
        f"max_turns={pins['TERMINALBENCH_MAX_TURNS']}",
        "--agent-kwarg",
        f"temperature={pins['TERMINALBENCH_TEMPERATURE']}",
        "--agent-kwarg",
        f"model_info={model_info}",
        "--agent-kwarg",
        f"llm_call_kwargs={call_kwargs}",
        "--n-attempts",
        pins["TERMINALBENCH_OFFICIAL_ATTEMPTS"],
        "--n-concurrent",
        "4",
        "--max-retries",
        "0",
        "--timeout-multiplier",
        pins["TERMINALBENCH_TIMEOUT_MULTIPLIER"],
        "--env",
        "docker",
        "--delete",
        "--yes",
    ]
    binding = runtime_binding(phase=phase)
    summary_dir = root / f"terminal-{phase}" / "summary"
    run_metadata = {
        "schema_version": 2,
        "run_spec": {
            "mode": "full",
            "label": "dynamo-vllm",
            "campaign_phase": phase,
            "dataset": pins["TERMINALBENCH_DATASET"],
            "dataset_revision": int(pins["TERMINALBENCH_DATASET_REVISION"]),
            "dataset_content_hash": pins["TERMINALBENCH_DATASET_CONTENT_HASH"],
            "dataset_version_id": pins["TERMINALBENCH_DATASET_VERSION_ID"],
            "expected_tasks": 89,
            "attempts_per_task": 5,
            "expected_trials": 445,
            "agent": pins["TERMINUS_AGENT"],
            "litellm_model": "openai/zai-org/GLM-5.2",
            "served_model": "zai-org/GLM-5.2",
            "api_base": endpoint,
            "n_concurrent": 4,
            "temperature": float(pins["TERMINALBENCH_TEMPERATURE"]),
            "top_p": float(pins["TERMINALBENCH_TOP_P"]),
            "max_turns": int(pins["TERMINALBENCH_MAX_TURNS"]),
            "max_context_tokens": max_context,
            "max_output_tokens": max_output,
            "timeout_multiplier": float(pins["TERMINALBENCH_TIMEOUT_MULTIPLIER"]),
            "job_name": f"dynamo-vllm-{phase}-terminalbench21-full-test",
            "job_dir": str((root / f"terminal-{phase}").resolve()),
            "runtime_deployment_sha256": binding["deployment_sha256"],
        },
        "pins": {
            "harbor_repository": pins["HARBOR_REPOSITORY"],
            "harbor_version": pins["HARBOR_VERSION"],
            "harbor_commit": pins["HARBOR_COMMIT"],
            "harbor_uv_lock_sha256": importer.TERMINAL_HARBOR_UV_LOCK_SHA256,
            "dataset": pins["TERMINALBENCH_DATASET"],
            "dataset_revision": int(pins["TERMINALBENCH_DATASET_REVISION"]),
            "dataset_content_hash": pins["TERMINALBENCH_DATASET_CONTENT_HASH"],
            "dataset_version_id": pins["TERMINALBENCH_DATASET_VERSION_ID"],
            "resolved_dataset": {
                "requested_ref": pins["TERMINALBENCH_DATASET"],
                "name": pins["TERMINALBENCH_DATASET"].rsplit("@", 1)[0],
                "version": f"sha256:{pins['TERMINALBENCH_DATASET_CONTENT_HASH']}",
                "dataset_version_id": pins["TERMINALBENCH_DATASET_VERSION_ID"],
                "content_hash": pins["TERMINALBENCH_DATASET_CONTENT_HASH"],
                "task_count": 89,
                "resolved_at": "2026-07-05T00:00:00Z",
                "task_refs": TEST_TERMINAL_TASK_REFS,
            },
        },
        "source": {
            "schema_version": 2,
            "generated_at": "2026-07-05T00:00:00Z",
            "source_commit": "1" * 40,
            "source_branch": "rmccormick/glm52",
            "source_clean": True,
            "source_changed_path_count": 0,
            "bundle_sha256": campaign_source_identity()["bundle_sha256"],
            "bundle_contents": ["campaign.env", "eval"],
            **{
                field: campaign_source_identity()[field]
                for field in (
                    "source_tree_sha256",
                    "eval_tree_sha256",
                    "campaign_env_sha256",
                    "source_file_count",
                    "eval_file_count",
                )
            },
            "source_files": {},
        },
        "campaign_source": campaign_source_identity(),
        "runtime_binding": binding,
        "harbor_environment": harbor_environment_identity(),
        "invocations": [
            {
                "command": command,
                "finished_at": "2026-07-05T01:00:00Z",
                "harbor_exit_code": 0,
                "elapsed_seconds": 3600,
            }
        ],
    }
    run_metadata_sha256 = hashlib.sha256(
        (json.dumps(run_metadata, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    trials_path = summary_dir / "trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_name",
                "trial_name",
                "status",
                "primary_reward",
                "result_sha256",
            ),
        )
        writer.writeheader()
        for task in tasks:
            for attempt in range(1, 6):
                passed = attempt <= task["passed_attempts"]
                trial_name = f"{task['task_name']}-attempt-{attempt}"
                writer.writerow(
                    {
                        "task_name": task["task_name"],
                        "trial_name": trial_name,
                        "status": "passed" if passed else "failed",
                        "primary_reward": 1.0 if passed else 0.0,
                        "result_sha256": hashlib.sha256(
                            f"{trial_name}:result".encode()
                        ).hexdigest(),
                    }
                )
    task_image_rows = []
    for task, task_ref in zip(tasks, TEST_TERMINAL_TASK_REFS):
        task_name = task["task_name"]
        repository = f"registry.example.test/terminal-bench/{task_ref['name']}"
        task_image_rows.append(
            {
                "task_name": task_name,
                "task_ref": copy.deepcopy(task_ref),
                "task_checksum": hashlib.sha256(
                    f"{task_name}:checksum".encode()
                ).hexdigest(),
                "task_toml_sha256": hashlib.sha256(
                    f"{task_name}:task.toml".encode()
                ).hexdigest(),
                "requested_ref": f"{repository}:2.1",
                "image_id": "sha256:"
                + hashlib.sha256(f"{task_name}:image-id".encode()).hexdigest(),
                "repo_digests": [
                    f"{repository}@sha256:"
                    + hashlib.sha256(f"{task_name}:repo-digest".encode()).hexdigest()
                ],
            }
        )
    task_images = {
        "schema_version": 1,
        "task_count": 89,
        "trial_count": 445,
        "tasks": task_image_rows,
    }
    task_images_path = summary_dir / "task-images.json"
    write_json(task_images_path, task_images)
    path = summary_dir / "summary.json"
    write_json(
        path,
        {
            "schema_version": 2,
            "generated_at": "2026-07-05T00:00:00Z",
            "validation": {
                "strict": True,
                "complete": True,
                "errors": [],
                "expected_tasks": 89,
                "expected_attempts_per_task": 5,
                "expected_trials": 445,
                "observed_tasks": 89,
                "observed_trials": 445,
                "expected_task_names": [task["task_name"] for task in tasks],
            },
            "score": {
                "mean_reward_all_trials": 50 / 445,
                "passed_attempts": 50,
                "failed_attempts": 395,
                "errored_attempts": 0,
                "no_reward_attempts": 0,
                "pass_at_k": {str(k): pass_value for k in range(1, 6)},
            },
            "input_hashes": {
                "job_result_sha256": "a" * 64,
                "job_config_sha256": "b" * 64,
                "job_lock_sha256": "c" * 64,
                "dataset_metadata_sha256": "d" * 64,
                "task_images_sha256": importer.sha256_file(task_images_path),
                "run_metadata_sha256": run_metadata_sha256,
            },
            "run_metadata": run_metadata,
            "task_images": task_images,
            "tasks": tasks,
        },
    )
    write_runtime_continuity(summary_dir, binding)
    return path


class ResultImportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous = importer.SWE_DATASETS["verified"]["scope_sha256"]
        importer.SWE_DATASETS["verified"]["scope_sha256"] = TEST_SWE_SCOPE_SHA256
        self.addCleanup(
            importer.SWE_DATASETS["verified"].__setitem__, "scope_sha256", previous
        )
        previous_refs = importer.TERMINAL_TASK_REFS_SHA256
        importer.TERMINAL_TASK_REFS_SHA256 = TEST_TERMINAL_TASK_REFS_SHA256
        self.addCleanup(setattr, importer, "TERMINAL_TASK_REFS_SHA256", previous_refs)
        previous_scored_ids = importer.BFCL_SCORED_IDS_SHA256
        previous_generated_ids = importer.BFCL_GENERATED_IDS_SHA256
        importer.BFCL_SCORED_IDS_SHA256 = TEST_BFCL_SCORED_IDS_SHA256
        importer.BFCL_GENERATED_IDS_SHA256 = TEST_BFCL_GENERATED_IDS_SHA256
        self.addCleanup(
            setattr, importer, "BFCL_SCORED_IDS_SHA256", previous_scored_ids
        )
        self.addCleanup(
            setattr, importer, "BFCL_GENERATED_IDS_SHA256", previous_generated_ids
        )
        source_guard = mock.patch.object(
            importer, "assert_pinned_report_sources", return_value=None
        )
        source_guard.start()
        self.addCleanup(source_guard.stop)

    def test_bfcl_import_is_atomic_idempotent_and_lineaged(self) -> None:
        artifact = bfcl_evidence(self.root)
        summary_path = self.root / "results" / "summary.json"
        write_json(summary_path, summary_template())

        data, changed = update_summary(
            summary_path, "dynamo-vllm", "bfcl-v4", "ab", artifact
        )
        self.assertTrue(changed)
        self.assertEqual(summary_path.stat().st_mode & 0o777, 0o644)
        validate(data)
        row = data["results"][0]
        self.assertAlmostEqual(row["metrics"]["overall_accuracy"], 0.7305)
        self.assertEqual(row["metrics"]["correct_cases"], 4000)
        self.assertEqual(len(row["evidence"]["sources"]), 9)
        for item in row["evidence"]["sources"]:
            self.assertTrue(item["path"].startswith("artifact://bfcl-v4/ab/"))
            self.assertEqual(len(item["sha256"]), 64)
        task_path = summary_path.parent / "task-level/bfcl-v4/ab/dynamo-vllm.jsonl"
        self.assertEqual(row["task_level"]["sha256"], importer.sha256_file(task_path))

        _, changed = update_summary(
            summary_path, "dynamo-vllm", "bfcl-v4", "ab", artifact
        )
        self.assertFalse(changed)

        compact = json.loads(artifact.read_text())
        compact["official_overall_csv_row"]["Overall Acc"] = "78.35%"
        write_json(artifact, compact)
        with self.assertRaisesRegex(EvidenceError, "Overall Acc"):
            update_summary(summary_path, "dynamo-vllm", "bfcl-v4", "ab", artifact)

    def test_bfcl_import_accepts_vllm_max_model_len_alias(self) -> None:
        artifact = bfcl_evidence(
            self.root, endpoint_context_fields={"max_model_len": 409600}
        )
        row = build_result(summary_template(), "dynamo-vllm", "bfcl-v4", "ab", artifact)
        self.assertAlmostEqual(row["metrics"]["overall_accuracy"], 0.7305)

    def test_bfcl_import_rejects_invalid_context_aliases(self) -> None:
        cases = (
            ({}, "non-null"),
            ({"context_window": None, "max_model_len": None}, "non-null"),
            ({"max_model_len": 262144}, "!= campaign"),
            (
                {"context_window": 409600, "max_model_len": 262144},
                "conflict",
            ),
            ({"max_model_len": 409600.0}, "must be integers"),
        )
        for fields, message in cases:
            with self.subTest(fields=fields):
                artifact = bfcl_evidence(self.root, endpoint_context_fields=fields)
                with self.assertRaisesRegex(EvidenceError, message):
                    build_result(
                        summary_template(),
                        "dynamo-vllm",
                        "bfcl-v4",
                        "ab",
                        artifact,
                    )

    def test_failed_evidence_does_not_modify_summary(self) -> None:
        artifact = bfcl_evidence(self.root, complete=False)
        summary_path = self.root / "summary.json"
        write_json(summary_path, summary_template())
        before = summary_path.read_bytes()
        with self.assertRaisesRegex(EvidenceError, "validation.status"):
            update_summary(summary_path, "dynamo-vllm", "bfcl-v4", "ab", artifact)
        self.assertEqual(summary_path.read_bytes(), before)

    def test_summary_write_failure_rolls_back_new_sidecars(self) -> None:
        artifact = bfcl_evidence(self.root)
        summary_path = self.root / "results" / "summary.json"
        write_json(summary_path, summary_template())
        before = summary_path.read_bytes()
        task_path = summary_path.parent / "task-level/bfcl-v4/ab/dynamo-vllm.jsonl"
        with mock.patch.object(
            importer, "atomic_write_json", side_effect=OSError("injected write failure")
        ):
            with self.assertRaisesRegex(OSError, "injected write failure"):
                update_summary(summary_path, "dynamo-vllm", "bfcl-v4", "ab", artifact)
        self.assertEqual(summary_path.read_bytes(), before)
        self.assertFalse(task_path.exists())

    def test_bfcl_import_rejects_pinned_identity_drift(self) -> None:
        cases = {
            "temperature": (1.0, "metadata.temperature"),
            "bfcl_patch_sha256": ("0" * 64, "metadata.bfcl_patch_sha256"),
            "glm52_openai_extra_body": ("{}", "glm52_openai_extra_body"),
            "glm52_openai_default_headers_sha256": (
                "0" * 64,
                "glm52_openai_default_headers_sha256",
            ),
        }
        for field, (value, message) in cases.items():
            with self.subTest(field=field):
                artifact = bfcl_evidence(self.root)
                metadata_path = artifact.parent / "metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata[field] = value
                write_json(metadata_path, metadata)
                with self.assertRaisesRegex(EvidenceError, message):
                    build_result(
                        summary_template(), "dynamo-vllm", "bfcl-v4", "ab", artifact
                    )

        artifact = bfcl_evidence(self.root)
        validation_path = artifact.parent / "complete-validation.json"
        validation = json.loads(validation_path.read_text())
        validation["generation"]["actual_ids_sha256"] = "0" * 64
        write_json(validation_path, validation)
        with self.assertRaisesRegex(EvidenceError, "actual_ids_sha256"):
            build_result(summary_template(), "dynamo-vllm", "bfcl-v4", "ab", artifact)

        artifact = bfcl_evidence(self.root)
        validation_path = artifact.parent / "complete-validation.json"
        validation = json.loads(validation_path.read_text())
        validation["scores"]["categories"]["irrelevance"]["expected_count"] += 1
        write_json(validation_path, validation)
        with self.assertRaisesRegex(EvidenceError, "irrelevance.expected_count"):
            build_result(summary_template(), "dynamo-vllm", "bfcl-v4", "ab", artifact)

    def test_swe_import_requires_generation_and_evaluation_closure(self) -> None:
        artifact = swe_evidence(self.root)
        row = build_result(
            summary_template(), "dynamo-vllm", "swebench-verified", "ab", artifact
        )
        self.assertEqual(row["metrics"]["benchmark_score"], 0.25)
        self.assertEqual(row["completeness"]["generated_units"], 500)
        self.assertEqual(
            [item["role"] for item in row["evidence"]["sources"]],
            [
                "swe-score",
                "swe-generation-validation",
                "swe-run-metadata",
                "swe-run-scope",
                "runtime-continuity",
                "swe-environment-freeze",
                "swe-environment-normalized-freeze",
            ],
        )

        generation_path = artifact.parent / "generation-summary.json"
        generation = json.loads(generation_path.read_text())
        generation["infrastructure_error_ids"] = ["instance-10"]
        write_json(generation_path, generation)
        with self.assertRaisesRegex(EvidenceError, "infrastructure_error_ids"):
            build_result(
                summary_template(), "dynamo-vllm", "swebench-verified", "ab", artifact
            )

    def test_swe_import_accepts_valid_empty_patch_failures(self) -> None:
        artifact = swe_evidence(self.root)
        score = json.loads(artifact.read_text())
        empty_id = score["unresolved_ids"].pop()
        score["empty_patch_ids"] = [empty_id]
        score["completed_instances"] -= 1
        write_json(artifact, score)
        row = build_result(
            summary_template(), "dynamo-vllm", "swebench-verified", "ab", artifact
        )
        self.assertEqual(row["metrics"]["failed_instances"], 375)

    def test_swe_import_requires_physical_canonical_environment(self) -> None:
        artifact = swe_evidence(self.root)
        freeze_path = artifact.parent / "environment.freeze.txt"
        metadata_path = artifact.parent / "run-metadata.json"
        generation_path = artifact.parent / "generation-summary.json"
        freeze_path.write_text(freeze_path.read_text() + "shadow-package==1.0\n")
        digest = importer.sha256_file(freeze_path)
        metadata = json.loads(metadata_path.read_text())
        generation = json.loads(generation_path.read_text())
        metadata["python_environment"]["freeze_sha256"] = digest
        generation["python_environment"]["freeze_sha256"] = digest
        write_json(metadata_path, metadata)
        write_json(generation_path, generation)
        with self.assertRaisesRegex(EvidenceError, "differ from constraints.lock"):
            build_result(
                summary_template(),
                "dynamo-vllm",
                "swebench-verified",
                "ab",
                artifact,
            )

    def test_swe_import_rejects_config_pin_dataset_and_source_drift(self) -> None:
        mutations = (
            (
                lambda metadata: metadata["configuration"]["files"][1].__setitem__(
                    "sha256", "0" * 64
                ),
                "configuration",
            ),
            (
                lambda metadata: metadata["pins"]["values"].__setitem__(
                    "SWEBENCH_COMMIT", "0" * 40
                ),
                "run metadata.pins",
            ),
            (
                lambda metadata: metadata["dataset"]["provenance"].__setitem__(
                    "revision", "0" * 40
                ),
                "provenance.revision",
            ),
            (
                lambda metadata: metadata["source"]["repositories"][
                    "mini_swe_agent"
                ].__setitem__("commit", "0" * 40),
                "run metadata.source",
            ),
            (
                lambda metadata: metadata["runtime_binding"]["content"][
                    "deployment"
                ].__setitem__("max_model_len", 262144),
                "runtime_binding.max_model_len",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                artifact = swe_evidence(self.root)
                metadata_path = artifact.parent / "run-metadata.json"
                metadata = json.loads(metadata_path.read_text())
                mutate(metadata)
                resign_runtime_binding(metadata["runtime_binding"])
                write_json(metadata_path, metadata)
                with self.assertRaisesRegex(EvidenceError, message):
                    build_result(
                        summary_template(),
                        "dynamo-vllm",
                        "swebench-verified",
                        "ab",
                        artifact,
                    )

    def test_swe_import_rejects_mixed_scope_ids(self) -> None:
        artifact = swe_evidence(self.root)
        scope_path = artifact.parent / "run-scope.json"
        scope = json.loads(scope_path.read_text())
        scope["target_ids"][-1] = "foreign-instance"
        write_json(scope_path, scope)
        with self.assertRaisesRegex(EvidenceError, "run scope SHA-256"):
            build_result(
                summary_template(), "dynamo-vllm", "swebench-verified", "ab", artifact
            )

    def test_swe_import_rejects_cross_variant_run_identity(self) -> None:
        artifact = swe_evidence(self.root)
        with self.assertRaisesRegex(EvidenceError, "run_name must begin with variant"):
            build_result(
                summary_template(), "vllm-serve", "swebench-verified", "ab", artifact
            )

    def test_terminal_import_recomputes_task_aggregates(self) -> None:
        artifact = terminal_evidence(self.root)
        row = build_result(
            summary_template(), "dynamo-vllm", "terminal-bench-2.1", "ab", artifact
        )
        self.assertEqual(row["metrics"]["passed_attempts"], 50)
        self.assertEqual(row["wall_time_seconds"], 3600.0)
        self.assertAlmostEqual(row["metrics"]["pass_at_1"], 10 / 89, places=8)
        self.assertEqual(
            [source["role"] for source in row["evidence"]["sources"]],
            [
                "terminalbench-summary",
                "runtime-continuity",
                "terminalbench-trials",
                "terminalbench-task-images",
            ],
        )
        task_image_identity = row["suite_identity"]["task_images"]
        self.assertEqual(task_image_identity["task_count"], 89)
        self.assertEqual(task_image_identity["trial_count"], 445)
        self.assertEqual(len(task_image_identity["task_image_map_sha256"]), 64)
        payload = json.loads(artifact.read_text())
        task_image_source = row["evidence"]["sources"][-1]
        self.assertEqual(
            task_image_source["sha256"], payload["input_hashes"]["task_images_sha256"]
        )

        payload["score"]["pass_at_k"]["5"] = 0.9
        write_json(artifact, payload)
        with self.assertRaisesRegex(EvidenceError, r"pass_at_k\[5\]"):
            build_result(
                summary_template(), "dynamo-vllm", "terminal-bench-2.1", "ab", artifact
            )

    def test_terminal_import_rejects_invalid_task_image_evidence(self) -> None:
        cases = (
            (
                "population",
                lambda evidence: evidence["tasks"].pop(),
                "full task population",
            ),
            (
                "task_ref",
                lambda evidence: evidence["tasks"][0]["task_ref"].__setitem__(
                    "ref", "sha256:" + "f" * 64
                ),
                "task_ref",
            ),
            (
                "task_checksum",
                lambda evidence: evidence["tasks"][0].__setitem__(
                    "task_checksum", "invalid"
                ),
                "task_checksum",
            ),
            (
                "requested_ref",
                lambda evidence: evidence["tasks"][0].__setitem__(
                    "requested_ref", "invalid image ref"
                ),
                "requested_ref",
            ),
            (
                "image_id",
                lambda evidence: evidence["tasks"][0].__setitem__(
                    "image_id", "sha256:invalid"
                ),
                "image_id",
            ),
            (
                "repo_digests",
                lambda evidence: evidence["tasks"][0].__setitem__("repo_digests", []),
                "repo_digests",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                artifact = terminal_evidence(self.root / name)
                payload = json.loads(artifact.read_text())
                task_images_path = artifact.parent / "task-images.json"
                evidence = json.loads(task_images_path.read_text())
                mutate(evidence)
                write_json(task_images_path, evidence)
                payload["task_images"] = evidence
                payload["input_hashes"]["task_images_sha256"] = importer.sha256_file(
                    task_images_path
                )
                write_json(artifact, payload)
                with self.assertRaisesRegex(EvidenceError, message):
                    build_result(
                        summary_template(),
                        "dynamo-vllm",
                        "terminal-bench-2.1",
                        "ab",
                        artifact,
                    )

    def test_terminal_import_requires_physical_matching_task_image_evidence(
        self,
    ) -> None:
        artifact = terminal_evidence(self.root / "embedded-mismatch")
        task_images_path = artifact.parent / "task-images.json"
        evidence = json.loads(task_images_path.read_text())
        evidence["tasks"][0]["requested_ref"] += "-different"
        write_json(task_images_path, evidence)
        with self.assertRaisesRegex(EvidenceError, "embedded task_images"):
            build_result(
                summary_template(), "dynamo-vllm", "terminal-bench-2.1", "ab", artifact
            )

        artifact = terminal_evidence(self.root / "input-hash")
        payload = json.loads(artifact.read_text())
        payload["input_hashes"]["task_images_sha256"] = "0" * 64
        write_json(artifact, payload)
        with self.assertRaisesRegex(EvidenceError, "task_images_sha256"):
            build_result(
                summary_template(), "dynamo-vllm", "terminal-bench-2.1", "ab", artifact
            )

        artifact = terminal_evidence(self.root / "symlink")
        task_images_path = artifact.parent / "task-images.json"
        target = artifact.parent / "task-images-target.json"
        task_images_path.rename(target)
        task_images_path.symlink_to(target)
        with self.assertRaisesRegex(EvidenceError, "must be a regular file"):
            build_result(
                summary_template(), "dynamo-vllm", "terminal-bench-2.1", "ab", artifact
            )

    def test_terminal_import_rejects_recipe_and_source_drift(self) -> None:
        mutations = (
            (
                lambda metadata: metadata["run_spec"].__setitem__("temperature", 0.0),
                "run_spec.temperature",
            ),
            (
                lambda metadata: metadata["run_spec"].__setitem__(
                    "max_context_tokens", 131072
                ),
                "run_spec.max_context_tokens",
            ),
            (
                lambda metadata: metadata["pins"].__setitem__(
                    "harbor_commit", "0" * 40
                ),
                "pins.harbor_commit",
            ),
            (
                lambda metadata: metadata["source"].__setitem__("source_clean", False),
                "source.source_clean",
            ),
            (
                lambda metadata: metadata["harbor_environment"]["packages"].append(
                    ["Harbor", "2.0.0"]
                ),
                "duplicate normalized package names",
            ),
            (
                lambda metadata: metadata["runtime_binding"]["content"][
                    "deployment"
                ].__setitem__("image", "unpublished:latest"),
                "runtime_binding.image",
            ),
            (
                lambda metadata: metadata["runtime_binding"]["content"]["deployment"][
                    "pods"
                ].__setitem__(
                    "extra",
                    metadata["runtime_binding"]["content"]["deployment"]["pods"][
                        "worker"
                    ],
                ),
                "pod roles",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                artifact = terminal_evidence(self.root)
                payload = json.loads(artifact.read_text())
                mutate(payload["run_metadata"])
                resign_runtime_binding(payload["run_metadata"]["runtime_binding"])
                write_json(artifact, payload)
                with self.assertRaisesRegex(EvidenceError, message):
                    build_result(
                        summary_template(),
                        "dynamo-vllm",
                        "terminal-bench-2.1",
                        "ab",
                        artifact,
                    )

        artifact = terminal_evidence(self.root)
        payload = json.loads(artifact.read_text())
        command = payload["run_metadata"]["invocations"][0]["command"]
        parser_index = command.index("parser_name=json")
        command[parser_index] = "parser_name=xml"
        write_json(artifact, payload)
        with self.assertRaisesRegex(EvidenceError, "agent-kwarg"):
            build_result(
                summary_template(), "dynamo-vllm", "terminal-bench-2.1", "ab", artifact
            )

        artifact = terminal_evidence(self.root)
        payload = json.loads(artifact.read_text())
        payload["run_metadata"]["pins"]["resolved_dataset"]["task_refs"][0]["ref"] = (
            "sha256:" + "f" * 64
        )
        write_json(artifact, payload)
        with self.assertRaisesRegex(EvidenceError, "task_refs SHA-256"):
            build_result(
                summary_template(), "dynamo-vllm", "terminal-bench-2.1", "ab", artifact
            )

    def test_replace_is_explicit_and_revalidated(self) -> None:
        artifact = bfcl_evidence(self.root)
        summary_path = self.root / "summary.json"
        write_json(summary_path, summary_template())
        update_summary(summary_path, "dynamo-vllm", "bfcl-v4", "ab", artifact)

        compact = json.loads(artifact.read_text())
        compact["categories"] = {"producer_note": "validated rerun"}
        write_json(artifact, compact)
        data, changed = update_summary(
            summary_path,
            "dynamo-vllm",
            "bfcl-v4",
            "ab",
            artifact,
            replace=True,
        )
        self.assertTrue(changed)
        self.assertAlmostEqual(
            data["results"][0]["metrics"]["overall_accuracy"], 0.7305
        )

    def test_paired_disagreements_are_sanitized_and_phase_scoped(self) -> None:
        summary = summary_template()
        records = {
            "dynamo-vllm": [
                {"id": "case-a", "outcome": "passed"},
                {"id": "case-b", "outcome": "failed", "private_path": "/raw/a"},
            ],
            "vllm-serve": [
                {"id": "case-a", "outcome": "failed"},
                {"id": "case-b", "outcome": "failed", "private_path": "/raw/b"},
            ],
        }
        rows = []
        for variant, task_records in records.items():
            logical = f"results/task-level/bfcl-v4/ab/{variant}.jsonl"
            payload = importer.jsonl_payload(task_records)
            output = self.root / "task-level" / "bfcl-v4" / "ab" / f"{variant}.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
            rows.append(
                {
                    "variant": variant,
                    "suite": "bfcl-v4",
                    "phase": "ab",
                    "status": "complete",
                    "task_level": {
                        "path": logical,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "records": 2,
                    },
                }
            )
        summary["results"] = rows
        entries, outputs = importer.build_paired_disagreements(summary, self.root, {})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["disagreement_records"], 1)
        disagreement_payload = next(iter(outputs.values())).decode()
        self.assertEqual(
            json.loads(disagreement_payload),
            {
                "id": "case-a",
                "dynamo_outcome": "passed",
                "native_outcome": "failed",
            },
        )
        self.assertNotIn("private_path", disagreement_payload)


if __name__ == "__main__":
    unittest.main()
