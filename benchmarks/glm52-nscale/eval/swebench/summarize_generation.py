#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize prediction and trajectory completeness without discarding raw artifacts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from manage_scope import load_dataset_ids, load_scope  # noqa: E402
from runtime_binding import canonical_sha256, validate_deployment  # noqa: E402


EXPECTED_MINI_VERSION = "2.4.4"
EXPECTED_TRAJECTORY_FORMAT = "mini-swe-agent-1.1"
VALID_MODEL_EXIT_STATUSES = {
    "Submitted",
    "LimitsExceeded",
    "TimeExceeded",
    "RepeatedFormatError",
}
EXPECTED_CONFIG_TYPES = {
    "agent_type": "minisweagent.run.benchmarks.utils.common.ProgressTrackingAgent",
    "environment_type": "minisweagent.environments.docker.DockerEnvironment",
    "model_type": "minisweagent.models.litellm_model.LitellmModel",
}
SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
REPO_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_drift(current: Any, expected: Any, prefix: str) -> list[str]:
    if isinstance(current, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(current.keys() | expected.keys()):
            child = f"{prefix}.{key}"
            if key not in current or key not in expected:
                differences.append(child)
            else:
                differences.extend(describe_drift(current[key], expected[key], child))
        return differences
    return [] if current == expected else [prefix]


def validate_trajectory(
    trajectory: Any,
    prediction: dict[str, Any],
    instance_id: str,
    expected_config: dict[str, Any],
    expected_image: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {
        "exit_status": "invalid_trajectory",
        "api_calls": 0,
        "effective_config_sha256": None,
    }
    if not isinstance(trajectory, dict):
        return ["trajectory is not a JSON object"], evidence
    if trajectory.get("trajectory_format") != EXPECTED_TRAJECTORY_FORMAT:
        failures.append(f"trajectory_format is not {EXPECTED_TRAJECTORY_FORMAT!r}")
    if trajectory.get("instance_id") != instance_id:
        failures.append("trajectory instance_id does not match prediction key")

    info = trajectory.get("info")
    if not isinstance(info, dict):
        return [*failures, "trajectory info is not a JSON object"], evidence
    if info.get("mini_version") != EXPECTED_MINI_VERSION:
        failures.append(f"info.mini_version is not {EXPECTED_MINI_VERSION!r}")
    status = info.get("exit_status")
    evidence["exit_status"] = status or "missing_exit_status"
    if status not in VALID_MODEL_EXIT_STATUSES:
        failures.append(f"exit_status {status!r} is not a clean model terminal status")
    for field in ("traceback", "exception_str"):
        if info.get(field):
            failures.append(
                f"trajectory records infrastructure exception field {field}"
            )

    model_stats = info.get("model_stats")
    calls = model_stats.get("api_calls") if isinstance(model_stats, dict) else None
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 1:
        failures.append("trajectory has no positive integer model_stats.api_calls")
    else:
        evidence["api_calls"] = calls

    submission = info.get("submission")
    patch = prediction.get("model_patch")
    if not isinstance(patch, str):
        failures.append("prediction model_patch is not a string")
        patch = ""
    if not isinstance(submission, str):
        failures.append("trajectory info.submission is not a string")
    elif submission != patch:
        failures.append("trajectory submission does not match prediction model_patch")

    config = info.get("config")
    if not isinstance(config, dict):
        failures.append("trajectory info.config is not a JSON object")
    else:
        for component in ("agent", "environment", "model"):
            if not isinstance(config.get(component), dict):
                failures.append(f"trajectory is missing info.config.{component}")
        model_config = config.get("model", {})
        trajectory_model = (
            model_config.get("model_name") if isinstance(model_config, dict) else None
        )
        if trajectory_model != prediction.get("model_name_or_path"):
            failures.append(
                "trajectory model_name does not match prediction model_name_or_path"
            )
        expected_keys = {"agent", "environment", "model", *EXPECTED_CONFIG_TYPES}
        if set(config) != expected_keys:
            failures.append(
                "trajectory info.config keys differ from the pinned effective schema"
            )
        for field, expected_type in EXPECTED_CONFIG_TYPES.items():
            if config.get(field) != expected_type:
                failures.append(f"trajectory {field} is not {expected_type!r}")
        environment = config.get("environment")
        normalized_environment = (
            {key: value for key, value in environment.items() if key != "image"}
            if isinstance(environment, dict)
            else {}
        )
        normalized_config = {
            "agent": config.get("agent"),
            "environment": normalized_environment,
            "model": config.get("model"),
        }
        evidence["effective_config_sha256"] = canonical_sha256(normalized_config)
        drift = describe_drift(normalized_config, expected_config, "info.config")
        if drift:
            failures.append(
                "trajectory effective config differs from immutable binding: "
                + ", ".join(drift)
            )
        if not isinstance(environment, dict) or environment.get(
            "image"
        ) != expected_image.get("requested_ref"):
            failures.append(
                "trajectory environment.image differs from immutable task-image evidence"
            )

    messages = trajectory.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        failures.append("trajectory does not contain the expected message history")
    elif not all(isinstance(message, dict) for message in messages):
        failures.append("trajectory messages are not JSON objects")
    else:
        if [messages[0].get("role"), messages[1].get("role")] != ["system", "user"]:
            failures.append("trajectory does not begin with system and user messages")
        final = messages[-1]
        final_extra = final.get("extra")
        if final.get("role") != "exit" or not isinstance(final_extra, dict):
            failures.append("trajectory does not end with an exit message")
        else:
            if final_extra.get("exit_status") != status:
                failures.append(
                    "final message exit_status does not match trajectory info"
                )
            if final_extra.get("submission") != submission:
                failures.append(
                    "final message submission does not match trajectory info"
                )
            for field in ("traceback", "exception_str"):
                if final_extra.get(field):
                    failures.append(
                        f"final message records infrastructure exception field {field}"
                    )
    return failures, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--run-metadata", required=True, type=Path)
    parser.add_argument("--task-images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    predictions_path = args.agent_dir / "preds.json"
    predictions = (
        json.loads(predictions_path.read_text()) if predictions_path.exists() else {}
    )
    if not isinstance(predictions, dict):
        raise TypeError("preds.json must be an instance_id keyed JSON object")
    dataset_ids = set(load_dataset_ids(args.dataset, args.expected))
    scope = load_scope(args.scope, args.dataset, args.expected)
    target_ids = set(scope["target_ids"])
    metadata = json.loads(args.run_metadata.read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 3:
        raise ValueError("run metadata must use schema version 3")
    if metadata.get("scope_sha256") != file_sha256(args.scope):
        raise ValueError("run metadata scope digest does not match run scope")
    dataset_binding = metadata.get("dataset")
    if not isinstance(dataset_binding, dict) or dataset_binding.get(
        "evaluator_jsonl_sha256"
    ) != file_sha256(args.dataset):
        raise ValueError("run metadata dataset digest does not match evaluator dataset")
    python_environment = metadata.get("python_environment")
    if (
        not isinstance(python_environment, dict)
        or python_environment.get("normalized_requirement_count") != 101
    ):
        raise ValueError("run metadata Python environment lock evidence is invalid")
    for field in (
        "constraints_lock_sha256",
        "freeze_sha256",
        "normalized_freeze_sha256",
    ):
        value = python_environment.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"run metadata Python environment {field} is invalid")
    runtime_wrapper = metadata.get("runtime_binding")
    if not isinstance(runtime_wrapper, dict) or set(runtime_wrapper) != {
        "file",
        "deployment_sha256",
        "content_sha256",
        "content",
    }:
        raise ValueError("run metadata has no runtime_binding object")
    runtime_binding = runtime_wrapper.get("content")
    if not isinstance(runtime_binding, dict) or set(runtime_binding) != {
        "deployment",
        "evaluator",
    }:
        raise ValueError(
            "runtime binding must contain deployment and evaluator objects"
        )
    binding_file = runtime_wrapper.get("file")
    if binding_file != "runtime-binding.json":
        raise ValueError("runtime binding file name is not runtime-binding.json")
    binding_path = args.run_metadata.parent / binding_file
    if (
        not binding_path.is_file()
        or json.loads(binding_path.read_text()) != runtime_binding
    ):
        raise ValueError("runtime binding file/content mismatch")
    deployment_binding = validate_deployment(
        runtime_binding.get("deployment"),
        campaign_phase=metadata.get("campaign_phase"),
        endpoint=metadata.get("endpoint"),
    )
    if runtime_wrapper.get("deployment_sha256") != canonical_sha256(deployment_binding):
        raise ValueError("runtime binding deployment digest mismatch")
    if runtime_wrapper.get("content_sha256") != canonical_sha256(runtime_binding):
        raise ValueError("runtime binding full-content digest mismatch")
    if (
        deployment_binding.get("variant") is None
        or deployment_binding.get("campaign_phase") != metadata.get("campaign_phase")
        or deployment_binding.get("served_model_name") != metadata.get("model")
        or deployment_binding.get("max_model_len") != 409600
        or deployment_binding.get("endpoint") != metadata.get("endpoint")
    ):
        raise ValueError("runtime binding deployment contract is inconsistent")
    evaluator_binding = runtime_binding.get("evaluator")
    expected_evaluator_fields = {
        "deployment_source_sha256",
        "runtime_family",
        "runtime_source_revision",
        "dynamo_enabled",
        "tensor_parallel_size",
        "generation",
        "evaluation",
        "effective_config_sha256",
        "effective_config_content_sha256",
        "effective_config",
        "endpoint_evidence",
        "campaign_source",
    }
    if (
        not isinstance(evaluator_binding, dict)
        or set(evaluator_binding) != expected_evaluator_fields
    ):
        raise ValueError("runtime binding evaluator config fields are invalid")
    if evaluator_binding.get("deployment_source_sha256") != runtime_wrapper.get(
        "deployment_sha256"
    ):
        raise ValueError("evaluator deployment source digest mismatch")
    campaign_source = metadata.get("campaign_source")
    if (
        not isinstance(campaign_source, dict)
        or evaluator_binding.get("campaign_source") != campaign_source
        or campaign_source.get("source_commit")
        != deployment_binding.get("recipe", {}).get("source_commit")
    ):
        raise ValueError("campaign evaluator source identity is inconsistent")
    endpoint_binding = evaluator_binding.get("endpoint_evidence")
    endpoint_content = (
        endpoint_binding.get("content") if isinstance(endpoint_binding, dict) else None
    )
    selected_model = (
        endpoint_content.get("selected_model_response")
        if isinstance(endpoint_content, dict)
        else None
    )
    if not isinstance(endpoint_binding, dict) or set(endpoint_binding) != {
        "file_sha256",
        "content_sha256",
        "content",
    }:
        raise ValueError("runtime binding endpoint evidence fields are invalid")
    if (
        not isinstance(endpoint_binding.get("file_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", endpoint_binding["file_sha256"]) is None
    ):
        raise ValueError("runtime binding endpoint evidence file digest is invalid")
    if isinstance(endpoint_content, dict) and endpoint_binding.get(
        "content_sha256"
    ) != canonical_sha256(endpoint_content):
        raise ValueError("runtime binding endpoint evidence digest mismatch")
    if (
        not isinstance(selected_model, dict)
        or selected_model.get("id") != metadata.get("model")
        or selected_model.get("context_window") != 409600
    ):
        raise ValueError("runtime binding endpoint model/context evidence is invalid")
    expected_config = evaluator_binding.get("effective_config")
    if not isinstance(expected_config, dict):
        raise ValueError("runtime binding effective config is missing")
    if evaluator_binding.get("effective_config_content_sha256") != canonical_sha256(
        expected_config
    ):
        raise ValueError("runtime binding effective config digest mismatch")
    generation_binding = evaluator_binding.get("generation")
    if not isinstance(generation_binding, dict) or set(generation_binding) != {
        "workers",
        "batch_size",
    }:
        raise ValueError("runtime binding generation execution config is invalid")
    for field in ("workers", "batch_size"):
        value = generation_binding.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"runtime binding generation {field} is invalid")
    if scope["full_run"] and generation_binding != {"workers": 16, "batch_size": 8}:
        raise ValueError("full-run generation concurrency is not pinned to 16/8")
    evaluation_binding = evaluator_binding.get("evaluation")
    if not isinstance(evaluation_binding, dict) or set(evaluation_binding) != {
        "workers",
        "timeout_seconds",
        "backend",
        "docker_platform",
    }:
        raise ValueError("runtime binding evaluation execution config is invalid")
    for field in ("workers", "timeout_seconds"):
        value = evaluation_binding.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"runtime binding evaluation {field} is invalid")
    if metadata.get("suite") == "pro":
        if evaluation_binding.get("backend") not in {"local", "modal"}:
            raise ValueError("SWE-bench Pro evaluation backend is invalid")
        if scope["full_run"] and evaluation_binding.get("backend") != "local":
            raise ValueError("full SWE-bench Pro runs require local Docker evaluation")
        if (
            scope["full_run"]
            and evaluation_binding.get("docker_platform") != "linux/amd64"
        ):
            raise ValueError("full SWE-bench Pro runs require linux/amd64 evaluation")
    elif (
        evaluation_binding.get("backend") != "official-swebench-docker"
        or evaluation_binding.get("docker_platform") is not None
    ):
        raise ValueError("standard SWE-bench evaluation execution config is invalid")

    task_evidence = json.loads(args.task_images.read_text())
    if not isinstance(task_evidence, dict) or task_evidence.get("schema_version") != 1:
        raise ValueError("task image evidence must use schema version 1")
    expected_task_header = {
        "suite": metadata.get("suite"),
        "dataset_sha256": file_sha256(args.dataset),
        "scope_sha256": file_sha256(args.scope),
    }
    for field, expected_value in expected_task_header.items():
        if task_evidence.get(field) != expected_value:
            raise ValueError(f"task image evidence {field} does not match run metadata")
    task_images = task_evidence.get("images")
    if not isinstance(task_images, dict):
        raise ValueError("task image evidence has no images map")
    missing_task_image_ids = sorted(target_ids - task_images.keys())
    unexpected_task_image_ids = sorted(task_images.keys() - target_ids)
    invalid_task_image_ids = []
    for instance_id, identity in sorted(task_images.items()):
        valid = isinstance(identity, dict)
        if valid:
            content = {
                "image_id": identity.get("image_id"),
                "repo_digests": identity.get("repo_digests"),
            }
            valid = (
                isinstance(identity.get("requested_ref"), str)
                and bool(identity["requested_ref"])
                and isinstance(content["image_id"], str)
                and bool(SHA256_ID.fullmatch(content["image_id"]))
                and isinstance(content["repo_digests"], list)
                and bool(content["repo_digests"])
                and content["repo_digests"] == sorted(set(content["repo_digests"]))
                and all(
                    isinstance(digest, str) and REPO_DIGEST.fullmatch(digest)
                    for digest in content["repo_digests"]
                )
                and identity.get("content_identity_sha256") == canonical_sha256(content)
            )
        if not valid:
            invalid_task_image_ids.append(instance_id)
    statuses: collections.Counter[str] = collections.Counter()
    instances = []
    total_calls = 0
    effective_config_hashes: set[str] = set()

    infrastructure_error_ids = []
    for instance_id in sorted(predictions):
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("prediction keys must be non-empty strings")
        prediction = predictions[instance_id]
        if not isinstance(prediction, dict):
            raise TypeError(f"prediction for {instance_id} must be a JSON object")
        if prediction.get("instance_id") != instance_id:
            raise ValueError(f"prediction key mismatch for {instance_id}")
        trajectory_path = args.agent_dir / instance_id / f"{instance_id}.traj.json"
        trajectory_failures = []
        if not trajectory_path.exists():
            trajectory = {}
            trajectory_failures.append("trajectory file is missing")
            evidence = {"exit_status": "missing_trajectory", "api_calls": 0}
        else:
            try:
                trajectory = json.loads(trajectory_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                trajectory = {}
                trajectory_failures.append(
                    f"trajectory cannot be read as JSON: {type(error).__name__}: {error}"
                )
                evidence = {"exit_status": "invalid_trajectory", "api_calls": 0}
            else:
                validation_failures, evidence = validate_trajectory(
                    trajectory,
                    prediction,
                    instance_id,
                    expected_config,
                    task_images.get(instance_id, {}),
                )
                trajectory_failures.extend(validation_failures)
        status = evidence["exit_status"]
        calls = evidence["api_calls"]
        if evidence.get("effective_config_sha256"):
            effective_config_hashes.add(evidence["effective_config_sha256"])
        patch = prediction.get("model_patch")
        patch = patch if isinstance(patch, str) else ""
        if trajectory_failures:
            infrastructure_error_ids.append(instance_id)
        statuses[status] += 1
        total_calls += calls
        instances.append(
            {
                "instance_id": instance_id,
                "exit_status": status,
                "api_calls": calls,
                "patch_bytes": len(patch.encode()),
                "trajectory": str(trajectory_path),
                "valid_model_result": not trajectory_failures,
                "validation_failures": trajectory_failures,
                "effective_config_sha256": evidence.get("effective_config_sha256"),
                "task_image": task_images.get(instance_id),
                "runtime_deployment_sha256": runtime_wrapper["deployment_sha256"],
                "runtime_content_sha256": runtime_wrapper["content_sha256"],
            }
        )

    missing_ids = sorted(target_ids - predictions.keys())
    unexpected_ids = sorted(predictions.keys() - target_ids)
    gate_failures = []
    if missing_ids:
        gate_failures.append(f"missing {len(missing_ids)} target predictions")
    if unexpected_ids:
        gate_failures.append(f"found {len(unexpected_ids)} unexpected predictions")
    if infrastructure_error_ids:
        gate_failures.append(
            "found "
            f"{len(infrastructure_error_ids)} predictions with infrastructure or "
            "trajectory errors"
        )
    if missing_task_image_ids:
        gate_failures.append(
            f"missing {len(missing_task_image_ids)} task image identities"
        )
    if unexpected_task_image_ids:
        gate_failures.append(
            f"found {len(unexpected_task_image_ids)} unexpected task image identities"
        )
    if invalid_task_image_ids:
        gate_failures.append(
            f"found {len(invalid_task_image_ids)} invalid task image identities"
        )
    expected_config_sha256 = canonical_sha256(expected_config)
    if effective_config_hashes and effective_config_hashes != {expected_config_sha256}:
        gate_failures.append(
            "trajectory effective configs are inconsistent with the immutable binding"
        )

    summary = {
        "scope": scope["scope"],
        "full_run": scope["full_run"],
        "dataset_instances": args.expected,
        "expected_instances": scope["target_instances"],
        "target_instances": scope["target_instances"],
        "excluded_dataset_instances": len(dataset_ids - target_ids),
        "predictions": len(predictions),
        "missing_predictions": len(missing_ids),
        "missing_prediction_ids": missing_ids,
        "unexpected_prediction_ids": unexpected_ids,
        "nonempty_patches": sum(item["patch_bytes"] > 0 for item in instances),
        "empty_patches": sum(item["patch_bytes"] == 0 for item in instances),
        "valid_model_empty_patches": sum(
            item["patch_bytes"] == 0 and item["valid_model_result"]
            for item in instances
        ),
        "infrastructure_error_ids": infrastructure_error_ids,
        "runtime_binding": runtime_wrapper,
        "python_environment": python_environment,
        "effective_config_sha256": expected_config_sha256,
        "trajectory_effective_config_sha256s": sorted(effective_config_hashes),
        "task_image_evidence": {
            "sha256": canonical_sha256(task_evidence),
            "images": task_images,
        },
        "missing_task_image_ids": missing_task_image_ids,
        "unexpected_task_image_ids": unexpected_task_image_ids,
        "invalid_task_image_ids": invalid_task_image_ids,
        "total_api_calls": total_calls,
        "exit_statuses": dict(sorted(statuses.items())),
        "complete": not gate_failures,
        "gate_failures": gate_failures,
        "instances": instances,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.require_complete and gate_failures:
        raise SystemExit(
            "generation completeness gate failed: " + "; ".join(gate_failures)
        )


if __name__ == "__main__":
    main()
