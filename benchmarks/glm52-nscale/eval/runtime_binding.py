#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical privacy-safe serving-runtime binding contract for all harnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
VARIANTS = {
    "dynamo-vllm": {
        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260701-5245c0f@sha256:c3336583c830ea5c3cf4bd5cc92cb57200b8f558398a18c3ac0f473f9b74dd1d",
        "endpoint": "http://glm52-dynamo-vllm-frontend:8000/v1",
        "service_name": "glm52-dynamo-vllm-frontend",
        "controller_kind": "DynamoGraphDeployment",
        "controller_name": "glm52-dynamo-vllm",
        "roles": {"frontend", "worker"},
        "dynamo": True,
    },
    "vllm-serve": {
        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260701-5245c0f@sha256:c3336583c830ea5c3cf4bd5cc92cb57200b8f558398a18c3ac0f473f9b74dd1d",
        "endpoint": "http://glm52-vllm-serve:8000/v1",
        "service_name": "glm52-vllm-serve",
        "controller_kind": "Deployment",
        "controller_name": "glm52-vllm-serve",
        "roles": {"worker"},
        "dynamo": False,
    },
    "dynamo-sglang": {
        "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:20260701-5245c0f@sha256:a60e09976d7bc26812fca94c24df0f76cb579f218bc2e2e689c717a784a1d5e5",
        "endpoint": "http://glm52-dynamo-sglang-frontend:8000/v1",
        "service_name": "glm52-dynamo-sglang-frontend",
        "controller_kind": "DynamoGraphDeployment",
        "controller_name": "glm52-dynamo-sglang",
        "roles": {"frontend", "worker"},
        "dynamo": True,
    },
    "sglang-serve": {
        "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:20260701-5245c0f@sha256:a60e09976d7bc26812fca94c24df0f76cb579f218bc2e2e689c717a784a1d5e5",
        "endpoint": "http://glm52-sglang-serve:8000/v1",
        "service_name": "glm52-sglang-serve",
        "controller_kind": "Deployment",
        "controller_name": "glm52-sglang-serve",
        "roles": {"worker"},
        "dynamo": False,
    },
}


class BindingError(ValueError):
    """Raised when runtime identity does not satisfy the campaign contract."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BindingError(f"{path} must be an object")
    if set(value) != expected:
        raise BindingError(
            f"{path} fields differ: expected {sorted(expected)}, got {sorted(value)}"
        )
    return value


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise BindingError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: Any, path: str) -> datetime:
    if not isinstance(value, str):
        raise BindingError(f"{path} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BindingError(f"{path} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BindingError(f"{path} must include a UTC offset")
    return parsed


def validate_deployment(
    value: Any,
    *,
    variant: str | None = None,
    campaign_phase: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    deployment = _exact_keys(
        value,
        {
            "schema_version",
            "variant",
            "campaign_phase",
            "served_model_name",
            "model_id",
            "model_revision",
            "max_model_len",
            "image",
            "endpoint",
            "service_name",
            "controller",
            "pods",
            "recipe",
            "hardware",
            "control_plane",
            "capture",
        },
        "runtime_binding",
    )
    if (
        isinstance(deployment["schema_version"], bool)
        or deployment["schema_version"] != 1
    ):
        raise BindingError("runtime_binding.schema_version must be 1")
    actual_variant = deployment["variant"]
    if not isinstance(actual_variant, str) or actual_variant not in VARIANTS:
        raise BindingError(f"unknown runtime binding variant: {actual_variant!r}")
    if variant is not None and actual_variant != variant:
        raise BindingError(f"runtime binding variant is not {variant!r}")
    definition = VARIANTS[actual_variant]
    phase = deployment["campaign_phase"]
    if not isinstance(phase, str) or phase not in {"validation", "ab", "ba"}:
        raise BindingError("runtime binding campaign_phase is invalid")
    if campaign_phase is not None and phase != campaign_phase:
        raise BindingError(f"runtime binding campaign_phase is not {campaign_phase!r}")
    expected_scalars = {
        "served_model_name": "zai-org/GLM-5.2",
        "model_id": "nvidia/GLM-5.2-NVFP4",
        "model_revision": "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
        "max_model_len": 409600,
        "image": definition["image"],
        "endpoint": definition["endpoint"],
        "service_name": definition["service_name"],
    }
    for field, expected in expected_scalars.items():
        if deployment[field] != expected:
            raise BindingError(
                f"runtime_binding.{field} must be {expected!r}, "
                f"got {deployment[field]!r}"
            )
    if endpoint is not None and deployment["endpoint"] != endpoint.rstrip("/"):
        raise BindingError("runtime binding endpoint differs from requested endpoint")

    controller = _exact_keys(
        deployment["controller"],
        {"kind", "name", "uid_sha256", "generation"},
        "runtime_binding.controller",
    )
    if controller["kind"] != definition["controller_kind"]:
        raise BindingError("runtime binding controller kind is wrong")
    if controller["name"] != definition["controller_name"]:
        raise BindingError("runtime binding controller name is wrong")
    _digest(controller["uid_sha256"], "runtime_binding.controller.uid_sha256")
    if (
        isinstance(controller["generation"], bool)
        or not isinstance(controller["generation"], int)
        or controller["generation"] <= 0
    ):
        raise BindingError("runtime binding controller generation must be positive")

    pods = deployment["pods"]
    if not isinstance(pods, dict) or set(pods) != definition["roles"]:
        raise BindingError(
            f"runtime binding pod roles must be {sorted(definition['roles'])}"
        )
    pod_fields = {
        "name_sha256",
        "uid_sha256",
        "node_name_sha256",
        "image_id",
        "argv_sha256",
        "model_manifest_sha256",
    }
    for role, raw_pod in pods.items():
        pod = _exact_keys(raw_pod, pod_fields, f"runtime_binding.pods.{role}")
        for field in pod_fields - {"image_id"}:
            _digest(pod[field], f"runtime_binding.pods.{role}.{field}")
        if (
            not isinstance(pod["image_id"], str)
            or IMAGE_ID_RE.fullmatch(pod["image_id"]) is None
        ):
            raise BindingError(f"runtime_binding.pods.{role}.image_id is invalid")

    recipe = _exact_keys(
        deployment["recipe"],
        {"source_commit", "template_sha256", "rendered_manifest_sha256"},
        "runtime_binding.recipe",
    )
    if (
        not isinstance(recipe["source_commit"], str)
        or COMMIT_RE.fullmatch(recipe["source_commit"]) is None
    ):
        raise BindingError("runtime binding recipe source_commit is invalid")
    _digest(recipe["template_sha256"], "runtime_binding.recipe.template_sha256")
    _digest(
        recipe["rendered_manifest_sha256"],
        "runtime_binding.recipe.rendered_manifest_sha256",
    )

    hardware = _exact_keys(
        deployment["hardware"],
        {
            "gpu_count",
            "gpu_model",
            "gpu_uuid_set_sha256",
            "driver_version",
            "gpu_memory_total_mib",
            "kernel_version",
            "kubelet_version",
            "container_runtime_version",
        },
        "runtime_binding.hardware",
    )
    if hardware["gpu_count"] != 4 or hardware["gpu_model"] != "NVIDIA B200":
        raise BindingError("runtime binding hardware must be 4x NVIDIA B200")
    _digest(
        hardware["gpu_uuid_set_sha256"],
        "runtime_binding.hardware.gpu_uuid_set_sha256",
    )
    for field in (
        "driver_version",
        "kernel_version",
        "kubelet_version",
        "container_runtime_version",
    ):
        if not isinstance(hardware[field], str) or not hardware[field]:
            raise BindingError(f"runtime_binding.hardware.{field} must be non-empty")
    memory = hardware["gpu_memory_total_mib"]
    if (
        not isinstance(memory, list)
        or not memory
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in memory
        )
    ):
        raise BindingError("runtime binding GPU memory identity is invalid")

    control_plane = deployment["control_plane"]
    if definition["dynamo"]:
        control_plane = _exact_keys(
            control_plane,
            {"dynamo_operator_image_digests", "grove_operator_image_digests"},
            "runtime_binding.control_plane",
        )
        for field, digests in control_plane.items():
            if not isinstance(digests, list) or not digests:
                raise BindingError(f"runtime_binding.control_plane.{field} is empty")
            for index, digest in enumerate(digests):
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
                ):
                    raise BindingError(
                        f"runtime_binding.control_plane.{field}[{index}] is invalid"
                    )
            if digests != sorted(set(digests)):
                raise BindingError(
                    f"runtime_binding.control_plane.{field} must be sorted and unique"
                )
    elif control_plane is not None:
        raise BindingError("native runtime binding control_plane must be null")

    capture = _exact_keys(
        deployment["capture"],
        {"sha256", "captured_at"},
        "runtime_binding.capture",
    )
    _digest(capture["sha256"], "runtime_binding.capture.sha256")
    _timestamp(capture["captured_at"], "runtime_binding.capture.captured_at")
    return deployment


def make_wrapper(
    deployment: Any,
    *,
    evaluator: dict[str, Any] | None,
    variant: str | None = None,
    campaign_phase: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    deployment = validate_deployment(
        deployment,
        variant=variant,
        campaign_phase=campaign_phase,
        endpoint=endpoint,
    )
    if evaluator is not None and not isinstance(evaluator, dict):
        raise BindingError("runtime binding evaluator must be an object or null")
    content = {"deployment": deployment, "evaluator": evaluator}
    return {
        "file": "runtime-binding.json",
        "deployment_sha256": canonical_sha256(deployment),
        "content_sha256": canonical_sha256(content),
        "content": content,
    }


def _validate_snapshot(
    value: Any, deployment: dict[str, Any], path: str
) -> dict[str, Any]:
    snapshot = _exact_keys(value, {"controller", "pods"}, path)
    controller = _exact_keys(
        snapshot["controller"],
        {"kind", "name", "uid_sha256", "generation"},
        f"{path}.controller",
    )
    _digest(controller["uid_sha256"], f"{path}.controller.uid_sha256")
    if (
        isinstance(controller["generation"], bool)
        or not isinstance(controller["generation"], int)
        or controller["generation"] <= 0
    ):
        raise BindingError(f"{path}.controller.generation must be positive")
    if controller != deployment["controller"]:
        raise BindingError(f"{path}.controller differs from deployment binding")
    pods = snapshot["pods"]
    if not isinstance(pods, dict) or set(pods) != set(deployment["pods"]):
        raise BindingError(f"{path}.pods roles differ from deployment binding")
    fields = {
        "name_sha256",
        "uid_sha256",
        "node_name_sha256",
        "image_id",
        "container_id_sha256",
        "restart_count",
    }
    for role, raw_pod in pods.items():
        pod = _exact_keys(raw_pod, fields, f"{path}.pods.{role}")
        for field in (
            "name_sha256",
            "uid_sha256",
            "node_name_sha256",
            "container_id_sha256",
        ):
            _digest(pod[field], f"{path}.pods.{role}.{field}")
        if (
            not isinstance(pod["image_id"], str)
            or IMAGE_ID_RE.fullmatch(pod["image_id"]) is None
        ):
            raise BindingError(f"{path}.pods.{role}.image_id is invalid")
        if (
            isinstance(pod["restart_count"], bool)
            or not isinstance(pod["restart_count"], int)
            or pod["restart_count"] != 0
        ):
            raise BindingError(f"{path}.pods.{role}.restart_count must be zero")
        deployment_pod = deployment["pods"][role]
        for field in ("name_sha256", "uid_sha256", "node_name_sha256", "image_id"):
            if pod[field] != deployment_pod[field]:
                raise BindingError(
                    f"{path}.pods.{role}.{field} differs from deployment binding"
                )
    return snapshot


def validate_continuity(
    value: Any, deployment: Any, *, require_success: bool = True
) -> dict[str, Any]:
    deployment = validate_deployment(deployment)
    continuity = _exact_keys(
        value,
        {
            "schema_version",
            "variant",
            "campaign_phase",
            "deployment_sha256",
            "command_exit_code",
            "stable",
            "pre_captured_at",
            "post_captured_at",
            "pre",
            "post",
        },
        "runtime_continuity",
    )
    if (
        isinstance(continuity["schema_version"], bool)
        or continuity["schema_version"] != 1
    ):
        raise BindingError("runtime_continuity.schema_version must be 1")
    if continuity["variant"] != deployment["variant"]:
        raise BindingError("runtime continuity variant differs from deployment")
    if continuity["campaign_phase"] != deployment["campaign_phase"]:
        raise BindingError("runtime continuity phase differs from deployment")
    if continuity["deployment_sha256"] != canonical_sha256(deployment):
        raise BindingError("runtime continuity deployment digest is wrong")
    exit_code = continuity["command_exit_code"]
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or not 0 <= exit_code <= 255
    ):
        raise BindingError("runtime continuity command_exit_code is invalid")
    pre_captured_at = _timestamp(
        continuity["pre_captured_at"], "runtime_continuity.pre_captured_at"
    )
    post_captured_at = _timestamp(
        continuity["post_captured_at"], "runtime_continuity.post_captured_at"
    )
    if post_captured_at < pre_captured_at:
        raise BindingError("runtime continuity timestamps are out of order")
    pre = _validate_snapshot(continuity["pre"], deployment, "runtime_continuity.pre")
    post = _validate_snapshot(continuity["post"], deployment, "runtime_continuity.post")
    if pre != post:
        raise BindingError("runtime continuity pre/post snapshots differ")
    if continuity["stable"] is not True:
        raise BindingError("runtime continuity stable marker contradicts snapshots")
    if require_success:
        if exit_code != 0:
            raise BindingError("runtime continuity command did not exit zero")
    return continuity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--phase", choices=("validation", "ab", "ba"))
    parser.add_argument("--endpoint")
    parser.add_argument("--continuity", type=Path)
    parser.add_argument("--allow-command-failure", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        deployment = json.loads(args.binding.read_text())
        wrapper = make_wrapper(
            deployment,
            evaluator=None,
            variant=args.variant,
            campaign_phase=args.phase,
            endpoint=args.endpoint,
        )
        if args.continuity:
            validate_continuity(
                json.loads(args.continuity.read_text()),
                wrapper["content"]["deployment"],
                require_success=not args.allow_command_failure,
            )
    except (OSError, json.JSONDecodeError, BindingError) as error:
        parser.error(str(error))
    rendered = json.dumps(wrapper, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
