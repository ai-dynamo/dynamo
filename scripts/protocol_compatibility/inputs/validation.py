# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate persisted assessment inputs before they affect history or gates.

These are direct-comparison schemas, deliberately separate from B1's legacy
upstream-pair registry. Unknown fields are retained for forward-compatible
annotations; every field that controls a gate is validated explicitly.
"""

from __future__ import annotations

import re
from typing import Any


def object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    return value


def text_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty text")
    return value


def list_value(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected an array")
    return value


def revision_pair(value: Any, label: str) -> None:
    pair = object_value(value, label)
    if set(pair) != {"dynamo", "vllm"}:
        raise ValueError(f"{label}: exact Dynamo and vLLM revisions required")
    for target, sha in pair.items():
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"{label}.{target}: full lowercase commit SHA required")


def fingerprint(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label}: SHA256 fingerprint required")


def validate_previous(previous: Any, scope: dict[str, Any]) -> None:
    previous = object_value(previous, "previous assessment")
    if previous.get("schema") not in {
        "dynamo-native-assessment/v1",
        "dynamo-native-assessment/v2",
    }:
        raise ValueError("previous assessment is not a direct comparison report")
    revision_pair(previous.get("revisions"), "previous revisions")
    previous_scope = dict(object_value(previous.get("scope"), "previous scope"))
    if (
        previous["schema"] == "dynamo-native-assessment/v1"
        and previous_scope.get("level") == "request_fields_and_selected_source_behavior"
    ):
        previous_scope["level"] = "request_contract"
    if previous_scope != scope:
        raise ValueError("previous assessment scope differs; establish a new baseline")
    identities = set()
    for key in ("findings", "retired_findings"):
        for item in list_value(previous.get(key, []), f"previous {key}"):
            item = object_value(item, f"previous {key} item")
            for field in ("identity", "endpoint", "path", "aspect", "observation"):
                text_value(item.get(field), f"previous finding {field}")
            identity = item["identity"]
            if identity in identities:
                raise ValueError(f"duplicate previous finding: {identity}")
            identities.add(identity)
            fingerprint(item.get("fingerprint"), f"previous finding {identity}")
            expected = f'{item["endpoint"]}#{item["path"]}:{item["aspect"]}'
            if identity != expected:
                raise ValueError(
                    f"previous finding identity does not match scope: {identity}"
                )
            if item.get("category") not in {
                "compatibility",
                "coverage",
                "dynamo_specific",
                "behavior",
            }:
                raise ValueError(f"invalid previous finding category: {identity}")
            if previous["schema"].endswith("/v2") and (
                item["category"] == "behavior" or item["aspect"] == "handling"
            ):
                raise ValueError(
                    "v2 contract history cannot contain investigation notes"
                )
            allowed = (
                {"new", "changed", "unchanged"}
                if key == "findings"
                else {"resolved", "no_longer_assessable"}
            )
            if item.get("lifecycle") not in allowed:
                raise ValueError(f"invalid previous finding lifecycle: {identity}")
    if "findings" not in previous:
        raise ValueError("previous assessment: findings required")
    selected = object_value(previous.get("selected_behavior", {}), "selected behavior")
    for key, group in selected.items():
        group = object_value(group, f"behavior group {key}")
        for field in ("endpoint", "path", "symbol"):
            text_value(group.get(field), f"behavior group {key}.{field}")
        object_value(group.get("methods"), f"behavior group {key}.methods")
        list_value(
            group.get("affected_fields"), f"behavior group {key}.affected_fields"
        )


def validate_registry(registry: Any, dispositions: frozenset[str]) -> None:
    registry = object_value(registry, "decision registry")
    if registry == {"schema": "dynamo-native-decisions/v1", "decisions": []}:
        return  # Empty historical registry grants no approvals.
    if (
        registry.get("schema") != "dynamo-native-decisions/v2"
        or registry.get("layer") != "contract"
    ):
        raise ValueError(
            "use dynamo-native-decisions/v2 with layer=contract; re-review legacy approvals"
        )
    identities = set()
    for record in list_value(registry.get("decisions"), "decisions"):
        record = object_value(record, "decision")
        identity = text_value(record.get("identity"), "decision identity")
        if identity.endswith(":handling") or "#@behavior/" in identity:
            raise ValueError("investigation notes cannot approve or block contracts")
        if identity in identities:
            raise ValueError(f"duplicate decision: {identity}")
        identities.add(identity)
        fingerprint(record.get("fingerprint"), f"decision {identity}")
        for key in ("rationale", "owner", "next_action"):
            text_value(record.get(key), f"decision {identity}.{key}")
        disposition = text_value(
            record.get("disposition"), f"decision {identity}.disposition"
        )
        if disposition not in dispositions:
            raise ValueError(f"decision {identity}: invalid disposition")
        if disposition in {"tracked_gap", "needs_runtime_evidence"}:
            text_value(record.get("tracking"), f"decision {identity}.tracking")
        object_value(record.get("scope"), f"decision {identity}.scope")
        revision_pair(
            record.get("reviewed_revisions"), f"decision {identity}.reviewed_revisions"
        )
        if type(record.get("carry_static_disposition", False)) is not bool:
            raise ValueError(
                f"decision {identity}: carry_static_disposition must be boolean"
            )
        evidence = list_value(record.get("evidence"), f"decision {identity}.evidence")
        if not evidence:
            raise ValueError(f"decision {identity}: evidence required")
        for entry in evidence:
            entry = object_value(entry, f"decision {identity} evidence")
            if entry.get("kind") not in ("source", "implementation", "runtime"):
                raise ValueError(f"decision {identity}: unknown evidence kind")
            text_value(entry.get("url"), f"decision {identity} evidence URL")
            if entry["kind"] == "runtime":
                revision_pair(
                    entry.get("revisions"), f"decision {identity} evidence revisions"
                )
                object_value(entry.get("scope"), f"decision {identity} evidence scope")


def validate_policy(policy: Any) -> None:
    policy = object_value(policy, "support policy")
    if policy.get("schema") != "dynamo-native-support-policy/v1":
        raise ValueError("unsupported support policy")
    required = list_value(
        policy.get("required_findings_absent"), "required_findings_absent"
    )
    for identity in required:
        text_value(identity, "required finding identity")
        if identity.endswith(":handling") or "#@behavior/" in identity:
            raise ValueError(
                "support policy must reference contract findings, not investigation"
            )
    if len(required) != len(set(required)):
        raise ValueError("duplicate support-policy finding identity")
