# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track finding history without treating lost coverage as resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..common.contracts import Contract, Finding
from ..common.provenance import digest


def contract_history(previous: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map v1 contract observations; retain moved evidence, never migrate approval."""
    if previous is None or previous["schema"] == "dynamo-native-assessment/v2":
        return previous
    result = deepcopy(previous)
    result["schema"] = "dynamo-native-assessment/v2"
    mappings = []
    moved = []
    # v1 materialized runtime admission vocabulary as if it were a typed field.
    # Preserve those observations as advisory, not permanent schema-coverage debt.
    inferred = {
        (endpoint, path)
        for endpoint, contract in previous.get("contracts", {})
        .get("dynamo", {})
        .get("endpoints", {})
        .items()
        for path, item in contract.get("fields", {}).items()
        if any(
            source.get("symbol") == "PASSTHROUGH_EXTRA_FIELDS"
            for source in item.get("source", [])
        )
    }
    for key in ("findings", "retired_findings"):
        kept = []
        for item in result.get(key, []):
            old_fingerprint = item["fingerprint"]
            if (
                item["aspect"] == "handling"
                or item["category"] == "behavior"
                or (
                    (item["endpoint"], item["path"]) in inferred
                    and not item["aspect"].startswith("native_coverage_")
                )
            ):
                moved.append(item)
                destination = "investigation"
            else:
                for side in ("native", "dynamo"):
                    value = item.get(side)
                    if isinstance(value, dict):
                        value.pop("handling", None)
                    elif isinstance(value, list):
                        for entry in value:
                            if isinstance(entry, dict):
                                entry.pop("handling", None)
                item["fingerprint"] = digest(
                    {"native": item["native"], "dynamo": item["dynamo"]}
                )
                item["decision"] = None
                item["decision_status"] = "missing"
                kept.append(item)
                destination = "contract"
            mappings.append(
                {
                    "identity": item["identity"],
                    "from_fingerprint": old_fingerprint,
                    "destination": destination,
                    "to_fingerprint": item["fingerprint"],
                    "approval_carried": False,
                }
            )
        result[key] = kept
    result["history_migration"] = {
        "from_schema": previous["schema"],
        "mappings": mappings,
        "legacy_investigation": moved,
    }
    return result


def assessable(
    old: dict[str, Any], current: list[Finding], native: Contract, dynamo: Contract
) -> bool:
    """A disappeared finding is resolved only if its comparison remains covered."""
    endpoint, path = old["endpoint"], old["path"]
    related_paths = {path}
    if old["aspect"] == "placement_candidate" and isinstance(old.get("dynamo"), dict):
        for item in old["dynamo"].get("nested_inputs", []):
            related_paths.add(".".join(item["wire_path"]))
    for contract in (native, dynamo):
        scope = contract.endpoints.get(endpoint)
        if scope is None or not scope.fields_complete:
            return False
    return not any(
        item.category == "coverage"
        and item.endpoint == endpoint
        and (
            item.path == "*"
            or any(
                item.path == related
                or related.startswith(item.path + ".")
                or item.path.startswith(related + ".")
                for related in related_paths
            )
        )
        for item in current
    )


def classify(
    findings: list[Finding],
    previous: dict[str, Any] | None,
    native: Contract,
    dynamo: Contract,
) -> list[dict[str, Any]]:
    """Return retired findings without erasing lost-coverage history."""
    if previous is None:
        return []
    if previous.get("schema") != "dynamo-native-assessment/v2":
        raise ValueError("previous assessment is not a direct comparison report")
    old = {
        item["identity"]: item
        for item in (
            previous["findings"]
            + [
                item
                for item in previous.get("retired_findings", [])
                if item["lifecycle"] == "no_longer_assessable"
            ]
        )
    }
    current_ids = {item.identity for item in findings}
    for item in findings:
        prior = old.get(item.identity)
        if prior is not None:
            item.lifecycle = (
                "unchanged" if prior["fingerprint"] == item.fingerprint else "changed"
            )
    return [
        {
            **item,
            "lifecycle": "resolved"
            if assessable(item, findings, native, dynamo)
            else "no_longer_assessable",
        }
        for identity, item in sorted(old.items())
        if identity not in current_ids
    ]
