# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track finding history without treating lost coverage as resolution."""

from __future__ import annotations

from typing import Any

from ..common.contracts import Contract, Finding


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
    if previous.get("schema") != "dynamo-native-assessment/v1":
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
