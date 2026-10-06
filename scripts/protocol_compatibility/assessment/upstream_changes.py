# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected source-change signals; not proof of changed runtime behavior."""

from __future__ import annotations

from typing import Any

from ..common.contracts import Finding, finding


def selected_behavior(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Request-class methods plus validator/normalizer helpers in selected sources.

    No whole-module bindings or unrelated helper changes require disposition.
    Class methods are grouped per declaring request class; affected field lists
    are candidate scope, not a claim of dataflow or behavioral impact.
    """
    result = {}
    requests = {
        "ChatCompletionRequest": "/v1/chat/completions",
        "CompletionRequest": "/v1/completions",
    }
    for path, module in snapshot["modules"].items():
        for name, contract in module["contract"]["classes"].items():
            if name not in requests:
                continue
            result[path + ":" + name] = {
                "endpoint": requests[name],
                "path": path,
                "symbol": name,
                "methods": dict(contract["methods"]),
                "affected_fields": sorted(contract["fields"]),
            }
        helpers = {
            name: value
            for name, value in module["contract"]["functions"].items()
            if name.startswith(("validate_", "_validate_", "normalize_", "_normalize_"))
        }
        if helpers and "/entrypoints/" in path:
            for endpoint in requests.values():
                result[path + ":helpers:" + endpoint] = {
                    "endpoint": endpoint,
                    "path": path,
                    "symbol": "validator/normalizer helpers",
                    "methods": helpers,
                    "affected_fields": ["unresolved helper-to-field dataflow"],
                }
    return result


def behavior_findings(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> list[Finding]:
    if previous is None:
        return []
    old = previous.get("selected_behavior", {})
    prior_findings = {
        item["identity"]: item
        for item in (
            previous["findings"] + previous.get("investigation", {}).get("notes", [])
        )
        if item["category"] == "behavior"
    }
    results = []
    for key in sorted(set(current) | set(old)):
        after, before = current.get(key), old.get(key)
        reference = after or before
        item = finding(
            reference["endpoint"],
            "@behavior/" + key,
            "implementation",
            "Selected upstream request validator/normalizer/helper implementation changed; runtime impact is unverified.",
            after,
            None,
            category="behavior",
            sources=[
                {
                    "path": reference["path"],
                    "symbol": reference["symbol"],
                    "target": "vllm",
                }
            ],
        )
        changed = (
            before is None or after is None or before["methods"] != after["methods"]
        )
        prior = prior_findings.get(item.identity)
        if changed or (prior is not None and prior["fingerprint"] == item.fingerprint):
            results.append(item)
    return results
