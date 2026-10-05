# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assess Dynamo directly against an explicitly pinned vLLM request contract.

Exit 0: extraction/review gates satisfied (not runtime parity or release approval).
Exit 1: assessment ran; review, required coverage, or support-policy action remains.
Exit 2: invalid inputs, tool/Git failure. Pins are never modified by this command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from protocol_assessment_inputs import object_value, validate_previous
from protocol_assessment_report import render_assessment
from protocol_comparison import Finding, build_assessment, finding
from protocol_drift import digest, tool_provenance
from protocol_dynamo_contract import extract_dynamo
from protocol_native_contract import extract_native


def selected_behavior(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Request-class methods plus validator/normalizer helpers in selected B1 files.

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
        for item in previous["findings"]
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


def read_json(path: Path | None) -> dict[str, Any] | None:
    return object_value(json.loads(path.read_text()), str(path)) if path else None


def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "output directory must be absent or empty; preserve prior assessments"
        )
    previous = read_json(args.previous)
    decisions = read_json(args.decisions)
    policy = read_json(args.support_policy)
    # Exact commits are required by both extractors. A candidate comparison is
    # read-only: no pin adoption, Git checkout, or engine execution occurs.
    native, snapshot = extract_native(args.upstream_repo, args.upstream_commit)
    dynamo = extract_dynamo(args.dynamo_repo, args.dynamo_commit, args.crate_cache)
    scope = {
        "target": "vllm",
        "endpoints": sorted(native.endpoints),
        "pipeline": args.pipeline,
        "level": "request_fields_and_selected_source_behavior",
    }
    if previous is not None:
        validate_previous(previous, scope)
    selected = selected_behavior(snapshot)
    report = build_assessment(
        native,
        dynamo,
        scope=scope,
        previous=previous,
        decisions=decisions,
        behavior=behavior_findings(selected, previous),
        policy=policy,
    )
    report["selected_behavior"] = selected
    report["provenance"] = {
        **tool_provenance(
            "assess_protocol_compatibility.py",
            "protocol_contract.py",
            "protocol_comparison.py",
            "protocol_native_contract.py",
            "protocol_dynamo_contract.py",
            "protocol_dynamo_handling.py",
            "protocol_rust_source.py",
            "protocol_assessment_report.py",
            "protocol_assessment_inputs.py",
            "protocol_drift.py",
            "protocol_dependencies.py",
            "generate_protocol_inventory.py",
        ),
        "native_snapshot_sha256": digest(snapshot),
        "previous_assessment_sha256": digest(previous) if previous else None,
        "decision_registry_sha256": digest(decisions) if args.decisions else None,
        "support_policy_sha256": digest(policy) if args.support_policy else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "report.json": report,
        "native-contract.json": native.to_dict(),
        "dynamo-contract.json": dynamo.to_dict(),
        "native-source.json": snapshot,
    }.items():
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    (args.output_dir / "report.md").write_text(render_assessment(report))
    print(
        f'Extraction: {report["gates"]["extraction"]}; review: {report["gates"]["review"]}; '
        f'{len(report["findings"])} findings. Runtime parity not established. See {args.output_dir / "report.md"}'
    )
    return report["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamo-repo", type=Path, required=True)
    parser.add_argument("--dynamo-commit", required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument(
        "--crate-cache",
        type=Path,
        help="Cargo .crate archives, verified against selected Cargo.lock",
    )
    parser.add_argument(
        "--pipeline", choices=("token", "text", "unspecified"), default="unspecified"
    )
    parser.add_argument(
        "--previous", type=Path, help="previous direct-assessment report.json"
    )
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--support-policy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (
        OSError,
        SyntaxError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Assessment tool error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
