# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestrate direct and configured-pin assessments without adopting source pins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common.provenance import digest, tool_provenance
from ..common.source import commit
from ..extraction.dynamo import extract_dynamo
from ..extraction.vllm import extract_native
from ..inputs.revisions import selected_pin
from ..inputs.validation import object_value, validate_previous
from ..reporting.report import code, render_assessment
from .policy import build_assessment


def read_json(path: Path | None) -> dict[str, Any] | None:
    return object_value(json.loads(path.read_text()), str(path)) if path else None


def assess(args: argparse.Namespace) -> int:
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
        "level": "request_contract",
    }
    if previous is not None:
        validate_previous(previous, scope)
    report = build_assessment(
        native,
        dynamo,
        scope=scope,
        previous=previous,
        decisions=decisions,
        policy=policy,
    )
    # Retain old advisory history for a separately installed investigation layer.
    report["selected_behavior"] = (
        previous.get("selected_behavior", {}) if previous else {}
    )
    report["provenance"] = {
        **tool_provenance(),
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
        f'Contract extraction: {report["gates"]["extraction"]}; review: {report["gates"]["review"]}; '
        f'{sum(item["category"] == "compatibility" for item in report["findings"])} declared differences; '
        f'{sum(item["category"] == "coverage" for item in report["findings"])} contract coverage gaps; '
        "Source investigation: not implemented. "
        f'Behavioral conformance: not assessed. See {args.output_dir / "report.md"}'
    )
    return report["exit_code"]


def run_configured(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("workflow output directory must be absent or empty")
    if args.previous and args.baseline_dynamo:
        raise ValueError("choose --previous or --baseline-dynamo, not both")
    current_sha = commit(args.dynamo_repo, args.dynamo_commit)
    current_pin = selected_pin(args.dynamo_repo, current_sha, args.platform)
    native_sha = commit(
        args.upstream_repo, args.upstream_candidate or current_pin["commit"]
    )
    previous = args.previous
    baseline_pin = None
    # A periodic candidate is compared against the configured version at the
    # same Dynamo revision unless an explicit retained assessment is supplied.
    baseline_sha = args.baseline_dynamo or (
        current_sha if args.upstream_candidate and not previous else None
    )
    common = {
        "dynamo_repo": args.dynamo_repo,
        "upstream_repo": args.upstream_repo,
        "crate_cache": args.crate_cache,
        "pipeline": args.pipeline,
        "decisions": args.decisions,
        "support_policy": args.support_policy,
    }
    if baseline_sha:
        baseline_sha = commit(args.dynamo_repo, baseline_sha)
        baseline_pin = selected_pin(args.dynamo_repo, baseline_sha, args.platform)
        baseline_dir = args.output_dir / "baseline"
        assess(
            argparse.Namespace(
                **common,
                dynamo_commit=baseline_sha,
                upstream_commit=baseline_pin["commit"],
                previous=None,
                output_dir=baseline_dir,
            )
        )
        previous = baseline_dir / "report.json"
    current_dir = args.output_dir / "current"
    result = assess(
        argparse.Namespace(
            **common,
            dynamo_commit=current_sha,
            upstream_commit=native_sha,
            previous=previous,
            output_dir=current_dir,
        )
    )
    workflow = {
        "schema": "dynamo-native-workflow/v1",
        "platform": args.platform,
        "dynamo_commit": current_sha,
        "configured_pin": current_pin,
        "baseline_dynamo_commit": baseline_sha,
        "baseline_pin": baseline_pin,
        "candidate_commit": native_sha,
        "candidate_adopted": False,
        "mode": "candidate"
        if args.upstream_candidate
        else "revision_comparison"
        if previous
        else "baseline",
        "current_report": "current/report.json",
        "exit_code": result,
        "provenance": tool_provenance(),
    }
    (args.output_dir / "workflow.json").write_text(
        json.dumps(workflow, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "report.md").write_text(
        "# Dynamo–vLLM assessment workflow\n\n"
        f"Platform: {code(args.platform)}; Dynamo: {code(current_sha)}; vLLM: {code(native_sha)}.\n\n"
        "[Open the current developer assessment](current/report.md). "
        + ("[Baseline assessment](baseline/report.md). " if baseline_sha else "")
        + f"\n\nCLI exit: {result}. Candidate checks do not adopt source pins. "
        "Static gates do not establish runtime parity or release approval.\n"
    )
    return result
