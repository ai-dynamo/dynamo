# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assess configured vLLM pins for a baseline, Dynamo change or version bump.

Select one configured platform explicitly. A candidate SHA overrides only this
assessment, never container versions or pins. Exit codes match the direct CLI:
0 static gates satisfied; 1 action required; 2 input/tool failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from assess_protocol_compatibility import run as assess
from check_protocol_pins import check_pins
from generate_protocol_inventory import OUTPUT, ROOT
from protocol_assessment_inputs import list_value, object_value, text_value
from protocol_assessment_report import code
from protocol_drift import commit, digest, git, tool_provenance

PINS = str((OUTPUT / "vllm_pins.json").relative_to(ROOT))


def selected_pin(repo: Path, revision: str, platform: str) -> dict:
    context = object_value(
        yaml.safe_load(git(repo, "show", f"{revision}:container/context.yaml")),
        "container context",
    )
    object_value(context.get("vllm"), "container context.vllm")
    pins = object_value(
        json.loads(git(repo, "show", f"{revision}:{PINS}")), "vLLM pins"
    )
    if pins.get("format_version") != 1 or pins.get("target") != "vllm":
        raise ValueError("unsupported vLLM pin schema")
    for pin in list_value(pins.get("versions"), "vLLM pin versions"):
        pin = object_value(pin, "vLLM pin")
        for key in ("version", "commit"):
            text_value(pin.get(key), f"vLLM pin {key}")
        for platform_name in list_value(pin.get("platforms"), "vLLM pin platforms"):
            text_value(platform_name, "vLLM platform")
    check_pins(context, pins)
    matching = [pin for pin in pins["versions"] if platform in pin["platforms"]]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one configured pin for platform {platform}")
    return {
        **matching[0],
        "context_sha256": digest(context),
        "pins_sha256": digest(pins),
    }


def run(args: argparse.Namespace) -> int:
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
        "provenance": tool_provenance(
            "run_protocol_assessment.py", "check_protocol_pins.py"
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamo-repo", type=Path, default=ROOT)
    parser.add_argument("--dynamo-commit", required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--baseline-dynamo")
    parser.add_argument("--upstream-candidate")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--crate-cache", type=Path)
    parser.add_argument(
        "--pipeline", choices=("token", "text", "unspecified"), default="unspecified"
    )
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--support-policy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    try:
        return run(parser.parse_args())
    except (
        OSError,
        SyntaxError,
        yaml.YAMLError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Assessment workflow error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
