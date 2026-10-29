# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a candidate suite result against a control run, case by case.

The candidate is normally Dynamo serving a model; the control is the native
engine (``vllm serve``, ``sglang serve``) serving the same model with the same
parser names, so every failure the two share belongs to the engine or the
model and every candidate-only failure is Dynamo's. Both inputs are
``suite-result.json`` files whose suite writes ``case_results`` (today:
``openrouter``).

    python3 -m benchmarks.tool_calling.e2e_verifier.compare \\
        --candidate dynamo/suite-result.json --control native/suite-result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FAILED_STATUSES = frozenset({"fail"})
UNSCORED_STATUSES = frozenset({"unscored", "missing"})


def _load_case_results(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a suite result object")
    case_results = payload.get("case_results")
    if not isinstance(case_results, dict) or not case_results:
        raise ValueError(
            f"{path} has no case_results; the {payload.get('suite')!r} suite "
            "does not write per-case outcomes"
        )
    return payload, case_results


def compare_case_results(
    candidate: Mapping[str, Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Split the union of case IDs into candidate-only, control-only, shared."""
    ids = sorted(set(candidate) | set(control))

    def failed(results: Mapping[str, Mapping[str, Any]], case_id: str) -> bool:
        return str(results.get(case_id, {}).get("status")) in FAILED_STATUSES

    def unscored(results: Mapping[str, Mapping[str, Any]], case_id: str) -> bool:
        return (
            str(results.get(case_id, {}).get("status", "missing")) in UNSCORED_STATUSES
        )

    candidate_only = [c for c in ids if failed(candidate, c) and not failed(control, c)]
    control_only = [c for c in ids if failed(control, c) and not failed(candidate, c)]
    shared = [c for c in ids if failed(candidate, c) and failed(control, c)]
    unscored_either = [c for c in ids if unscored(candidate, c) or unscored(control, c)]

    def kinds(
        results: Mapping[str, Mapping[str, Any]], case_ids: list[str]
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case_id in case_ids:
            kind = str(results.get(case_id, {}).get("failure_kind") or "unknown")
            counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "case_count": len(ids),
        "candidate_only_failures": candidate_only,
        "control_only_failures": control_only,
        "shared_failures": shared,
        "unscored_in_either": unscored_either,
        "candidate_only_failure_kinds": kinds(candidate, candidate_only),
        "control_only_failure_kinds": kinds(control, control_only),
        "shared_failure_kinds": kinds(candidate, shared),
        # Candidate-only failures are the ones the serving stack under test
        # introduced; this is the number a relative verdict would gate on.
        "candidate_regressions": len(candidate_only),
    }


def compare_files(candidate_path: Path, control_path: Path) -> dict[str, Any]:
    candidate, candidate_cases = _load_case_results(candidate_path)
    control, control_cases = _load_case_results(control_path)
    if candidate.get("suite") != control.get("suite"):
        raise ValueError(
            f"suite mismatch: candidate={candidate.get('suite')!r}, "
            f"control={control.get('suite')!r}"
        )
    comparison = compare_case_results(candidate_cases, control_cases)
    comparison.update(
        {
            "suite": candidate.get("suite"),
            "model": candidate.get("model"),
            "candidate": {
                "runtime": candidate.get("runtime"),
                "score": (candidate.get("summary") or {}).get("score"),
                "path": str(candidate_path),
            },
            "control": {
                "runtime": control.get("runtime"),
                "score": (control.get("summary") or {}).get("score"),
                "path": str(control_path),
            },
        }
    )
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write the comparison JSON here")
    args = parser.parse_args(argv)
    try:
        comparison = compare_files(args.candidate, args.control)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
