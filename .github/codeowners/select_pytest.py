# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve changed paths to pytest marker expressions via CODEOWNERS areas."""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import ResolvedModel, compute_resolution  # noqa: E402

BACKEND_MARKERS = frozenset({"vllm", "sglang", "trtllm"})
SMOKE_MARKER = "unit"
LANES = (*sorted(BACKEND_MARKERS), "generic")


@dataclass(frozen=True)
class MarkerClause:
    """One changed file's marker constraint, split by marker dimension."""

    backends: tuple[str, ...]
    features: tuple[str, ...]

    @classmethod
    def from_markers(cls, markers: set[str]) -> MarkerClause:
        return cls(
            backends=tuple(sorted(markers & BACKEND_MARKERS)),
            features=tuple(sorted(markers - BACKEND_MARKERS)),
        )

    def expression(self) -> str:
        groups: list[str] = []
        if self.backends:
            groups.append(_or_group(self.backends))
        if self.features:
            groups.append(_or_group(self.features))
        return " and ".join(groups)


@dataclass(frozen=True)
class LaneSelection:
    mode: str
    expression: str = ""


@dataclass(frozen=True)
class SelectionPlan:
    mode: str
    lanes: dict[str, LaneSelection]
    clauses: tuple[str, ...]
    areas: tuple[str, ...]
    changed_test_files: tuple[str, ...]
    fallback_reasons: tuple[str, ...]


def selected_tests_by_lane(
    plan: SelectionPlan, records: list[dict]
) -> dict[str, list[str]]:
    """Resolve the exact feature-level node IDs selected in each test lane."""
    selected: dict[str, list[str]] = {lane: [] for lane in LANES}
    for record in records:
        nodeid = str(record.get("nodeid", ""))
        markers = set(record.get("markers", []))
        for lane, selection in plan.lanes.items():
            in_lane = (
                not (markers & BACKEND_MARKERS)
                if lane == "generic"
                else lane in markers
            )
            if not in_lane or selection.mode == "none":
                continue
            if (
                selection.mode == "full"
                or SMOKE_MARKER in markers
                or _matches_marker_expression(selection.expression, markers)
            ):
                selected[lane].append(nodeid)
    return {lane: sorted(set(nodeids)) for lane, nodeids in selected.items()}


def selected_pr_tests(plan: SelectionPlan, records: list[dict]) -> dict[str, list[str]]:
    """Resolve node IDs for the PR workflow's concrete backend test jobs."""
    lane_tests = selected_tests_by_lane(plan, records)
    markers_by_nodeid = {
        str(record.get("nodeid", "")): set(record.get("markers", []))
        for record in records
    }
    jobs: dict[str, list[str]] = {}
    job_markers = {
        "cpu": ({"pre_merge"}, {"gpu_0"}),
        "gpu-1": ({"pre_merge"}, {"gpu_1"}),
        "multi-gpu": ({"pre_merge"}, {"gpu_2"}),
    }
    for lane in sorted(BACKEND_MARKERS):
        for job, (allowed_lifecycle, allowed_hardware) in job_markers.items():
            jobs[f"{lane}-{job}"] = [
                nodeid
                for nodeid in lane_tests[lane]
                if markers_by_nodeid[nodeid] & allowed_lifecycle
                and markers_by_nodeid[nodeid] & allowed_hardware
            ]
    jobs["vllm-4-gpu"] = [
        nodeid
        for nodeid in lane_tests["vllm"]
        if markers_by_nodeid[nodeid] & {"pre_merge", "post_merge"}
        and "gpu_4" in markers_by_nodeid[nodeid]
    ]
    return jobs


def _matches_marker_expression(expression: str, markers: set[str]) -> bool:
    """Evaluate the selector's small ``and``/``or`` marker expressions."""
    alternatives = expression.replace("(", "").replace(")", "").split(" or ")
    return any(
        all(marker.strip() in markers for marker in alternative.split(" and "))
        for alternative in alternatives
    )


def _or_group(markers: tuple[str, ...]) -> str:
    if len(markers) == 1:
        return markers[0]
    return f"({' or '.join(markers)})"


def _is_test_file(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return path.endswith(".py") and (
        "tests" in parts or name.startswith("test_") or name.endswith("_test.py")
    )


def _lane_selection(clauses: set[MarkerClause], lane: str) -> LaneSelection:
    expressions: set[str] = set()
    for clause in clauses:
        if lane == "generic":
            if clause.backends:
                continue
        elif clause.backends and lane not in clause.backends:
            continue

        # The caller's backend job already adds its framework marker. A
        # backend-only clause therefore means the whole lane is required.
        if not clause.features:
            return LaneSelection(mode="full")
        expressions.add(_or_group(clause.features))

    if not expressions:
        return LaneSelection(mode="none")
    return LaneSelection(
        mode="markers",
        expression=" or ".join(
            f"({expression})" if " or " in expression else expression
            for expression in sorted(expressions)
        ),
    )


def build_plan(model: ResolvedModel, paths: list[str]) -> SelectionPlan:
    """Build a conservative selection plan for ``paths``."""
    clauses: set[MarkerClause] = set()
    matched_labels: set[str] = set()
    fallback_reasons: list[str] = []

    for path in sorted(set(paths)):
        areas = model.matching_areas(path)
        if not areas:
            fallback_reasons.append(f"{path}: no explicit ownership area")
            continue
        matched_labels.update(area.label for area in areas)
        markers = {marker for area in areas for marker in area.pytest_markers}
        if markers:
            clauses.add(MarkerClause.from_markers(markers))
            continue
        if any(area.pytest_mode == "fallback" for area in areas):
            labels = ", ".join(area.label for area in areas)
            fallback_reasons.append(f"{path}: no marker mapping ({labels})")

    changed_tests = tuple(sorted(path for path in set(paths) if _is_test_file(path)))
    if fallback_reasons or not paths:
        reasons = fallback_reasons or ["no changed paths were provided"]
        return SelectionPlan(
            mode="full",
            lanes={lane: LaneSelection(mode="full") for lane in LANES},
            clauses=tuple(sorted(clause.expression() for clause in clauses)),
            areas=tuple(sorted(matched_labels)),
            changed_test_files=changed_tests,
            fallback_reasons=tuple(reasons),
        )

    lanes = {lane: _lane_selection(clauses, lane) for lane in LANES}
    mode = "markers" if clauses else "none"
    return SelectionPlan(
        mode=mode,
        lanes=lanes,
        clauses=tuple(sorted(clause.expression() for clause in clauses)),
        areas=tuple(sorted(matched_labels)),
        changed_test_files=changed_tests,
        fallback_reasons=(),
    )


def _write_github_output(path: Path, plan: SelectionPlan) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"mode={plan.mode}\n")
        output.write(f"areas={','.join(plan.areas)}\n")
        output.write(
            "changed_test_files="
            f"{json.dumps(plan.changed_test_files, separators=(',', ':'))}\n"
        )
        for lane, selection in plan.lanes.items():
            output.write(f"{lane}_mode={selection.mode}\n")
            output.write(f"{lane}_expression={selection.expression}\n")


def _write_summary(
    path: Path,
    paths: list[str],
    plan: SelectionPlan,
    selected_tests: dict[str, list[str]] | None = None,
    pr_tests: dict[str, list[str]] | None = None,
    inventory_status: str = "",
) -> None:
    with path.open("a", encoding="utf-8") as summary:
        summary.write("## CODEOWNERS pytest selection\n\n")
        summary.write(f"- Plan mode: `{plan.mode}`\n")
        summary.write(
            f"- Matched areas: `{', '.join(plan.areas) if plan.areas else 'none'}`\n"
        )
        summary.write(f"- Changed paths evaluated: {len(set(paths))}\n")
        if plan.clauses:
            summary.write(f"- Marker clauses: `{' or '.join(plan.clauses)}`\n")
        for lane, selection in plan.lanes.items():
            detail = (
                selection.expression if selection.mode == "markers" else selection.mode
            )
            summary.write(f"- `{lane}`: `{detail}`\n")
        if plan.fallback_reasons:
            summary.write("\nFull-suite fallback reasons:\n")
            for reason in plan.fallback_reasons:
                summary.write(f"- {reason}\n")
        if pr_tests is not None:
            summary.write("\n### Exact shadow selection\n\n")
            summary.write(
                "These are the exact node IDs selected by each PR backend job's "
                "smoke/feature, lifecycle, and GPU-marker intersection. Multi-GPU jobs "
                "still honor the repository's `RUN_MULTIGPU_TESTS` switch.\n\n"
            )
            if inventory_status and inventory_status != "success":
                summary.write(
                    f"> Marker collection status was `{inventory_status}`; this "
                    "inventory may be partial.\n\n"
                )
            for lane, nodeids in pr_tests.items():
                shown = nodeids[:200]
                summary.write(
                    f"<details><summary><code>{lane}</code>: "
                    f"{len(nodeids)} test(s)</summary>\n\n<pre>"
                )
                summary.write("\n".join(html.escape(nodeid) for nodeid in shown))
                if len(nodeids) > len(shown):
                    summary.write(
                        f"\n... {len(nodeids) - len(shown)} more in the JSON artifact"
                    )
                summary.write("</pre>\n\n</details>\n\n")
        summary.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--areas", required=True, type=Path)
    parser.add_argument("--paths", nargs="*", default=[])
    parser.add_argument(
        "--paths-from-env",
        help="Read a shell-style changed-files list from this environment variable",
    )
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--marker-report", type=Path)
    parser.add_argument("--marker-report-status", default="")
    parser.add_argument("--selection-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = list(args.paths)
    if args.paths_from_env:
        paths.extend(shlex.split(os.environ.get(args.paths_from_env, "")))
    spec = yaml.safe_load(args.areas.read_text(encoding="utf-8")) or {}
    model = compute_resolution(spec)
    plan = build_plan(model, paths)
    selected_tests = None
    pr_tests = None
    marker_report = None
    if args.marker_report and args.marker_report.exists():
        marker_report = json.loads(args.marker_report.read_text(encoding="utf-8"))
        records = marker_report.get("collected_tests", marker_report.get("tests", []))
        selected_tests = selected_tests_by_lane(plan, records)
        pr_tests = selected_pr_tests(plan, records)
    if args.github_output:
        _write_github_output(args.github_output, plan)
    if args.summary:
        _write_summary(
            args.summary,
            paths,
            plan,
            selected_tests,
            pr_tests,
            args.marker_report_status,
        )
    result = {
        "plan": asdict(plan),
        "selected_tests": selected_tests,
        "selected_pr_tests": pr_tests,
    }
    if args.selection_json:
        args.selection_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    console_result = result
    if selected_tests is not None and pr_tests is not None:
        console_result = {
            "plan": asdict(plan),
            "selected_test_counts": {
                lane: len(nodeids) for lane, nodeids in selected_tests.items()
            },
            "selected_pr_test_counts": {
                job: len(nodeids) for job, nodeids in pr_tests.items()
            },
        }
    print(json.dumps(console_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
