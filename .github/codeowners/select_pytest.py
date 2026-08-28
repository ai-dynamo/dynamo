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
from pytest_markers import FRAMEWORK_MARKERS  # noqa: E402

BACKEND_MARKERS = FRAMEWORK_MARKERS
SMOKE_MARKER = "unit"
LANES = (*sorted(BACKEND_MARKERS), "generic")
NON_TEST_DOC_SUFFIXES = frozenset({".md", ".mdx", ".rst", ".txt"})


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
    ignored_paths: tuple[str, ...]
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


def _is_non_test_documentation(path: str) -> bool:
    """Match documentation extensions already excluded by CI path filters."""
    return Path(path).suffix.lower() in NON_TEST_DOC_SUFFIXES


def _lane_selection(clauses: set[MarkerClause], lane: str) -> LaneSelection:
    features: set[str] = set()
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
        features.update(clause.features)

    if not features:
        return LaneSelection(mode="none")
    return LaneSelection(
        mode="markers",
        expression=" or ".join(sorted(features)),
    )


def build_plan(model: ResolvedModel, paths: list[str]) -> SelectionPlan:
    """Build a conservative selection plan for ``paths``."""
    clauses: set[MarkerClause] = set()
    matched_labels: set[str] = set()
    fallback_reasons: list[str] = []
    unique_paths = sorted(set(paths))
    ignored_paths = tuple(
        path for path in unique_paths if _is_non_test_documentation(path)
    )
    selection_paths = [path for path in unique_paths if path not in ignored_paths]

    for path in selection_paths:
        areas = model.matching_areas(path)
        if not areas:
            fallback_reasons.append(f"{path}: no explicit ownership area")
            continue
        matched_labels.update(area.label for area in areas)
        markers = {marker for area in areas for marker in area.pytest_markers}
        if markers:
            clauses.add(MarkerClause.from_markers(markers))
            continue
        labels = ", ".join(area.label for area in areas)
        fallback_reasons.append(f"{path}: no marker mapping ({labels})")

    changed_tests = tuple(path for path in selection_paths if _is_test_file(path))
    if fallback_reasons or not unique_paths:
        reasons = fallback_reasons or ["no changed paths were provided"]
        return SelectionPlan(
            mode="full",
            lanes={lane: LaneSelection(mode="full") for lane in LANES},
            clauses=tuple(sorted(clause.expression() for clause in clauses)),
            areas=tuple(sorted(matched_labels)),
            changed_test_files=changed_tests,
            ignored_paths=ignored_paths,
            fallback_reasons=tuple(reasons),
        )

    if not selection_paths:
        return SelectionPlan(
            mode="none",
            lanes={lane: LaneSelection(mode="none") for lane in LANES},
            clauses=(),
            areas=(),
            changed_test_files=(),
            ignored_paths=ignored_paths,
            fallback_reasons=(),
        )

    lanes = {lane: _lane_selection(clauses, lane) for lane in LANES}
    mode = "markers" if clauses else "none"
    return SelectionPlan(
        mode=mode,
        lanes=lanes,
        clauses=tuple(sorted(clause.expression() for clause in clauses)),
        areas=tuple(sorted(matched_labels)),
        changed_test_files=changed_tests,
        ignored_paths=ignored_paths,
        fallback_reasons=(),
    )


def _write_github_output(
    path: Path, plan: SelectionPlan, *, apply_selection: bool
) -> None:
    """Export only backend feature expressions consumed by the PR workflow."""
    with path.open("a", encoding="utf-8") as output:
        for lane in sorted(BACKEND_MARKERS):
            selection = plan.lanes[lane]
            features = (
                selection.expression
                if apply_selection and selection.mode == "markers"
                else ""
            )
            output.write(f"{lane}_features={features}\n")


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
        result = {
            "full": "full suite (no feature filtering)",
            "markers": "selective feature markers",
            "none": "no pytest selection",
        }[plan.mode]
        summary.write(f"- Selection result: `{result}`\n")
        summary.write(
            f"- Matched areas: `{', '.join(plan.areas) if plan.areas else 'none'}`\n"
        )
        summary.write(f"- Changed paths evaluated: {len(set(paths))}\n")
        if plan.ignored_paths:
            summary.write(
                f"- Ignored non-test documentation: {len(plan.ignored_paths)} path(s)\n"
            )
        summary.write("\n### Backend feature markers\n\n")
        for lane in sorted(BACKEND_MARKERS):
            selection = plan.lanes[lane]
            if selection.mode == "markers":
                detail = selection.expression
            elif selection.mode == "full":
                detail = "none (full suite)"
            else:
                detail = "none (lane not selected)"
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
                "still honor the repository's `RUN_MULTIGPU_TESTS` switch. Pytest "
                "`skip` and `skipif` conditions are evaluated later at runtime, so a "
                "listed item may still skip.\n\n"
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
    parser.add_argument(
        "--apply-selection",
        action="store_true",
        help="Export selected feature expressions; otherwise export empty shadow outputs",
    )
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
        _write_github_output(
            args.github_output, plan, apply_selection=args.apply_selection
        )
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
