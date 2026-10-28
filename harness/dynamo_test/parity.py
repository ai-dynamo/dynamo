# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The marker-parity gate: prove a change did not alter what CI selects.

Every migration step in the test-framework work risks the same failure, and it is
invisible: a refactor that stops selecting forty GPU tests produces a *greener*
run than one that breaks them. Nothing in a passing CI report distinguishes
"these tests passed" from "these tests were never chosen".

This module makes that difference checkable. It collects the suite the way CI
does, for every ``-m`` expression CI actually uses, on both the base and the
change, and diffs the node-id sets.

## Three things it does not do, each for a measured reason

**It does not use ``--collect-only``.** ``tests/conftest.py`` guards both the
``--max-vram-gib`` deselection and ``write_test_meta`` on
``not config.option.collectonly``, so a collect-only run is blind to the only
mechanism that removes tests. Measured on ``tests/serve`` with
``--max-vram-gib 8``: ``--collect-only`` reports 6 selected and 0 deselected,
``--dry-run`` reports 3 and 3. A gate built on the former passes while CI runs
half as much.

**It does not use ``git stash``.** On a clean tree — which CI always has — a
stash is a no-op, and the gate then compares HEAD to HEAD and always passes. The
base is an explicit worktree at the merge base.

**It does not invent the marker expressions.** They are literals in the caller
workflows under ``gpu_test_markers:`` and ``cpu_only_test_markers:``, and the
callee wraps the GPU ones as ``({0}) and not profiled_vram_gib``. Both are read
from the files rather than restated here, so the gate cannot drift from the CI
it is supposed to model.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MarkerExpression",
    "SelectionDiff",
    "read_marker_expressions",
    "collect_selection",
    "compare",
    "main",
]

_MARKER_LINE = re.compile(r"^\s*(gpu_test_markers|cpu_only_test_markers):\s*(.+?)\s*$")
# The callee's runtime transform, read from the file rather than restated.
_TRANSFORM = re.compile(r"format\(\s*'\(\{0\}\)\s*and\s*(?P<suffix>[^']+)'")


@dataclass(frozen=True)
class MarkerExpression:
    """One ``-m`` expression CI runs, and where it came from."""

    expression: str
    lane: str
    source: str

    def variants(self, suffix: str | None) -> tuple[str, ...]:
        """The expression, plus the transformed form if the callee wraps it."""
        if suffix and self.lane == "gpu_test_markers":
            return (self.expression, f"({self.expression}) and {suffix}")
        return (self.expression,)


def read_marker_expressions(repo: Path) -> tuple[list[MarkerExpression], str | None]:
    """Every ``-m`` expression in the workflows, and the callee's transform."""
    workflows = repo / ".github" / "workflows"
    found: dict[tuple[str, str], MarkerExpression] = {}
    suffix: str | None = None

    for path in sorted(workflows.glob("*.y*ml")):
        text = path.read_text()
        transform = _TRANSFORM.search(text)
        if transform:
            suffix = transform.group("suffix").strip()
        for n, line in enumerate(text.splitlines(), 1):
            match = _MARKER_LINE.match(line)
            if not match:
                continue
            lane, raw = match.group(1), match.group(2).strip()
            # Skip the callee's own parameter declarations and pass-throughs.
            if not raw or raw.startswith("${{") or raw.startswith("#"):
                continue
            expression = raw.strip("'\"")
            if not expression or expression.endswith(":"):
                continue
            found[(lane, expression)] = MarkerExpression(
                expression, lane, f"{path.name}:{n}"
            )
    return sorted(found.values(), key=lambda e: (e.lane, e.expression)), suffix


def collect_selection(
    repo: Path,
    expression: str,
    *,
    paths: list[str] | None = None,
    vram_gib: float | None = None,
    harness: Path | None = None,
    timeout: float = 900.0,
) -> dict:
    """Run collection the way CI would and return what it would execute.

    Uses ``--dry-run`` rather than ``--collect-only`` — see the module docstring.
    """
    harness = harness or Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        # A distinct TMPDIR per run: `write_test_meta` writes to a fixed name,
        # so two runs sharing a temp directory clobber each other's metadata.
        out = Path(tmp) / "selection.json"
        argv = [
            sys.executable,
            "-m",
            "pytest",
            *(paths or ["tests"]),
            "-p",
            "dynamo_test.pytest_plugin",
            f"--dynamo-selection-out={out}",
            "--dry-run",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            expression,
        ]
        if vram_gib is not None:
            argv += ["--max-vram-gib", str(vram_gib)]
        env = {
            **_clean_env(),
            "PYTHONPATH": str(harness),
            "TMPDIR": tmp,
        }
        result = subprocess.run(
            argv, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout
        )
        if not out.exists():
            return {
                "error": "collection produced no selection file",
                "returncode": result.returncode,
                "stderr": result.stderr[-2000:],
                "tests": [],
                "selected": 0,
            }
        return json.loads(out.read_text())


def _clean_env() -> dict:
    import os

    keep = ("PATH", "HOME", "LANG", "LC_ALL", "VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES")
    return {k: v for k, v in os.environ.items() if k in keep or k.startswith("GIT_")}


@dataclass(frozen=True)
class SelectionDiff:
    """What one expression selects on each side, and how that differs."""

    expression: str
    lane: str
    base_count: int
    head_count: int
    lost: tuple[str, ...]
    gained: tuple[str, ...]
    marker_changed: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.lost and not self.marker_changed

    def describe(self) -> str:
        if self.is_clean:
            gained = f", +{len(self.gained)} new" if self.gained else ""
            return f"  OK   {self.base_count:5d} -> {self.head_count:<5d}{gained}  {self.expression}"
        parts = []
        if self.lost:
            parts.append(f"LOST {len(self.lost)}")
        if self.marker_changed:
            parts.append(f"MARKERS CHANGED {len(self.marker_changed)}")
        return (
            f"  FAIL {self.base_count:5d} -> {self.head_count:<5d} "
            f"({', '.join(parts)})  {self.expression}"
        )


def compare(base: dict, head: dict, expression: str, lane: str) -> SelectionDiff:
    """Diff two selections.

    Losing a test is a failure. *Gaining* one is not — that is what adding a test
    looks like. A marker whose argument changed is also a failure, because
    ``profiled_vram_gib`` feeds the VRAM scheduler and a changed number silently
    moves a test between lanes.
    """
    base_by_id = {t["nodeid"]: t for t in base.get("tests", [])}
    head_by_id = {t["nodeid"]: t for t in head.get("tests", [])}

    lost = sorted(set(base_by_id) - set(head_by_id))
    gained = sorted(set(head_by_id) - set(base_by_id))
    changed = sorted(
        nodeid
        for nodeid in set(base_by_id) & set(head_by_id)
        if base_by_id[nodeid].get("marker_args")
        != head_by_id[nodeid].get("marker_args")
        or base_by_id[nodeid].get("markers") != head_by_id[nodeid].get("markers")
    )
    return SelectionDiff(
        expression=expression,
        lane=lane,
        base_count=len(base_by_id),
        head_count=len(head_by_id),
        lost=tuple(lost),
        gained=tuple(gained),
        marker_changed=tuple(changed),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dynamo-test-parity",
        description="Prove a change did not alter what CI selects.",
    )
    parser.add_argument("--repo", default=".", help="the working tree to check")
    parser.add_argument(
        "--base",
        required=True,
        help="a checkout of the merge base. Use an explicit worktree: a git "
        "stash on a clean tree is a no-op and compares HEAD to HEAD.",
    )
    parser.add_argument("--paths", nargs="*", default=["tests"])
    parser.add_argument("--max-vram-gib", type=float, default=None)
    parser.add_argument(
        "--expression",
        action="append",
        help="check only this expression; repeatable. Defaults to every "
        "expression found in the workflows.",
    )
    args = parser.parse_args(argv)

    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    if args.expression:
        expressions = [
            MarkerExpression(e, "gpu_test_markers", "<cli>") for e in args.expression
        ]
        suffix = None
    else:
        expressions, suffix = read_marker_expressions(repo)
        print(
            f"{len(expressions)} marker expression(s) from the workflows"
            + (f"; callee transform: 'and {suffix}'" if suffix else "")
        )

    diffs: list[SelectionDiff] = []
    for expression in expressions:
        for variant in expression.variants(suffix):
            head = collect_selection(
                repo, variant, paths=args.paths, vram_gib=args.max_vram_gib
            )
            before = collect_selection(
                base, variant, paths=args.paths, vram_gib=args.max_vram_gib
            )
            diff = compare(before, head, variant, expression.lane)
            diffs.append(diff)
            print(diff.describe())

    broken = [d for d in diffs if not d.is_clean]
    print()
    if not broken:
        print(f"marker parity holds across {len(diffs)} expression(s)")
        return 0

    print(f"{len(broken)} expression(s) select differently:\n")
    for diff in broken:
        print(f"  {diff.expression}")
        for nodeid in diff.lost[:20]:
            print(f"    no longer selected: {nodeid}")
        for nodeid in diff.marker_changed[:20]:
            print(f"    markers changed:    {nodeid}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
