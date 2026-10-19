# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

SELECTION_ENV = "DYNAMO_CI_TEST_SELECTION"
MODE_ENV = "DYNAMO_CI_TEST_SELECTION_MODE"
CHILD_ENV = "DYNAMO_GPU_PARALLEL_WORKER"


def _framework_selection(framework: str) -> tuple[dict, list[str], list[str]]:
    selection = json.loads(os.environ[SELECTION_ENV])
    if not isinstance(selection, dict) or selection.get("version") != 1:
        raise ValueError("selection must be a version 1 JSON object")
    if selection.get("mode") not in {"shadow", "selective"}:
        raise ValueError("selection mode must be shadow or selective")
    frameworks = selection.get("frameworks")
    if not isinstance(frameworks, dict) or framework not in frameworks:
        raise ValueError(f"selection has no entry for framework {framework!r}")
    decision = frameworks[framework]
    if not isinstance(decision, dict) or decision.get("decision") not in {
        "full",
        "selective",
    }:
        raise ValueError(f"selection has an invalid entry for framework {framework!r}")
    markers = decision.get("markers", [])
    changed_test_files = decision.get("changed_test_files", [])
    feature_markers = selection.get("feature_markers", [])
    reasons = decision.get("reasons", [])
    if not all(
        isinstance(value, list)
        for value in (markers, changed_test_files, feature_markers, reasons)
    ):
        raise ValueError("selection markers, paths, and reasons must be arrays")
    if not all(
        isinstance(value, str)
        for value in markers + changed_test_files + feature_markers + reasons
    ):
        raise ValueError("selection arrays must contain strings")
    return decision, feature_markers, changed_test_files


def _relative_test_path(config: pytest.Config, item: pytest.Item) -> str:
    path = Path(str(item.path)).resolve()
    try:
        return path.relative_to(Path(str(config.rootpath)).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_report(
    config: pytest.Config,
    framework: str,
    test_type: str,
    pytest_marks: str,
    decision: str,
    reasons: list[str],
    collected: list[pytest.Item],
    would_run: list[pytest.Item],
    would_skip: list[pytest.Item],
    intentional_empty: bool,
) -> None:
    safe_test_type = re.sub(r"[^A-Za-z0-9_.-]", "_", test_type or "unknown")
    report_dir = os.environ.get("DYNAMO_CI_TEST_SELECTION_DIR")
    output_dir = (
        Path(report_dir) if report_dir else Path(str(config.rootpath)) / "test-results"
    )
    output = output_dir / f"ci_test_selection_{safe_test_type}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "framework": framework,
        "test_type": test_type,
        "pytest_marks": pytest_marks,
        "decision": decision,
        "reasons": reasons,
        "collected": len(collected),
        "would_run": [item.nodeid for item in would_run],
        "would_skip": [item.nodeid for item in would_skip],
        "intentional_empty": intentional_empty,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _stderr_line(message: str) -> None:
    try:
        print(message.replace("\n", " "), file=sys.stderr)
    except Exception:
        pass


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    original = list(items)
    try:
        raw = os.environ.get(SELECTION_ENV)
        if not raw or os.environ.get(CHILD_ENV) == "1":
            return

        framework = os.environ.get("DYNAMO_TEST_FRAMEWORK", "")
        test_type = os.environ.get("TEST_TYPE", "unknown")
        pytest_marks = os.environ.get("PYTEST_MARKS", "")
        mode = "selective" if os.environ.get(MODE_ENV) == "selective" else "shadow"
        reasons: list[str] = []
        selection_decision = "full"
        feature_markers: list[str] = []
        changed_test_files: list[str] = []
        selected_markers: list[str] = []
        diagnostic: str | None = None

        if not framework:
            raise ValueError("DYNAMO_TEST_FRAMEWORK is not set")
        try:
            decision, feature_markers, changed_test_files = _framework_selection(
                framework
            )
            selection_decision = decision["decision"]
            selected_markers = decision["markers"]
            reasons = decision["reasons"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            reasons = [f"invalid-selection: {exc}"]
            diagnostic = f"GPU test selection unavailable; keeping all tests: {exc}"

        if selection_decision == "full":
            if not reasons:
                reasons = ["full-selection"]
            if diagnostic is None:
                diagnostic = (
                    f"GPU test selection is full for "
                    f"{framework or 'unknown framework'}: {', '.join(reasons)}"
                )
            would_run = original
            would_skip: list[pytest.Item] = []
        else:
            changed_files = set(changed_test_files)
            feature_names = set(feature_markers)
            selected_names = set(selected_markers)
            would_run = []
            would_skip = []
            for item in original:
                item_markers = {marker.name for marker in item.iter_markers()}
                keep = (
                    item.get_closest_marker("smoke") is not None
                    or bool(item_markers.intersection(selected_names))
                    or _relative_test_path(config, item) in changed_files
                    or not item_markers.intersection(feature_names)
                )
                (would_run if keep else would_skip).append(item)

        _write_report(
            config,
            framework,
            test_type,
            pytest_marks,
            selection_decision,
            reasons,
            original,
            would_run,
            would_skip,
            False,
        )
        if diagnostic:
            _stderr_line(diagnostic)
        if mode == "shadow":
            print(
                f"GPU test selection shadow report ({framework or 'unknown'}, "
                f"{test_type}): collected={len(original)}, "
                f"would_run={len(would_run)}, would_skip={len(would_skip)}"
            )
            for item in would_skip:
                print(f"  would skip: {item.nodeid}")
            return

        if mode == "selective" and selection_decision == "selective" and would_skip:
            config.hook.pytest_deselected(items=would_skip)
            items[:] = would_run
            if original and not would_run:
                _write_report(
                    config,
                    framework,
                    test_type,
                    pytest_marks,
                    selection_decision,
                    reasons,
                    original,
                    would_run,
                    would_skip,
                    True,
                )

    except Exception as exc:
        try:
            items[:] = original
        except Exception as restore_exc:
            _stderr_line(
                "GPU test selection failed open; keeping all tests: "
                f"{exc}; restoring items also failed: {restore_exc}"
            )
        else:
            _stderr_line(f"GPU test selection failed open; keeping all tests: {exc}")
