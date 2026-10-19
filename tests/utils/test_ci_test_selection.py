# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]
pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("test-results/ci_test_selection_pre_merge_gpu.json")

TEST_MODULE = """\
import pytest

@pytest.mark.smoke
def test_smoke():
    pass

@pytest.mark.router
def test_selected_feature():
    pass

@pytest.mark.multimodal
def test_other_feature():
    pass

def test_unclassified():
    pass
"""


def _selection(
    *,
    markers: list[str] | None = None,
    changed_test_files: list[str] | None = None,
    frameworks: dict | None = None,
) -> dict:
    return {
        "version": 1,
        "mode": "shadow",
        "event": "pull_request",
        "pr": 42,
        "feature_markers": ["multimodal", "router"],
        "frameworks": frameworks
        or {
            "vllm": {
                "decision": "selective",
                "markers": markers or ["router"],
                "changed_test_files": changed_test_files or [],
                "reasons": ["feature:router:lib/kv-router/src/lib.rs"],
            }
        },
    }


def _run(
    pytester,
    monkeypatch,
    selection: dict | str | None = None,
    *,
    mode: str = "shadow",
    framework: str = "vllm",
    child: bool = False,
    mark_expression: str | None = None,
):
    pytester.makepyfile(test_items=TEST_MODULE)
    pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(ROOT), pythonpath]).rstrip(os.pathsep)
    )
    monkeypatch.setenv("DYNAMO_TEST_FRAMEWORK", framework)
    monkeypatch.setenv("TEST_TYPE", "pre_merge_gpu")
    monkeypatch.setenv("PYTEST_MARKS", "pre_merge and vllm and gpu_1")
    monkeypatch.setenv("DYNAMO_CI_TEST_SELECTION_MODE", mode)
    monkeypatch.delenv("DYNAMO_GPU_PARALLEL_WORKER", raising=False)
    if child:
        monkeypatch.setenv("DYNAMO_GPU_PARALLEL_WORKER", "1")
    if selection is None:
        monkeypatch.delenv("DYNAMO_CI_TEST_SELECTION", raising=False)
    else:
        monkeypatch.setenv(
            "DYNAMO_CI_TEST_SELECTION",
            selection if isinstance(selection, str) else json.dumps(selection),
        )
    args = ["-q", "-p", "tests.utils.ci_test_selection"]
    if mark_expression:
        args.extend(["-m", mark_expression])
    return pytester.runpytest_subprocess(*args, "test_items.py")


def _output(result) -> str:
    return result.stdout.str() + result.stderr.str()


def _read_report(pytester) -> dict:
    return json.loads((pytester.path / REPORT).read_text(encoding="utf-8"))


def test_shadow_writes_report_without_deselecting(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, _selection())
    assert result.ret == 0, _output(result)
    report = _read_report(pytester)
    assert report["pytest_marks"] == "pre_merge and vllm and gpu_1"
    assert report["collected"] == 4
    assert len(report["would_run"]) == 3
    assert report["would_skip"] == ["test_items.py::test_other_feature"]
    output = _output(result)
    assert "would_skip=1" in output
    assert "would skip: test_items.py::test_other_feature" in output
    assert "4 passed" in output


def test_selective_deselects_excluded_feature(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, _selection(), mode="selective")
    assert result.ret == 0, _output(result)
    assert "3 passed, 1 deselected" in _output(result)
    assert _read_report(pytester)["would_skip"] == ["test_items.py::test_other_feature"]


def test_changed_test_file_keeps_its_feature_tests(pytester, monkeypatch):
    result = _run(
        pytester, monkeypatch, _selection(changed_test_files=["test_items.py"])
    )
    assert result.ret == 0, _output(result)
    report = _read_report(pytester)
    assert report["would_skip"] == []
    assert len(report["would_run"]) == 4


@pytest.mark.parametrize(
    "selection",
    [
        "{invalid json",
        "[]",
        _selection(frameworks={"sglang": {"decision": "selective"}}),
    ],
)
def test_invalid_json_or_missing_framework_fails_open(pytester, monkeypatch, selection):
    result = _run(pytester, monkeypatch, selection)
    assert result.ret == 0, _output(result)
    assert "keeping all tests" in _output(result)
    report = _read_report(pytester)
    assert report["decision"] == "full"
    assert report["collected"] == 4
    assert len(report["would_run"]) == 4
    assert report["would_skip"] == []
    assert "4 passed" in _output(result)


def test_child_process_is_a_no_op(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, _selection(), child=True)
    assert result.ret == 0, _output(result)
    assert not (pytester.path / REPORT).exists()
    assert "GPU test selection shadow report" not in _output(result)


def test_hook_observes_items_after_pytest_mark_filter(pytester, monkeypatch):
    result = _run(
        pytester, monkeypatch, _selection(markers=[]), mark_expression="smoke"
    )
    assert result.ret == 0, _output(result)
    report = _read_report(pytester)
    assert report["collected"] == 1
    assert report["would_run"] == ["test_items.py::test_smoke"]
    assert report["would_skip"] == []
