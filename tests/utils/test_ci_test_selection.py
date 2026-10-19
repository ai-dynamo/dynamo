# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.utils import ci_test_selection

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
    decision: str = "selective",
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
                "decision": decision,
                "markers": ["router"] if markers is None else markers,
                "changed_test_files": (
                    [] if changed_test_files is None else changed_test_files
                ),
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
    pytester.makeini(
        """
[pytest]
filterwarnings = error
markers =
    smoke: marker used by selection tests
    router: marker used by selection tests
    multimodal: marker used by selection tests
"""
    )
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
    args = ["-q", "-W", "error", "-p", "tests.utils.ci_test_selection"]
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


def test_full_decision_with_warnings_as_errors_keeps_every_test(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, _selection(decision="full"))
    assert result.ret == 0, _output(result)
    assert "4 passed" in _output(result)
    assert "full for vllm" in _output(result)
    report = _read_report(pytester)
    assert report["decision"] == "full"
    assert len(report["would_run"]) == 4
    assert report["would_skip"] == []


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


def test_report_write_failure_keeps_all_tests(pytester, monkeypatch):
    (pytester.path / "test-results").write_text("not a directory", encoding="utf-8")
    result = _run(pytester, monkeypatch, _selection(), mode="selective")
    output = _output(result)
    assert result.ret == 0, output
    assert "GPU test selection failed open; keeping all tests:" in output
    assert "4 passed" in output
    assert "1 deselected" not in output


def test_hook_observes_items_after_pytest_mark_filter(pytester, monkeypatch):
    result = _run(
        pytester, monkeypatch, _selection(markers=[]), mark_expression="smoke"
    )
    assert result.ret == 0, _output(result)
    report = _read_report(pytester)
    assert report["collected"] == 1
    assert report["would_run"] == ["test_items.py::test_smoke"]
    assert report["would_skip"] == []


def test_vram_orchestrator_receives_post_selection_session_items(monkeypatch, tmp_path):
    from tests import conftest as dynamo_conftest

    class Item:
        def __init__(self, name: str, markers: list[str]):
            self.nodeid = f"test_items.py::{name}"
            self.path = tmp_path / "test_items.py"
            self.fixturenames = []
            self._markers = markers

        def iter_markers(self):
            return [SimpleNamespace(name=name) for name in self._markers]

        def get_closest_marker(self, name: str):
            return SimpleNamespace(name=name) if name in self._markers else None

    class Config:
        rootpath = tmp_path
        hook = SimpleNamespace(pytest_deselected=lambda items: None)

        def __init__(self):
            self.stash = {
                dynamo_conftest._gpu_slots_key: 2,
                dynamo_conftest._gpu_indices_key: [0, 1],
            }
            self.pluginmanager = SimpleNamespace(get_plugin=lambda name: None)

        @staticmethod
        def getoption(name, default=None):
            return 24 if name == "max_vram_gib" else default

        @staticmethod
        def getini(name):
            return ".pytest_cache"

    items = [
        Item("test_smoke", ["smoke"]),
        Item("test_selected_feature", ["router"]),
        Item("test_other_feature", ["multimodal"]),
        Item("test_unclassified", []),
    ]
    selection = _selection()
    monkeypatch.setenv("DYNAMO_CI_TEST_SELECTION", json.dumps(selection))
    monkeypatch.setenv("DYNAMO_CI_TEST_SELECTION_MODE", "selective")
    monkeypatch.setenv("DYNAMO_TEST_FRAMEWORK", "vllm")
    monkeypatch.setenv("TEST_TYPE", "pre_merge_gpu")
    monkeypatch.delenv("DYNAMO_GPU_PARALLEL_WORKER", raising=False)

    config = Config()
    ci_test_selection.pytest_collection_modifyitems(config, items)
    selected_ids = [item.nodeid for item in items]
    assert selected_ids == [
        "test_items.py::test_smoke",
        "test_items.py::test_selected_feature",
        "test_items.py::test_unclassified",
    ]

    calls = {}

    def run_parallel(**kwargs):
        calls.update(kwargs)
        return 0

    parallel = ModuleType("tests.utils.pytest_parallel_gpu")
    parallel.run_parallel = run_parallel
    vram = ModuleType("tests.utils.vram_utils")
    vram.load_test_meta = lambda: {}
    monkeypatch.setitem(sys.modules, "tests.utils.pytest_parallel_gpu", parallel)
    monkeypatch.setitem(sys.modules, "tests.utils.vram_utils", vram)
    session = SimpleNamespace(config=config, items=items, testsfailed=0)

    assert dynamo_conftest.pytest_runtestloop(session) is True
    assert calls["test_ids"] == selected_ids
