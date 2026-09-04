# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the selection recorder.

Run as real pytest subprocesses against a throwaway project, because the thing
under test is hook ordering — and hook ordering cannot be exercised from inside
the run whose hooks you are testing.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = str(Path(__file__).resolve().parents[1])

# A project with a conftest that deselects, then clears the item list — the two
# behaviours in the real repository that defeat a naive observer.
CONFTEST = """
import pytest

def pytest_addoption(parser):
    parser.addoption("--max-vram-gib", type=float, default=None)
    parser.addoption("--dry-run", action="store_true", default=False)

def pytest_configure(config):
    config.addinivalue_line("markers", "vram(n): profiled VRAM")

def pytest_collection_modifyitems(config, items):
    limit = config.getoption("--max-vram-gib")
    # Guarded on collectonly exactly as the real conftest guards it, which is
    # what makes a --collect-only gate blind.
    if limit is not None and not config.option.collectonly:
        keep, drop = [], []
        for item in items:
            mark = item.get_closest_marker("vram")
            (keep if mark and mark.args and mark.args[0] <= limit else drop).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep
    if config.getoption("--dry-run"):
        items.clear()      # the real conftest does this too
        return
"""

TESTS = """
import pytest

@pytest.mark.vram(4)
def test_small(): pass

@pytest.mark.vram(40)
def test_large(): pass

def test_unmarked(): pass
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "conftest.py").write_text(CONFTEST)
    (tmp_path / "test_things.py").write_text(TESTS)
    return tmp_path


def collect(project, *args):
    out = project / "selection.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(project),
            "-p",
            "dynamo_test.pytest_plugin",
            f"--dynamo-selection-out={out}",
            "-q",
            "-p",
            "no:cacheprovider",
            *args,
        ],
        cwd=project,
        env={"PYTHONPATH": HARNESS, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    return json.loads(out.read_text())


def ids(payload):
    return sorted(t["nodeid"].split("::")[-1] for t in payload["tests"])


def test_records_every_test_when_nothing_deselects(project):
    got = collect(project, "--dry-run")
    assert got["selected"] == 3
    assert ids(got) == ["test_large", "test_small", "test_unmarked"]


def test_it_sees_a_deselection_that_collect_only_cannot(project):
    """The reason the gate does not use --collect-only.

    Both the deselection and the metadata write are guarded on
    `not config.option.collectonly`, so a collect-only run is structurally blind
    to the one mechanism that removes tests. A gate built on it reports the same
    number before and after a change that halves what CI runs.
    """
    dry = collect(project, "--dry-run", "--max-vram-gib", "8")
    collect_only = collect(project, "--collect-only", "--max-vram-gib", "8")

    assert dry["deselected"] == 2 and dry["selected"] == 1
    assert ids(dry) == ["test_small"]

    assert collect_only["deselected"] == 0 and collect_only["selected"] == 3
    assert dry["selected"] != collect_only["selected"]


def test_it_survives_items_clear(project):
    """`items.clear()` in the dry-run branch defeats a trylast observer.

    Snapshotting first and subtracting recorded deselections is correct
    regardless of what any later hook does to the list.
    """
    got = collect(project, "--dry-run", "--max-vram-gib", "8")
    assert got["collected"] == 3, "the snapshot must predate items.clear()"
    assert got["selected"] == 1


def test_an_unmarked_test_is_deselected_by_a_vram_limit(project):
    """Unknown VRAM is treated as unsafe, and the gate must see that too."""
    got = collect(project, "--dry-run", "--max-vram-gib", "100")
    assert "test_unmarked" not in ids(got)
    assert ids(got) == ["test_large", "test_small"]


def test_markers_and_their_arguments_are_recorded(project):
    """A marker whose *argument* changed is a selection change on the GPU lane."""
    got = collect(project, "--dry-run")
    large = next(t for t in got["tests"] if t["nodeid"].endswith("test_large"))
    assert "vram" in large["markers"]
    assert large["marker_args"]["vram"] == ["40"]


def test_the_plugin_is_inert_without_the_option(project):
    """No output file requested, no plugin registered, no behaviour change."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(project),
            "-p",
            "dynamo_test.pytest_plugin",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=project,
        env={"PYTHONPATH": HARNESS, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
    assert not (project / "selection.json").exists()
