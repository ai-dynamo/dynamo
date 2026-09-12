# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the marker-parity gate.

The gate's only job is to notice that a change altered what CI selects, so the
tests that matter are the ones where it must fail: a test that quietly stopped
being selected, and a VRAM number that quietly moved a test between lanes.
"""

import pathlib
import shutil

import pytest
from dynamo_test.parity import (
    MarkerExpression,
    collect_selection,
    compare,
    read_marker_expressions,
)

CONFTEST = """
import pytest

def pytest_addoption(parser):
    parser.addoption("--max-vram-gib", type=float, default=None)
    parser.addoption("--dry-run", action="store_true", default=False)

def pytest_configure(config):
    for m in ("gpu_1", "gpu_0", "pre_merge", "vllm", "profiled_vram_gib"):
        config.addinivalue_line("markers", m + "(*a): x")

# trylast, exactly as tests/conftest.py:660 declares it. That decorator is why
# pytest's own -m deselection runs *before* this hook, and therefore why the
# snapshot-and-subtract approach observes it. A fixture without it models a
# different repository.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    limit = config.getoption("--max-vram-gib")
    if limit is not None and not config.option.collectonly:
        keep, drop = [], []
        for item in items:
            mark = item.get_closest_marker("profiled_vram_gib")
            (keep if mark and mark.args and mark.args[0] <= limit else drop).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep
    if config.getoption("--dry-run"):
        items.clear()
        return
"""

BASE_TESTS = """
import pytest

@pytest.mark.pre_merge
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(20)
def test_alpha(): pass

@pytest.mark.pre_merge
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(20)
def test_beta(): pass

@pytest.mark.pre_merge
@pytest.mark.vllm
@pytest.mark.gpu_0
def test_gamma(): pass
"""


def make_repo(root: pathlib.Path, tests: str) -> pathlib.Path:
    """A project shaped like the real one.

    The conftest goes **inside** ``tests/``, not at the root. That is where the
    real one lives, and it matters: hook order decides whether pytest's own
    ``-m`` deselection runs before or after the conftest's
    ``pytest_collection_modifyitems``. With the conftest at the root, the
    ``items.clear()`` in its dry-run branch happens first and ``-m`` never
    applies — which is not how the repository behaves, and a fixture that gets
    this wrong makes the gate assert the wrong thing.
    """
    root.mkdir(parents=True, exist_ok=True)
    suite = root / "tests"
    suite.mkdir(exist_ok=True)
    (suite / "conftest.py").write_text(CONFTEST)
    (suite / "test_things.py").write_text(tests)
    return root


@pytest.fixture
def base(tmp_path):
    return make_repo(tmp_path / "base", BASE_TESTS)


EXPR = "pre_merge and vllm and gpu_1"


def selections(base_repo, head_repo, expression=EXPR, vram=None):
    return (
        collect_selection(base_repo, expression, vram_gib=vram),
        collect_selection(head_repo, expression, vram_gib=vram),
    )


def test_an_unchanged_tree_has_parity(base, tmp_path):
    head = tmp_path / "head"
    shutil.copytree(base, head)
    b, h = selections(base, head)
    diff = compare(b, h, EXPR, "gpu_test_markers")
    assert diff.is_clean
    assert diff.base_count == diff.head_count == 2


def test_a_test_that_stops_being_selected_fails_the_gate(base, tmp_path):
    """The failure mode the gate exists for.

    Dropping a marker makes CI *greener*, because the test is no longer chosen.
    Nothing else in a passing report distinguishes that from success.
    """
    head = make_repo(
        tmp_path / "head",
        BASE_TESTS.replace(
            "@pytest.mark.gpu_1\n@pytest.mark.profiled_vram_gib(20)\ndef test_beta",
            "@pytest.mark.gpu_0\n@pytest.mark.profiled_vram_gib(20)\ndef test_beta",
        ),
    )
    b, h = selections(base, head)
    diff = compare(b, h, EXPR, "gpu_test_markers")
    assert not diff.is_clean
    assert diff.base_count == 2 and diff.head_count == 1
    assert any(n.endswith("test_beta") for n in diff.lost)
    assert "LOST 1" in diff.describe()


def test_a_changed_vram_number_fails_the_gate(base, tmp_path):
    """`profiled_vram_gib` feeds the VRAM scheduler.

    The same test, still selected by the same expression, but a changed number
    moves it between lanes — so equal node-id sets are not enough.
    """
    head = make_repo(
        tmp_path / "head",
        BASE_TESTS.replace(
            "@pytest.mark.profiled_vram_gib(20)\ndef test_alpha",
            "@pytest.mark.profiled_vram_gib(80)\ndef test_alpha",
        ),
    )
    b, h = selections(base, head)
    diff = compare(b, h, EXPR, "gpu_test_markers")
    assert not diff.is_clean
    assert diff.base_count == diff.head_count == 2  # same tests...
    assert any(
        n.endswith("test_alpha") for n in diff.marker_changed
    )  # ...different lane
    assert "MARKERS CHANGED" in diff.describe()


def test_adding_a_test_is_not_a_failure(base, tmp_path):
    """Gaining coverage is what adding a test looks like."""
    head = make_repo(
        tmp_path / "head",
        BASE_TESTS + "\n@pytest.mark.pre_merge\n@pytest.mark.vllm\n@pytest.mark.gpu_1\n"
        "@pytest.mark.profiled_vram_gib(20)\ndef test_delta(): pass\n",
    )
    b, h = selections(base, head)
    diff = compare(b, h, EXPR, "gpu_test_markers")
    assert diff.is_clean
    assert len(diff.gained) == 1
    assert "+1 new" in diff.describe()


def test_the_gate_sees_through_the_vram_deselection(base, tmp_path):
    """With a VRAM limit the two 20 GiB tests drop out — and the gate sees it.

    This is the case `--collect-only` cannot observe at all.
    """
    head = tmp_path / "head"
    shutil.copytree(base, head)
    b, h = selections(base, head, vram=8)
    # `gamma` goes to the -m filter, `alpha` and `beta` to the VRAM limit.
    assert b["deselected"] == 3 and b["selected"] == 0
    assert compare(b, h, EXPR, "gpu_test_markers").is_clean


# ---------------------------------------------- reading CI's own expressions


REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (REPO / ".github" / "workflows").is_dir(),
    reason="the workflows are not present next to the harness",
)
def test_the_real_marker_expressions_are_recoverable():
    """They are literals in the callers, so the gate reads them rather than
    restating them and drifting from the CI it models."""
    expressions, suffix = read_marker_expressions(REPO)
    assert len(expressions) > 20, f"only found {len(expressions)}"
    text = {e.expression for e in expressions}
    assert any("gpu_1" in e for e in text)
    assert any("pre_merge" in e for e in text)
    # The callee wraps the GPU lane; without this the gate models the wrong run.
    assert suffix == "not profiled_vram_gib", suffix
    assert all(e.source.endswith(tuple("0123456789")) for e in expressions)


@pytest.mark.skipif(
    not (REPO / ".github" / "workflows").is_dir(),
    reason="the workflows are not present next to the harness",
)
def test_the_transform_is_applied_only_to_the_gpu_lane():
    gpu = MarkerExpression("pre_merge and gpu_1", "gpu_test_markers", "x:1")
    cpu = MarkerExpression("pre_merge and gpu_0", "cpu_only_test_markers", "x:2")
    assert gpu.variants("not profiled_vram_gib") == (
        "pre_merge and gpu_1",
        "(pre_merge and gpu_1) and not profiled_vram_gib",
    )
    assert cpu.variants("not profiled_vram_gib") == ("pre_merge and gpu_0",)
