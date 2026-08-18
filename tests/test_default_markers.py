# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the repository-wide default-marker hook."""

from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]
pytest_plugins = ["pytester"]


class _FakeItem:
    """Minimal stand-in for a collected pytest item."""

    def __init__(self, *markers: str):
        self.markers = set(markers)

    def get_closest_marker(self, name: str):
        return name if name in self.markers else None

    def add_marker(self, marker) -> None:
        self.markers.add(marker.name)


@pytest.fixture
def root_conftest(request: pytest.FixtureRequest) -> ModuleType:
    """Return the repository-root conftest without importing an ambiguous name."""
    root_path = Path(__file__).parents[1] / "conftest.py"
    for plugin in request.config.pluginmanager.get_plugins():
        plugin_file = getattr(plugin, "__file__", None)
        if plugin_file and Path(plugin_file).resolve() == root_path.resolve():
            return plugin
    pytest.fail("Repository-root conftest plugin was not loaded")


def test_unmarked_test_gets_both_defaults(root_conftest):
    item = _FakeItem()
    root_conftest.pytest_itemcollected(item)
    assert item.markers == {"pre_merge", "gpu_0"}


def test_unmarked_test_in_sibling_collection_root_gets_both_defaults(pytester):
    """The root hook must cover trees that do not load tests/conftest.py."""
    root_conftest_path = Path(__file__).parents[1] / "conftest.py"
    pytester.makeconftest(root_conftest_path.read_text())
    sibling_root = pytester.path / "components" / "src"
    sibling_root.mkdir(parents=True)
    sibling_test = sibling_root / "test_unmarked.py"
    sibling_test.write_text("def test_unmarked():\n    pass\n")

    result = pytester.runpytest(
        "-o",
        "addopts=",
        "-m",
        "pre_merge and gpu_0",
        sibling_test,
    )

    result.assert_outcomes(passed=1)


def test_machine_marker_is_not_overridden(root_conftest):
    """A gpu_2 test must not also become gpu_0 and match CPU-only selectors."""
    item = _FakeItem("gpu_2")
    root_conftest.pytest_itemcollected(item)
    assert item.markers == {"gpu_2", "pre_merge"}


def test_non_gpu_machine_markers_are_recognized(root_conftest):
    """xpu/h100/k8s tests declare hardware and must keep it."""
    item = _FakeItem("xpu_1", "post_merge")
    root_conftest.pytest_itemcollected(item)
    assert item.markers == {"xpu_1", "post_merge"}


def test_defaults_disabled_by_env(monkeypatch, root_conftest):
    """The marker gate relies on this opt-out to see authored markers only."""
    monkeypatch.setenv(root_conftest._NO_DEFAULT_MARKERS_ENV, "1")
    item = _FakeItem()
    root_conftest.pytest_itemcollected(item)
    assert item.markers == set()


def test_marker_gate_opts_out_with_the_same_env_var(root_conftest):
    """A typo in the report's copy of the name would silently re-mask the gate."""
    source = Path(__file__).parent / "report_pytest_markers.py"
    assert root_conftest._NO_DEFAULT_MARKERS_ENV in source.read_text()
