# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the default-marker hook in tests/conftest.py."""

from pathlib import Path

import pytest

from tests.conftest import _NO_DEFAULT_MARKERS_ENV, pytest_itemcollected

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _FakeItem:
    """Minimal stand-in for a collected pytest item."""

    def __init__(self, *markers: str):
        self.markers = set(markers)

    def get_closest_marker(self, name: str):
        return name if name in self.markers else None

    def add_marker(self, marker) -> None:
        self.markers.add(marker.name)


def test_unmarked_test_gets_both_defaults():
    item = _FakeItem()
    pytest_itemcollected(item)
    assert item.markers == {"pre_merge", "gpu_0"}


def test_machine_marker_is_not_overridden():
    """A gpu_2 test must not also become gpu_0 and match CPU-only selectors."""
    item = _FakeItem("gpu_2")
    pytest_itemcollected(item)
    assert item.markers == {"gpu_2", "pre_merge"}


def test_non_gpu_machine_markers_are_recognized():
    """xpu/h100/k8s tests declare hardware and must keep it."""
    item = _FakeItem("xpu_1", "post_merge")
    pytest_itemcollected(item)
    assert item.markers == {"xpu_1", "post_merge"}


def test_defaults_disabled_by_env(monkeypatch):
    """The marker gate relies on this opt-out to see authored markers only."""
    monkeypatch.setenv(_NO_DEFAULT_MARKERS_ENV, "1")
    item = _FakeItem()
    pytest_itemcollected(item)
    assert item.markers == set()


def test_marker_gate_opts_out_with_the_same_env_var():
    """A typo in the report's copy of the name would silently re-mask the gate."""
    source = Path(__file__).parent / "report_pytest_markers.py"
    assert _NO_DEFAULT_MARKERS_ENV in source.read_text()
