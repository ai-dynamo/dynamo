# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OSRB CSV diff.

Run from the repo root with the compliance package on the path:

    PYTHONPATH=container python -m pytest container/compliance/tests/test_diff_osrb_csv.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compliance.diff_osrb_csv import (
    ADDED,
    BASELINE_UNAVAILABLE,
    LICENSE_CHANGED,
    REMOVED,
    VERSION_AND_LICENSE_CHANGED,
    VERSION_CHANGED,
    baseline_unavailable_rows,
    compute_diff,
    read_osrb_csv,
    write_diff_csv,
)
from compliance.generators.common import Component, write_merged_csv

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _write_csv(path: Path, components: list[Component]) -> Path:
    return write_merged_csv(components, {}, path)


def _c(eco: str, name: str, version: str, spdx: str = "MIT") -> Component:
    return Component(ecosystem=eco, name=name, version=version, spdx=spdx)


def _rows(
    components: list[Component], tmp_path: Path, fname: str
) -> list[dict[str, str]]:
    return read_osrb_csv(_write_csv(tmp_path / fname, components))


class TestChangeClassification:
    def test_added(self, tmp_path):
        cur = _rows([_c("python", "brand-new", "1.0.0")], tmp_path, "cur.csv")
        base = _rows([], tmp_path, "base.csv")
        diff = compute_diff(cur, base)
        assert [r["change"] for r in diff] == [ADDED]
        assert diff[0]["new_version"] == "1.0.0"
        assert diff[0]["old_version"] == ""

    def test_removed(self, tmp_path):
        cur = _rows([], tmp_path, "cur.csv")
        base = _rows([_c("python", "gone", "2.0.0")], tmp_path, "base.csv")
        diff = compute_diff(cur, base)
        assert [r["change"] for r in diff] == [REMOVED]
        assert diff[0]["old_version"] == "2.0.0"
        assert diff[0]["new_version"] == ""

    def test_version_changed(self, tmp_path):
        cur = _rows([_c("rust", "serde", "1.1.0", "MIT")], tmp_path, "cur.csv")
        base = _rows([_c("rust", "serde", "1.0.0", "MIT")], tmp_path, "base.csv")
        diff = compute_diff(cur, base)
        assert [r["change"] for r in diff] == [VERSION_CHANGED]
        assert (diff[0]["old_version"], diff[0]["new_version"]) == ("1.0.0", "1.1.0")

    def test_license_changed_only(self, tmp_path):
        cur = _rows([_c("rust", "serde", "1.0.0", "Apache-2.0")], tmp_path, "cur.csv")
        base = _rows([_c("rust", "serde", "1.0.0", "MIT")], tmp_path, "base.csv")
        diff = compute_diff(cur, base)
        assert [r["change"] for r in diff] == [LICENSE_CHANGED]
        assert (diff[0]["old_spdx"], diff[0]["new_spdx"]) == ("MIT", "Apache-2.0")

    def test_version_and_license_changed(self, tmp_path):
        cur = _rows([_c("rust", "serde", "1.1.0", "Apache-2.0")], tmp_path, "cur.csv")
        base = _rows([_c("rust", "serde", "1.0.0", "MIT")], tmp_path, "base.csv")
        diff = compute_diff(cur, base)
        assert [r["change"] for r in diff] == [VERSION_AND_LICENSE_CHANGED]

    def test_identical_is_empty(self, tmp_path):
        comps = [_c("python", "requests", "2.0.0")]
        assert (
            compute_diff(
                _rows(comps, tmp_path, "c.csv"), _rows(comps, tmp_path, "b.csv")
            )
            == []
        )


class TestOrdering:
    def test_bucket_then_ecosystem_then_name(self, tmp_path):
        cur = _rows(
            [
                _c("python", "z-added", "1.0.0"),
                _c("rust", "a-added", "1.0.0"),
                _c("python", "verbump", "2.0.0", "MIT"),
                _c("rust", "verbump2", "2.0.0", "Apache-2.0"),  # version+license
                _c("go", "liconly", "1.0.0", "BSD-3-Clause"),  # license only
            ],
            tmp_path,
            "cur.csv",
        )
        base = _rows(
            [
                _c("python", "verbump", "1.0.0", "MIT"),
                _c("rust", "verbump2", "1.0.0", "MIT"),  # version+license changed
                _c("go", "liconly", "1.0.0", "MIT"),  # license only changed
                _c("dpkg", "removed-pkg", "3.0.0"),
            ],
            tmp_path,
            "base.csv",
        )
        diff = compute_diff(cur, base)
        got = [(r["change"], r["ecosystem"], r["name"]) for r in diff]
        assert got == [
            # additions: rust before python (ecosystem order), then name
            (ADDED, "rust", "a-added"),
            (ADDED, "python", "z-added"),
            # version+license change bucket
            (VERSION_AND_LICENSE_CHANGED, "rust", "verbump2"),
            # version-only change bucket
            (VERSION_CHANGED, "python", "verbump"),
            # license-only change bucket
            (LICENSE_CHANGED, "go", "liconly"),
            # removals
            (REMOVED, "dpkg", "removed-pkg"),
        ]


class TestMultiVersion:
    def test_multi_version_mismatch_degrades_to_add_remove(self, tmp_path):
        cur = _rows(
            [_c("python", "pkg", "1.0.0"), _c("python", "pkg", "2.0.0")],
            tmp_path,
            "cur.csv",
        )
        base = _rows([_c("python", "pkg", "1.0.0")], tmp_path, "base.csv")
        diff = compute_diff(cur, base)
        # 1.0.0 unchanged (present both sides), 2.0.0 is a new version -> added
        assert [(r["change"], r["new_version"]) for r in diff] == [(ADDED, "2.0.0")]


class TestBaselineUnavailable:
    def test_marker_rows(self):
        rows = baseline_unavailable_rows("release baseline v1.2.1 (expired)")
        assert len(rows) == 1
        assert rows[0]["change"] == BASELINE_UNAVAILABLE
        assert rows[0]["name"] == "release baseline v1.2.1 (expired)"

    def test_write_and_reread(self, tmp_path):
        out = tmp_path / "x.diff.csv"
        write_diff_csv(baseline_unavailable_rows("lbl"), out)
        text = out.read_text()
        assert text.startswith(
            "change,ecosystem,name,old_version,new_version,old_spdx,new_spdx,source_url,notes\n"
        )
        assert BASELINE_UNAVAILABLE in text
