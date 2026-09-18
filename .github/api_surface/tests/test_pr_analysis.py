# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ops/api_surface/pr_analysis.py.

The analyzer is pure -- it consumes two in-memory snapshots -- so every test
builds synthetic :class:`SurfaceSnapshot` objects. No git checkout or network.
"""

from __future__ import annotations

import json

import pytest
from api_surface.models import SurfaceSnapshot, SurfaceSymbol
from api_surface.pr_analysis import IMPACT_FORMATS, analyze_impact, render_impact


def _sym(
    sid: str, *, stability: str = "stable", deprecated: bool = False, sig: str = ""
) -> SurfaceSymbol:
    return SurfaceSymbol(
        surface="python",
        kind="function",
        id=sid,
        signature=sig,
        stability=stability,
        deprecated=deprecated,
    )


def _snap(ref: str, symbols: list[SurfaceSymbol]) -> SurfaceSnapshot:
    return SurfaceSnapshot(
        release="", ref=ref, coverage={"python": True}, symbols=symbols
    )


def _base() -> SurfaceSnapshot:
    return _snap(
        "v1.2.0",
        [
            _sym("python:m.StableGone"),
            _sym("python:m.PreviewGone", stability="preview"),
            _sym("python:m.DepGone", deprecated=True),
            _sym("python:m.Changed", sig="(a)"),
            _sym("python:m.Keep"),
        ],
    )


def _head() -> SurfaceSnapshot:
    return _snap(
        "pr-head",
        [
            _sym("python:m.Changed", sig="(a, b)"),
            _sym("python:m.Keep"),
            _sym("python:m.NewThing"),
        ],
    )


class TestClassification:
    @pytest.mark.unit
    def test_stable_removal_is_breaking(self) -> None:
        data = analyze_impact(_base(), _head()).data
        ids = {c["id"] for c in data["breaking"]}
        assert "python:m.StableGone" in ids
        assert "python:m.Changed" in ids  # signature change on stable

    @pytest.mark.unit
    def test_preview_and_deprecated_removals_are_informational(self) -> None:
        data = analyze_impact(_base(), _head()).data
        info_ids = {c["id"] for c in data["informational"]}
        breaking_ids = {c["id"] for c in data["breaking"]}
        assert "python:m.PreviewGone" in info_ids
        assert "python:m.DepGone" in info_ids
        assert "python:m.PreviewGone" not in breaking_ids
        assert "python:m.DepGone" not in breaking_ids

    @pytest.mark.unit
    def test_addition_is_additive(self) -> None:
        data = analyze_impact(_base(), _head()).data
        assert {c["id"] for c in data["additive"]} == {"python:m.NewThing"}

    @pytest.mark.unit
    def test_counts_and_gate_signal(self) -> None:
        result = analyze_impact(_base(), _head())
        counts = result.data["counts"]
        assert counts["breaking"] == 2
        assert counts["informational"] == 2
        assert counts["additive"] == 1
        assert result.metadata["has_breaking"] is True
        assert result.data["base_ref"] == "v1.2.0"
        assert result.data["head_ref"] == "pr-head"

    @pytest.mark.unit
    def test_no_breaking_when_only_additions(self) -> None:
        base = _snap("a", [_sym("python:m.Keep")])
        head = _snap("b", [_sym("python:m.Keep"), _sym("python:m.New")])
        result = analyze_impact(base, head)
        assert result.metadata["has_breaking"] is False
        assert result.data["counts"]["additive"] == 1

    @pytest.mark.unit
    def test_stable_rename_is_breaking(self) -> None:
        base = _snap("a", [_sym("python:m.old", sig="(value: int)")])
        head = _snap("b", [_sym("python:m.new", sig="(value: int)")])

        result = analyze_impact(base, head)

        assert result.metadata["has_breaking"] is True
        assert result.data["breaking"][0]["change_type"] == "renamed_function"
        assert result.data["breaking"][0]["id"] == "python:m.old"
        assert result.data["breaking"][0]["relocated_to"] == "python:m.new"

    @pytest.mark.unit
    def test_deprecated_rename_is_informational(self) -> None:
        base = _snap(
            "a",
            [_sym("python:m.old", sig="(value: int)", deprecated=True)],
        )
        head = _snap("b", [_sym("python:m.new", sig="(value: int)")])

        result = analyze_impact(base, head)

        assert result.metadata["has_breaking"] is False
        assert result.data["informational"][0]["change_type"] == "renamed_function"

    @pytest.mark.unit
    def test_stable_relocation_is_breaking(self) -> None:
        detail = {"rust:old": True, "rust:new": True}
        base = SurfaceSnapshot(
            release="",
            ref="a",
            coverage={"rust": True},
            coverage_detail=detail,
            symbols=[
                SurfaceSymbol(
                    surface="rust",
                    kind="struct",
                    id="rust:old::api::Thing",
                    signature="pub struct Thing",
                )
            ],
        )
        head = SurfaceSnapshot(
            release="",
            ref="b",
            coverage={"rust": True},
            coverage_detail=detail,
            symbols=[
                SurfaceSymbol(
                    surface="rust",
                    kind="struct",
                    id="rust:new::api::Thing",
                    signature="pub struct Thing",
                )
            ],
        )

        result = analyze_impact(base, head)

        assert result.metadata["has_breaking"] is True
        assert result.data["breaking"][0]["change_type"] == "relocated"

    @pytest.mark.unit
    def test_unknown_stability_is_protected(self) -> None:
        corrupted = _sym("python:m.Gone")
        corrupted.stability = "unknown"

        result = analyze_impact(_snap("a", [corrupted]), _snap("b", []))

        assert result.metadata["has_breaking"] is True

    @pytest.mark.unit
    def test_uncovered_surface_not_read_as_removal(self) -> None:
        """A surface absent from head coverage must not register mass removals."""
        base = _snap("a", [_sym("python:m.A")])
        head = SurfaceSnapshot(
            release="", ref="b", coverage={"python": False}, symbols=[]
        )
        result = analyze_impact(base, head)
        assert result.data["counts"]["breaking"] == 0


class TestRendering:
    @pytest.mark.unit
    def test_md_lists_breaking_and_summary(self) -> None:
        out = render_impact(analyze_impact(_base(), _head()).data, "md")
        assert "# API Surface Impact" in out
        assert "## Breaking Changes (2)" in out
        assert "python:m.StableGone" in out
        assert "## Added (1)" in out

    @pytest.mark.unit
    def test_md_no_breaking_states_none(self) -> None:
        base = _snap("a", [_sym("python:m.Keep")])
        head = _snap("b", [_sym("python:m.Keep"), _sym("python:m.New")])
        out = render_impact(analyze_impact(base, head).data, "md")
        assert "None. No stable public API was removed or altered." in out

    @pytest.mark.unit
    def test_slack_format_native(self) -> None:
        out = render_impact(analyze_impact(_base(), _head()).data, "slack")
        assert out.startswith("*API Surface Impact:*")
        assert ":warning:" in out
        assert "•" in out

    @pytest.mark.unit
    def test_json_roundtrips(self) -> None:
        data = analyze_impact(_base(), _head()).data
        out = render_impact(data, "json")
        assert json.loads(out)["counts"]["breaking"] == 2

    @pytest.mark.unit
    def test_relocated_branch_rendered(self) -> None:
        """render_impact surfaces relocated entries with the new id (data-driven)."""
        data = {
            "base_ref": "a",
            "head_ref": "b",
            "counts": {
                "breaking": 0,
                "informational": 0,
                "additive": 0,
                "relocated": 1,
                "total": 1,
            },
            "breaking": [],
            "informational": [],
            "additive": [],
            "relocated": [
                {
                    "id": "rust:old::Foo",
                    "surface": "rust",
                    "change_type": "relocated",
                    "relocated_to": "rust:new::Foo",
                }
            ],
            "coverage_gaps": [],
        }
        out = render_impact(data, "md")
        assert "## Relocated (1)" in out
        assert "rust:old::Foo -> rust:new::Foo" in out

    @pytest.mark.unit
    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            render_impact({}, "xml")

    @pytest.mark.unit
    def test_formats_constant(self) -> None:
        assert {"md", "slack", "json"} == IMPACT_FORMATS


class TestWaivers:
    @pytest.mark.unit
    def test_waived_stable_removal_clears_has_breaking(self) -> None:
        from api_surface.suppressions import Suppression, SuppressionSet

        waivers = SuppressionSet(
            rules=[
                Suppression(id="python:m.StableGone", reason="never public"),
                Suppression(id="python:m.Changed", reason="signature widened"),
            ]
        )

        result = analyze_impact(_base(), _head(), suppressions=waivers)

        assert result.metadata["has_breaking"] is False
        assert result.data["breaking"] == []
        assert result.data["counts"]["waived"] == 2
        waived = {c["id"]: c["reason"] for c in result.data["waived"]}
        assert waived["python:m.StableGone"] == "never public"
        assert waived["python:m.Changed"] == "signature widened"

    @pytest.mark.unit
    def test_expired_waiver_still_blocks(self) -> None:
        from api_surface.suppressions import Suppression, SuppressionSet

        waivers = SuppressionSet(
            rules=[
                Suppression(
                    id="python:m.StableGone",
                    reason="never public",
                    until="1.3.0",
                )
            ]
        )

        result = analyze_impact(_base(), _head(), release="1.4.0", suppressions=waivers)

        assert result.metadata["has_breaking"] is True
        assert "python:m.StableGone" in {c["id"] for c in result.data["breaking"]}
        assert result.data["counts"]["waived"] == 0

    @pytest.mark.unit
    def test_md_render_lists_waived_section(self) -> None:
        from api_surface.suppressions import Suppression, SuppressionSet

        waivers = SuppressionSet(
            rules=[Suppression(id="python:m.StableGone", reason="never public")]
        )
        data = analyze_impact(_base(), _head(), suppressions=waivers).data

        out = render_impact(data, "md")

        assert "- Waived: 1" in out
        assert "## Waived (1)" in out
        assert "python:m.StableGone" in out
        assert "never public" in out
