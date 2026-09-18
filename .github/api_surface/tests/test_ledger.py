# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ops/api_surface/ledger.py - lifecycle merge + persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from api_surface.ledger import load_ledger, load_provenance, merge_changes, save_ledger
from api_surface.models import (
    SurfaceChange,
    SurfaceLedger,
    SurfaceSnapshot,
    SurfaceSymbol,
)

CONFIG = Path(__file__).with_name("crate-provenance.yaml")


def _sym(sid: str, surface: str = "python", **kw) -> SurfaceSymbol:
    return SurfaceSymbol(surface=surface, kind="function", id=sid, **kw)


def _snap(
    release: str, symbols: list[SurfaceSymbol], coverage: dict[str, bool]
) -> SurfaceSnapshot:
    return SurfaceSnapshot(
        release=release, ref=f"v{release}", coverage=coverage, symbols=symbols
    )


def _entry_by_id(ledger: SurfaceLedger, sid: str):
    return next(e for e in ledger.entries if e.id == sid)


class TestLifecycle:
    @pytest.mark.unit
    def test_added_then_stable(self) -> None:
        """Debut release -> added; surviving to a later release -> stable."""
        s10 = _snap("1.0.0", [_sym("python:m.A")], {"python": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        e = _entry_by_id(l10, "python:m.A")
        assert e.status == "added"
        assert e.added_in == "1.0.0"

        s11 = _snap("1.1.0", [_sym("python:m.A")], {"python": True})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11)
        e = _entry_by_id(l11, "python:m.A")
        assert e.status == "stable"
        assert e.added_in == "1.0.0"
        assert e.last_seen == "1.1.0"

    @pytest.mark.unit
    def test_removal_via_snapshot_absence(self) -> None:
        """Absent from a covered surface in the new snapshot -> removed."""
        s10 = _snap("1.0.0", [_sym("python:m.A")], {"python": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        s11 = _snap("1.1.0", [], {"python": True})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11)
        e = _entry_by_id(l11, "python:m.A")
        assert e.status == "removed"
        assert e.removed_in == "1.1.0"

    @pytest.mark.unit
    def test_coverage_gap_does_not_remove(self) -> None:
        """If the surface is uncovered in the new snapshot, no removal occurs."""
        s10 = _snap("1.0.0", [_sym("crd:K@v1.x", surface="crd")], {"crd": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        s11 = _snap("1.1.0", [], {"crd": False})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11)
        assert _entry_by_id(l11, "crd:K@v1.x").status == "added"

    @pytest.mark.unit
    def test_relocation_not_removal(self) -> None:
        """A crate that left the monorepo is relocated, not removed."""
        prov = load_provenance(CONFIG)
        s10 = _snap(
            "1.0.0",
            [_sym("rust:dynamo-parsers::lib::parse", surface="rust")],
            {"rust": True},
        )
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10, provenance=prov)
        s11 = _snap("1.1.0", [], {"rust": True})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11, provenance=prov)
        e = _entry_by_id(l11, "rust:dynamo-parsers::lib::parse")
        assert e.status == "relocated"
        assert e.relocated_to == "ai-dynamo/dynamo-parsers"
        assert e.relocated_in == "1.1.0"
        assert e.removed_in == ""

    @pytest.mark.unit
    def test_native_marker_deprecation(self) -> None:
        """A symbol carrying a native marker becomes deprecated (source=marker)."""
        s = _snap(
            "1.2.0",
            [_sym("python:m.old", deprecated=True, deprecated_note="use new")],
            {"python": True},
        )
        ledger = merge_changes(SurfaceLedger(), [], "1.2.0", snapshot=s)
        e = _entry_by_id(ledger, "python:m.old")
        assert e.status == "deprecated"
        assert e.deprecated_in == "1.2.0"
        assert e.source == "marker"
        assert e.note == "use new"


class TestSymbolRelocation:
    @pytest.mark.unit
    def test_diff_relocation_change_sets_relocated_not_removed(self) -> None:
        """A diff-sourced ``relocated`` change marks the origin id relocated to
        the new id, overriding the transient snapshot-absence removal."""
        old_id = "rust:dynamo-runtime::storage::Cache"
        new_id = "rust:dynamo-llm::storage::Cache"
        s10 = _snap("1.0.0", [_sym(old_id, surface="rust")], {"rust": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        change = SurfaceChange(
            change_type="relocated",
            id=old_id,
            surface="rust",
            relocated_to=new_id,
            source="diff",
            summary=f"Relocated {old_id} -> {new_id}",
        )
        s11 = _snap("1.1.0", [_sym(new_id, surface="rust")], {"rust": True})
        l11 = merge_changes(l10, [change], "1.1.0", snapshot=s11)
        origin = _entry_by_id(l11, old_id)
        assert origin.status == "relocated"
        assert origin.relocated_to == new_id
        assert origin.relocated_in == "1.1.0"
        assert origin.removed_in == ""
        # The destination id is present and tracked as a fresh add.
        assert _entry_by_id(l11, new_id).status == "added"

    @pytest.mark.unit
    def test_relocation_change_is_idempotent(self) -> None:
        """Re-merging the same release leaves the relocated entry unchanged."""
        old_id = "rust:dynamo-runtime::storage::Cache"
        new_id = "rust:dynamo-llm::storage::Cache"
        s10 = _snap("1.0.0", [_sym(old_id, surface="rust")], {"rust": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        change = SurfaceChange(
            change_type="relocated",
            id=old_id,
            surface="rust",
            relocated_to=new_id,
            source="diff",
        )
        s11 = _snap("1.1.0", [_sym(new_id, surface="rust")], {"rust": True})
        once = merge_changes(l10, [change], "1.1.0", snapshot=s11)
        twice = merge_changes(once, [change], "1.1.0", snapshot=s11)
        assert [e.to_dict() for e in once.sorted_entries()] == [
            e.to_dict() for e in twice.sorted_entries()
        ]


class TestRustMarkerReconcile:
    @pytest.mark.unit
    def test_marker_reconciles_to_impl_qualified_entry(self) -> None:
        """A marker id lacking the impl/inline-mod segment folds onto the unique
        fully-qualified rust entry sharing its crate and trailing name."""
        full_id = "rust:dynamo-llm::storage::PinnedStorage::new"
        s = _snap("1.2.0", [_sym(full_id, surface="rust")], {"rust": True})
        ledger = merge_changes(SurfaceLedger(), [], "1.2.0", snapshot=s)
        # The marker scanner only knows the file-level module path.
        marker = SurfaceChange(
            change_type="deprecated_api",
            id="rust:dynamo-llm::storage::new",
            surface="rust",
            summary="use with_capacity",
            source="marker",
            confidence="high",
        )
        merged = merge_changes(ledger, [marker], "1.2.0", snapshot=s)
        e = _entry_by_id(merged, full_id)
        assert e.status == "deprecated"
        assert e.note == "use with_capacity"
        # No spurious second entry for the under-qualified marker id.
        assert not any(e.id == "rust:dynamo-llm::storage::new" for e in merged.entries)

    @pytest.mark.unit
    def test_ambiguous_marker_keeps_own_entry(self) -> None:
        """When >1 rust entry shares the (crate, trailing name), do not guess."""
        s = _snap(
            "1.2.0",
            [
                _sym("rust:dynamo-llm::a::Foo::new", surface="rust"),
                _sym("rust:dynamo-llm::b::Bar::new", surface="rust"),
            ],
            {"rust": True},
        )
        ledger = merge_changes(SurfaceLedger(), [], "1.2.0", snapshot=s)
        marker = SurfaceChange(
            change_type="deprecated_api",
            id="rust:dynamo-llm::new",
            surface="rust",
            source="marker",
        )
        merged = merge_changes(ledger, [marker], "1.2.0", snapshot=s)
        # Ambiguous -> the marker keeps its own id rather than guessing.
        assert any(e.id == "rust:dynamo-llm::new" for e in merged.entries)
        assert _entry_by_id(merged, "rust:dynamo-llm::a::Foo::new").status == "added"


class TestReviewGating:
    @pytest.mark.unit
    def test_label_source_needs_review(self) -> None:
        """A label/LLM-sourced change lands needs_review (no snapshot path)."""
        change = SurfaceChange(
            change_type="deprecated_api",
            id="python:m.future",
            surface="python",
            summary="planned deprecation announced in PR",
            source="label",
            pr_number=4242,
        )
        ledger = merge_changes(SurfaceLedger(), [change], "1.2.0")
        e = _entry_by_id(ledger, "python:m.future")
        assert e.status == "deprecated"
        assert e.review_status == "needs_review"
        assert e.pr_number == 4242
        # A label-only new entry carries honest provenance, not the default diff.
        assert e.source == "label"

    @pytest.mark.unit
    @pytest.mark.parametrize("surface", ["config", "env", "helm"])
    def test_heuristic_removal_needs_review(self, surface: str) -> None:
        """A removal on a regex/line-scan surface is parked needs_review, not auto."""
        s10 = _snap("1.0.0", [_sym(f"{surface}:X", surface=surface)], {surface: True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        s11 = _snap("1.1.0", [], {surface: True})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11)
        e = _entry_by_id(l11, f"{surface}:X")
        assert e.status == "removed"
        assert e.review_status == "needs_review"

    @pytest.mark.unit
    def test_parsed_surface_removal_stays_auto(self) -> None:
        """A removal on an AST/parsed surface (python) is asserted, not gated."""
        s10 = _snap("1.0.0", [_sym("python:m.A")], {"python": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        s11 = _snap("1.1.0", [], {"python": True})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11)
        e = _entry_by_id(l11, "python:m.A")
        assert e.status == "removed"
        assert e.review_status == "auto"

    @pytest.mark.unit
    def test_reappearance_clears_heuristic_review_flag(self) -> None:
        """A config symbol that reappears (the ApiError class) clears needs_review."""
        s10 = _snap("1.0.0", [_sym("config:C", surface="config")], {"config": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        s11 = _snap("1.1.0", [], {"config": True})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11)
        assert _entry_by_id(l11, "config:C").review_status == "needs_review"
        # Reappears at 1.2.0 -> added again, review flag cleared.
        s12 = _snap("1.2.0", [_sym("config:C", surface="config")], {"config": True})
        l12 = merge_changes(l11, [], "1.2.0", snapshot=s12)
        e = _entry_by_id(l12, "config:C")
        assert e.status == "added"
        assert e.review_status == "auto"

    @pytest.mark.unit
    def test_confirmed_heuristic_removal_is_not_demoted(self) -> None:
        """A human-confirmed removal stays confirmed across a re-merge."""
        s10 = _snap("1.0.0", [_sym("env:DYN_X", surface="env")], {"env": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        s11 = _snap("1.1.0", [], {"env": True})
        l11 = merge_changes(l10, [], "1.1.0", snapshot=s11)
        _entry_by_id(l11, "env:DYN_X").review_status = "confirmed"
        l11b = merge_changes(l11, [], "1.1.0", snapshot=s11)
        assert _entry_by_id(l11b, "env:DYN_X").review_status == "confirmed"


class TestIdempotency:
    @pytest.mark.unit
    def test_rerun_same_release_is_identical(self) -> None:
        """Merging the same release twice yields an identical ledger body."""
        s10 = _snap("1.0.0", [_sym("python:m.A"), _sym("python:m.B")], {"python": True})
        l10 = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s10)
        s11 = _snap("1.1.0", [_sym("python:m.A")], {"python": True})
        once = merge_changes(l10, [], "1.1.0", snapshot=s11)
        twice = merge_changes(once, [], "1.1.0", snapshot=s11)

        def bodies(ledger: SurfaceLedger):
            return [e.to_dict() for e in ledger.sorted_entries()]

        assert bodies(once) == bodies(twice)


class TestPersistence:
    @pytest.mark.unit
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        assert load_ledger(tmp_path / "nope.json").entries == []

    @pytest.mark.unit
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        s = _snap("1.0.0", [_sym("python:m.A")], {"python": True})
        ledger = merge_changes(SurfaceLedger(), [], "1.0.0", snapshot=s)
        path = tmp_path / "api_surface" / "ledger.json"
        save_ledger(ledger, path)
        assert path.exists()
        restored = load_ledger(path)
        assert [e.to_dict() for e in restored.sorted_entries()] == [
            e.to_dict() for e in ledger.sorted_entries()
        ]

    @pytest.mark.unit
    def test_provenance_loads_external_crates(self) -> None:
        prov = load_provenance(CONFIG)
        assert prov["dynamo-parsers"]["status"] == "external"
        assert prov["fastokens"]["repo"] == "ai-dynamo/fastokens"
