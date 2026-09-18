# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for models/api_surface.py - API surface tracker data models."""

from __future__ import annotations

from pathlib import Path

import pytest
from api_surface.models import (
    BREAKING_CHANGE_TYPES,
    CHANGE_TYPES,
    LEDGER_SCHEMA_VERSION,
    LIFECYCLE_STATUSES,
    REVIEW_STATUSES,
    SNAPSHOT_SCHEMA_VERSION,
    SOURCES,
    STABILITY_TIERS,
    SURFACES,
    LedgerEntry,
    SurfaceChange,
    SurfaceExtractor,
    SurfaceLedger,
    SurfaceSnapshot,
    SurfaceSymbol,
)
from api_surface.results import OperationResult

# =============================================================================
# Constants / taxonomy reuse
# =============================================================================


class TestConstants:
    """The vocabulary stays aligned with the release-notes taxonomy."""

    @pytest.mark.unit
    def test_change_types_superset_of_breaking(self) -> None:
        """CHANGE_TYPES extends, never replaces, BREAKING_CHANGE_TYPES."""
        assert BREAKING_CHANGE_TYPES <= CHANGE_TYPES
        assert {
            "added",
            "removed",
            "signature_changed",
            "relocated",
            "served_version_removed",
        } <= (CHANGE_TYPES)

    @pytest.mark.unit
    def test_enum_sets_are_frozen_shapes(self) -> None:
        """Each enum set has the exact expected members."""
        assert {"stable", "preview", "experimental"} == STABILITY_TIERS
        assert {"auto", "needs_review", "confirmed"} == REVIEW_STATUSES
        assert {"diff", "marker", "llm", "label"} == SOURCES
        assert {
            "stable",
            "added",
            "deprecated",
            "scheduled_for_removal",
            "removed",
            "relocated",
        } == LIFECYCLE_STATUSES
        assert "python" in SURFACES and "crd" in SURFACES and "rust" in SURFACES

    @pytest.mark.unit
    def test_schema_versions(self) -> None:
        """Schema versions are positive ints."""
        assert SNAPSHOT_SCHEMA_VERSION >= 1
        assert LEDGER_SCHEMA_VERSION >= 1


# =============================================================================
# SurfaceSymbol
# =============================================================================


class TestSurfaceSymbol:
    """Tests for SurfaceSymbol dataclass."""

    @pytest.mark.unit
    def test_basic_creation_defaults(self) -> None:
        """Required fields set; defaults apply."""
        sym = SurfaceSymbol(
            surface="python",
            kind="method",
            id="python:dynamo.runtime.DistributedRuntime.create",
        )
        assert sym.surface == "python"
        assert sym.kind == "method"
        assert sym.signature == ""
        assert sym.component == ""
        assert sym.stability == "stable"
        assert sym.deprecated is False
        assert sym.deprecated_note == ""
        assert sym.metadata == {}

    @pytest.mark.unit
    def test_to_dict_all_fields(self) -> None:
        """to_dict emits every field with a copied metadata dict."""
        sym = SurfaceSymbol(
            surface="crd",
            kind="field",
            id="crd:DynamoGraphDeployment@v1alpha1.spec.services[].replicas",
            signature="integer",
            component="Operator",
            stability="preview",
            deprecated=True,
            deprecated_note="use scaling.replicas",
            metadata={"version": "v1alpha1", "served": True, "storage": True},
        )
        d = sym.to_dict()
        assert d == {
            "surface": "crd",
            "kind": "field",
            "id": "crd:DynamoGraphDeployment@v1alpha1.spec.services[].replicas",
            "signature": "integer",
            "component": "Operator",
            "stability": "preview",
            "deprecated": True,
            "deprecated_note": "use scaling.replicas",
            "metadata": {"version": "v1alpha1", "served": True, "storage": True},
        }
        d["metadata"]["served"] = False
        assert sym.metadata["served"] is True

    @pytest.mark.unit
    def test_round_trip(self) -> None:
        """from_dict(to_dict(x)) reconstructs an equal symbol."""
        sym = SurfaceSymbol(
            surface="metric",
            kind="metric",
            id="metric:frontend_service::REQUESTS_TOTAL",
            signature="requests_total",
            component="Frontend",
            metadata={"name": "requests_total", "module": "frontend_service"},
        )
        assert SurfaceSymbol.from_dict(sym.to_dict()) == sym

    @pytest.mark.unit
    def test_invalid_stability_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="stability"):
            SurfaceSymbol(
                surface="python",
                kind="function",
                id="python:m.f",
                stability="typo",
            )

    @pytest.mark.unit
    def test_non_boolean_deprecation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="deprecated"):
            SurfaceSymbol(
                surface="python",
                kind="function",
                id="python:m.f",
                deprecated="false",  # type: ignore[arg-type]
            )


# =============================================================================
# SurfaceSnapshot
# =============================================================================


class TestSurfaceSnapshot:
    """Tests for SurfaceSnapshot dataclass."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        """Defaults: repo, schema_version, empty coverage/symbols."""
        snap = SurfaceSnapshot(release="1.2.0", ref="v1.2.0")
        assert snap.repo == "ai-dynamo/dynamo"
        assert snap.schema_version == SNAPSHOT_SCHEMA_VERSION
        assert snap.coverage == {}
        assert snap.symbols == []

    @pytest.mark.unit
    def test_to_dict_sorts_symbols_and_coverage(self) -> None:
        """Symbols and coverage are emitted in sorted order for clean diffs."""
        snap = SurfaceSnapshot(
            release="1.2.0",
            ref="v1.2.0",
            coverage={"python": True, "crd": True, "rust": False},
            symbols=[
                SurfaceSymbol(surface="python", kind="class", id="python:z.Z"),
                SurfaceSymbol(surface="python", kind="class", id="python:a.A"),
            ],
        )
        d = snap.to_dict()
        assert [s["id"] for s in d["symbols"]] == ["python:a.A", "python:z.Z"]
        assert list(d["coverage"].keys()) == ["crd", "python", "rust"]

    @pytest.mark.unit
    def test_round_trip(self) -> None:
        """Round trip preserves release/ref/coverage/symbols."""
        snap = SurfaceSnapshot(
            release="1.1.0",
            ref="v1.1.0",
            generated_at="2026-04-29T00:00:00Z",
            coverage={"python": True, "helm": True},
            symbols=[
                SurfaceSymbol(surface="helm", kind="value", id="helm:platform:x.y")
            ],
        )
        restored = SurfaceSnapshot.from_dict(snap.to_dict())
        assert restored.release == snap.release
        assert restored.coverage == snap.coverage
        assert restored.symbols == snap.symbols

    @pytest.mark.unit
    def test_future_schema_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="snapshot schema"):
            SurfaceSnapshot.from_dict(
                {
                    "schema_version": SNAPSHOT_SCHEMA_VERSION + 1,
                    "release": "1.2.0",
                    "ref": "v1.2.0",
                    "symbols": [],
                }
            )


# =============================================================================
# SurfaceChange
# =============================================================================


class TestSurfaceChange:
    """Tests for SurfaceChange dataclass."""

    @pytest.mark.unit
    def test_diff_event_has_no_pr(self) -> None:
        """A code-grounded diff event defaults to source=diff, pr_number=None."""
        change = SurfaceChange(change_type="removed", id="python:dynamo.llm.OldClass")
        assert change.source == "diff"
        assert change.pr_number is None
        assert change.confidence == "medium"

    @pytest.mark.unit
    def test_round_trip(self) -> None:
        """Round trip preserves all fields including pr_number and similarity."""
        change = SurfaceChange(
            change_type="renamed_function",
            id="python:dynamo.runtime.foo",
            surface="python",
            component="Runtime",
            from_signature="foo(a, b)",
            to_signature="bar(a, b)",
            summary="renamed foo -> bar",
            confidence="high",
            source="marker",
            pr_number=4321,
            similarity=0.94,
        )
        assert SurfaceChange.from_dict(change.to_dict()) == change


# =============================================================================
# LedgerEntry
# =============================================================================


class TestLedgerEntry:
    """Tests for LedgerEntry dataclass."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        """Defaults: stable/auto/medium/diff, empty lifecycle fields."""
        entry = LedgerEntry(id="python:dynamo.runtime.X", surface="python")
        assert entry.status == "stable"
        assert entry.review_status == "auto"
        assert entry.confidence == "medium"
        assert entry.source == "diff"
        assert entry.signature_history == []

    @pytest.mark.unit
    def test_computed_properties(self) -> None:
        """is_deprecated and needs_review reflect status/review_status."""
        stable = LedgerEntry(id="a", surface="python")
        assert stable.is_deprecated is False
        assert stable.needs_review is False

        dep = LedgerEntry(id="b", surface="python", status="deprecated")
        sched = LedgerEntry(id="c", surface="python", status="scheduled_for_removal")
        review = LedgerEntry(id="d", surface="python", review_status="needs_review")
        assert dep.is_deprecated is True
        assert sched.is_deprecated is True
        assert review.needs_review is True

    @pytest.mark.unit
    def test_to_dict_includes_computed(self) -> None:
        """to_dict carries the computed properties per CLAUDE.md contract rule."""
        entry = LedgerEntry(id="b", surface="python", status="deprecated")
        d = entry.to_dict()
        assert d["is_deprecated"] is True
        assert d["needs_review"] is False

    @pytest.mark.unit
    def test_round_trip_ignores_computed(self) -> None:
        """from_dict(to_dict(x)) reconstructs an equal entry."""
        entry = LedgerEntry(
            id="rust:dynamo-parsers::lib::parse",
            surface="rust",
            kind="crate_item",
            component="Runtime",
            status="relocated",
            stability="stable",
            review_status="confirmed",
            confidence="high",
            source="marker",
            added_in="1.0.0",
            deprecated_in="1.1.0",
            removal_target="1.3.0",
            relocated_to="ai-dynamo/dynamo-parsers",
            relocated_in="1.2.0",
            last_seen="1.2.0",
            note="moved out of monorepo",
            migration_guidance="depend on the dynamo-parsers crate",
            pr_number=9999,
            signature_history=[{"release": "1.0.0", "signature": "parse(s: &str)"}],
        )
        assert LedgerEntry.from_dict(entry.to_dict()) == entry


# =============================================================================
# SurfaceLedger
# =============================================================================


class TestSurfaceLedger:
    """Tests for SurfaceLedger envelope."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        """Default ledger carries the current schema version and no entries."""
        ledger = SurfaceLedger()
        assert ledger.schema_version == LEDGER_SCHEMA_VERSION
        assert ledger.entries == []

    @pytest.mark.unit
    def test_to_dict_sorts_entries(self) -> None:
        """Entries are emitted sorted by id for reviewable git diffs."""
        ledger = SurfaceLedger(
            entries=[
                LedgerEntry(id="python:z", surface="python"),
                LedgerEntry(id="python:a", surface="python"),
            ]
        )
        d = ledger.to_dict()
        assert [e["id"] for e in d["entries"]] == ["python:a", "python:z"]

    @pytest.mark.unit
    def test_round_trip(self) -> None:
        """Round trip preserves schema version and entries."""
        ledger = SurfaceLedger(
            generated_at="2026-05-27T00:00:00Z",
            entries=[LedgerEntry(id="python:a", surface="python", added_in="1.0.0")],
        )
        restored = SurfaceLedger.from_dict(ledger.to_dict())
        assert restored.schema_version == ledger.schema_version
        assert restored.entries == ledger.entries


# =============================================================================
# Extractor protocol
# =============================================================================


class TestSurfaceExtractor:
    """The frozen extractor contract Phase B subagents implement."""

    @pytest.mark.unit
    def test_conforming_callable_is_recognized(self) -> None:
        """A callable with the right signature satisfies the runtime protocol."""

        def extract(repo_path: Path, release: str) -> OperationResult:
            return OperationResult(
                data={"symbols": []},
                metadata={"surface": "python", "covered": True},
            )

        assert isinstance(extract, SurfaceExtractor)
