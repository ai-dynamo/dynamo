# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the C3 signal layer: stability, suppressions, validate, review.

Covers ``ops/api_surface/{stability,suppressions,validate}.py`` plus the review
+ schema-migration seams in ``ledger.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from api_surface.ledger import (
    confirm_review,
    load_ledger,
    migrate_ledger_dict,
    pending_review,
    save_ledger,
)
from api_surface.models import (
    LEDGER_SCHEMA_VERSION,
    LedgerEntry,
    SurfaceLedger,
    SurfaceSymbol,
)
from api_surface.stability import infer_stability, normalize_stability
from api_surface.suppressions import (
    Suppression,
    SuppressionSet,
    load_suppressions,
    version_key,
)
from api_surface.validate import validate_ledger


def _sym(sid: str, surface: str = "python", **kw) -> SurfaceSymbol:
    return SurfaceSymbol(surface=surface, kind="function", id=sid, **kw)


def _entry(sid: str, **kw) -> LedgerEntry:
    return LedgerEntry(id=sid, surface=kw.pop("surface", "python"), **kw)


# =============================================================================
# stability
# =============================================================================


class TestStabilityInference:
    @pytest.mark.unit
    def test_plain_symbol_is_stable(self) -> None:
        assert infer_stability(_sym("python:dynamo.runtime.create")) == "stable"

    @pytest.mark.unit
    def test_crd_alpha_version_is_experimental(self) -> None:
        sym = _sym(
            "crd:Foo@v1alpha1.spec", surface="crd", metadata={"version": "v1alpha1"}
        )
        assert infer_stability(sym) == "experimental"

    @pytest.mark.unit
    def test_crd_beta_version_is_preview(self) -> None:
        sym = _sym(
            "crd:Foo@v1beta2.spec", surface="crd", metadata={"version": "v1beta2"}
        )
        assert infer_stability(sym) == "preview"

    @pytest.mark.unit
    def test_crd_stable_version_stays_stable(self) -> None:
        sym = _sym("crd:Foo@v1.spec", surface="crd", metadata={"version": "v1"})
        assert infer_stability(sym) == "stable"

    @pytest.mark.unit
    def test_experimental_keyword_in_id(self) -> None:
        assert (
            infer_stability(_sym("python:dynamo.experimental.thing")) == "experimental"
        )

    @pytest.mark.unit
    def test_preview_keyword_in_id(self) -> None:
        assert infer_stability(_sym("env:DYN_PREVIEW_MODE")) == "preview"

    @pytest.mark.unit
    def test_keyword_needs_word_boundary(self) -> None:
        """``alphabet`` / ``betamax`` must not trip the alpha/beta keywords."""
        assert infer_stability(_sym("python:dynamo.alphabet.parse")) == "stable"
        assert infer_stability(_sym("python:dynamo.betamax.run")) == "stable"

    @pytest.mark.unit
    def test_least_stable_signal_wins(self) -> None:
        """An experimental keyword beats a preview CRD version."""
        sym = _sym(
            "crd:Foo@v1beta1.experimental.field",
            surface="crd",
            metadata={"version": "v1beta1"},
        )
        assert infer_stability(sym) == "experimental"

    @pytest.mark.unit
    def test_normalize_only_downgrades(self) -> None:
        """An extractor-asserted tier is never promoted by inference."""
        sym = _sym("python:dynamo.plain", stability="experimental")
        normalize_stability(sym)
        assert sym.stability == "experimental"

    @pytest.mark.unit
    def test_normalize_demotes_stable_default(self) -> None:
        sym = _sym("python:dynamo.experimental.x", stability="stable")
        normalize_stability(sym)
        assert sym.stability == "experimental"


class TestDeclaredPublicGate:
    """Python/Rust need a positive ``declared_public`` signal to stay ``stable``."""

    @pytest.mark.unit
    def test_undeclared_python_is_experimental(self) -> None:
        sym = _sym(
            "python:dynamo.planner.utils.helper", metadata={"declared_public": False}
        )
        assert infer_stability(sym) == "experimental"

    @pytest.mark.unit
    def test_declared_python_is_stable(self) -> None:
        sym = _sym(
            "python:dynamo.frontend.Frontend", metadata={"declared_public": True}
        )
        assert infer_stability(sym) == "stable"

    @pytest.mark.unit
    def test_absent_flag_stays_stable(self) -> None:
        """No flag => treated as declared, so the historical default holds."""
        assert infer_stability(_sym("python:dynamo.runtime.create")) == "stable"

    @pytest.mark.unit
    def test_undeclared_rust_is_experimental(self) -> None:
        sym = _sym(
            "rust:dynamo-kv-router::approx::PruneManager",
            surface="rust",
            metadata={"declared_public": False},
        )
        assert infer_stability(sym) == "experimental"

    @pytest.mark.unit
    def test_gate_does_not_touch_public_by_construction_surfaces(self) -> None:
        """env/helm/etc. are public by construction: the gate never demotes them."""
        for surface, sid in (
            ("env", "env:DYN_FOO"),
            ("helm", "helm:platform:global.x"),
        ):
            sym = _sym(sid, surface=surface, metadata={"declared_public": False})
            assert infer_stability(sym) == "stable"

    @pytest.mark.unit
    def test_normalize_demotes_undeclared_python(self) -> None:
        sym = _sym(
            "python:dynamo.planner.utils.helper",
            stability="stable",
            metadata={"declared_public": False},
        )
        normalize_stability(sym)
        assert sym.stability == "experimental"


# =============================================================================
# suppressions
# =============================================================================


class TestSuppressions:
    @pytest.mark.unit
    def test_exact_id_match(self) -> None:
        s = Suppression(id="python:m.A", reason="never public")
        assert s.matches("python:m.A")
        assert not s.matches("python:m.B")

    @pytest.mark.unit
    def test_glob_pattern_match(self) -> None:
        s = Suppression(
            pattern="helm:platform:components.planner.*", reason="alpha chart"
        )
        assert s.matches("helm:platform:components.planner.replicas")
        assert not s.matches("helm:platform:components.router.replicas")

    @pytest.mark.unit
    def test_surface_guard(self) -> None:
        s = Suppression(pattern="*", surface="helm", reason="x")
        assert s.matches("helm:c:k", surface="helm")
        assert not s.matches("python:m.A", surface="python")

    @pytest.mark.unit
    def test_until_expiry(self) -> None:
        s = Suppression(id="python:m.A", until="1.2.0", reason="x")
        assert s.matches("python:m.A", release="1.2.0")
        assert s.matches("python:m.A", release="1.1.0")
        assert not s.matches("python:m.A", release="1.3.0")

    @pytest.mark.unit
    def test_version_key_ordering(self) -> None:
        assert version_key("1.10.0") > version_key("1.9.0")
        assert version_key("bad") == (0,)

    @pytest.mark.unit
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "suppressions.yaml"
        path.write_text(
            "suppressions:\n"
            '  - id: "python:m.A"\n'
            '    reason: "never public"\n'
            '  - pattern: "helm:*"\n'
            '    reason: "chart churn"\n'
        )
        sset = load_suppressions(path)
        assert len(sset.rules) == 2
        assert sset.is_suppressed("python:m.A")
        assert sset.is_suppressed("helm:platform:x")
        assert not sset.is_suppressed("crd:K@v1.x")

    @pytest.mark.unit
    def test_load_missing_is_empty(self, tmp_path: Path) -> None:
        assert load_suppressions(tmp_path / "absent.yaml").rules == []

    @pytest.mark.unit
    def test_load_rejects_rule_without_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "suppressions.yaml"
        path.write_text('suppressions:\n  - id: "python:m.A"\n')

        with pytest.raises(ValueError, match="reason"):
            load_suppressions(path)

    @pytest.mark.unit
    def test_load_rejects_rule_with_id_and_pattern(self, tmp_path: Path) -> None:
        path = tmp_path / "suppressions.yaml"
        path.write_text(
            "suppressions:\n"
            '  - id: "python:m.A"\n'
            '    pattern: "python:*"\n'
            '    reason: "ambiguous waiver"\n'
        )

        with pytest.raises(ValueError, match="exactly one"):
            load_suppressions(path)

    @pytest.mark.unit
    def test_load_rejects_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "suppressions.yaml"
        path.write_text("suppressions: [\n", encoding="utf-8")

        with pytest.raises(ValueError, match="failed to load"):
            load_suppressions(path)


# =============================================================================
# validate
# =============================================================================


class TestValidate:
    @pytest.mark.unit
    def test_stable_removed_without_deprecation_is_violation(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry(
                    "python:m.A",
                    status="removed",
                    stability="stable",
                    removed_in="1.2.0",
                )
            ]
        )
        result = validate_ledger(ledger)
        assert result.metadata["violation_count"] == 1
        assert result.data["violations"][0]["kind"] == "removed_without_deprecation"

    @pytest.mark.unit
    def test_deprecated_then_removed_is_clean(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry(
                    "python:m.A",
                    status="removed",
                    stability="stable",
                    deprecated_in="1.1.0",
                    removed_in="2.0.0",
                )
            ]
        )
        assert validate_ledger(ledger).metadata["violation_count"] == 0

    @pytest.mark.unit
    def test_premature_removal_is_violation(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry(
                    "python:m.A",
                    status="removed",
                    stability="stable",
                    deprecated_in="1.1.0",
                    removal_target="1.5.0",
                    removed_in="1.2.0",
                )
            ]
        )
        result = validate_ledger(ledger)
        assert result.data["violations"][0]["kind"] == "premature_removal"

    @pytest.mark.unit
    def test_experimental_removal_is_not_a_violation(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry(
                    "python:m.A",
                    status="removed",
                    stability="experimental",
                    removed_in="1.2.0",
                )
            ]
        )
        assert validate_ledger(ledger).metadata["violation_count"] == 0

    @pytest.mark.unit
    def test_suppression_waives_violation(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry(
                    "python:m.A",
                    status="removed",
                    stability="stable",
                    removed_in="1.2.0",
                )
            ]
        )
        sset = SuppressionSet(
            rules=[Suppression(id="python:m.A", reason="never public")]
        )
        result = validate_ledger(ledger, suppressions=sset)
        assert result.metadata["violation_count"] == 0
        assert result.data["waived"] == 1

    @pytest.mark.unit
    def test_expired_suppression_does_not_waive_current_deprecation(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry(
                    "python:m.A",
                    status="deprecated",
                    stability="stable",
                    deprecated_in="1.2.0",
                    last_seen="1.4.0",
                )
            ]
        )
        suppressions = SuppressionSet(
            rules=[
                Suppression(
                    id="python:m.A",
                    reason="temporary exception",
                    until="1.3.0",
                )
            ]
        )

        result = validate_ledger(ledger, suppressions=suppressions)

        assert result.metadata["violation_count"] == 1
        assert result.data["waived"] == 0


# =============================================================================
# review + migration
# =============================================================================


class TestReview:
    @pytest.mark.unit
    def test_pending_lists_needs_review(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry("python:m.A", review_status="needs_review"),
                _entry("python:m.B", review_status="auto"),
            ]
        )
        pending = pending_review(ledger)
        assert [e.id for e in pending] == ["python:m.A"]

    @pytest.mark.unit
    def test_confirm_promotes_only_needs_review(self) -> None:
        ledger = SurfaceLedger(
            entries=[
                _entry("python:m.A", review_status="needs_review"),
                _entry("python:m.B", review_status="auto"),
            ]
        )
        confirmed = confirm_review(
            ledger, ["python:m.A", "python:m.B", "python:m.missing"]
        )
        assert confirmed == 1
        assert ledger.entries[0].review_status == "confirmed"
        assert ledger.entries[1].review_status == "auto"


class TestSchemaMigration:
    @pytest.mark.unit
    def test_missing_version_stamps_current(self) -> None:
        migrated = migrate_ledger_dict({"entries": []})
        assert migrated["schema_version"] == LEDGER_SCHEMA_VERSION

    @pytest.mark.unit
    def test_future_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="ledger schema"):
            migrate_ledger_dict({"schema_version": 99, "entries": []})

    @pytest.mark.unit
    def test_load_upgrades_unversioned_file(self, tmp_path: Path) -> None:
        ledger = SurfaceLedger(
            entries=[_entry("python:m.A", status="added", added_in="1.0.0")]
        )
        path = tmp_path / "ledger.json"
        save_ledger(ledger, path)
        restored = load_ledger(path)
        assert restored.schema_version == LEDGER_SCHEMA_VERSION
        assert restored.entries[0].id == "python:m.A"
