# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ops/api_surface/diff.py - the code-grounded diff engine."""

from __future__ import annotations

import pytest
from api_surface import diff
from api_surface.diff import canonicalize_signature, diff_snapshots
from api_surface.models import SurfaceSnapshot, SurfaceSymbol


def _sym(sid: str, signature: str = "", surface: str = "python", **kw) -> SurfaceSymbol:
    return SurfaceSymbol(
        surface=surface, kind="function", id=sid, signature=signature, **kw
    )


def _snap(
    release: str,
    symbols: list[SurfaceSymbol],
    coverage: dict[str, bool],
    coverage_detail: dict[str, bool] | None = None,
) -> SurfaceSnapshot:
    return SurfaceSnapshot(
        release=release,
        ref=f"v{release}",
        coverage=coverage,
        coverage_detail=coverage_detail or {},
        symbols=symbols,
    )


def _rust(sid: str, signature: str) -> SurfaceSymbol:
    return _sym(sid, signature, surface="rust")


class TestCanonicalize:
    @pytest.mark.unit
    def test_collapses_whitespace(self) -> None:
        assert canonicalize_signature("foo(a,   b )") == "foo(a, b )"
        assert canonicalize_signature("  x \n y ") == "x y"


class TestDiffSnapshots:
    @pytest.mark.unit
    def test_added_and_removed(self) -> None:
        """A symbol only in new is added; one only in old is removed."""
        old = _snap("1.0.0", [_sym("python:m.A"), _sym("python:m.C")], {"python": True})
        new = _snap("1.1.0", [_sym("python:m.A"), _sym("python:m.B")], {"python": True})
        result = diff_snapshots(old, new)
        events = {(c["change_type"], c["id"]) for c in result.data["changes"]}
        assert ("added", "python:m.B") in events
        assert ("removed", "python:m.C") in events
        assert result.metadata["change_count"] == 2

    @pytest.mark.unit
    def test_signature_changed(self) -> None:
        """Same id, different canonical signature -> signature_changed."""
        old = _snap("1.0.0", [_sym("python:m.f", "f(a)")], {"python": True})
        new = _snap("1.1.0", [_sym("python:m.f", "f(a, b)")], {"python": True})
        changes = diff_snapshots(old, new).data["changes"]
        assert len(changes) == 1
        assert changes[0]["change_type"] == "signature_changed"
        assert changes[0]["from_signature"] == "f(a)"
        assert changes[0]["to_signature"] == "f(a, b)"

    @pytest.mark.unit
    def test_cosmetic_signature_not_changed(self) -> None:
        """Whitespace-only signature differences do not register."""
        old = _snap("1.0.0", [_sym("python:m.f", "f(a, b)")], {"python": True})
        new = _snap("1.1.0", [_sym("python:m.f", "f(a,  b)")], {"python": True})
        assert diff_snapshots(old, new).data["changes"] == []

    @pytest.mark.unit
    def test_rename_detected_as_single_event(self) -> None:
        """A removed+added pair with near-identical signatures is one rename."""
        old = _snap(
            "1.0.0",
            [_sym("python:m.old_name", "fn(self, request: Req) -> Resp")],
            {"python": True},
        )
        new = _snap(
            "1.1.0",
            [_sym("python:m.new_name", "fn(self, request: Req) -> Resp")],
            {"python": True},
        )
        changes = diff_snapshots(old, new).data["changes"]
        assert len(changes) == 1
        assert changes[0]["change_type"] == "renamed_function"
        assert changes[0]["id"] == "python:m.old_name"
        assert changes[0]["relocated_to"] == "python:m.new_name"
        assert changes[0]["similarity"] >= 0.85

    @pytest.mark.unit
    def test_coverage_gap_is_not_removal(self) -> None:
        """A surface absent from the new snapshot is a gap, never a removal."""
        old = _snap(
            "1.0.0",
            [_sym("python:m.A"), _sym("crd:K@v1alpha1.spec.x", surface="crd")],
            {"python": True, "crd": True},
        )
        new = _snap("1.1.0", [_sym("python:m.A")], {"python": True, "crd": False})
        result = diff_snapshots(old, new)
        assert result.data["changes"] == []
        assert "crd" in result.data["coverage_gaps"]

    @pytest.mark.unit
    def test_rust_crate_gated_by_coverage_detail(self) -> None:
        """A rust crate not covered in both snapshots emits no removals for it."""
        old = _snap(
            "1.0.0",
            [_rust("rust:dynamo-kvbm::a::Foo", "struct Foo")],
            {"rust": True},
            coverage_detail={"rust:dynamo-kvbm": True},
        )
        # The crate was split/renamed at the new tag -> not covered there.
        new = _snap(
            "1.1.0",
            [],
            {"rust": True},
            coverage_detail={"rust:dynamo-kvbm": False},
        )
        result = diff_snapshots(old, new)
        assert result.data["changes"] == []


class TestRelocation:
    @pytest.mark.unit
    def test_relocation_detected_not_removal(self) -> None:
        """A rust symbol that moves crate but keeps its path is one relocation."""
        detail = {"rust:dynamo-runtime": True, "rust:dynamo-llm": True}
        old = _snap(
            "1.0.0",
            [_rust("rust:dynamo-runtime::storage::Cache", "pub struct Cache")],
            {"rust": True},
            coverage_detail=detail,
        )
        new = _snap(
            "1.1.0",
            [_rust("rust:dynamo-llm::storage::Cache", "pub struct Cache")],
            {"rust": True},
            coverage_detail=detail,
        )
        changes = diff_snapshots(old, new).data["changes"]
        assert len(changes) == 1
        reloc = changes[0]
        assert reloc["change_type"] == "relocated"
        assert reloc["id"] == "rust:dynamo-runtime::storage::Cache"
        assert reloc["relocated_to"] == "rust:dynamo-llm::storage::Cache"
        types = {c["change_type"] for c in changes}
        assert "removed" not in types and "added" not in types

    @pytest.mark.unit
    def test_constructor_collision_not_relocation(self) -> None:
        """Different paths sharing only a `new() -> Self` body do not relocate."""
        detail = {"rust:dynamo-runtime": True, "rust:dynamo-llm": True}
        old = _snap(
            "1.0.0",
            [_rust("rust:dynamo-runtime::a::Foo::new", "pub fn new() -> Self")],
            {"rust": True},
            coverage_detail=detail,
        )
        new = _snap(
            "1.1.0",
            [_rust("rust:dynamo-llm::b::Bar::new", "pub fn new() -> Self")],
            {"rust": True},
            coverage_detail=detail,
        )
        types = {c["change_type"] for c in diff_snapshots(old, new).data["changes"]}
        assert "relocated" not in types
        assert types == {"removed", "added"}


class TestRenameNameGuard:
    @pytest.mark.unit
    def test_cross_container_rename_blocked(self) -> None:
        """Identical signatures in different containers are not paired as rename."""
        old = _snap(
            "1.0.0", [_sym("python:m.A.foo", "fn(self) -> int")], {"python": True}
        )
        new = _snap(
            "1.1.0", [_sym("python:m.B.bar", "fn(self) -> int")], {"python": True}
        )
        types = {c["change_type"] for c in diff_snapshots(old, new).data["changes"]}
        assert types == {"removed", "added"}

    @pytest.mark.unit
    def test_same_container_rename_detected(self) -> None:
        """A rename within the same container is still a single rename event."""
        old = _snap(
            "1.0.0", [_sym("python:m.A.foo", "fn(self) -> int")], {"python": True}
        )
        new = _snap(
            "1.1.0", [_sym("python:m.A.bar", "fn(self) -> int")], {"python": True}
        )
        changes = diff_snapshots(old, new).data["changes"]
        assert len(changes) == 1
        assert changes[0]["change_type"] == "renamed_function"
        assert changes[0]["id"] == "python:m.A.foo"
        assert changes[0]["relocated_to"] == "python:m.A.bar"


class TestContainerRename:
    @pytest.mark.unit
    def test_renamed_class_pairs_shared_members_as_relocations(self) -> None:
        """A class renamed in lockstep relocates shared members, not removes them.

        Mirrors backend.EngineConfig -> backend.LlmRegistration: members present
        on both sides relocate; members on only one side stay removed / added.
        """
        old = _snap(
            "1.0.0",
            [
                _sym("python:m.b.EngineConfig.foo", "fn(self) -> int"),
                _sym("python:m.b.EngineConfig.bar", "fn(self) -> str"),
                _sym("python:m.b.EngineConfig.baz", "fn(self) -> bool"),
                _sym("python:m.b.EngineConfig.dropped", "fn(self) -> None"),
            ],
            {"python": True},
        )
        new = _snap(
            "1.1.0",
            [
                _sym("python:m.b.LlmRegistration.foo", "fn(self) -> int"),
                _sym("python:m.b.LlmRegistration.bar", "fn(self) -> str"),
                _sym("python:m.b.LlmRegistration.baz", "fn(self) -> bool"),
                _sym("python:m.b.LlmRegistration.brand_new", "fn(self) -> int"),
            ],
            {"python": True},
        )
        changes = diff_snapshots(old, new).data["changes"]
        by_type: dict[str, list[dict]] = {}
        for c in changes:
            by_type.setdefault(c["change_type"], []).append(c)
        relocated = {(c["id"], c["relocated_to"]) for c in by_type.get("relocated", [])}
        assert relocated == {
            ("python:m.b.EngineConfig.foo", "python:m.b.LlmRegistration.foo"),
            ("python:m.b.EngineConfig.bar", "python:m.b.LlmRegistration.bar"),
            ("python:m.b.EngineConfig.baz", "python:m.b.LlmRegistration.baz"),
        }
        assert [c["id"] for c in by_type.get("removed", [])] == [
            "python:m.b.EngineConfig.dropped"
        ]
        assert [c["id"] for c in by_type.get("added", [])] == [
            "python:m.b.LlmRegistration.brand_new"
        ]

    @pytest.mark.unit
    def test_too_few_shared_members_not_a_rename(self) -> None:
        """Two sibling classes sharing under the floor stay removed + added."""
        old = _snap(
            "1.0.0",
            [
                _sym("python:m.X.a", "fn(self) -> int"),
                _sym("python:m.X.b", "fn(self) -> int"),
            ],
            {"python": True},
        )
        new = _snap(
            "1.1.0",
            [
                _sym("python:m.Y.a", "fn(self) -> int"),
                _sym("python:m.Y.b", "fn(self) -> int"),
            ],
            {"python": True},
        )
        types = {c["change_type"] for c in diff_snapshots(old, new).data["changes"]}
        assert "relocated" not in types
        assert types == {"removed", "added"}

    @pytest.mark.unit
    def test_rust_excluded_from_container_rename(self) -> None:
        """Rust structs sharing trait-method leaves must NOT fabricate a move.

        Two distinct rust types under one module expose the same trait method
        leaves (``a`` / ``b`` / ``c``); the member-leaf-overlap heuristic would
        falsely pair them (the real ``KvRouter`` <-> ``WorkerSelector`` bug), so
        rust is excluded and these stay removed + added. Genuine rust moves are
        the job of the strict same-path relocation detector.
        """
        detail = {"rust:dynamo-llm": True}
        old = _snap(
            "1.0.0",
            [
                _rust("rust:dynamo-llm::m::Router::a", "pub fn a(&self)"),
                _rust("rust:dynamo-llm::m::Router::b", "pub fn b(&self)"),
                _rust("rust:dynamo-llm::m::Router::c", "pub fn c(&self)"),
            ],
            {"rust": True},
            coverage_detail=detail,
        )
        new = _snap(
            "1.1.0",
            [
                _rust("rust:dynamo-llm::m::Selector::a", "pub fn a(&self)"),
                _rust("rust:dynamo-llm::m::Selector::b", "pub fn b(&self)"),
                _rust("rust:dynamo-llm::m::Selector::c", "pub fn c(&self)"),
            ],
            {"rust": True},
            coverage_detail=detail,
        )
        types = {c["change_type"] for c in diff_snapshots(old, new).data["changes"]}
        assert "relocated" not in types
        assert types == {"removed", "added"}


class TestCrateMerge:
    @pytest.mark.unit
    def test_clean_merge_relocates_not_add_plus_drop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crate cleanly folded into another under a preserved prefix relocates.

        The old crate is gated out at the new ref (absent), so the merge pass
        reads the ungated sets: the old symbol pairs with its new home as one
        `relocated`, and the new id is not also reported as a bare addition.
        ``CRATE_MERGES`` is monkeypatched so the test exercises the mechanism
        independent of the live (intentionally empty) config.
        """
        monkeypatch.setattr(diff, "CRATE_MERGES", {"dynamo-old": ("dynamo-llm", "old")})
        detail_old = {"rust:dynamo-old": True, "rust:dynamo-llm": True}
        detail_new = {"rust:dynamo-old": False, "rust:dynamo-llm": True}
        old = _snap(
            "1.2.0",
            [
                _rust("rust:dynamo-old::common::Foo", "pub struct Foo"),
                _rust("rust:dynamo-llm::engine::Run", "pub fn run()"),
            ],
            {"rust": True},
            coverage_detail=detail_old,
        )
        new = _snap(
            "1.3.0",
            [
                _rust("rust:dynamo-llm::old::common::Foo", "pub struct Foo"),
                _rust("rust:dynamo-llm::engine::Run", "pub fn run()"),
            ],
            {"rust": True},
            coverage_detail=detail_new,
        )
        changes = diff_snapshots(old, new).data["changes"]
        assert {c["change_type"] for c in changes} == {"relocated"}
        reloc = changes[0]
        assert reloc["id"] == "rust:dynamo-old::common::Foo"
        assert reloc["relocated_to"] == "rust:dynamo-llm::old::common::Foo"

    @pytest.mark.unit
    def test_empty_merge_map_is_a_noop(self) -> None:
        """With no configured merges, a vanished crate is left to the gate."""
        detail_old = {"rust:dynamo-old": True, "rust:dynamo-llm": True}
        detail_new = {"rust:dynamo-old": False, "rust:dynamo-llm": True}
        old = _snap(
            "1.2.0",
            [_rust("rust:dynamo-old::common::Foo", "pub struct Foo")],
            {"rust": True},
            coverage_detail=detail_old,
        )
        new = _snap(
            "1.3.0",
            [_rust("rust:dynamo-llm::old::common::Foo", "pub struct Foo")],
            {"rust": True},
            coverage_detail=detail_new,
        )
        changes = diff_snapshots(old, new).data["changes"]
        # Gated old crate -> no removal; new home surfaces as a plain addition.
        assert [c["change_type"] for c in changes] == ["added"]


class TestSorting:
    @pytest.mark.unit
    def test_changes_are_sorted(self) -> None:
        """Output is deterministically sorted by (surface, id, change_type)."""
        old = _snap("1.0.0", [], {"python": True})
        new = _snap(
            "1.1.0",
            [_sym("python:m.Z"), _sym("python:m.A"), _sym("python:m.M")],
            {"python": True},
        )
        ids = [c["id"] for c in diff_snapshots(old, new).data["changes"]]
        assert ids == ["python:m.A", "python:m.M", "python:m.Z"]
