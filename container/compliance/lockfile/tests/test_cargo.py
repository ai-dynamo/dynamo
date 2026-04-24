# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Cargo.lock parser and workspace-member discovery."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from container.compliance.lockfile.cargo import (
    _expand_workspace_member_paths,
    get_workspace_members,
    parse_cargo_lock,
)

# ---------- parse_cargo_lock ----------


def test_parse_cargo_lock_drops_workspace_members() -> None:
    """Workspace members (first-party crates) must be filtered out."""
    content = dedent(
        """
        [[package]]
        name = "dynamo-runtime"
        version = "0.1.0"
        dependencies = [
         "serde 1.0.0",
        ]

        [[package]]
        name = "serde"
        version = "1.0.0"
        source = "registry+https://github.com/rust-lang/crates.io-index"
        """
    )

    pkgs = parse_cargo_lock(content, workspace_members=["dynamo-runtime"])
    by_name = {p["name"]: p for p in pkgs}

    assert "dynamo-runtime" not in by_name
    assert by_name["serde"]["version"] == "1.0.0"
    assert by_name["serde"]["ecosystem"] == "cargo"
    assert by_name["serde"]["source"] == (
        "registry+https://github.com/rust-lang/crates.io-index"
    )
    assert by_name["serde"]["is_direct"] is False


def test_parse_cargo_lock_falls_back_to_source_heuristic() -> None:
    """When workspace_members is None, sourceless crates are dropped as workspace."""
    content = dedent(
        """
        [[package]]
        name = "local-only"
        version = "0.1.0"

        [[package]]
        name = "from-registry"
        version = "1.0.0"
        source = "registry+https://example.com"
        """
    )
    pkgs = {p["name"]: p for p in parse_cargo_lock(content)}
    assert "local-only" not in pkgs
    assert pkgs["from-registry"]["source"] == "registry+https://example.com"


def test_parse_cargo_lock_empty_returns_empty_list() -> None:
    assert parse_cargo_lock("") == []
    assert parse_cargo_lock("   \n  ") == []


def test_parse_cargo_lock_record_has_exact_contract_keys() -> None:
    content = dedent(
        """
        [[package]]
        name = "foo"
        version = "1.2.3"
        source = "registry+https://example.com"
        """
    )
    pkgs = parse_cargo_lock(content, workspace_members=[])
    assert len(pkgs) == 1
    assert set(pkgs[0].keys()) == {
        "name",
        "version",
        "ecosystem",
        "is_direct",
        "source",
    }


# ---------- _expand_workspace_member_paths ----------


def test_workspace_literal_path_passes_through() -> None:
    assert _expand_workspace_member_paths("components/api", []) == ["components/api"]


def test_workspace_glob_expands_against_cargo_toml_paths() -> None:
    """`lib/*` should expand to every dir containing Cargo.toml one level deep."""
    paths = [
        "Cargo.toml",
        "lib/foo/Cargo.toml",
        "lib/foo/src/main.rs",
        "lib/bar/Cargo.toml",
        "lib/bar/README.md",
        "components/api/Cargo.toml",
    ]
    expanded = _expand_workspace_member_paths("lib/*", paths)
    assert expanded == ["lib/bar", "lib/foo"]


def test_workspace_glob_with_no_matches_returns_empty() -> None:
    assert _expand_workspace_member_paths("nonexistent/*", ["lib/foo/Cargo.toml"]) == []


def test_workspace_glob_unsupported_pattern_raises() -> None:
    with pytest.raises(ValueError):
        _expand_workspace_member_paths("lib/*/thing", ["lib/foo/thing/Cargo.toml"])


# ---------- get_workspace_members ----------


def test_get_workspace_members_literal_paths(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        dedent(
            """
            [workspace]
            members = [
                "crates/alpha",
                "crates/beta",
            ]
            """
        )
    )
    (tmp_path / "crates" / "alpha").mkdir(parents=True)
    (tmp_path / "crates" / "alpha" / "Cargo.toml").write_text(
        '[package]\nname = "alpha-crate"\nversion = "0.1.0"\n'
    )
    (tmp_path / "crates" / "beta").mkdir(parents=True)
    (tmp_path / "crates" / "beta" / "Cargo.toml").write_text(
        '[package]\nname = "beta-crate"\nversion = "0.1.0"\n'
    )

    assert get_workspace_members(str(tmp_path)) == ["alpha-crate", "beta-crate"]


def test_get_workspace_members_glob(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        dedent(
            """
            [workspace]
            members = [
                "lib/*",
            ]
            """
        )
    )
    (tmp_path / "lib" / "foo").mkdir(parents=True)
    (tmp_path / "lib" / "foo" / "Cargo.toml").write_text(
        '[package]\nname = "foo-crate"\nversion = "0.1.0"\n'
    )
    (tmp_path / "lib" / "bar").mkdir(parents=True)
    (tmp_path / "lib" / "bar" / "Cargo.toml").write_text(
        '[package]\nname = "bar-crate"\nversion = "0.1.0"\n'
    )

    members = get_workspace_members(str(tmp_path))
    assert set(members) == {"foo-crate", "bar-crate"}


def test_get_workspace_members_missing_cargo_toml_returns_empty(
    tmp_path: Path,
) -> None:
    assert get_workspace_members(str(tmp_path)) == []
