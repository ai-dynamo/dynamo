# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for extractor helpers (workspace globs + multi go.mod discovery)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent

import pytest
from dynamo_attributions import extractor as extractor_mod
from dynamo_attributions.extractor import (
    _expand_workspace_member_paths,
    extract_transitive,
)
from dynamo_attributions.types import Ecosystem

# ---------- _expand_workspace_member_paths ----------


def test_workspace_literal_path_passes_through():
    assert _expand_workspace_member_paths("components/api", []) == ["components/api"]


def test_workspace_glob_expands_against_cargo_toml_paths():
    """Finding #20 — `lib/*` should expand to every dir containing Cargo.toml."""
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


def test_workspace_glob_with_no_matches_returns_empty():
    assert _expand_workspace_member_paths("nonexistent/*", ["lib/foo/Cargo.toml"]) == []


# ---------- _git_read / extract_transitive ----------


def _run_git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _init_git_repo(tmp_path: Path) -> None:
    _run_git(tmp_path, "init", "-q", "-b", "main")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test")
    _run_git(tmp_path, "config", "commit.gpgsign", "false")


def _commit_all(tmp_path: Path, msg: str = "test") -> None:
    _run_git(tmp_path, "add", "-A")
    _run_git(tmp_path, "commit", "-q", "--no-gpg-sign", "-m", msg)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    _init_git_repo(tmp_path)
    return tmp_path


def test_extract_transitive_finds_multiple_go_mods(fake_repo: Path) -> None:
    """Finding #17 — go.mod files outside deploy/operator must be discovered."""
    (fake_repo / "deploy" / "operator").mkdir(parents=True)
    (fake_repo / "deploy" / "operator" / "go.mod").write_text(dedent("""
            module example.com/operator
            require github.com/spf13/cobra v1.8.0
            """))
    (fake_repo / "tools" / "scanner").mkdir(parents=True)
    (fake_repo / "tools" / "scanner" / "go.mod").write_text(dedent("""
            module example.com/scanner
            require (
                github.com/stretchr/testify v1.9.0
            )
            """))
    _commit_all(fake_repo)

    tree = extract_transitive(
        dynamo_path=str(fake_repo),
        branch="main",
        ecosystem=Ecosystem.GO,
    )
    names = {p.name for p in tree.all_packages()}
    assert names == {"github.com/spf13/cobra", "github.com/stretchr/testify"}


def test_extract_transitive_dedupes_go_modules_across_files(fake_repo: Path) -> None:
    (fake_repo / "deploy" / "operator").mkdir(parents=True)
    (fake_repo / "deploy" / "operator" / "go.mod").write_text(
        "module example.com/op\nrequire github.com/spf13/cobra v1.8.0\n"
    )
    (fake_repo / "tools").mkdir(parents=True)
    (fake_repo / "tools" / "go.mod").write_text(
        "module example.com/tool\nrequire github.com/spf13/cobra v1.7.0\n"
    )
    _commit_all(fake_repo)

    tree = extract_transitive(
        dynamo_path=str(fake_repo),
        branch="main",
        ecosystem=Ecosystem.GO,
    )
    names = [p.name for p in tree.all_packages()]
    assert names.count("github.com/spf13/cobra") == 1


def test_extract_transitive_ignores_python_ecosystem(
    fake_repo: Path, monkeypatch
) -> None:
    """When ecosystem=PYTHON, no Rust/Go work happens."""
    called = []

    def _fake_read(*args, **kwargs):
        called.append(args)
        return ""

    monkeypatch.setattr(extractor_mod, "_git_read", _fake_read)
    monkeypatch.setattr(extractor_mod, "_git_ls_tree", lambda *a, **kw: [])

    tree = extract_transitive(
        dynamo_path=str(fake_repo),
        branch="main",
        ecosystem=Ecosystem.PYTHON,
    )
    assert tree.all_packages() == []
    assert called == []  # neither Cargo.lock nor go.mod was read
