# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the go.mod parser and multi-module discovery."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from container.compliance.lockfile.gomod import list_go_mod_files, parse_go_mod

# ---------- parse_go_mod ----------


def test_parse_go_mod_block_form() -> None:
    content = dedent(
        """
        module example.com/operator
        go 1.21

        require (
            k8s.io/api v0.28.0
            github.com/pkg/errors v0.9.1 // indirect
        )
        """
    )
    pkgs = {p["name"]: p for p in parse_go_mod(content)}

    assert pkgs["k8s.io/api"]["version"] == "v0.28.0"
    assert pkgs["k8s.io/api"]["is_direct"] is True
    assert pkgs["k8s.io/api"]["ecosystem"] == "golang"
    assert pkgs["github.com/pkg/errors"]["is_direct"] is False


def test_parse_go_mod_single_line_require() -> None:
    """Single-line `require module v1.2.3` directives must parse."""
    content = dedent(
        """
        module example.com/cli
        go 1.21

        require github.com/spf13/cobra v1.8.0
        require github.com/stretchr/testify v1.9.0 // indirect
        """
    )
    pkgs = {p["name"]: p for p in parse_go_mod(content)}

    assert pkgs["github.com/spf13/cobra"]["version"] == "v1.8.0"
    assert pkgs["github.com/spf13/cobra"]["is_direct"] is True
    assert pkgs["github.com/stretchr/testify"]["is_direct"] is False


def test_parse_go_mod_mixed_forms_dedupe() -> None:
    """Block + single-line forms should not produce duplicates."""
    content = dedent(
        """
        module example.com/svc

        require github.com/pkg/errors v0.9.1

        require (
            k8s.io/api v0.28.0
            github.com/pkg/errors v0.9.1 // indirect
        )
        """
    )
    pkgs = parse_go_mod(content)
    names = [p["name"] for p in pkgs]
    assert names.count("github.com/pkg/errors") == 1
    assert {p["name"] for p in pkgs} == {"github.com/pkg/errors", "k8s.io/api"}


def test_parse_go_mod_empty_returns_empty_list() -> None:
    assert parse_go_mod("") == []
    assert parse_go_mod("   \n  ") == []


def test_parse_go_mod_record_has_exact_contract_keys() -> None:
    content = "module example.com/x\nrequire github.com/foo/bar v1.0.0\n"
    pkgs = parse_go_mod(content)
    assert len(pkgs) == 1
    assert set(pkgs[0].keys()) == {
        "name",
        "version",
        "ecosystem",
        "is_direct",
        "source",
    }
    assert pkgs[0]["source"] == ""


def test_parse_go_mod_drops_local_replace_directive() -> None:
    """Modules replaced with local filesystem paths must be filtered out."""
    content = dedent(
        """
        module example.com/svc

        require (
            example.com/internal v0.0.0
            github.com/pkg/errors v0.9.1
        )

        replace example.com/internal => ./internal
        """
    )
    pkgs = {p["name"] for p in parse_go_mod(content)}
    assert pkgs == {"github.com/pkg/errors"}


def test_parse_go_mod_keeps_remote_replace_directive() -> None:
    """replace -> a remote module path should NOT drop the require entry."""
    content = dedent(
        """
        module example.com/svc

        require github.com/foo/bar v1.0.0

        replace github.com/foo/bar => github.com/fork/bar v1.0.1
        """
    )
    pkgs = {p["name"] for p in parse_go_mod(content)}
    assert "github.com/foo/bar" in pkgs


def test_parse_go_mod_block_form_local_replace() -> None:
    content = dedent(
        """
        module example.com/svc

        require (
            example.com/internal v0.0.0
            github.com/pkg/errors v0.9.1
        )

        replace (
            example.com/internal => ../internal
        )
        """
    )
    pkgs = {p["name"] for p in parse_go_mod(content)}
    assert pkgs == {"github.com/pkg/errors"}


# ---------- list_go_mod_files ----------


def test_list_go_mod_files_finds_multiple(tmp_path: Path) -> None:
    (tmp_path / "deploy" / "operator").mkdir(parents=True)
    (tmp_path / "deploy" / "operator" / "go.mod").write_text(
        "module example.com/operator\n"
    )
    (tmp_path / "tools" / "scanner").mkdir(parents=True)
    (tmp_path / "tools" / "scanner" / "go.mod").write_text(
        "module example.com/scanner\n"
    )
    # Root-level go.mod too, to check it is included.
    (tmp_path / "go.mod").write_text("module example.com/root\n")

    result = list_go_mod_files(str(tmp_path))
    assert result == [
        "deploy/operator/go.mod",
        "go.mod",
        "tools/scanner/go.mod",
    ]


def test_list_go_mod_files_skips_git_dir(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    # A stray go.mod inside .git must not be surfaced.
    (tmp_path / ".git" / "go.mod").write_text("module example.com/hidden\n")
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "go.mod").write_text("module example.com/svc\n")

    assert list_go_mod_files(str(tmp_path)) == ["svc/go.mod"]


def test_list_go_mod_files_empty_tree(tmp_path: Path) -> None:
    assert list_go_mod_files(str(tmp_path)) == []
