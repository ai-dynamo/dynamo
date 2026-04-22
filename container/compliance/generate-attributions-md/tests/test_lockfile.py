# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lockfile parsers."""

from __future__ import annotations

from textwrap import dedent

from dynamo_attributions.lockfile import parse_cargo_lock, parse_go_mod
from dynamo_attributions.types import Ecosystem


def test_parse_cargo_lock_marks_workspace_members_direct() -> None:
    content = dedent("""
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
        """)

    pkgs = parse_cargo_lock(content, workspace_members=["dynamo-runtime"])
    by_name = {p.name: p for p in pkgs}

    assert by_name["dynamo-runtime"].is_direct is True
    assert by_name["dynamo-runtime"].ecosystem == Ecosystem.RUST
    assert by_name["serde"].is_direct is False
    assert by_name["dynamo-runtime"].dependencies == ["serde"]


def test_parse_cargo_lock_falls_back_to_source_heuristic() -> None:
    """When workspace_members is None, treat sourceless crates as direct."""
    content = dedent("""
        [[package]]
        name = "local-only"
        version = "0.1.0"

        [[package]]
        name = "from-registry"
        version = "1.0.0"
        source = "registry+https://example.com"
        """)
    pkgs = {p.name: p for p in parse_cargo_lock(content)}
    assert pkgs["local-only"].is_direct is True
    assert pkgs["from-registry"].is_direct is False


def test_parse_go_mod_block_form() -> None:
    content = dedent("""
        module example.com/operator
        go 1.21

        require (
            k8s.io/api v0.28.0
            github.com/pkg/errors v0.9.1 // indirect
        )
        """)
    pkgs = {p.name: p for p in parse_go_mod(content)}

    assert pkgs["k8s.io/api"].version == "v0.28.0"
    assert pkgs["k8s.io/api"].is_direct is True
    assert pkgs["k8s.io/api"].ecosystem == Ecosystem.GO
    assert pkgs["github.com/pkg/errors"].is_direct is False


def test_parse_go_mod_single_line_require() -> None:
    """Finding #19: single-line `require module v1.2.3` directives."""
    content = dedent("""
        module example.com/cli
        go 1.21

        require github.com/spf13/cobra v1.8.0
        require github.com/stretchr/testify v1.9.0 // indirect
        """)
    pkgs = {p.name: p for p in parse_go_mod(content)}

    assert pkgs["github.com/spf13/cobra"].version == "v1.8.0"
    assert pkgs["github.com/spf13/cobra"].is_direct is True
    assert pkgs["github.com/stretchr/testify"].is_direct is False


def test_parse_go_mod_mixed_forms_dedupe() -> None:
    """Block + single-line forms should not produce duplicates."""
    content = dedent("""
        module example.com/svc

        require github.com/pkg/errors v0.9.1

        require (
            k8s.io/api v0.28.0
            github.com/pkg/errors v0.9.1 // indirect
        )
        """)
    pkgs = parse_go_mod(content)
    names = [p.name for p in pkgs]
    assert names.count("github.com/pkg/errors") == 1
    # First-seen wins; block-form is processed before single-line, so it stays direct
    # via the indirect=False from block parsing of the singleline-only form? No:
    # block runs first, so the block entry (// indirect) wins.
    assert {p.name for p in pkgs} == {"github.com/pkg/errors", "k8s.io/api"}


def test_parse_go_mod_empty_returns_empty_list() -> None:
    assert parse_go_mod("") == []
    assert parse_go_mod("   \n  ") == []


def test_parse_cargo_lock_empty_returns_empty_list() -> None:
    assert parse_cargo_lock("") == []
