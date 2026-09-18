# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scoping the API freshness gate to what a branch changed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import api_freshness
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

SOURCES = ("lib/bindings/python/src", "components/src", "Cargo.toml")
OUTPUTS = frozenset({"docs/fern/pages/reference/api/python/README.mdx"})
SCRIPT_DIR = "docs/fern/scripts"


def _attributable(changed: frozenset[str] | None) -> bool:
    return api_freshness._attributable(
        changed,
        sources=SOURCES,
        output_paths=OUTPUTS,
        script_dir=SCRIPT_DIR,
    )


@pytest.mark.parametrize(
    "changed,expected,reason",
    [
        (frozenset({"lib/runtime/src/lib.rs"}), False, "Rust-only branch"),
        (frozenset({"docs/fern/pages/kubernetes/foo.mdx"}), False, "unrelated page"),
        (frozenset(), False, "empty diff"),
        (frozenset({"components/src/dynamo/common/storage.py"}), True, "API source"),
        (
            frozenset({"docs/fern/pages/reference/api/python/README.mdx"}),
            True,
            "output",
        ),
        (frozenset({"docs/fern/scripts/api_rendering.py"}), True, "generator"),
        (frozenset({"Cargo.toml"}), True, "source matched as an exact file"),
    ],
)
def test_attribution(changed: frozenset[str], expected: bool, reason: str) -> None:
    assert _attributable(changed) is expected, reason


def test_unresolvable_range_keeps_the_strict_gate() -> None:
    """None must not become a silent pass.

    A shallow clone without the base commit, or a bad ref, cannot tell us
    whether the branch caused the drift. Treating that as "not the branch's
    fault" would disable the gate exactly when it is least observable.
    """
    assert _attributable(None) is True


def test_a_source_prefix_does_not_match_a_sibling_by_string_prefix() -> None:
    """``components/src`` must not swallow ``components/srcfoo``.

    The match is anchored on a path separator rather than a bare prefix, so a
    directory whose name merely starts the same way is not attributed.
    """
    assert _attributable(frozenset({"components/srcfoo/thing.py"})) is False
    assert _attributable(frozenset({"components/src/thing.py"})) is True


def test_changed_paths_returns_none_for_an_unknown_ref(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert api_freshness.changed_paths("no-such-ref", tmp_path) is None


def test_changed_paths_returns_none_outside_a_repository(tmp_path: Path) -> None:
    assert api_freshness.changed_paths("HEAD", tmp_path) is None


def test_blames_branch_normalises_paths_to_repo_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public entry point does the path normalisation its callers used to.

    Guards the consolidation: each generator passes absolute Paths and trusts
    this to render them repo-relative before matching.
    """
    monkeypatch.setattr(
        api_freshness,
        "changed_paths",
        lambda since, repo_root: frozenset({"docs/fern/pages/out.mdx"}),
    )
    assert (
        api_freshness.blames_branch(
            "base",
            repo_root=tmp_path,
            sources=(),
            outputs=[tmp_path / "docs" / "fern" / "pages" / "out.mdx"],
            script_dir=tmp_path / "docs" / "fern" / "scripts",
        )
        is True
    )
