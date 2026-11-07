# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the source commit that hatch_build.py writes into every _version.py."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    # Each test runs git subprocesses (well under a second locally).
    pytest.mark.timeout(10),
    pytest.mark.unit,
]

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_VERSION = "1.2.3.dev20260101"
VALID_SHA = "0123456789abcdef0123456789abcdef01234567"
COMPONENTS = ("alpha", "beta")


class _FakeApp:
    def __init__(self) -> None:
        """Collect build warnings without requiring a Hatch application."""
        self.warnings: list[str] = []

    def display_warning(self, message: str) -> None:
        """Keep warning text so tests can check the missing-commit message."""
        self.warnings.append(message)


class _StubBuildHookInterface:
    """Stand-in for hatchling's base class; hatchling is not installed in test images."""


def _load_hook(tree: Path, monkeypatch: pytest.MonkeyPatch, app: _FakeApp):
    """Copy hatch_build.py into tree and return a VersionWriterHook rooted there."""
    shutil.copy(REPO_ROOT / "hatch_build.py", tree / "hatch_build.py")
    for component in COMPONENTS:
        (tree / "components" / "src" / "dynamo" / component).mkdir(parents=True)

    interface = types.ModuleType("hatchling.builders.hooks.plugin.interface")
    interface.BuildHookInterface = _StubBuildHookInterface  # type: ignore[attr-defined]
    for name in (
        "hatchling",
        "hatchling.builders",
        "hatchling.builders.hooks",
        "hatchling.builders.hooks.plugin",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, interface.__name__, interface)

    spec = importlib.util.spec_from_file_location(
        f"hatch_build_under_test_{id(tree)}", tree / "hatch_build.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the build hook from {tree / 'hatch_build.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hook = module.VersionWriterHook()
    hook.root = str(tree)
    hook.metadata = types.SimpleNamespace(version=METADATA_VERSION)
    hook.app = app
    return hook


def _read_versions(tree: Path) -> dict[str, tuple[str | None, str | None]]:
    """Read each component's version and commit without importing its package."""
    result = {}
    for component in COMPONENTS:
        text = (
            tree / "components" / "src" / "dynamo" / component / "_version.py"
        ).read_text()
        version = re.search(r'^__version__ = "([^"]*)"$', text, re.MULTILINE)
        commit = re.search(r'^__git_commit__ = "([^"]*)"$', text, re.MULTILINE)
        result[component] = (
            version.group(1) if version else None,
            commit.group(1) if commit else None,
        )
    return result


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated source tree with no inherited commit metadata."""
    root = tmp_path / "src"
    root.mkdir()
    monkeypatch.delenv("DYNAMO_GIT_COMMIT", raising=False)
    monkeypatch.delenv("CI", raising=False)
    # Stop git from discovering an enclosing repository, so "no .git" holds.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    return root


def test_explicit_commit_without_git(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normalize an explicit SHA and write it to every component without Git."""
    monkeypatch.setenv("DYNAMO_GIT_COMMIT", VALID_SHA.upper())
    app = _FakeApp()
    _load_hook(tree, monkeypatch, app).initialize("standard", {})

    assert _read_versions(tree) == {
        c: (METADATA_VERSION, VALID_SHA) for c in COMPONENTS
    }
    assert app.warnings == []


def test_falls_back_to_git_head(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the source checkout's HEAD while preserving the version suffix."""
    git = [
        "git",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run([*git, "init", "-q"], cwd=tree, check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "x"], cwd=tree, check=True
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    short = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    app = _FakeApp()
    _load_hook(tree, monkeypatch, app).initialize("standard", {})

    expected = (f"{METADATA_VERSION}+{short}", head)
    assert _read_versions(tree) == {c: expected for c in COMPONENTS}
    assert app.warnings == []


def test_no_commit_writes_empty_and_warns(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep builds usable without a commit and explain how to supply one."""
    app = _FakeApp()
    _load_hook(tree, monkeypatch, app).initialize("standard", {})

    assert _read_versions(tree) == {c: (METADATA_VERSION, "") for c in COMPONENTS}
    assert len(app.warnings) == 1
    assert "DYNAMO_GIT_COMMIT" in app.warnings[0]


def test_malformed_explicit_commit_raises(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a short explicit SHA instead of recording ambiguous metadata."""
    monkeypatch.setenv("DYNAMO_GIT_COMMIT", VALID_SHA[:7])
    hook = _load_hook(tree, monkeypatch, _FakeApp())
    with pytest.raises(ValueError, match="DYNAMO_GIT_COMMIT"):
        hook.initialize("standard", {})


def test_keeps_commit_from_existing_version_file(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carry the sdist's recorded commit into every wheel component."""
    # Models a wheel built from an unpacked sdist: no .git, no env var, but the
    # sdist build already wrote the commit into _version.py.
    hook = _load_hook(tree, monkeypatch, _FakeApp())
    (tree / "components" / "src" / "dynamo" / COMPONENTS[1] / "_version.py").write_text(
        f'__version__ = "{METADATA_VERSION}"\n__git_commit__ = "{VALID_SHA}"\n'
    )
    app = _FakeApp()
    hook.app = app
    hook.initialize("standard", {})

    assert _read_versions(tree) == {
        c: (METADATA_VERSION, VALID_SHA) for c in COMPONENTS
    }
    assert app.warnings == []


def test_conflicting_existing_commits_raise(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when existing component files disagree about their source commit."""
    hook = _load_hook(tree, monkeypatch, _FakeApp())
    other_sha = "f" * 40
    for component, sha in zip(COMPONENTS, (VALID_SHA, other_sha)):
        (tree / "components" / "src" / "dynamo" / component / "_version.py").write_text(
            f'__version__ = "{METADATA_VERSION}"\n__git_commit__ = "{sha}"\n'
        )

    with pytest.raises(ValueError, match="record different commits"):
        hook.initialize("standard", {})


def test_unreadable_existing_version_file_raises(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate read errors so an unreadable commit is not treated as absent."""
    hook = _load_hook(tree, monkeypatch, _FakeApp())
    # A directory in place of _version.py: opening it fails with an error
    # other than "file not found".
    (tree / "components" / "src" / "dynamo" / COMPONENTS[0] / "_version.py").mkdir()
    read_existing = type(hook).initialize.__globals__["read_existing_git_commit"]
    components = [str(tree / "components" / "src" / "dynamo" / c) for c in COMPONENTS]

    with pytest.raises(IsADirectoryError):
        read_existing(components)


def test_ignores_commit_of_enclosing_checkout(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avoid attributing an unpacked source tree to an unrelated parent checkout."""
    # Models an unpacked source tree placed inside an unrelated checkout.
    outer = tree.parent
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(outer.parent))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=outer, check=True)
    subprocess.run(
        [
            *git,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        cwd=outer,
        check=True,
    )

    app = _FakeApp()
    _load_hook(tree, monkeypatch, app).initialize("standard", {})

    assert {commit for _, commit in _read_versions(tree).values()} == {""}
    assert len(app.warnings) == 1
