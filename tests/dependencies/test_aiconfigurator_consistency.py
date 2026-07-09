# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep every Dynamo AIC dependency on the same release candidate."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.aiconfigurator,
]

ROOT = Path(__file__).resolve().parents[2]
DIRECT_REFERENCE_RE = re.compile(
    r"^aiconfigurator\s*@\s*git\+(?P<url>.+)@(?P<ref>[^@\s]+)$"
)
VERSION_REFERENCE_RE = re.compile(
    r"^aiconfigurator\s*(?:==|>=)\s*(?P<version>[^,;\s]+)$"
)


def _normalized_git_url(value: str) -> str:
    parsed = urlsplit(value)
    path = parsed.path.removesuffix(".git")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _python_candidate(requirement: str) -> tuple[str, ...]:
    requirement = requirement.strip()
    direct = DIRECT_REFERENCE_RE.fullmatch(requirement)
    if direct:
        return (
            "git",
            _normalized_git_url(direct.group("url")),
            direct.group("ref"),
        )
    version = VERSION_REFERENCE_RE.fullmatch(requirement)
    if version:
        return ("version", version.group("version"))
    raise AssertionError(f"unsupported aiconfigurator requirement: {requirement!r}")


def _cargo_candidate(dependency: object) -> tuple[str, ...]:
    if isinstance(dependency, str):
        return ("version", dependency.lstrip("="))
    assert isinstance(dependency, dict), dependency
    if "git" in dependency:
        revision = dependency.get("rev") or dependency.get("tag")
        assert (
            revision
        ), f"git dependency must use an immutable rev or tag: {dependency}"
        return ("git", _normalized_git_url(str(dependency["git"])), str(revision))
    version = dependency.get("version")
    assert version, f"unsupported Cargo dependency: {dependency}"
    return ("version", str(version).lstrip("="))


def _project_requirement(path: Path, *, extra: str | None = None) -> str:
    with path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    requirements = (
        project["optional-dependencies"][extra] if extra else project["dependencies"]
    )
    matches = [
        requirement
        for requirement in requirements
        if requirement.strip().startswith("aiconfigurator")
    ]
    assert len(matches) == 1, f"expected one AIC requirement in {path}: {matches}"
    return matches[0]


def _planner_requirement() -> str:
    path = ROOT / "container/deps/requirements.planner.txt"
    matches = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("aiconfigurator")
    ]
    assert len(matches) == 1, f"expected one AIC requirement in {path}: {matches}"
    return matches[0]


def test_all_aiconfigurator_dependencies_use_one_candidate() -> None:
    with (ROOT / "Cargo.toml").open("rb") as handle:
        root_cargo = tomllib.load(handle)
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings_cargo = tomllib.load(handle)

    candidates = {
        "root Cargo workspace": _cargo_candidate(
            root_cargo["workspace"]["dependencies"]["aiconfigurator-core"]
        ),
        "Python bindings Cargo": _cargo_candidate(
            bindings_cargo["dependencies"]["aiconfigurator-core"]
        ),
        "ai-dynamo[mocker]": _python_candidate(
            _project_requirement(ROOT / "pyproject.toml", extra="mocker")
        ),
        "benchmarks": _python_candidate(
            _project_requirement(ROOT / "benchmarks/pyproject.toml")
        ),
        "planner image": _python_candidate(_planner_requirement()),
    }

    expected = candidates["root Cargo workspace"]
    mismatches = {
        consumer: candidate
        for consumer, candidate in candidates.items()
        if candidate != expected
    }
    assert not mismatches, f"AIC candidates differ; expected {expected}: {mismatches}"
