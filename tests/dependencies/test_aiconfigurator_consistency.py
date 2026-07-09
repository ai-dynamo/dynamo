# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep every Dynamo AIC dependency on the same release candidate."""

from __future__ import annotations

import json
import re
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

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


def _normalized_git_url(value: str) -> str:
    parsed = urlsplit(value)
    path = parsed.path.removesuffix(".git")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _version_candidate(specifiers: SpecifierSet) -> tuple[str, ...]:
    normalized = tuple(sorted(str(specifier) for specifier in specifiers))
    assert normalized, "AIC version dependency must include a constraint"
    return ("version", *normalized)


def _python_candidate(requirement: str) -> tuple[str, ...]:
    requirement = requirement.strip()
    parsed = Requirement(requirement)
    assert parsed.name == "aiconfigurator", parsed
    assert parsed.marker is None, "AIC candidate must not be conditional"
    if parsed.url:
        direct = DIRECT_REFERENCE_RE.fullmatch(requirement)
        assert direct, f"unsupported AIC direct reference: {requirement!r}"
        return (
            "git",
            _normalized_git_url(direct.group("url")),
            direct.group("ref"),
        )
    return _version_candidate(parsed.specifier)


def _cargo_version_candidate(requirement: str) -> tuple[str, ...]:
    requirement = requirement.strip()
    if requirement.startswith("="):
        return _version_candidate(SpecifierSet(f"=={requirement[1:]}"))

    version = Version(requirement.removeprefix("^"))
    if version.major > 0:
        upper = Version(f"{version.major + 1}.0.0")
    elif version.minor > 0:
        upper = Version(f"0.{version.minor + 1}.0")
    else:
        upper = Version(f"0.0.{version.micro + 1}")
    return _version_candidate(SpecifierSet(f">={version},<{upper}"))


def _cargo_candidate(dependency: object) -> tuple[str, ...]:
    if isinstance(dependency, str):
        return _cargo_version_candidate(dependency)
    assert isinstance(dependency, dict), dependency
    if "git" in dependency:
        revision = dependency.get("rev") or dependency.get("tag")
        assert (
            revision
        ), f"git dependency must use an immutable rev or tag: {dependency}"
        return ("git", _normalized_git_url(str(dependency["git"])), str(revision))
    version = dependency.get("version")
    assert version, f"unsupported Cargo dependency: {dependency}"
    return _cargo_version_candidate(str(version))


def _cargo_lock_candidate(path: Path) -> tuple[str, ...]:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [
        package for package in packages if package["name"] == "aiconfigurator-core"
    ]
    assert len(matches) == 1, f"expected one aiconfigurator-core package in {path}"

    package = matches[0]
    version = str(package["version"])
    source = str(package.get("source", ""))
    if source.startswith("git+"):
        repository, separator, commit = source.removeprefix("git+").partition("#")
        assert separator and re.fullmatch(r"[0-9a-fA-F]{40}", commit), source
        repository = repository.split("?", 1)[0]
        return ("git", _normalized_git_url(repository), commit.lower(), version)
    return ("version", version)


def _installed_candidate() -> tuple[str, ...]:
    distribution = metadata.distribution("aiconfigurator")
    version = distribution.version
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url.get("vcs_info")
        if vcs_info:
            commit = str(vcs_info["commit_id"])
            assert re.fullmatch(r"[0-9a-fA-F]{40}", commit), vcs_info
            return (
                "git",
                _normalized_git_url(str(direct_url["url"])),
                commit.lower(),
                version,
            )
    return ("version", version)


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

    resolved = {
        "root Cargo.lock": _cargo_lock_candidate(ROOT / "Cargo.lock"),
        "Python bindings Cargo.lock": _cargo_lock_candidate(
            ROOT / "lib/bindings/python/Cargo.lock"
        ),
        "installed aiconfigurator": _installed_candidate(),
    }
    expected_resolved = resolved["root Cargo.lock"]
    resolved_mismatches = {
        consumer: candidate
        for consumer, candidate in resolved.items()
        if candidate != expected_resolved
    }
    assert not resolved_mismatches, (
        f"resolved AIC candidates differ; expected {expected_resolved}: "
        f"{resolved_mismatches}"
    )


def test_version_candidates_preserve_constraint_semantics() -> None:
    assert _python_candidate("aiconfigurator>=0.10.0,<0.11.0") == _cargo_candidate(
        "0.10.0"
    )
    assert _python_candidate("aiconfigurator==0.10.0") == _cargo_candidate("=0.10.0")
    assert _python_candidate("aiconfigurator>=0.10.0") != _cargo_candidate("0.10.0")
