# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep every Dynamo AIC dependency on one source revision and release."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
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
AIC_PACKAGES = {"aiconfigurator", "aiconfigurator-core"}
SourceCandidate = tuple[str, str, str]


def _normalized_git_url(value: str) -> str:
    parsed = urlsplit(value)
    path = parsed.path.removesuffix(".git")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _immutable_revision(value: str) -> str:
    assert re.fullmatch(
        r"[0-9a-fA-F]{40}", value
    ), f"AIC source dependency must use a full commit SHA: {value!r}"
    return value.lower()


def _python_source_candidate(requirement: str, *, package: str) -> SourceCandidate:
    parsed = Requirement(requirement)
    assert canonicalize_name(parsed.name) == canonicalize_name(package), parsed
    assert parsed.marker is None, "AIC source dependency must not be conditional"
    assert parsed.url and parsed.url.startswith(
        "git+"
    ), f"AIC source dependency must be a git reference: {requirement!r}"

    url_without_fragment = parsed.url.removeprefix("git+").split("#", 1)[0]
    repository, separator, revision = url_without_fragment.rpartition("@")
    assert (
        separator and repository and revision
    ), f"unsupported AIC direct reference: {requirement!r}"
    return (
        "git",
        _normalized_git_url(repository),
        _immutable_revision(revision),
    )


def _python_exact_version(requirement: str, *, package: str) -> Version:
    parsed = Requirement(requirement)
    assert canonicalize_name(parsed.name) == canonicalize_name(package), parsed
    assert parsed.marker is None, "AIC release dependency must not be conditional"
    assert (
        parsed.url is None
    ), f"AIC release dependency must be index-resolvable: {parsed}"

    specifiers = list(parsed.specifier)
    assert (
        len(specifiers) == 1 and specifiers[0].operator == "=="
    ), f"AIC release dependency must use one exact version: {parsed}"
    assert (
        "*" not in specifiers[0].version
    ), f"AIC release dependency must not use a wildcard: {parsed}"
    return Version(specifiers[0].version)


def _cargo_source_candidate(dependency: object) -> SourceCandidate:
    assert isinstance(dependency, dict), dependency
    repository = dependency.get("git")
    revision = dependency.get("rev")
    assert (
        repository and revision
    ), f"AIC Cargo dependency must use an immutable git rev: {dependency}"
    return (
        "git",
        _normalized_git_url(str(repository)),
        _immutable_revision(str(revision)),
    )


def _cargo_lock_candidate(path: Path) -> tuple[SourceCandidate, Version]:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [
        package for package in packages if package["name"] == "aiconfigurator-core"
    ]
    assert len(matches) == 1, f"expected one aiconfigurator-core package in {path}"

    package = matches[0]
    source = str(package.get("source", ""))
    assert source.startswith(
        "git+"
    ), f"aiconfigurator-core must resolve from the pinned git source in {path}: {source}"
    repository_and_query, separator, revision = source.removeprefix("git+").partition(
        "#"
    )
    assert separator, source
    repository = repository_and_query.split("?", 1)[0]
    return (
        (
            "git",
            _normalized_git_url(repository),
            _immutable_revision(revision),
        ),
        Version(str(package["version"])),
    )


def _project_requirements(path: Path, *, extra: str | None = None) -> dict[str, str]:
    with path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    requirements = (
        project["optional-dependencies"][extra] if extra else project["dependencies"]
    )
    return _aic_requirements(requirements, source=str(path))


def _requirements_file(path: Path) -> dict[str, str]:
    requirements = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _aic_requirements(requirements, source=str(path))


def _aic_requirements(requirements: list[str], *, source: str) -> dict[str, str]:
    matches: dict[str, str] = {}
    for requirement in requirements:
        parsed = Requirement(requirement)
        package = canonicalize_name(parsed.name)
        if package not in AIC_PACKAGES:
            continue
        assert package not in matches, f"duplicate {package} requirement in {source}"
        matches[package] = requirement
    return matches


def _container_source_candidate(expected_repository: str) -> SourceCandidate:
    context = (ROOT / "container/context.yaml").read_text(encoding="utf-8")
    revisions = re.findall(
        r"^\s*aiconfigurator_ref:\s*[\"']?([0-9a-fA-F]{40})[\"']?\s*$",
        context,
        flags=re.MULTILINE,
    )
    assert len(revisions) == 1, "expected one immutable aiconfigurator_ref"
    return ("git", expected_repository, _immutable_revision(revisions[0]))


def test_all_aiconfigurator_dependencies_use_one_candidate() -> None:
    with (ROOT / "Cargo.toml").open("rb") as handle:
        root_cargo = tomllib.load(handle)
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings_cargo = tomllib.load(handle)

    benchmark_requirements = _project_requirements(ROOT / "benchmarks/pyproject.toml")
    assert set(benchmark_requirements) == {"aiconfigurator-core"}

    source_candidates = {
        "root Cargo workspace": _cargo_source_candidate(
            root_cargo["workspace"]["dependencies"]["aiconfigurator-core"]
        ),
        "Python bindings Cargo": _cargo_source_candidate(
            bindings_cargo["dependencies"]["aiconfigurator-core"]
        ),
        "benchmarks": _python_source_candidate(
            benchmark_requirements["aiconfigurator-core"],
            package="aiconfigurator-core",
        ),
    }
    expected_source = source_candidates["root Cargo workspace"]
    source_candidates["container wheel build"] = _container_source_candidate(
        expected_source[1]
    )

    resolved = {
        "root Cargo.lock": _cargo_lock_candidate(ROOT / "Cargo.lock"),
        "Python bindings Cargo.lock": _cargo_lock_candidate(
            ROOT / "lib/bindings/python/Cargo.lock"
        ),
    }
    source_candidates.update(
        {consumer: candidate[0] for consumer, candidate in resolved.items()}
    )
    source_mismatches = {
        consumer: candidate
        for consumer, candidate in source_candidates.items()
        if candidate != expected_source
    }
    assert (
        not source_mismatches
    ), f"AIC source candidates differ; expected {expected_source}: {source_mismatches}"

    root_requirements = _project_requirements(ROOT / "pyproject.toml", extra="mocker")
    aisimulate_requirements = _project_requirements(ROOT / "aisimulate/pyproject.toml")
    planner_requirements = _requirements_file(
        ROOT / "container/deps/requirements.planner.txt"
    )
    assert set(root_requirements) == {"aiconfigurator-core"}
    assert set(aisimulate_requirements) == {"aiconfigurator"}
    assert set(planner_requirements) == AIC_PACKAGES

    versions = {
        "root Cargo.lock": resolved["root Cargo.lock"][1],
        "Python bindings Cargo.lock": resolved["Python bindings Cargo.lock"][1],
        "ai-dynamo[mocker]": _python_exact_version(
            root_requirements["aiconfigurator-core"],
            package="aiconfigurator-core",
        ),
        "aisimulate upper": _python_exact_version(
            aisimulate_requirements["aiconfigurator"],
            package="aiconfigurator",
        ),
        "planner upper": _python_exact_version(
            planner_requirements["aiconfigurator"],
            package="aiconfigurator",
        ),
        "planner core": _python_exact_version(
            planner_requirements["aiconfigurator-core"],
            package="aiconfigurator-core",
        ),
    }
    expected_version = versions["root Cargo.lock"]
    version_mismatches = {
        consumer: version
        for consumer, version in versions.items()
        if version != expected_version
    }
    assert not version_mismatches, (
        f"AIC release versions differ; expected {expected_version}: "
        f"{version_mismatches}"
    )


def test_installed_aiconfigurator_packages_match_declared_release() -> None:
    _, expected_version = _cargo_lock_candidate(ROOT / "Cargo.lock")
    installed = {
        "aiconfigurator": Version(metadata.version("aiconfigurator")),
        "aiconfigurator-core": Version(metadata.version("aiconfigurator-core")),
    }
    mismatches = {
        package: version
        for package, version in installed.items()
        if version != expected_version
    }
    assert (
        not mismatches
    ), f"installed AIC versions differ; expected {expected_version}: {mismatches}"


def test_candidate_parsing_preserves_source_and_version_semantics() -> None:
    source = _python_source_candidate(
        "aiconfigurator-core @ "
        "git+https://github.com/ai-dynamo/aiconfigurator.git@"
        "f3b78eab442377b2693ee52dcdcdc6dd82c21f9e#subdirectory=aic-core",
        package="aiconfigurator-core",
    )
    assert source == (
        "git",
        "https://github.com/ai-dynamo/aiconfigurator",
        "f3b78eab442377b2693ee52dcdcdc6dd82c21f9e",
    )
    assert _python_exact_version(
        "aiconfigurator-core==0.11.0", package="aiconfigurator-core"
    ) == Version("0.11.0")
