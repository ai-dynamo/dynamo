# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover the published Rust API surface from repository release data."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gen_llms_tables import parse_data_module

RustCrateGroup = Literal["core", "supporting", "development", "deprecated"]

VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)")

SOURCE_BASE = "https://github.com/ai-dynamo/dynamo/tree/main"
CORE_CRATES = frozenset(
    {
        "dynamo-runtime",
        "dynamo-llm",
        "dynamo-kv-router",
        "dynamo-memory",
        "kvbm-logical",
    }
)
DEVELOPMENT_CRATES = frozenset({"dynamo-mocker"})
DEPRECATED_CRATES = frozenset({"dynamo-async-openai"})
GROUP_ORDER: dict[RustCrateGroup, int] = {
    "core": 0,
    "supporting": 1,
    "development": 2,
    "deprecated": 3,
}


@dataclass(frozen=True)
class RustCrate:
    """One published Rust crate in the current release inventory."""

    name: str
    summary: str
    meta: str
    version: str
    group: RustCrateGroup
    docs_href: str
    crates_href: str
    install_command: str
    member_path: str | None
    source_href: str | None
    badge: str | None


@dataclass(frozen=True)
class RustBinding:
    """One source-only language binding around the Rust core."""

    name: str
    language: str
    summary: str
    member_path: str
    source_href: str


@dataclass(frozen=True)
class RustReference:
    """Complete generated model for the Rust API landing page."""

    workspace_version: str
    release_tag: str
    crates: tuple[RustCrate, ...]
    bindings: tuple[RustBinding, ...]


def discover_rust_reference(repo_root: Path, data_path: Path) -> RustReference:
    """Build the Rust reference from Cargo metadata and release artifacts."""
    workspace = _load_toml(repo_root / "Cargo.toml")
    version = str(workspace["workspace"]["package"]["version"])
    members = _workspace_member_paths(repo_root, workspace)
    release_data = parse_data_module(data_path)
    release_tag = validate_release_tag(release_data, version)
    crates = [
        _crate_from_artifact(item, members)
        for item in release_data["ARTIFACTS"]
        if item.get("category") == "crate"
    ]
    crates.sort(key=lambda crate: (GROUP_ORDER[crate.group], crate.name))
    return RustReference(
        workspace_version=version,
        release_tag=release_tag,
        crates=tuple(crates),
        bindings=_bindings(),
    )


def _load_toml(path: Path) -> dict[str, object]:
    """Load one UTF-8 TOML document."""
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _workspace_member_paths(
    repo_root: Path,
    workspace: dict[str, object],
) -> dict[str, str]:
    """Map each workspace package name to its repository-relative path."""
    workspace_table = workspace.get("workspace")
    if not isinstance(workspace_table, dict):
        raise ValueError("Cargo.toml is missing [workspace]")
    raw_members = workspace_table.get("members")
    if not isinstance(raw_members, list):
        raise ValueError("Cargo.toml is missing workspace.members")
    return _member_name_map(repo_root, [str(member) for member in raw_members])


def _member_name_map(repo_root: Path, members: list[str]) -> dict[str, str]:
    """Resolve workspace paths to package names."""
    names: dict[str, str] = {}
    for member_path in members:
        manifest = _load_toml(repo_root / member_path / "Cargo.toml")
        package = manifest.get("package")
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        if isinstance(name, str):
            names[name] = member_path
    return names


def validate_release_tag(data: dict[str, object], workspace_version: str) -> str:
    """Return the shipped release tag that crate links are pinned to.

    Crate versions follow the newest release that actually shipped, not the
    workspace version: main carries the next development version for months
    before any crate is published under it, so the tag normally lags. A tag
    ahead of the workspace is fail-closed -- it would pin docs.rs links at
    crates that were never published.
    """
    current_tag = data.get("CURRENT_TAG")
    if not isinstance(current_tag, str):
        raise ValueError("release data is missing a string CURRENT_TAG")
    if _version_key(current_tag) > _version_key(workspace_version):
        raise ValueError(
            f"release artifact tag {current_tag!r} is ahead of workspace "
            f"version {workspace_version!r}"
        )
    return current_tag


def _version_key(version: str) -> tuple[int, ...]:
    """Order X.Y.Z versions, ignoring any prerelease suffix."""
    match = VERSION_PATTERN.match(version)
    if match is None:
        raise ValueError(f"cannot parse release version {version!r}")
    return tuple(int(part) for part in match.groups())


def _crate_from_artifact(
    artifact: dict[str, object],
    member_paths: dict[str, str],
) -> RustCrate:
    """Adapt one releases.data.ts crate artifact into the typed model."""
    name = _required_text(artifact, "name")
    crates_href = _required_text(artifact, "href")
    version = _artifact_version(name, crates_href)
    member_path = member_paths.get(name)
    return RustCrate(
        name=name,
        summary=_required_text(artifact, "description"),
        meta=_required_text(artifact, "meta"),
        version=version,
        group=_crate_group(name),
        docs_href=f"https://docs.rs/{name}/{version}",
        crates_href=crates_href,
        install_command=_install_command(artifact),
        member_path=member_path,
        source_href=f"{SOURCE_BASE}/{member_path}" if member_path else None,
        badge=_optional_text(artifact, "badge"),
    )


def _required_text(item: dict[str, object], key: str) -> str:
    """Read one required string field."""
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"crate artifact has invalid {key!r}: {value!r}")
    return value


def _optional_text(item: dict[str, object], key: str) -> str | None:
    """Read one optional string field."""
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"crate artifact has invalid {key!r}: {value!r}")
    return value


def _artifact_version(name: str, href: str) -> str:
    """Extract and validate the version pinned in a crates.io artifact URL."""
    prefix = f"https://crates.io/crates/{name}/"
    if not href.startswith(prefix):
        raise ValueError(f"unexpected crates.io URL for {name}: {href}")
    version = href.removeprefix(prefix)
    if not version or "/" in version:
        raise ValueError(f"invalid crates.io version for {name}: {version!r}")
    return version


def _install_command(artifact: dict[str, object]) -> str:
    """Read the canonical cargo-add command from the first artifact tag."""
    tags = artifact.get("tags")
    if not isinstance(tags, list) or not tags or not isinstance(tags[0], dict):
        raise ValueError("crate artifact is missing install tags")
    return _required_text(tags[0], "clipboard")


def _crate_group(name: str) -> RustCrateGroup:
    """Assign the compact browser group for one published crate."""
    if name in CORE_CRATES:
        return "core"
    if name in DEVELOPMENT_CRATES:
        return "development"
    if name in DEPRECATED_CRATES:
        return "deprecated"
    return "supporting"


def _bindings() -> tuple[RustBinding, ...]:
    """Return source-only bindings maintained in this repository."""
    return (
        _binding(
            "dynamo-codegen",
            "Python",
            "PyO3 code generation for the dynamo._core module.",
            "lib/bindings/python/codegen",
        ),
        _binding(
            "libdynamo_llm",
            "C",
            "C bindings for the Dynamo LLM library.",
            "lib/bindings/c",
        ),
    )


def _binding(
    name: str,
    language: str,
    summary: str,
    member_path: str,
) -> RustBinding:
    """Build one repository binding record."""
    return RustBinding(
        name=name,
        language=language,
        summary=summary,
        member_path=member_path,
        source_href=f"{SOURCE_BASE}/{member_path}",
    )
