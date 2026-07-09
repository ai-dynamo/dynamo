# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import REPO

def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            REPO / "bench" / "results" / f"valkey-router-aiperf-{stamp}-{os.getpid()}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty to prevent mixed artifacts: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash one configured executable or shared object without loading it at once."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def rust_build_profile_from_path(path: Path) -> str | None:
    """Infer a Cargo profile only from an unambiguous ``target`` path segment."""

    profiles: list[str] = []
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "target":
            continue
        # Cargo uses target/{debug,release} for native builds and may insert a
        # target triple between those components for cross compilation.
        for candidate in parts[index + 1 : index + 3]:
            if candidate in {"debug", "release"}:
                profiles.append(candidate)
                break
    return profiles[0] if profiles and len(set(profiles)) == 1 else None


def file_provenance(path: Path) -> dict[str, Any]:
    """Return stable identity metadata for one benchmark executable artifact."""

    configured_path = path.expanduser().absolute()
    resolved_path = configured_path.resolve(strict=True)
    stat = resolved_path.stat()
    profile = rust_build_profile_from_path(resolved_path)
    return {
        "configured_path": str(configured_path),
        "resolved_path": str(resolved_path),
        "sha256": sha256_file(resolved_path),
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "mtime_ns": stat.st_mtime_ns,
        "rust_build_profile": profile,
        "rust_build_profile_source": "resolved_path" if profile else None,
    }


def enrich_dynamo_core_build_profile(record: dict[str, Any]) -> None:
    """Identify an installed ``dynamo._core`` copy by matching Cargo output."""

    if record["rust_build_profile"] is not None:
        record["matched_build_artifact_path"] = record["resolved_path"]
        return

    loaded_path = Path(record["resolved_path"])
    loaded_size = record["size_bytes"]
    loaded_sha256 = record["sha256"]
    matches: list[tuple[str, str]] = []
    target_root = REPO / "lib" / "bindings" / "python" / "target"
    for profile in ("debug", "release"):
        candidate = target_root / profile / "lib_core.so"
        try:
            resolved_candidate = candidate.resolve(strict=True)
            candidate_stat = resolved_candidate.stat()
        except OSError:
            continue
        # Debug extensions can exceed a gigabyte. Size is sufficient to reject
        # a non-match without an unnecessary second full-file hash.
        if candidate_stat.st_size != loaded_size:
            continue
        if resolved_candidate == loaded_path:
            candidate_sha256 = loaded_sha256
        else:
            candidate_sha256 = sha256_file(resolved_candidate)
        if candidate_sha256 == loaded_sha256:
            matches.append((profile, str(resolved_candidate)))

    record["matched_build_artifact_paths"] = [path for _, path in matches]
    if len(matches) == 1:
        profile, matched_path = matches[0]
        record["rust_build_profile"] = profile
        record["rust_build_profile_source"] = "target_artifact_sha256"
        record["matched_build_artifact_path"] = matched_path
    else:
        record["matched_build_artifact_path"] = None
        if matches:
            record["rust_build_profile_source"] = "ambiguous_target_artifact_sha256"


def python_runtime_provenance() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "resolved_executable": str(Path(sys.executable).resolve(strict=True)),
        "version": sys.version,
        "implementation": sys.implementation.name,
    }


def git_provenance() -> dict[str, Any]:
    record: dict[str, Any] = {"revision": None, "dirty": None}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            check=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=REPO,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    record["revision"] = revision.stdout.strip() or None
    record["dirty"] = bool(status.stdout)
    return record


def benchmark_provenance(
    args: argparse.Namespace, *, include_valkey_artifacts: bool
) -> dict[str, Any]:
    """Capture the exact native code and interpreter used by this process."""

    core = importlib.import_module("dynamo._core")
    core_import_path = getattr(core, "__file__", None)
    if not isinstance(core_import_path, str) or not core_import_path:
        raise RuntimeError("dynamo._core does not expose a loadable shared-object path")
    core_record = file_provenance(Path(core_import_path))
    core_record["import_path"] = core_import_path
    core_record["build_git_revision"] = getattr(
        core, "__build_git_revision__", None
    )
    core_record["build_git_dirty"] = getattr(core, "__build_git_dirty__", None)
    enrich_dynamo_core_build_profile(core_record)
    if include_valkey_artifacts:
        dynkv_module = file_provenance(args.dynkv_module)
        valkey_server = file_provenance(args.valkey_server)
    else:
        dynkv_module = {
            "configured_path": str(args.dynkv_module),
            "used_by_planned_arms": False,
        }
        valkey_server = {
            "configured_path": str(args.valkey_server),
            "used_by_planned_arms": False,
        }
    return {
        "benchmark_harness": file_provenance(Path(__file__)),
        "dynamo_core": core_record,
        "dynkv_module": dynkv_module,
        "valkey_server": valkey_server,
        "aiperf": file_provenance(args.aiperf),
        "python": python_runtime_provenance(),
        "git": git_provenance(),
    }


def release_core_provenance_error(provenance: Any) -> str | None:
    """Return why a provenance record cannot prove a release core build."""

    if not isinstance(provenance, Mapping):
        return "benchmark provenance is not a mapping"
    core = provenance.get("dynamo_core")
    if not isinstance(core, Mapping):
        return "benchmark provenance has no dynamo._core record"
    profile = core.get("rust_build_profile")
    if profile != "release":
        return (
            "active dynamo._core build profile is "
            f"{profile!r}; performance comparisons require a proven release build"
        )
    build_revision = core.get("build_git_revision")
    if not isinstance(build_revision, str) or not build_revision:
        return "active dynamo._core does not expose its source revision"
    build_dirty = core.get("build_git_dirty")
    if build_dirty is not False:
        return (
            "active dynamo._core was built from a dirty or unproven source "
            f"checkout ({build_dirty!r})"
        )
    git = provenance.get("git")
    source_revision = git.get("revision") if isinstance(git, Mapping) else None
    if build_revision != source_revision:
        return (
            "active dynamo._core source revision "
            f"{build_revision!r} does not match benchmark source revision "
            f"{source_revision!r}"
        )
    return None


def provenance_change_errors(
    campaign: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    """Reject build/runtime identity changes before or during one arm."""

    def comparison_identity(record: Mapping[str, Any]) -> dict[str, Any]:
        identity = dict(record)
        git = identity.get("git")
        if isinstance(git, Mapping):
            # Benchmark artifacts may live under the worktree and make an
            # otherwise clean checkout dirty. Revision and binary hashes are
            # identity; the dirty boolean is descriptive metadata only.
            comparable_git = dict(git)
            comparable_git.pop("dirty", None)
            identity["git"] = comparable_git
        return identity

    errors: list[str] = []
    for label, record in (("campaign", campaign), ("before", before), ("after", after)):
        if error := release_core_provenance_error(record):
            errors.append(f"{label} provenance: {error}")
    campaign_identity = comparison_identity(campaign)
    before_identity = comparison_identity(before)
    after_identity = comparison_identity(after)
    if before_identity != campaign_identity:
        errors.append("arm-start provenance differs from campaign provenance")
    if after_identity != before_identity:
        errors.append("arm-end provenance differs from arm-start provenance")
    return errors


@contextlib.contextmanager
def environment(overrides: Mapping[str, str | None]) -> Iterator[None]:
    """Temporarily set process-wide variables inherited by wrapper subprocesses."""

    old_values = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
