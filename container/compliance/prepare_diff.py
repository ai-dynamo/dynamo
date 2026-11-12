#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an OSRB comparison using one exact run's artifact and matching inventories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .diff_osrb_csv import (
    BASELINE_BOOTSTRAP,
    baseline_unavailable_rows,
    compute_diff,
    read_osrb_csv,
    write_diff_csv,
)


class ComparisonError(ValueError):
    """Evidence is missing, ambiguous, or belongs to a different build."""


def inventory_name(prefix: str, arch: str, sha: str, cuda_version: str = "") -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ComparisonError("An exact 40-character source SHA is required")
    for value in (prefix, arch, cuda_version):
        if value and not re.fullmatch(r"[a-zA-Z0-9_.-]+", value):
            raise ComparisonError(f"Invalid inventory identity: {value!r}")
    image = f"{prefix}-{cuda_version}" if cuda_version else prefix
    return f"osrb-{image}-{arch}-{sha[:8]}.csv"


def select_osrb_csv(
    root: Path, prefix: str, arch: str, sha: str, cuda_version: str = ""
) -> Path:
    """Select exactly one full CSV; historical folders and other variants cannot match."""
    name = inventory_name(prefix, arch, sha, cuda_version)
    candidates = [root / f"linux_{arch}" / name, root / name]
    matches = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise ComparisonError(f"Expected one {name} in {root}; found {len(matches)}")
    return matches[0]


def validate_csv(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "ecosystem",
            "name",
            "version",
            "spdx",
            "source_url",
            "notes",
        ]:
            raise ComparisonError(f"Invalid OSRB inventory columns: {path}")
        count = 0
        for row in reader:
            if None in row or any(row[field] is None for field in reader.fieldnames):
                raise ComparisonError(f"Malformed OSRB inventory row: {path}")
            if not all(
                row[field].strip() for field in ("ecosystem", "name", "version", "spdx")
            ):
                raise ComparisonError(f"Incomplete OSRB inventory row: {path}")
            count += 1
        if not count:
            raise ComparisonError(f"Empty OSRB inventory: {path}")


def validate_artifact(
    metadata: dict, artifact_id: str, run_id: str, sha: str, prefix: str
) -> None:
    expected_name = f"compliance-{sha}-{prefix}"
    provenance = metadata["workflow_run"]
    if (
        str(metadata["id"]) != artifact_id
        or metadata["name"] != expected_name
        or metadata["expired"]
        or metadata["size_in_bytes"] <= 0
        or str(provenance["id"]) != run_id
        or provenance["head_sha"] != sha
    ):
        raise ComparisonError(
            "Baseline artifact does not match the selected run, SHA, and variant"
        )


def download_baseline(
    repo: str, artifact_id: str, run_id: str, sha: str, prefix: str
) -> bytes:
    if not artifact_id.isdigit() or not run_id.isdigit():
        raise ComparisonError("Baseline artifact and generating run IDs are required")
    endpoint = f"/repos/{repo}/actions/artifacts/{artifact_id}"
    response = subprocess.run(["gh", "api", endpoint], capture_output=True, check=True)
    metadata = json.loads(response.stdout)
    validate_artifact(metadata, artifact_id, run_id, sha, prefix)
    archive = subprocess.run(
        ["gh", "api", f"{endpoint}/zip"], capture_output=True, check=True
    ).stdout
    digest = metadata.get("digest")
    if digest and digest != "sha256:" + hashlib.sha256(archive).hexdigest():
        raise ComparisonError(
            "Downloaded baseline archive does not match its GitHub artifact digest"
        )
    return archive


def _baseline_member(
    archive: ZipFile, prefix: str, arch: str, sha: str, cuda_version: str
) -> str:
    name = inventory_name(prefix, arch, sha, cuda_version)
    candidates = {f"linux_{arch}/{name}", name}
    matches = [member for member in archive.namelist() if member in candidates]
    if len(matches) != 1:
        raise ComparisonError(
            f"Expected one matching baseline inventory {name}; found {len(matches)}"
        )
    return matches[0]


def prepare_comparison(
    root: Path,
    repo: str,
    prefix: str,
    platforms: str,
    current_sha: str,
    cuda_version: str = "",
    base_sha: str = "",
    base_label: str = "",
    artifact_id: str = "",
    run_id: str = "",
    allow_bootstrap: bool = False,
    bootstrap_legacy_operator_arm64: bool = False,
) -> None:
    platforms_list = platforms.split(",")
    if not platforms_list or any(
        not re.fullmatch(r"linux/(amd64|arm64)", p) for p in platforms_list
    ):
        raise ComparisonError(f"Unsupported platforms: {platforms}")
    arches = [p.split("/")[1] for p in platforms_list]
    if len(set(arches)) != len(arches):
        raise ComparisonError("Duplicate comparison architectures")
    current = {
        arch: select_osrb_csv(root, prefix, arch, current_sha, cuda_version)
        for arch in arches
    }
    for path in current.values():
        validate_csv(path)
    bootstrap = not (base_sha or artifact_id or run_id)
    if bootstrap and not allow_bootstrap:
        raise ComparisonError(
            "Expected baseline is missing; explicit first-of-kind bootstrap is required"
        )
    if allow_bootstrap and not bootstrap:
        raise ComparisonError("Bootstrap cannot bypass a selected baseline")
    baseline = {}
    migration_arches = set()
    if not bootstrap:
        archive_bytes = download_baseline(repo, artifact_id, run_id, base_sha, prefix)
        # Read only exact inventory members. Never unpack arbitrary archive paths.
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            for arch, path in current.items():
                try:
                    member = _baseline_member(
                        archive, prefix, arch, base_sha, cuda_version
                    )
                except ComparisonError:
                    # Old operator artifacts genuinely never contained ARM64.
                    # This explicit migration cannot exempt a new-format or
                    # malformed ARM64 inventory, another image, or a failed API.
                    if not (
                        bootstrap_legacy_operator_arm64
                        and prefix == "operator"
                        and arch == "arm64"
                        and not cuda_version
                        and "build-provenance.json" not in archive.namelist()
                        and not any("arm64" in name for name in archive.namelist())
                    ):
                        raise
                    old_amd64 = _baseline_member(archive, prefix, "amd64", base_sha, "")
                    migration_evidence = path.parent / "baseline" / Path(old_amd64).name
                    migration_evidence.parent.mkdir(exist_ok=True)
                    migration_evidence.write_bytes(archive.read(old_amd64))
                    validate_csv(migration_evidence)
                    migration_arches.add(arch)
                    continue
                directory = path.parent / "baseline"
                directory.mkdir(exist_ok=True)
                base_path = directory / Path(member).name
                base_path.write_bytes(archive.read(member))
                validate_csv(base_path)
                baseline[arch] = base_path
    for arch, path in current.items():
        base_path = baseline.get(arch)
        arch_bootstrap = bootstrap or arch in migration_arches
        bootstrap_reason = (
            "Explicit migration from legacy AMD64-only operator artifact; "
            "this is the first ARM64 inventory"
            if arch in migration_arches
            else "Explicit first-of-kind bootstrap; no comparison baseline exists"
        )
        if arch_bootstrap:
            rows = baseline_unavailable_rows(
                base_label or "explicit first-of-kind bootstrap"
            )
            rows[0]["change"] = BASELINE_BOOTSTRAP
            rows[0]["notes"] = bootstrap_reason
        else:
            rows = compute_diff(read_osrb_csv(path), read_osrb_csv(base_path))
        write_diff_csv(rows, path.with_suffix(".diff.csv"))
        directory = path.parent / "baseline"
        directory.mkdir(exist_ok=True)
        metadata = {
            "status": "bootstrap" if arch_bootstrap else "compared",
            "bootstrap_reason": bootstrap_reason if arch_bootstrap else None,
            "repository": repo,
            "source_sha": base_sha,
            "label": base_label,
            "run_id": run_id,
            "artifact_id": artifact_id,
            "csv": base_path.name if base_path else None,
        }
        (directory / "baseline.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        description = (
            bootstrap_reason
            if arch_bootstrap
            else f"Compared against `{base_path.name}` from [run {run_id}](https://github.com/{repo}/actions/runs/{run_id}), artifact `{artifact_id}`, commit `{base_sha}`."
        )
        (directory / "BASELINE.md").write_text(
            f"# OSRB diff baseline\n\n{description}\n\n{base_label}\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--platforms", required=True)
    parser.add_argument("--current-sha", required=True)
    parser.add_argument("--cuda-version", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--base-label", default="")
    parser.add_argument("--artifact-id", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--allow-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-legacy-operator-arm64", action="store_true")
    args = parser.parse_args()
    try:
        prepare_comparison(
            args.current_dir,
            args.repo,
            args.prefix,
            args.platforms,
            args.current_sha,
            args.cuda_version,
            args.base_sha,
            args.base_label,
            args.artifact_id,
            args.run_id,
            args.allow_bootstrap,
            args.bootstrap_legacy_operator_arm64,
        )
    except (
        ComparisonError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        BadZipFile,
    ) as exc:
        parser.exit(1, f"OSRB comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
