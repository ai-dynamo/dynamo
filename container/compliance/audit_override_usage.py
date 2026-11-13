#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report actual override use across a declared nightly matrix; never delete entries."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import subprocess
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from .overrides import Override, load_records
from .policy.review import classify
from .policy.validate import parse_spdx


def _api(endpoint: str) -> dict:
    return json.loads(
        subprocess.run(
            ["gh", "api", endpoint], capture_output=True, text=True, check=True
        ).stdout
    )


def _artifacts(repo: str, run: dict) -> list[dict]:
    records = []
    page = 1
    while True:
        payload = _api(
            f"/repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100&page={page}"
        )
        records.extend(payload["artifacts"])
        if page * 100 >= payload["total_count"]:
            return records
        page += 1


def download_latest(
    repo: str, expected: list[dict]
) -> tuple[dict, dict[str, bytes], list[str]]:
    runs = _api(
        f"/repos/{repo}/actions/workflows/nightly-ci.yml/runs?branch=main&event=schedule&status=completed&per_page=100"
    )["workflow_runs"]
    for run in sorted(runs, key=lambda r: r["id"], reverse=True):
        if (
            run["head_branch"] != "main"
            or run["event"] != "schedule"
            or run["head_repository"]["full_name"] != repo
            or run["path"].split("@")[0] != ".github/workflows/nightly-ci.yml"
        ):
            continue
        artifacts = _artifacts(repo, run)
        if not any(
            a["name"].startswith(f"compliance-{run['head_sha']}-") for a in artifacts
        ):
            continue
        archives = {}
        problems = []
        for image in expected:
            name = f"compliance-{run['head_sha']}-{image['prefix']}"
            matching = [a for a in artifacts if a["name"] == name and not a["expired"]]
            if len(matching) != 1:
                problems.append(f"{name}: missing, expired, or ambiguous artifact")
                continue
            artifact = matching[0]
            if (
                artifact["workflow_run"]["id"] != run["id"]
                or artifact["workflow_run"]["head_sha"] != run["head_sha"]
            ):
                problems.append(f"{name}: artifact provenance mismatch")
                continue
            try:
                raw = subprocess.run(
                    [
                        "gh",
                        "api",
                        f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip",
                    ],
                    capture_output=True,
                    check=True,
                ).stdout
                if (
                    not artifact.get("digest")
                    or artifact["digest"] != "sha256:" + hashlib.sha256(raw).hexdigest()
                ):
                    problems.append(f"{name}: artifact digest missing or mismatched")
                    continue
                archives[image["prefix"]] = raw
            except subprocess.CalledProcessError:
                problems.append(f"{name}: artifact download failed")
        catalog = _api(
            f"/repos/{repo}/contents/container/compliance/license_overrides.yaml?ref={run['head_sha']}"
        )
        run["override_catalog_sha256"] = hashlib.sha256(
            base64.b64decode(catalog["content"])
        ).hexdigest()
        return run, archives, problems
    raise ValueError("No completed scheduled nightly has compliance artifacts")


def _member(archive: ZipFile, name: str, arch: str) -> str:
    paths = [
        path for path in archive.namelist() if path in {name, f"linux_{arch}/{name}"}
    ]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {name} for {arch}; found {len(paths)}")
    return paths[0]


def _base_components(components: list[dict]) -> list[dict]:
    ecosystems = {"pypi": "python", "cargo": "rust", "golang": "go", "deb": "dpkg"}
    result = []
    for component in components:
        purl = component.get("purl", "")
        kind = purl.split("/", 1)[0].removeprefix("pkg:")
        if kind in ecosystems:
            result.append(
                {
                    "ecosystem": ecosystems[kind],
                    "name": component["name"],
                    "version": str(component.get("version", "")),
                    "origin": "base",
                }
            )
    return result


def inspect_archive(
    raw: bytes, image: dict, sha: str, catalog_sha256: str = ""
) -> tuple[list[dict], list[dict], list[str]]:
    observations, inventory, problems = [], [], []
    with ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Artifact contains an unsafe archive path")
        for arch in image["architectures"]:
            label = f"{image['prefix']}/{arch}"
            try:
                cuda = f"-{image['cuda']}" if image["cuda"] else ""
                csv_name = f"osrb-{image['prefix']}{cuda}-{arch}-{sha[:8]}.csv"
                current = _member(archive, csv_name, arch)
                rows = list(csv.DictReader(io.StringIO(archive.read(current).decode())))
                if (
                    not rows
                    or not {"ecosystem", "name", "version", "spdx"} <= rows[0].keys()
                ):
                    raise ValueError("Missing or malformed full inventory")
                inventory.extend({**row, "origin": label} for row in rows)
                telemetry_path = _member(archive, "override-usage.json", arch)
                # A flat telemetry member cannot serve both architectures.
                if len(image["architectures"]) > 1 and not telemetry_path.startswith(
                    f"linux_{arch}/"
                ):
                    raise ValueError("Multiarch telemetry is not architecture-specific")
                telemetry = json.loads(archive.read(telemetry_path))
                if telemetry["schema_version"] != 1:
                    raise ValueError("Unsupported override-use schema")
                if (
                    not catalog_sha256
                    or telemetry["override_catalog_sha256"] != catalog_sha256
                ):
                    raise ValueError(
                        "Override telemetry does not match the source commit's catalog"
                    )
                observed_keys = {
                    (r["ecosystem"], r["name"], r["version"])
                    for r in telemetry["observations"]
                }
                if any(
                    (r["ecosystem"], r["name"], r["version"]) not in observed_keys
                    for r in rows
                ):
                    raise ValueError(
                        "Telemetry does not cover every package in the current inventory"
                    )
                for row in telemetry["observations"]:
                    observations.append({**row, "origin": label})
                    inventory.append({**row, "origin": label})
                base = telemetry["baseline"]
                if base is not None:
                    if base["file"] != "override-base.cdx.json":
                        raise ValueError("Unexpected associated base inventory path")
                    member = str(PurePosixPath(telemetry_path).parent / base["file"])
                    base_bytes = archive.read(member)
                    if hashlib.sha256(base_bytes).hexdigest() != base["sha256"]:
                        raise ValueError("Associated base inventory hash mismatch")
                    inventory.extend(
                        _base_components(json.loads(base_bytes)["components"])
                    )
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                problems.append(f"{label}: {exc}")
    return observations, inventory, problems


def summarize(
    records: tuple[Override, ...],
    observations: list[dict],
    inventory: list[dict],
    problems: list[str],
) -> dict:
    result = []
    complete = not problems
    for record in records:
        matched = [
            r
            for r in observations
            if r["ecosystem"] == record.ecosystem
            and r["name"] == record.name
            and (record.version is None or r["version"] == record.version)
        ]
        present = [
            r
            for r in inventory
            if r["ecosystem"] == record.ecosystem
            and r["name"] == record.name
            and (record.version is None or r["version"] == record.version)
        ]
        configured = [
            r
            for r in matched
            if r["override_license"] is not None
            and r["override_version"] == record.version
            and (
                record.copyright_sha256 is None
                or r.get("copyright_sha256") == record.copyright_sha256
            )
        ]
        applied = [r for r in configured if r["applied"]]
        conflicts = []
        for row in applied:
            raw = row["raw_license"]
            if row["override_license"] != record.license:
                conflicts.append(
                    "Applied historical override differs from current catalog"
                )
            elif raw and raw != "UNKNOWN":
                try:
                    if parse_spdx(raw) != parse_spdx(record.license):
                        conflicts.append(
                            "Raw license metadata differs from the applied correction; verify its evidence"
                        )
                except ValueError:
                    pass  # Malformed raw metadata is the reason for many corrections.
        status = (
            "conflict"
            if conflicts
            else "applied"
            if applied
            else "matched_unused"
            if configured
            else "incomplete_coverage"
            if present
            else "unobserved"
        )
        result.append(
            {
                **asdict(record),
                "status": status,
                "applied": bool(applied),
                "legacy_unversioned": record.version is None,
                "observed_versions": sorted({r["version"] for r in present}),
                "conflicts": sorted(set(conflicts)),
                "classification": classify(record.license),
                "cleanup_review_candidate": complete
                and status in {"unobserved", "matched_unused"},
            }
        )
    if any(row["status"] == "incomplete_coverage" for row in result):
        complete = False
        problems = [
            *problems,
            "Catalog matches lack actual-use telemetry for one or more observed packages",
        ]
        for row in result:
            row["cleanup_review_candidate"] = False
    return {
        "coverage_complete": complete,
        "coverage_problems": problems,
        "automatic_deletion": False,
        "scope_note": "Nightly coverage does not prove unused across retained release lines or optional images",
        "overrides": result,
    }


def load_matrix(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    images = document["images"]
    if document["schema_version"] != 1 or not isinstance(images, list) or not images:
        raise ValueError("Expected a nonempty schema-version-1 image matrix")
    prefixes = set()
    for image in images:
        if (
            not isinstance(image["prefix"], str)
            or not image["prefix"]
            or image["prefix"] in prefixes
        ):
            raise ValueError("Matrix image prefixes must be nonempty and unique")
        prefixes.add(image["prefix"])
        arches = image["architectures"]
        if (
            not isinstance(arches, list)
            or not arches
            or len(set(arches)) != len(arches)
            or not set(arches) <= {"amd64", "arm64"}
        ):
            raise ValueError("Each matrix image needs unique supported architectures")
        if not isinstance(image["cuda"], str):
            raise ValueError("CUDA matrix identity must be a string")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = load_matrix(args.matrix)
    run, archives, problems = download_latest(args.repo, expected)
    observations, inventory = [], []
    for image in expected:
        if image["prefix"] not in archives:
            continue
        try:
            obs, inv, missing = inspect_archive(
                archives[image["prefix"]],
                image,
                run["head_sha"],
                run["override_catalog_sha256"],
            )
            observations.extend(obs)
            inventory.extend(inv)
            problems.extend(missing)
        except (BadZipFile, ValueError) as exc:
            problems.append(f"{image['prefix']}: {exc}")
    report = summarize(load_records(args.overrides), observations, inventory, problems)
    report.update(
        {
            "schema_version": 1,
            "repository": args.repo,
            "run_id": run["id"],
            "source_sha": run["head_sha"],
            "expected_matrix": expected,
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Coverage complete: {report['coverage_complete']}; {len(problems)} missing or invalid evidence item(s). No entries removed."
    )
    return 0  # Report-only; incomplete coverage is never a cleanup authorization.


if __name__ == "__main__":
    raise SystemExit(main())
