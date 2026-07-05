#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attest immutable Docker identities for one completed Terminal-Bench job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 is used by local unit-test tooling.
    tomllib = None  # type: ignore[assignment]


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA256_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REPO_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TASK_REF_FIELDS = {"org", "name", "ref"}
TASK_FIELDS = {
    "task_name",
    "task_ref",
    "task_checksum",
    "task_toml_sha256",
    "requested_ref",
    "image_id",
    "repo_digests",
}


class TaskImageError(ValueError):
    """Raised when task-image evidence is missing or inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TaskImageError(f"expected a JSON object in {path}")
    return value


def scoped_task_refs(
    dataset_metadata: dict[str, Any], expected_tasks: int
) -> list[dict[str, str]]:
    refs = dataset_metadata.get("task_refs")
    if not isinstance(refs, list):
        raise TaskImageError("dataset metadata task_refs must be a list")
    if dataset_metadata.get("task_count") != len(refs):
        raise TaskImageError("dataset metadata task_count does not match task_refs")
    if expected_tasks < 1 or expected_tasks > len(refs):
        raise TaskImageError("expected task count is outside the pinned dataset")

    validated: list[dict[str, str]] = []
    for index, ref in enumerate(refs[:expected_tasks]):
        if not isinstance(ref, dict) or set(ref) != TASK_REF_FIELDS:
            raise TaskImageError(f"dataset task ref {index} fields are invalid")
        org = ref["org"]
        name = ref["name"]
        digest = ref["ref"]
        if (
            not isinstance(org, str)
            or COMPONENT_RE.fullmatch(org) is None
            or not isinstance(name, str)
            or COMPONENT_RE.fullmatch(name) is None
            or not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or SHA256_RE.fullmatch(digest.removeprefix("sha256:")) is None
        ):
            raise TaskImageError(f"dataset task ref {index} is invalid")
        validated.append({"org": org, "name": name, "ref": digest})
    names = [f"{ref['org']}/{ref['name']}" for ref in validated]
    if len(names) != len(set(names)):
        raise TaskImageError("scoped dataset task names are not unique")
    return validated


def collect_result_identities(
    job_dir: Path,
    refs: list[dict[str, str]],
    expected_attempts: int,
) -> list[dict[str, Any]]:
    if expected_attempts < 1:
        raise TaskImageError("expected attempts must be positive")
    expected = {f"{ref['org']}/{ref['name']}": ref for ref in refs}
    by_task: dict[str, list[dict[str, str]]] = {name: [] for name in expected}
    checksums: dict[str, set[str]] = {name: set() for name in expected}
    seen_trials: set[str] = set()
    paths = sorted(
        path for path in job_dir.glob("*/result.json") if path.parent.name != "summary"
    )
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
            raise TaskImageError(f"trial result is not a regular file: {path}")
        result = load_json(path)
        task_name = result.get("task_name")
        trial_name = result.get("trial_name")
        task_id = result.get("task_id")
        task_checksum = result.get("task_checksum")
        if not isinstance(task_name, str) or task_name not in expected:
            raise TaskImageError(f"unexpected result task identity: {task_name!r}")
        if (
            not isinstance(trial_name, str)
            or not trial_name
            or any(character in trial_name for character in "\r\n\0")
            or path.parent.name != trial_name
        ):
            raise TaskImageError(f"result trial identity/path mismatch: {path}")
        if trial_name in seen_trials:
            raise TaskImageError(f"duplicate result trial identity: {trial_name}")
        seen_trials.add(trial_name)
        if task_id != expected[task_name]:
            raise TaskImageError(
                f"result task_id differs from pinned ref for {task_name}"
            )
        if (
            not isinstance(task_checksum, str)
            or SHA256_RE.fullmatch(task_checksum) is None
        ):
            raise TaskImageError(f"result task checksum is invalid for {trial_name}")
        checksums[task_name].add(task_checksum)
        by_task[task_name].append(
            {"trial_name": trial_name, "result_sha256": file_sha256(path)}
        )

    expected_trials = len(refs) * expected_attempts
    if len(paths) != expected_trials:
        raise TaskImageError(
            f"expected {expected_trials} result files, found {len(paths)}"
        )
    identities: list[dict[str, Any]] = []
    for ref in refs:
        task_name = f"{ref['org']}/{ref['name']}"
        trials = sorted(by_task[task_name], key=lambda trial: trial["trial_name"])
        if len(trials) != expected_attempts:
            raise TaskImageError(
                f"expected {expected_attempts} results for {task_name}, "
                f"found {len(trials)}"
            )
        if len(checksums[task_name]) != 1:
            raise TaskImageError(
                f"task checksum differs across attempts for {task_name}"
            )
        identities.append(
            {
                "task_name": task_name,
                "task_ref": ref,
                "task_checksum": next(iter(checksums[task_name])),
                "trials": trials,
            }
        )
    return identities


def inspect_image(requested_ref: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", requested_ref],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TaskImageError(
            f"could not inspect Docker image {requested_ref!r}"
        ) from error
    try:
        documents = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise TaskImageError(
            f"Docker inspect returned invalid JSON for {requested_ref!r}"
        ) from error
    if (
        not isinstance(documents, list)
        or len(documents) != 1
        or not isinstance(documents[0], dict)
    ):
        raise TaskImageError(
            f"Docker inspect returned an invalid object for {requested_ref!r}"
        )
    return documents[0]


def image_identity(requested_ref: str, inspected: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(requested_ref, str)
        or not requested_ref
        or any(character.isspace() or character == "\0" for character in requested_ref)
    ):
        raise TaskImageError("task.toml environment.docker_image is invalid")
    image_id = inspected.get("Id")
    repo_digests = inspected.get("RepoDigests")
    if not isinstance(image_id, str) or SHA256_ID_RE.fullmatch(image_id) is None:
        raise TaskImageError(f"Docker image ID is invalid for {requested_ref!r}")
    if (
        not isinstance(repo_digests, list)
        or not repo_digests
        or not all(
            isinstance(digest, str) and REPO_DIGEST_RE.fullmatch(digest) is not None
            for digest in repo_digests
        )
    ):
        raise TaskImageError(
            f"Docker image has no valid RepoDigests for {requested_ref!r}"
        )
    canonical_digests = sorted(set(repo_digests))
    if len(canonical_digests) != len(repo_digests):
        raise TaskImageError(
            f"Docker image RepoDigests contain duplicates for {requested_ref!r}"
        )
    return {
        "requested_ref": requested_ref,
        "image_id": image_id,
        "repo_digests": canonical_digests,
    }


def build_evidence(
    job_dir: Path,
    dataset_metadata: dict[str, Any],
    package_cache_dir: Path,
    expected_tasks: int,
    expected_attempts: int,
    *,
    inspector: Callable[[str], dict[str, Any]] = inspect_image,
    toml_loader: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if toml_loader is None:
        if tomllib is None:
            raise TaskImageError(
                "capture requires Python 3.11+ (use the pinned Harbor interpreter)"
            )
        toml_loader = tomllib.loads
    refs = scoped_task_refs(dataset_metadata, expected_tasks)
    result_identities = collect_result_identities(job_dir, refs, expected_attempts)
    tasks: list[dict[str, Any]] = []
    for result_identity in result_identities:
        ref = result_identity["task_ref"]
        task_dir = (
            package_cache_dir
            / ref["org"]
            / ref["name"]
            / ref["ref"].removeprefix("sha256:")
        )
        task_toml = task_dir / "task.toml"
        if not task_toml.is_file() or task_toml.is_symlink():
            raise TaskImageError(
                f"content-addressed task.toml is missing for {result_identity['task_name']}"
            )
        try:
            document = toml_loader(task_toml.read_text())
        except Exception as error:
            raise TaskImageError(
                "could not parse content-addressed task.toml for "
                f"{result_identity['task_name']}"
            ) from error
        if not isinstance(document, dict):
            raise TaskImageError(
                f"content-addressed task.toml is invalid for {result_identity['task_name']}"
            )
        environment = document.get("environment")
        requested_ref = (
            environment.get("docker_image") if isinstance(environment, dict) else None
        )
        if not isinstance(requested_ref, str):
            raise TaskImageError(
                f"task.toml has no environment.docker_image for {result_identity['task_name']}"
            )
        tasks.append(
            {
                **{
                    field: result_identity[field]
                    for field in ("task_name", "task_ref", "task_checksum")
                },
                "task_toml_sha256": file_sha256(task_toml),
                **image_identity(requested_ref, inspector(requested_ref)),
            }
        )
    return {
        "schema_version": 1,
        "task_count": len(tasks),
        "trial_count": len(refs) * expected_attempts,
        "tasks": tasks,
    }


def validate_evidence(
    evidence: Any,
    dataset_metadata: dict[str, Any],
    job_dir: Path,
    expected_tasks: int,
    expected_attempts: int,
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema_version",
        "task_count",
        "trial_count",
        "tasks",
    }:
        raise TaskImageError("task-image evidence fields are invalid")
    if isinstance(evidence["schema_version"], bool) or evidence["schema_version"] != 1:
        raise TaskImageError("task-image evidence schema_version must be 1")
    if evidence["task_count"] != expected_tasks or isinstance(
        evidence["task_count"], bool
    ):
        raise TaskImageError("task-image evidence task_count is invalid")
    expected_trial_count = expected_tasks * expected_attempts
    if evidence["trial_count"] != expected_trial_count or isinstance(
        evidence["trial_count"], bool
    ):
        raise TaskImageError("task-image evidence trial_count is invalid")

    refs = scoped_task_refs(dataset_metadata, expected_tasks)
    expected_identities = collect_result_identities(job_dir, refs, expected_attempts)
    tasks = evidence["tasks"]
    if not isinstance(tasks, list) or len(tasks) != expected_tasks:
        raise TaskImageError("task-image evidence tasks are invalid")
    for index, (task, expected_identity) in enumerate(zip(tasks, expected_identities)):
        if not isinstance(task, dict) or set(task) != TASK_FIELDS:
            raise TaskImageError(f"task-image evidence task {index} fields are invalid")
        for field in ("task_name", "task_ref", "task_checksum"):
            if task[field] != expected_identity[field]:
                raise TaskImageError(
                    f"task-image evidence {field} differs from results at task {index}"
                )
        if (
            not isinstance(task["task_toml_sha256"], str)
            or SHA256_RE.fullmatch(task["task_toml_sha256"]) is None
        ):
            raise TaskImageError(
                f"task-image evidence task_toml_sha256 is invalid at task {index}"
            )
        canonical_image = image_identity(
            task["requested_ref"],
            {"Id": task["image_id"], "RepoDigests": task["repo_digests"]},
        )
        if canonical_image != {
            field: task[field]
            for field in ("requested_ref", "image_id", "repo_digests")
        }:
            raise TaskImageError(
                f"task-image evidence image identity is not canonical at task {index}"
            )
    return evidence


def write_exclusive_or_verify(path: Path, evidence: dict[str, Any]) -> None:
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise TaskImageError(f"refusing existing non-regular evidence path: {path}")
        if path.read_text() != payload:
            raise TaskImageError(
                f"refusing to overwrite different task-image evidence: {path}"
            )
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--job-dir", type=Path, required=True)
    result.add_argument("--dataset-metadata", type=Path, required=True)
    result.add_argument("--package-cache-dir", type=Path, required=True)
    result.add_argument("--expected-tasks", type=int, required=True)
    result.add_argument("--expected-attempts", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        evidence = build_evidence(
            args.job_dir.resolve(),
            load_json(args.dataset_metadata.resolve()),
            args.package_cache_dir.resolve(),
            args.expected_tasks,
            args.expected_attempts,
        )
        validate_evidence(
            evidence,
            load_json(args.dataset_metadata.resolve()),
            args.job_dir.resolve(),
            args.expected_tasks,
            args.expected_attempts,
        )
        write_exclusive_or_verify(args.output.resolve(), evidence)
    except (
        OSError,
        json.JSONDecodeError,
        TaskImageError,
    ) as error:
        print(f"task image attestation error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "task_count": evidence["task_count"],
                "trial_count": evidence["trial_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
