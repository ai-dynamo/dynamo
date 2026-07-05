#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture immutable Docker content identities for SWE task images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from manage_scope import load_dataset_ids, load_scope


SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
REPO_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_rows(dataset: Path, expected: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in dataset.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["instance_id"]] = row
    if len(rows) != expected:
        raise ValueError(f"expected {expected} unique dataset rows, got {len(rows)}")
    return rows


def task_image_ref(instance_id: str, row: dict[str, Any]) -> str:
    if image_name := row.get("image_name"):
        if not isinstance(image_name, str):
            raise ValueError(f"invalid image_name for {instance_id}")
        return image_name
    docker_id = instance_id.replace("__", "_1776_").lower()
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest"


def inspect_image(reference: str, docker: str = "docker") -> dict[str, Any]:
    output = subprocess.run(
        [docker, "image", "inspect", reference],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    values = json.loads(output)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError(f"unexpected docker image inspect output for {reference}")
    return image_identity(reference, values[0])


def image_identity(reference: str, attributes: dict[str, Any]) -> dict[str, Any]:
    image_id = attributes.get("Id")
    repo_digests = sorted(set(attributes.get("RepoDigests") or []))
    if not isinstance(image_id, str) or not SHA256_ID.fullmatch(image_id):
        raise ValueError(f"image {reference} has no immutable sha256 image ID")
    if not repo_digests or not all(
        isinstance(digest, str) and REPO_DIGEST.fullmatch(digest)
        for digest in repo_digests
    ):
        raise ValueError(f"image {reference} has no valid immutable RepoDigest")
    content = {"image_id": image_id, "repo_digests": repo_digests}
    return {
        "requested_ref": reference,
        **content,
        "content_identity_sha256": canonical_sha256(content),
    }


def load_image_map(path: Path) -> dict[str, dict[str, Any]]:
    evidence = json.loads(path.read_text())
    images = evidence.get("images") if isinstance(evidence, dict) else None
    if not isinstance(images, dict) or not images:
        raise ValueError("task image evidence has no images map")
    return images


def normalize_registry_reference(reference: str) -> str:
    """Normalize Docker Hub's equivalent qualified and unqualified names."""
    for prefix in ("docker.io/", "index.docker.io/"):
        if reference.startswith(prefix):
            return reference[len(prefix) :]
    return reference


def install_docker_image_guard(docker_module: Any, evidence_path: Path) -> None:
    """Reject any evaluator image whose pulled/local content differs from generation."""
    expected_by_ref = {
        normalize_registry_reference(identity["requested_ref"]): identity
        for identity in load_image_map(evidence_path).values()
    }
    image_collection = docker_module.models.images.ImageCollection
    original_get = image_collection.get
    original_pull = image_collection.pull

    def validate(reference: str, image: Any) -> Any:
        normalized_reference = normalize_registry_reference(reference)
        expected = expected_by_ref.get(normalized_reference)
        is_task_image = normalized_reference.startswith(
            "swebench/sweb.eval."
        ) or normalized_reference.startswith("jefzda/sweap-images:")
        if is_task_image and expected is None:
            raise RuntimeError(
                f"evaluator requested task image absent from generation map: {reference}"
            )
        if expected is not None:
            actual = image_identity(expected["requested_ref"], image.attrs)
            if actual != expected:
                raise RuntimeError(
                    f"evaluator task image identity differs from generation: {reference}"
                )
        return image

    def guarded_get(collection: Any, name: str) -> Any:
        return validate(name, original_get(collection, name))

    def guarded_pull(
        collection: Any, repository: str, *args: Any, **kwargs: Any
    ) -> Any:
        image = original_pull(collection, repository, *args, **kwargs)
        tag = kwargs.get("tag")
        if tag is None and args:
            tag = args[0]
        reference = f"{repository}:{tag}" if tag else repository
        if isinstance(image, list):
            return [validate(reference, item) for item in image]
        return validate(reference, image)

    image_collection.get = guarded_get
    image_collection.pull = guarded_pull


def load_or_initialize(
    output: Path, suite: str, dataset: Path, scope: Path
) -> dict[str, Any]:
    expected_header = {
        "schema_version": 1,
        "suite": suite,
        "dataset_sha256": file_sha256(dataset),
        "scope_sha256": file_sha256(scope),
    }
    if not output.exists():
        return {**expected_header, "images": {}}
    evidence = json.loads(output.read_text())
    if not isinstance(evidence, dict) or not isinstance(evidence.get("images"), dict):
        raise ValueError("task image evidence must be an object with an images map")
    actual_header = {key: evidence.get(key) for key in expected_header}
    if actual_header != expected_header:
        raise ValueError("task image evidence dataset/scope identity drifted")
    return evidence


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload)
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def capture(args: argparse.Namespace) -> None:
    dataset_ids = load_dataset_ids(args.dataset, args.expected)
    rows = load_rows(args.dataset, args.expected)
    scope = load_scope(args.scope, args.dataset, args.expected)
    pattern = re.compile(args.batch_filter)
    selected = [item for item in scope["target_ids"] if pattern.match(item)]
    if not selected:
        raise ValueError("batch filter selects no scoped instances")
    evidence = load_or_initialize(args.output, args.suite, args.dataset, args.scope)
    images = evidence["images"]
    for instance_id in selected:
        if instance_id not in dataset_ids:
            raise ValueError(f"scoped instance is absent from dataset: {instance_id}")
        reference = task_image_ref(instance_id, rows[instance_id])
        try:
            current = inspect_image(reference, args.docker)
        except subprocess.CalledProcessError:
            if instance_id in images:
                continue
            raise ValueError(
                f"task image disappeared before identity capture: {reference}"
            ) from None
        if instance_id in images and images[instance_id] != current:
            raise ValueError(f"task image identity changed for {instance_id}")
        images[instance_id] = current
    atomic_write(args.output, evidence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", required=True, choices=("verified", "multilingual", "pro")
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--batch-filter", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docker", default="docker")
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
