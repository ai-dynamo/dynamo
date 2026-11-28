#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish approved HF snapshots and hydrate verified, offline pytest caches."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
PR_BUCKET = "dynamo-ci-210086341041-us-west-2-pr-models"
APPROVED_BUCKET = "dynamo-ci-210086341041-us-west-2-approved-test-models"
BUCKETS = {PR_BUCKET, APPROVED_BUCKET}
PREFIX = "approved_test_models/v1"
MAX_INDEX_BYTES = 16 * 1024 * 1024


def inventory(path):
    """An immutable inventory determines its own S3 namespace."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
        raise ValueError("Unsupported approved-model inventory")
    seen = set()
    for model in data["models"]:
        name = model.get("id", "")
        if not MODEL_ID.fullmatch(name) or name in seen:
            raise ValueError("Invalid or duplicate approved model ID")
        if not SHA.fullmatch(model.get("revision", "")):
            raise ValueError(f"Model revision must be a full immutable commit: {name}")
        if not isinstance(model.get("ignore_patterns", []), list) or not all(
            isinstance(pattern, str) for pattern in model.get("ignore_patterns", [])
        ):
            raise ValueError("Model exclusions must be a list of glob strings")
        seen.add(name)
    if not seen:
        raise ValueError("The approved-model inventory is empty")
    return data["models"], hashlib.sha256(raw).hexdigest()


def relative_path(value):
    """Reject S3/index paths capable of escaping a snapshot directory."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\0" in value
        or str(path) != value
    ):
        raise ValueError("Invalid model file path")
    return path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_prefix(inventory_digest, model):
    return f"{PREFIX}/{inventory_digest}/{model['id']}/{model['revision']}"


def select(models, requested):
    if not requested:
        return models
    names = set(requested)
    available = {model["id"] for model in models}
    if not names <= available:
        raise ValueError(f"Models are not approved: {sorted(names - available)}")
    return [model for model in models if model["id"] in names]


def publish(s3, models, inventory_digest, buckets, download, work_dir):
    """Publish each model to BOTH stores; the completion index is written last."""
    if set(buckets) != BUCKETS or len(buckets) != 2:
        raise ValueError("Publication must hydrate both approved model stores")
    for model in models:
        with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
            root = Path(temporary)
            snapshot = Path(
                download(
                    model["id"],
                    revision=model["revision"],
                    cache_dir=str(root),
                    ignore_patterns=model.get("ignore_patterns", []),
                )
            )
            snapshot.resolve().relative_to(root.resolve())
            entries = []
            for path in sorted(snapshot.rglob("*")):
                if not path.is_file():
                    continue
                # HF stores snapshot symlinks into its sibling blobs directory.
                # Follow only links whose target stays in this isolated download.
                path.resolve().relative_to(root.resolve())
                name = path.relative_to(snapshot).as_posix()
                relative_path(name)
                entries.append(
                    {
                        "path": name,
                        "size": path.stat().st_size,
                        "sha256": file_digest(path),
                    }
                )
            if not entries:
                raise ValueError(f"Empty HF snapshot: {model['id']}")
            index = {
                "schema_version": 1,
                "inventory_sha256": inventory_digest,
                "model": model,
                "files": entries,
            }
            payload = json.dumps(index, sort_keys=True).encode()
            prefix = model_prefix(inventory_digest, model)
            for bucket in buckets:
                for entry in entries:
                    s3.upload_file(
                        str(snapshot / entry["path"]),
                        bucket,
                        f"{prefix}/files/{entry['path']}",
                        ExtraArgs={"ServerSideEncryption": "AES256"},
                    )
                s3.put_object(
                    Bucket=bucket,
                    Key=f"{prefix}/index.json",
                    Body=payload,
                    ContentType="application/json",
                    ServerSideEncryption="AES256",
                )


def hydrate(s3, models, inventory_digest, bucket, destination):
    """Install regular snapshot files and HF refs, checking size and SHA-256."""
    if bucket not in BUCKETS:
        raise ValueError("Hydration requires an approved model store")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for model in models:
        prefix = model_prefix(inventory_digest, model)
        response = s3.get_object(Bucket=bucket, Key=f"{prefix}/index.json")
        body = response["Body"]
        try:
            raw = body.read(MAX_INDEX_BYTES + 1)
        finally:
            body.close()
        if len(raw) > MAX_INDEX_BYTES:
            raise ValueError("Model completion index exceeds the size limit")
        index = json.loads(raw)
        if (
            index.get("schema_version") != 1
            or index.get("inventory_sha256") != inventory_digest
            or index.get("model") != model
            or not index.get("files")
        ):
            raise ValueError(
                "Model completion index does not match the approved inventory"
            )
        cache = destination / "hub" / ("models--" + model["id"].replace("/", "--"))
        snapshot = cache / "snapshots" / model["revision"]
        seen = set()
        for entry in index["files"]:
            name = entry.get("path", "")
            relative_path(name)
            if name in seen or not DIGEST.fullmatch(entry.get("sha256", "")):
                raise ValueError("Invalid or duplicate model file entry")
            if type(entry.get("size")) is not int or entry["size"] < 0:
                raise ValueError("Invalid model file size")
            seen.add(name)
            path = snapshot / name
            path.resolve().relative_to(destination)
            if (
                path.is_file()
                and path.stat().st_size == entry["size"]
                and file_digest(path) == entry["sha256"]
            ):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                s3.download_file(bucket, f"{prefix}/files/{name}", str(temporary_path))
                if (
                    temporary_path.stat().st_size != entry["size"]
                    or file_digest(temporary_path) != entry["sha256"]
                ):
                    raise ValueError(
                        f"Model cache integrity check failed: {model['id']}/{name}"
                    )
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
        refs = cache / "refs"
        refs.resolve().relative_to(destination)
        refs.mkdir(parents=True, exist_ok=True)
        # Existing tests load the model ID at its default ref; weights remain
        # pinned by this local ref even when upstream main subsequently moves.
        ref = refs / "main"
        ref.resolve().relative_to(destination)
        ref.write_text(model["revision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "publish", "hydrate"])
    parser.add_argument("--manifest", default=".github/approved-test-models.json")
    parser.add_argument("--bucket", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--destination", default="/models")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--region", default="us-west-2")
    args = parser.parse_args()
    models, digest = inventory(args.manifest)
    models = select(models, args.model)
    if args.command == "check":
        print(f"Validated {len(models)} approved models; inventory {digest}")
        return
    # Keep dependencies optional: local pytest does not import this tool or
    # instantiate an AWS client unless cache hydration is explicitly requested.
    import boto3

    s3 = boto3.client("s3", region_name=args.region)
    if args.command == "publish":
        from huggingface_hub import snapshot_download

        def download(repo, **kwargs):
            return snapshot_download(
                repo, token=os.environ.get("HF_TOKEN") or False, **kwargs
            )

        publish(s3, models, digest, args.bucket, download, args.work_dir)
    else:
        if len(args.bucket) != 1:
            parser.error("hydrate requires exactly one --bucket")
        hydrate(s3, models, digest, args.bucket[0], args.destination)


if __name__ == "__main__":
    main()
