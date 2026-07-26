# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot object store.

The round-trip test runs against a live S3 endpoint and is skipped
unless GMS_SNAPSHOT_S3_BUCKET (and credentials/endpoint) are configured. The
remaining tests cover the pure key-derivation and path-safety helpers.
"""

from __future__ import annotations

import os
import uuid

import pytest
from gpu_memory_service.snapshot.objectstore import (
    MANIFEST_FILENAME,
    S3ArtifactStore,
    make_key_prefix,
    store_from_env,
)

# Pre-merge unit tests; no GPU and no framework-specific runtime required.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_make_key_prefix():
    assert (
        make_key_prefix("team-a", "sha256:abc", "3") == "team-a/sha256:abc/versions/3"
    )
    assert make_key_prefix("", "sha256:abc") == "sha256:abc/versions/1"
    assert make_key_prefix("/p/", "/id/", "  ") == "p/id/versions/1"


def test_normalize_prefix():
    assert S3ArtifactStore._normalize_prefix(" /a/b/ ") == "a/b"
    assert S3ArtifactStore._normalize_prefix("") == ""
    assert S3ArtifactStore._normalize_prefix("/") == ""


def test_safe_dest_accepts_nested():
    dest = S3ArtifactStore._safe_dest("/tmp/restore", "shards/shard_0.bin")
    assert dest == os.path.abspath("/tmp/restore/shards/shard_0.bin")


@pytest.mark.parametrize(
    "rel", ["../escape", "shards/../../escape", "../../etc/passwd"]
)
def test_safe_dest_rejects_traversal(rel):
    with pytest.raises(ValueError):
        S3ArtifactStore._safe_dest("/tmp/restore", rel)


def test_store_from_env_returns_none_without_bucket(monkeypatch):
    monkeypatch.delenv("GMS_SNAPSHOT_S3_BUCKET", raising=False)
    assert store_from_env() is None


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    not os.environ.get("GMS_SNAPSHOT_S3_BUCKET"),
    reason="set GMS_SNAPSHOT_S3_BUCKET and credentials to run the live round-trip",
)
def test_s3_round_trip(tmp_path):
    store = store_from_env()
    assert store is not None

    src = tmp_path / "snapshot"
    (src / "shards").mkdir(parents=True)
    (src / "shards" / "shard_0.bin").write_bytes(b"weights")
    (src / "gms_metadata.json").write_text("{}")
    (src / MANIFEST_FILENAME).write_text('{"version": 1}')

    # Unique per run so concurrent/repeat runs against a shared bucket never collide.
    prefix = f"dynamo-test/gms/{uuid.uuid4().hex}"
    try:
        store.upload_tree(str(src), prefix)
        assert store.exists(prefix) is True

        dst = tmp_path / "restore"
        store.download_tree(prefix, str(dst))
        assert (dst / "shards" / "shard_0.bin").read_bytes() == b"weights"
        assert (dst / MANIFEST_FILENAME).read_text() == '{"version": 1}'
    finally:
        store.remove(prefix)
    assert store.exists(prefix) is False
