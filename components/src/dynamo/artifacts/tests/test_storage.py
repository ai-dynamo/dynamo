# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from dynamo.artifacts.storage import (
    ArtifactStorageError,
    ManagedFsspecTarget,
    PresignedHttpPutTarget,
    put_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.asyncio
async def test_managed_fsspec_writes_exact_profile_object(monkeypatch) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"training":{"url":"memory://artifacts/run"}}',
    )
    payload = b"artifact-bytes"

    receipt = await put_artifact(
        payload,
        ManagedFsspecTarget(profile="training", object_key="request-1/output.dynexp"),
    )

    import fsspec

    with fsspec.open("memory://artifacts/run/request-1/output.dynexp", "rb") as stored:
        assert stored.read() == payload
    assert receipt.actual_bytes == len(payload)
    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
    assert receipt.object_id == "training:request-1/output.dynexp"


@pytest.mark.asyncio
@pytest.mark.parametrize("object_key", ["", "/absolute", "../escape", "a/../b", "a\\b"])
async def test_managed_fsspec_rejects_unsafe_object_keys(monkeypatch, object_key: str) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"training":{"url":"memory://artifacts/run"}}',
    )
    with pytest.raises(ArtifactStorageError, match="object_key"):
        await put_artifact(
            b"data", ManagedFsspecTarget(profile="training", object_key=object_key)
        )


@pytest.mark.asyncio
async def test_presigned_put_preserves_exact_url_and_hides_capability() -> None:
    url = "https://storage.example/object%2Fname?X-Signature=secret&part=1&part=2"
    fs = MagicMock()
    payload = b"artifact"
    target = PresignedHttpPutTarget(
        url=url,
        max_bytes=1024,
        required_headers={"content-type": "application/octet-stream"},
        object_id="opaque-1",
    )

    with patch("dynamo.artifacts.storage.HTTPFileSystem", return_value=fs):
        receipt = await put_artifact(payload, target)

    fs.pipe_file.assert_called_once_with(
        url,
        payload,
        method="put",
        headers={
            "content-type": "application/octet-stream",
            "content-length": str(len(payload)),
        },
        allow_redirects=False,
    )
    assert "secret" not in repr(target)
    assert receipt.object_id == "opaque-1"


@pytest.mark.asyncio
async def test_presigned_put_rejects_oversize_before_network() -> None:
    fs = MagicMock()
    target = PresignedHttpPutTarget(
        url="https://storage.example/object?signature=secret",
        max_bytes=3,
        object_id="opaque-1",
    )
    with patch("dynamo.artifacts.storage.HTTPFileSystem", return_value=fs):
        with pytest.raises(ArtifactStorageError, match="max_bytes"):
            await put_artifact(b"four", target)
    fs.pipe_file.assert_not_called()


def test_presigned_put_rejects_insecure_url_and_unapproved_headers() -> None:
    with pytest.raises(ArtifactStorageError, match="HTTPS"):
        PresignedHttpPutTarget(
            url="http://storage.example/object", max_bytes=1024, object_id="opaque"
        )
    with pytest.raises(ArtifactStorageError, match="header"):
        PresignedHttpPutTarget(
            url="https://storage.example/object",
            max_bytes=1024,
            required_headers={"authorization": "secret"},
            object_id="opaque",
        )
