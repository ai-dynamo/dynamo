# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, Mock, call

import pytest

from dynamo.common import metadata_upload
from dynamo.common.metadata_upload import MetadataUploader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def test_url_only_settings_keep_existing_shape() -> None:
    uploader = MetadataUploader.from_settings({"url": " s3://primary/prefix "})

    assert uploader == MetadataUploader(url="s3://primary/prefix")
    assert uploader.fallback_url is None


@pytest.mark.parametrize(
    ("fallback_url", "message"),
    [
        ("", "metadata_upload.fallback_url must not be empty"),
        ("   ", "metadata_upload.fallback_url must not be empty"),
        (1, "metadata_upload.fallback_url must be a string"),
    ],
)
def test_fallback_url_must_be_a_nonempty_string(fallback_url, message) -> None:
    with pytest.raises(ValueError, match=message):
        MetadataUploader.from_settings(
            {"url": "s3://primary/prefix", "fallback_url": fallback_url}
        )


@pytest.mark.asyncio
async def test_primary_failure_uploads_same_serialized_payload_to_fallback(
    monkeypatch,
) -> None:
    payload = b"serialized-metadata"
    serialize = Mock(return_value=payload)
    upload = AsyncMock(side_effect=[OSError("primary unavailable"), "uploaded"])
    monkeypatch.setattr(metadata_upload, "_serialize_metadata", serialize)
    monkeypatch.setattr(metadata_upload, "_upload_bytes", upload)
    uploader = MetadataUploader.from_settings(
        {
            "url": " s3://primary/prefix ",
            "fallback_url": " file:///fallback/prefix ",
        }
    )

    assert uploader is not None
    await uploader.upload_choice(2, {"request_id": "abc"})

    serialize.assert_called_once_with({"request_id": "abc"})
    assert upload.await_args_list == [
        call("s3://primary/prefix", "choice_2.msgpack.zst", payload),
        call("file:///fallback/prefix", "choice_2.msgpack.zst", payload),
    ]


@pytest.mark.asyncio
async def test_successful_primary_does_not_use_fallback(monkeypatch) -> None:
    upload = AsyncMock(return_value="uploaded")
    monkeypatch.setattr(
        metadata_upload, "_serialize_metadata", Mock(return_value=b"data")
    )
    monkeypatch.setattr(metadata_upload, "_upload_bytes", upload)
    uploader = MetadataUploader(
        url="s3://primary/prefix", fallback_url="file:///fallback/prefix"
    )

    await uploader.upload_choice(0, {"request_id": "abc"})

    upload.assert_awaited_once_with(
        "s3://primary/prefix", "choice_0.msgpack.zst", b"data"
    )


@pytest.mark.asyncio
async def test_cancelled_primary_does_not_use_fallback(monkeypatch) -> None:
    upload = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(
        metadata_upload, "_serialize_metadata", Mock(return_value=b"data")
    )
    monkeypatch.setattr(metadata_upload, "_upload_bytes", upload)
    uploader = MetadataUploader(
        url="s3://primary/prefix", fallback_url="file:///fallback/prefix"
    )

    with pytest.raises(asyncio.CancelledError):
        await uploader.upload_choice(0, {"request_id": "abc"})

    upload.assert_awaited_once_with(
        "s3://primary/prefix", "choice_0.msgpack.zst", b"data"
    )
