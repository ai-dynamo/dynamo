# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for out-of-band metadata uploads."""

import msgspec
import pytest
import zstandard as zstd

from dynamo.common.metadata_upload import MetadataUploader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _read_zstd_payload(path):
    raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return msgspec.msgpack.decode(raw)


def test_metadata_uploader_parses_backend_request():
    uploader = MetadataUploader.from_backend_request(
        {
            "extra_args": {
                "nvext": {
                    "metadata_upload": {
                        "url": "s3://bucket/root/rollouts",
                        "fallback_url": " file:///var/tmp/rollouts ",
                    }
                }
            }
        }
    )

    assert uploader is not None
    assert uploader.url == "s3://bucket/root/rollouts"
    assert uploader.fallback_url == "file:///var/tmp/rollouts"

    uploader = MetadataUploader.from_backend_request(
        {"nvext": {"metadata_upload": {"url": "s3://bucket/root/rollouts"}}}
    )
    assert uploader is not None
    assert uploader.fallback_url is None


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({}, "metadata_upload.url is required"),
        ({"url": ""}, "metadata_upload.url must not be empty"),
        ({"url": 1}, "metadata_upload.url must be a string"),
        (
            {"url": "s3://bucket/root/rollouts", "fallback_url": ""},
            "metadata_upload.fallback_url must not be empty",
        ),
        (
            {"url": "s3://bucket/root/rollouts", "fallback_url": 1},
            "metadata_upload.fallback_url must be a string",
        ),
        ("invalid", "metadata_upload must be an object"),
        (
            {"url": "s3://bucket/root/rollouts", "format": "json"},
            "metadata_upload.format is not supported",
        ),
    ],
)
def test_metadata_uploader_rejects_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        MetadataUploader.from_backend_request({"nvext": {"metadata_upload": settings}})


@pytest.mark.asyncio
async def test_metadata_upload_normalizes_tensor_values(tmp_path):
    torch = pytest.importorskip("torch")
    uploader = MetadataUploader(url=(tmp_path / "metadata/tensors").as_uri())
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    await uploader.upload_choice(0, {"router_tensor": tensor})

    payload = _read_zstd_payload(tmp_path / "metadata/tensors/choice_0.msgpack.zst")
    uploaded_tensor = payload["metadata"]["router_tensor"]
    assert uploaded_tensor["type"] == "tensor"
    assert uploaded_tensor["dtype"] == "int32"
    assert uploaded_tensor["shape"] == [2, 2]
    assert uploaded_tensor["data"] == tensor.numpy().tobytes()

    bfloat_tensor = torch.tensor([1, 2], dtype=torch.bfloat16)
    await uploader.upload_choice(1, {"router_tensor": bfloat_tensor})

    payload = _read_zstd_payload(tmp_path / "metadata/tensors/choice_1.msgpack.zst")
    uploaded_tensor = payload["metadata"]["router_tensor"]
    assert uploaded_tensor["dtype"] == "bfloat16"
    assert uploaded_tensor["shape"] == [2]
    assert uploaded_tensor["data"] == bfloat_tensor.view(torch.uint8).numpy().tobytes()


@pytest.mark.asyncio
async def test_metadata_upload_normalizes_numpy_values(tmp_path):
    np = pytest.importorskip("numpy")
    uploader = MetadataUploader(url=(tmp_path / "metadata/arrays").as_uri())
    array = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    await uploader.upload_choice(0, {"scores": array[:, ::-1]})

    payload = _read_zstd_payload(tmp_path / "metadata/arrays/choice_0.msgpack.zst")
    uploaded_array = payload["metadata"]["scores"]
    expected = np.ascontiguousarray(array[:, ::-1])
    assert uploaded_array["type"] == "ndarray"
    assert uploaded_array["dtype"] == "float32"
    assert uploaded_array["shape"] == [2, 2]
    assert uploaded_array["data"] == expected.tobytes()


@pytest.mark.asyncio
async def test_metadata_upload_does_not_write_fallback_when_primary_succeeds(tmp_path):
    primary_path = tmp_path / "metadata/primary"
    fallback_path = tmp_path / "metadata/fallback"
    uploader = MetadataUploader(
        url=primary_path.as_uri(),
        fallback_url=fallback_path.as_uri(),
    )

    await uploader.upload_choice(0, {"id": "request-1"})

    assert (primary_path / "choice_0.msgpack.zst").is_file()
    assert not fallback_path.exists()


@pytest.mark.asyncio
async def test_metadata_upload_uses_fallback_after_primary_failure(tmp_path, caplog):
    primary_path = tmp_path / "primary-is-a-file"
    primary_path.write_bytes(b"primary sentinel")
    fallback_path = tmp_path / "metadata/fallback"
    uploader = MetadataUploader(
        url=primary_path.as_uri(),
        fallback_url=fallback_path.as_uri(),
    )

    with caplog.at_level("WARNING"):
        await uploader.upload_choice(2, {"id": "request-2"})

    assert primary_path.read_bytes() == b"primary sentinel"
    assert "Primary metadata upload failed" in caplog.text
    payload = _read_zstd_payload(fallback_path / "choice_2.msgpack.zst")
    assert payload["metadata"] == {"id": "request-2"}


@pytest.mark.asyncio
async def test_metadata_upload_propagates_fallback_failure(tmp_path):
    primary_path = tmp_path / "primary-is-a-file"
    fallback_path = tmp_path / "fallback-is-a-file"
    primary_path.write_bytes(b"primary sentinel")
    fallback_path.write_bytes(b"fallback sentinel")
    uploader = MetadataUploader(
        url=primary_path.as_uri(),
        fallback_url=fallback_path.as_uri(),
    )

    with pytest.raises(OSError) as exc_info:
        await uploader.upload_choice(0, {"id": "request-3"})

    assert str(fallback_path) in str(exc_info.value)
    assert primary_path.read_bytes() == b"primary sentinel"
    assert fallback_path.read_bytes() == b"fallback sentinel"
