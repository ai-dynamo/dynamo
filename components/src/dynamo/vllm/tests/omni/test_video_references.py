# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from dynamo.common.http.url_validator import UrlValidationPolicy
    from dynamo.common.protocols.video_protocol import NvCreateVideoRequest
    from dynamo.vllm.omni.video_references import VideoReferenceMaterializer
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_materializes_typed_data_references_in_per_type_order():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[
            {"type": "image", "source": "data:image/png;base64,aW1hZ2U="},
            {"type": "audio", "source": "data:audio/mpeg;base64,YXVkaW8="},
            {"type": "image", "source": "data:image/jpeg;base64,aW1hZ2Uy"},
        ],
    )

    materialized = await VideoReferenceMaterializer().materialize(request)
    assert materialized is not None
    root = Path(materialized.temporary_directory.name)
    try:
        assert [
            Path(path).suffix for path in materialized.multi_modal_data["image"]
        ] == [
            ".png",
            ".jpg",
        ]
        assert Path(materialized.multi_modal_data["audio"][0]).suffix == ".mp3"
        assert [
            Path(path).read_bytes() for path in materialized.multi_modal_data["image"]
        ] == [
            b"image",
            b"image2",
        ]
    finally:
        materialized.cleanup()
    assert not root.exists()


@pytest.mark.asyncio
async def test_generic_omni_data_scalarizes_singletons():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[
            {"type": "image", "source": "data:image/png;base64,aW1hZ2U="},
            {"type": "audio", "source": "data:audio/wav;base64,YXVkaW8="},
        ],
    )

    materialized = await VideoReferenceMaterializer().materialize(request)
    assert materialized is not None
    try:
        omni_data = materialized.as_omni_data()
        assert isinstance(omni_data["image"], str)
        assert isinstance(omni_data["audio"], str)
    finally:
        materialized.cleanup()


@pytest.mark.asyncio
async def test_generic_omni_data_rejects_unsupported_multiplicity():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[
            {"type": "image", "source": "data:image/png;base64,aW1hZ2U="},
            {"type": "image", "source": "data:image/png;base64,aW1hZ2Uy"},
        ],
    )

    materialized = await VideoReferenceMaterializer().materialize(request)
    assert materialized is not None
    try:
        with pytest.raises(ValueError, match="at most one image"):
            materialized.as_omni_data()
        assert len(materialized.as_omni_data(allow_multiple=True)["image"]) == 2
    finally:
        materialized.cleanup()


@pytest.mark.asyncio
async def test_rejects_cumulative_reference_size(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.omni.video_references._MAX_TOTAL_REFERENCE_BYTES", 5
    )
    request = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[
            {"type": "image", "source": "data:image/png;base64,YWFh"},
            {"type": "audio", "source": "data:audio/wav;base64,YmJi"},
        ],
    )

    with pytest.raises(ValueError, match="total limit"):
        await VideoReferenceMaterializer().materialize(request)


@pytest.mark.asyncio
async def test_rejects_more_than_twelve_references_before_materialization():
    request = SimpleNamespace(input_references=[None] * 13)

    with pytest.raises(ValueError, match="at most 12"):
        await VideoReferenceMaterializer().materialize(request)


@pytest.mark.asyncio
async def test_uses_validated_local_path_without_copy(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    request = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[{"type": "video", "source": str(source)}],
    )
    materializer = VideoReferenceMaterializer(
        url_policy=UrlValidationPolicy(allowed_local_path=str(tmp_path))
    )

    materialized = await materializer.materialize(request)
    assert materialized is not None
    try:
        assert materialized.multi_modal_data == {"video": [str(source.resolve())]}
    finally:
        materialized.cleanup()
    assert source.exists()


def test_unknown_remote_suffix_uses_media_type_default():
    reference = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[
            {"type": "video", "source": "https://example.com/download.bin?format=mp4"}
        ],
    ).input_references[0]

    assert VideoReferenceMaterializer._suffix(reference) == ".mp4"
