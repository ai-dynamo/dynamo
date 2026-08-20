# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

try:
    from dynamo.common.http.url_validator import UrlValidationPolicy
    from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
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
async def test_materializes_typed_data_references_in_order():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="MiniMaxAI/MiniMax-H3",
        input_references=[
            {"type": "image", "source": "data:image/png;base64,aW1hZ2U="},
            {"type": "audio", "source": "data:audio/mpeg;base64,YXVkaW8="},
            {"type": "image", "source": "data:image/jpeg;base64,aW1hZ2Uy"},
        ],
        nvext=VideoNvExt(task="ref2va"),
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
async def test_uses_validated_local_path_without_copy(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    request = NvCreateVideoRequest(
        prompt="cat",
        model="MiniMaxAI/MiniMax-H3",
        input_references=[{"type": "video", "source": str(source)}],
        nvext=VideoNvExt(task="ref2va"),
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
        model="MiniMaxAI/MiniMax-H3",
        input_references=[
            {"type": "video", "source": "https://example.com/download.bin?format=mp4"}
        ],
        nvext=VideoNvExt(task="ref2va"),
    ).input_references[0]

    assert VideoReferenceMaterializer._suffix(reference) == ".mp4"


@pytest.mark.parametrize(
    ("task", "references", "message"),
    [
        (
            "t2va",
            [{"type": "image", "source": "data:image/png;base64,AA=="}],
            "does not accept",
        ),
        (
            "fl2va",
            [{"type": "audio", "source": "data:audio/wav;base64,AA=="}],
            "only one or two image",
        ),
        (
            "ref2va",
            [{"type": "audio", "source": "data:audio/wav;base64,AA=="}],
            "at least one image or video",
        ),
    ],
)
@pytest.mark.asyncio
async def test_rejects_invalid_h3_reference_contract(task, references, message):
    request = NvCreateVideoRequest(
        prompt="cat",
        model="MiniMaxAI/MiniMax-H3",
        input_references=references,
        nvext=VideoNvExt(task=task),
    )
    with pytest.raises(ValueError, match=message):
        await VideoReferenceMaterializer().materialize(request)
