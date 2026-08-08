# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""process_openai_request must let client-error types from image loading
propagate (so the frontend returns a 4xx) instead of swallowing them to None."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

# multimodal_processor imports tensorrt_llm, whose import needs CUDA. -m selection
# happens after collection, so this module is still imported on the CPU-only step;
# skip at collection there to avoid an ImportError. GPU runs it via the gpu_1 marker.
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )

from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError
from dynamo.trtllm import multimodal_processor as mmp
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        UrlValidationError("blocked IP literal"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.png"),
    ],
)
async def test_client_errors_propagate(error) -> None:
    # Mock tokenizer skips tokenizer_factory; mock loader forces the failure.
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    processor.image_loader.load_image_batch = AsyncMock(side_effect=error)

    request = {
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/x.png"}]}
    }
    with pytest.raises(type(error)):
        await processor.process_openai_request(
            request, embeddings=None, ep_disaggregated_params=None
        )


@pytest.mark.asyncio
async def test_internal_video_uses_dynamo_fetcher_when_allowed(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    fetch = AsyncMock(return_value=b"video bytes")
    load_video = AsyncMock(return_value=object())
    monkeypatch.setattr(mmp, "fetch_bytes", fetch, raising=False)
    monkeypatch.setattr(mmp, "async_load_video", load_video)
    url = "http://169.254.169.254/latest/meta-data/"

    await processor.process_openai_request(
        {
            "multi_modal_data": {"video_url": [{"Url": url}]},
            "token_ids": [1],
        },
        embeddings=None,
        ep_disaggregated_params=None,
    )

    fetch.assert_awaited_once_with(url, 30.0, policy=processor._url_policy)
    assert load_video.await_args.args[0] != url


def test_nvdec_video_data_builds_pt_videodata(monkeypatch) -> None:
    """_nvdec_video_data turns NVDEC's (T,H,W,3) uint8 RGB frames into TRT-LLM's
    VideoData "pt" form: a list of (C,H,W) float32 [0,1] tensors + metadata that
    matches the vendor async_load_video output. Decode is mocked."""
    import numpy as np

    from dynamo.common.multimodal import nvdec_decoder

    n, h, w = 3, 4, 5
    # Mod in the default int dtype, then cast: `uint8_array % 256` raises
    # OverflowError under NumPy 2.x / NEP 50 (256 is out of uint8 range).
    frames_np = (np.arange(n * h * w * 3) % 256).astype(np.uint8).reshape(n, h, w, 3)
    meta = {"total_num_frames": 24, "fps": 12.0, "frames_indices": [0, 8, 16]}
    # _nvdec_video_data imports decode_video_nvdec lazily from this module, so
    # patching the source attribute is picked up on the fresh import.
    monkeypatch.setattr(
        nvdec_decoder,
        "decode_video_nvdec",
        lambda content, num_frames: (frames_np, meta),
    )

    vd = mmp._nvdec_video_data(b"h264-bytes", 3)

    assert isinstance(vd.frames, list) and len(vd.frames) == n
    f0 = vd.frames[0]
    assert isinstance(f0, torch.Tensor)
    assert tuple(f0.shape) == (3, h, w)  # (C, H, W)
    assert f0.dtype == torch.float32
    assert 0.0 <= float(f0.min()) and float(f0.max()) <= 1.0
    expected = torch.from_numpy(frames_np[0].astype("float32") * (1.0 / 255.0)).permute(
        2, 0, 1
    )
    assert torch.allclose(f0, expected, atol=1e-6)
    assert set(vd.metadata) == {
        "total_num_frames",
        "fps",
        "duration",
        "frames_indices",
    }
    assert vd.metadata["total_num_frames"] == 24
    assert vd.metadata["fps"] == 12.0
    assert vd.metadata["duration"] == 24 / 12.0
    assert vd.metadata["frames_indices"] == [0, 8, 16]
    assert vd.audio is None


@pytest.mark.asyncio
async def test_h264_video_routes_through_nvdec(monkeypatch) -> None:
    """H.264 video input is hardware-decoded via NVDEC; the vendor
    async_load_video is bypassed. Other codecs fall back to it (covered by
    test_internal_video_uses_dynamo_fetcher_when_allowed)."""
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    sentinel = object()
    monkeypatch.setattr(mmp, "fetch_bytes", AsyncMock(return_value=b"h264 bytes"))
    monkeypatch.setattr(mmp, "probe_video_codec", lambda content: "h264")
    monkeypatch.setattr(mmp, "should_use_nvdec", lambda codec: codec == "h264")
    nvdec = MagicMock(return_value=sentinel)
    monkeypatch.setattr(mmp, "_nvdec_video_data", nvdec)
    load_video = AsyncMock(return_value=object())
    monkeypatch.setattr(mmp, "async_load_video", load_video)

    await processor.process_openai_request(
        {
            "multi_modal_data": {
                "video_url": [{"Url": "http://169.254.169.254/v.mp4"}]
            },
            "token_ids": [1],
        },
        embeddings=None,
        ep_disaggregated_params=None,
    )

    nvdec.assert_called_once()  # the NVDEC transform ran ...
    assert nvdec.call_args.args[0] == b"h264 bytes"
    load_video.assert_not_awaited()  # ... and the vendor decoder was bypassed


@pytest.mark.asyncio
async def test_vp9_video_reports_an_unsupported_codec_error(monkeypatch) -> None:
    """VP9 has no decoder in this image, and the request must say so.

    The TensorRT-LLM images ship no cv2, and TensorRT-LLM decodes video through
    it (`_load_video_by_cv2`). NVDEC covers H.264/H.265 only, so VP8/VP9/AV1 have
    no decode path at all here -- a documented consequence of the codec-compliant
    build, not an accident.

    Assert the failure rather than leave the gap untested, the approach NVIDIA
    DALI took when it trimmed its own FFmpeg build (NVIDIA/DALI#6352). Without
    this, a base-image bump that quietly reintroduced cv2 -- or a routing change
    that sent VP9 somewhere unexpected -- would go unnoticed, since no test
    exercises a codec this image cannot decode.

    The contract asserted is what a client sees: HTTP 400 naming the video,
    not a 500 or a hang.

    Confirmed against a real TensorRT-LLM image (1.3.0rc22, no cv2 installed),
    which returned exactly:

        HTTP 400 ... Failed to load video (<url>): OpenCV (cv2) is required for
        video decoding but is not installed. Install it with
        `pip install opencv-python-headless`.

    Only the prefix and the 400 are ours (`multimodal_processor.py`); the rest is
    TensorRT-LLM's own guard, so this asserts the wording we own and that the
    upstream reason is *preserved* rather than swallowed -- not the vendor's
    exact phrasing, which any version bump may reword.
    """
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    monkeypatch.setattr(mmp, "fetch_bytes", AsyncMock(return_value=b"vp9 bytes"))
    monkeypatch.setattr(mmp, "probe_video_codec", lambda content: "vp9")
    # Real routing: only H.264/H.265 are hardware-decoded.
    monkeypatch.setattr(
        mmp, "should_use_nvdec", lambda codec: codec in ("h264", "hevc")
    )
    nvdec = MagicMock()
    monkeypatch.setattr(mmp, "_nvdec_video_data", nvdec)
    # The vendor loader failing on an image with no cv2, reproducing the message
    # observed on hardware verbatim. Our handler catches bare `Exception`, so the
    # type is immaterial -- the message is what reaches the client.
    upstream_error = ImportError(
        "OpenCV (cv2) is required for video decoding but is not installed. "
        "Install it with `pip install opencv-python-headless`."
    )
    monkeypatch.setattr(mmp, "async_load_video", AsyncMock(side_effect=upstream_error))

    with pytest.raises(HttpStatusError) as excinfo:
        await processor.process_openai_request(
            {
                "multi_modal_data": {
                    "video_url": [{"Url": "http://169.254.169.254/v.webm"}]
                },
                "token_ids": [1],
            },
            embeddings=None,
            ep_disaggregated_params=None,
        )

    # Contract change with the actionable-error work: a missing DECODER is
    # deployment configuration, not a bad request, so this is now a 500 (the
    # ImportError wrap classifies it), not the generic 400.
    assert excinfo.value.status == 500
    message = str(excinfo.value)
    # Ours: the prefix and the offending URL, so the client knows which input failed.
    assert "Failed to load video" in message
    assert "http://169.254.169.254/v.webm" in message
    # The upstream reason is preserved verbatim rather than swallowed. Asserted as
    # "whatever the cause said survives", not as specific vendor wording.
    assert str(upstream_error) in message
    # And the actionable guidance is present alongside it.
    assert "install_media_decoders trtllm" in message

    nvdec.assert_not_called()  # VP9 must never take the hardware path


@pytest.mark.asyncio
async def test_video_error_never_echoes_an_inline_media_payload(
    monkeypatch,
) -> None:
    """A data: video URL carries the whole payload inline, so it must never be
    echoed back in the error.

    Measured on the real TRT-LLM image before this bound existed: a 250 KB
    inline video produced a 683,213-character message -- the payload twice
    (once from our text, once from HttpStatusError's own format string), about
    1500x the 457 characters of actionable guidance -- returned to the client
    and written to every log sink that recorded the failure.
    """
    import base64

    payload = base64.b64encode(b"\x00" * (64 * 1024)).decode()
    url = f"data:video/mp4;base64,{payload}"

    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    monkeypatch.setattr(
        mmp,
        "async_load_video",
        AsyncMock(side_effect=ImportError("OpenCV (cv2) is required")),
    )

    with pytest.raises(HttpStatusError) as exc_info:
        await processor.process_openai_request(
            {
                "multi_modal_data": {"video_url": [{"Url": url}]},
                "token_ids": [1],
            },
            embeddings=None,
            ep_disaggregated_params=None,
        )

    message = str(exc_info.value)
    assert payload not in message
    assert payload not in str(exc_info.value.url)
    # Still identifies the source by type and size, and stays actionable.
    assert "data:video/mp4" in message
    assert "payload elided" in message
    assert "install_media_decoders trtllm" in message
    assert len(message) < 2_000, f"error message is unbounded: {len(message)} chars"


@pytest.mark.asyncio
async def test_video_missing_decoder_error_is_actionable(monkeypatch) -> None:
    """cv2 absent: the vendor loader's ImportError must surface as guidance
    naming the bounded spec and installer, not a bare module error."""
    from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    fetch = AsyncMock(return_value=b"video bytes")
    load_video = AsyncMock(
        side_effect=ImportError("OpenCV (cv2) is required for video decoding")
    )
    monkeypatch.setattr(mmp, "fetch_bytes", fetch, raising=False)
    monkeypatch.setattr(mmp, "async_load_video", load_video)

    with pytest.raises(HttpStatusError) as exc_info:
        await processor.process_openai_request(
            {
                "multi_modal_data": {
                    "video_url": [{"Url": "http://169.254.169.254/x.mp4"}]
                },
                "token_ids": [1],
            },
            embeddings=None,
            ep_disaggregated_params=None,
        )

    # A missing decoder is a deployment gap: 500, never the generic 400.
    assert exc_info.value.status == 500
    msg = str(exc_info.value)
    assert VALIDATED_SPECS["opencv-python-headless"] in msg
    assert "install_media_decoders trtllm" in msg
    # The vendor loader's own text survives as the cause.
    assert "OpenCV (cv2) is required for video decoding" in msg
