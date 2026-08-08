# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import numpy as np
import pytest
import torch

from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SamplingOptions,
    SglangMultimodalRequest,
    StopConditions,
)
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    Modality,
    MultimodalEncodeWorkerHandler,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    EmbeddingsProcessor,
    _build_mm_items,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(Modality is None, reason="SGLang Modality is required"),
]


def test_extract_media_urls_supports_video_urls():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    image_urls, video_urls = handler._extract_media_urls(
        {
            "multi_modal_data": {
                "video_url": [
                    {"Url": "https://example.com/clip.mp4"},
                    "file:///tmp/local.mp4",
                ]
            }
        }
    )

    assert image_urls == []
    assert video_urls == ["https://example.com/clip.mp4", "file:///tmp/local.mp4"]


def test_extract_media_urls_supports_mixed_image_and_video():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    image_urls, video_urls = handler._extract_media_urls(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/image.png"}],
                "video_url": [{"Url": "https://example.com/clip.mp4"}],
            }
        }
    )

    assert image_urls == ["https://example.com/image.png"]
    assert video_urls == ["https://example.com/clip.mp4"]


@pytest.mark.multimodal
def test_extract_media_urls_rejects_multimodal_cache_uuid():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    with pytest.raises(ValueError, match="supported only by the vLLM backend"):
        handler._extract_media_urls(
            {
                "multi_modal_uuids": {"image_url": ["cached-image"]},
            }
        )


@pytest.mark.asyncio
async def test_build_mm_items_routes_video_to_video_data():
    embeddings = torch.arange(24, dtype=torch.float16).reshape(6, 4)

    class _FakeEmbeddingsProcessor:
        async def process_embeddings(self, request):
            return embeddings, 17

        @staticmethod
        def create_multimodal_image_item(
            embeddings,
            image_grid_thw,
        ):
            return EmbeddingsProcessor.create_multimodal_image_item(
                embeddings,
                image_grid_thw,
            )

        @staticmethod
        def create_multimodal_video_item(
            embeddings,
            video_grid_thw,
            second_per_grid_ts=None,
            video_timestamps=None,
        ):
            return EmbeddingsProcessor.create_multimodal_video_item(
                embeddings,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                video_timestamps=video_timestamps,
            )

    request = SglangMultimodalRequest(
        request=PreprocessedRequest(
            token_ids=[151652, 151656, 151653],
            stop_conditions=StopConditions(max_tokens=32),
            sampling_options=SamplingOptions(temperature=0.0),
        ),
        multimodal_inputs=[
            MultiModalGroup(
                multimodal_input=MultiModalInput(),
                video_grid_thw=[2, 3, 4],
                second_per_grid_ts=0.5,
                video_timestamps=[0.25, 0.75],
                num_mm_tokens=6,
            )
        ],
    )

    image_items, video_items, combined_embeddings, tensor_id = await _build_mm_items(
        request, _FakeEmbeddingsProcessor()
    )

    assert tensor_id == 17
    assert torch.equal(combined_embeddings, embeddings)
    assert image_items == []
    assert len(video_items) == 1

    mm_item = video_items[0]
    assert mm_item["modality"] == "VIDEO"
    assert torch.equal(mm_item["video_grid_thw"], torch.tensor([[2, 3, 4]]))
    assert torch.equal(
        mm_item["second_per_grid_ts"], torch.tensor([0.5], dtype=torch.float32)
    )
    assert mm_item["video_timestamps"] == [[0.25, 0.75]]


async def test_nvdec_video_metadata_shim_stamps_valid_metadata():
    """The metadata shim gives pre-decoded ndarray frames a valid metadata dict.

    SGLang's ``preprocess_video`` returns ``(frames, None)`` for a pre-decoded
    ndarray, and transformers >= 5.12 strict-rejects the resulting
    ``video_metadata=[None]``. ``_install_nvdec_video_metadata_shim`` wraps it so
    the ndarray carries a valid dict instead (validated end-to-end on gpu-ts
    against the real ``MMEncoder._encode``: before FAIL -> after PASS).
    """
    es = pytest.importorskip("sglang.srt.disaggregation.encode_server")
    from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
        _NVDEC_SHIM_FPS,
        _install_nvdec_video_metadata_shim,
    )

    saved = es.preprocess_video
    try:
        _install_nvdec_video_metadata_shim()
        assert getattr(es.preprocess_video, "_dynamo_nvdec_shim", False)

        # Idempotent: a second install must not double-wrap.
        wrapped = es.preprocess_video
        _install_nvdec_video_metadata_shim()
        assert es.preprocess_video is wrapped

        frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        video, meta = await es.preprocess_video(frames)
        assert video is frames
        assert meta == {
            "fps": _NVDEC_SHIM_FPS,
            "duration": 5 / _NVDEC_SHIM_FPS,
            "total_num_frames": 5,
            "frames_indices": [0, 1, 2, 3, 4],
            "video_backend": "nvdec",
        }

        # Non-ndarray inputs keep None metadata (the shim only touches pixels).
        _, meta_none = await es.preprocess_video("not-an-array")
        assert meta_none is None
    finally:
        es.preprocess_video = saved


# ---------------------------------------------------------------------------
# Unsupported-codec preflight in _maybe_nvdec_decoder.
# ---------------------------------------------------------------------------


def _bare_handler():
    """Handler skeleton for exercising _maybe_nvdec_decoder in isolation."""
    from dynamo.common.http.url_validator import UrlValidationPolicy
    from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
        MultimodalEncodeWorkerHandler,
    )

    handler = object.__new__(MultimodalEncodeWorkerHandler)
    handler._url_policy = UrlValidationPolicy.from_env()
    return handler


def _selective_import(present: set[str]):
    """Fake importlib.import_module for the preflight's real-import probe:
    decoder modules import only when listed in `present` (a broken native
    install behaves exactly like an absent one -- ImportError either way)."""
    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name in ("torchcodec", "decord"):
            if name in present:
                return object()
            raise ImportError(f"No module named '{name}' (or broken install)")
        return real(name, *args, **kwargs)

    return fake


@pytest.mark.asyncio
async def test_vp9_without_software_decoder_is_actionable(monkeypatch):
    """A VP9 request on an image with neither torchcodec nor decord must fail
    HERE with guidance, not deep inside SGLang with the video payload repr
    embedded in a bare "No module named 'decord'" (observed on a real image).

    Raising through _maybe_nvdec_decoder also proves the broad fallback
    except-clause does not swallow the preflight error and pass the bytes on
    anyway.
    """
    import dynamo.sglang.request_handlers.multimodal.encode_worker_handler as ewh
    from dynamo.common.multimodal.codec_errors import MissingMediaDecoderError
    from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

    async def fake_validate(url, policy):
        return url

    async def fake_fetch(url, timeout, policy=None):
        return b"vp9-bytes"

    monkeypatch.setattr(ewh, "validate_media_url", fake_validate)
    monkeypatch.setattr(ewh, "fetch_bytes", fake_fetch)
    monkeypatch.setattr(ewh, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(ewh, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(ewh.importlib, "import_module", _selective_import(set()))

    handler = _bare_handler()
    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await handler._maybe_nvdec_decoder("https://example.com/clip.webm")

    msg = str(exc_info.value)
    assert "'vp9'" in msg
    assert VALIDATED_SPECS["decord2"] in msg
    assert "install_media_decoders sglang" in msg


@pytest.mark.asyncio
async def test_vp9_with_software_decoder_passes_bytes_through(monkeypatch):
    """With decord importable the preflight stays silent and the validated
    bytes are handed to SGLang exactly as before."""
    import dynamo.sglang.request_handlers.multimodal.encode_worker_handler as ewh

    async def fake_validate(url, policy):
        return url

    async def fake_fetch(url, timeout, policy=None):
        return b"vp9-bytes"

    monkeypatch.setattr(ewh, "validate_media_url", fake_validate)
    monkeypatch.setattr(ewh, "fetch_bytes", fake_fetch)
    monkeypatch.setattr(ewh, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(ewh, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(ewh.importlib, "import_module", _selective_import({"decord"}))

    handler = _bare_handler()
    result = await handler._maybe_nvdec_decoder("https://example.com/clip.webm")

    assert result == b"vp9-bytes"
