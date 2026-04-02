# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for output_formatter.py — modality-specific formatters."""

from unittest.mock import MagicMock

import pytest

try:
    from dynamo.vllm.omni.output_formatter import (
        DiffusionFormatter,
        TextFormatter,
        _build_completion_usage,
        _error_chunk,
    )
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


# ── TextFormatter ──────────────────────────────────────────


def _make_request_output(text="hello world", finish_reason=None):
    output = MagicMock()
    output.text = text
    output.finish_reason = finish_reason
    output.token_ids = [1, 2, 3]  # 3 completion tokens
    ro = MagicMock()
    ro.outputs = [output]
    ro.prompt_token_ids = [
        10,
        20,
        30,
        40,
        50,
    ]  # 5 prompt tokens (different from completion)
    return ro


class TestTextFormatter:
    def test_delta_text(self):
        f = TextFormatter(model_name="test-model")
        chunk = f.format(
            _make_request_output("hello world"), "req-1", previous_text="hello "
        )
        assert chunk["choices"][0]["delta"]["content"] == "world"

    def test_no_outputs_returns_error(self):
        f = TextFormatter(model_name="test-model")
        ro = MagicMock()
        ro.outputs = []
        chunk = f.format(ro, "req-1")
        assert "Error" in chunk["choices"][0]["delta"]["content"]

    def test_finish_reason_included(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("done", finish_reason="stop")
        chunk = f.format(ro, "req-1")
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in chunk

    def test_finish_reason_abort_normalized(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("done", finish_reason="abort")
        chunk = f.format(ro, "req-1")
        assert chunk["choices"][0]["finish_reason"] == "cancelled"

    def test_finish_reason_none_when_not_finished(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("partial")
        chunk = f.format(ro, "req-1")
        assert chunk["choices"][0]["finish_reason"] is None

    def test_model_name_in_response(self):
        f = TextFormatter(model_name="my-model")
        chunk = f.format(_make_request_output(), "req-1")
        assert chunk["model"] == "my-model"

    def test_usage_has_prompt_and_completion_tokens(self):
        f = TextFormatter(model_name="test-model")
        ro = _make_request_output("done", finish_reason="stop")
        chunk = f.format(ro, "req-1")
        assert chunk["usage"]["prompt_tokens"] == 5  # 5 prompt token IDs
        assert chunk["usage"]["completion_tokens"] == 3  # 3 completion token IDs
        assert chunk["usage"]["total_tokens"] == 8


# ── Helpers ────────────────────────────────────────────────


class TestErrorChunk:
    def test_error_chunk_format(self):
        chunk = _error_chunk("req-1", "my-model", "something broke")
        assert chunk["choices"][0]["delta"]["content"] == "Error: something broke"
        assert chunk["choices"][0]["finish_reason"] == "error"
        assert chunk["model"] == "my-model"


# ── DiffusionFormatter ─────────────────────────────────────


def _make_diffusion_formatter():
    return DiffusionFormatter(
        model_name="test-model", media_fs=None, media_http_url=None
    )


class TestDiffusionFormatterPrepareImages:
    @pytest.mark.asyncio
    async def test_b64_json(self):
        f = _make_diffusion_formatter()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"fake_png_data")
        results = await f._prepare_images([img], "req-1", "b64_json")
        assert len(results) == 1
        assert results[0].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_b64_default_when_none(self):
        f = _make_diffusion_formatter()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"data")
        results = await f._prepare_images([img], "req-1", None)
        assert results[0].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_invalid_format(self):
        f = _make_diffusion_formatter()
        with pytest.raises(ValueError, match="Invalid response format"):
            await f._prepare_images([MagicMock()], "req-1", "invalid")

    @pytest.mark.asyncio
    async def test_multiple_images(self):
        f = _make_diffusion_formatter()
        imgs = [MagicMock() for _ in range(3)]
        for img in imgs:
            img.save = lambda b, format: b.write(b"px")
        results = await f._prepare_images(imgs, "req-1", "b64_json")
        assert len(results) == 3


class TestDiffusionFormatterImage:
    @pytest.mark.asyncio
    async def test_chat_completion_format(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"px")
        chunk = await f._encode_image(
            [img], "req-1", request_type=RequestType.CHAT_COMPLETION
        )
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["choices"][0]["delta"]["content"][0]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_image_generation_b64_format(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"px")
        chunk = await f._encode_image(
            [img],
            "req-1",
            response_format="b64_json",
            request_type=RequestType.IMAGE_GENERATION,
        )
        assert chunk["data"][0]["b64_json"] is not None

    @pytest.mark.asyncio
    async def test_image_generation_default_format_returns_b64(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"px")
        chunk = await f._encode_image(
            [img],
            "req-1",
            response_format=None,
            request_type=RequestType.IMAGE_GENERATION,
        )
        assert chunk["data"][0]["b64_json"] is not None

    @pytest.mark.asyncio
    async def test_empty_images_returns_error(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        chunk = await f._encode_image(
            [], "req-1", request_type=RequestType.IMAGE_GENERATION
        )
        assert "Error" in chunk["choices"][0]["delta"]["content"]


class TestDiffusionFormatterVideo:
    @pytest.mark.asyncio
    async def test_empty_frames_returns_none(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        stage = MagicMock()
        stage.images = []
        result = await f.format(
            stage, "req-1", request_type=RequestType.VIDEO_GENERATION
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_error_returns_failed_status(self):
        from unittest.mock import patch

        f = _make_diffusion_formatter()
        with patch(
            "dynamo.common.utils.video_utils.normalize_video_frames",
            side_effect=RuntimeError("boom"),
        ):
            chunk = await f._encode_video([MagicMock()], "req-1", fps=16)
        assert chunk["status"] == "failed"
        assert "boom" in chunk["error"]


class TestBuildCompletionUsage:
    def test_basic(self):
        ro = _make_request_output("hello", finish_reason="stop")
        usage = _build_completion_usage(ro)
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 3
        assert usage["total_tokens"] == 8

    def test_no_prompt_tokens(self):
        ro = _make_request_output()
        ro.prompt_token_ids = None
        usage = _build_completion_usage(ro)
        assert usage["prompt_tokens"] is None
        assert usage["total_tokens"] is None


# ── AudioFormatter ─────────────────────────────────────────


class TestAudioFormatterExtractTensor:
    def test_extracts_from_audio_key(self):
        import numpy as np

        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        mm = {"audio": np.array([0.1, -0.2, 0.3], dtype=np.float32), "sr": 24000}
        audio_np, sr = f._extract_audio_tensor(mm)
        assert sr == 24000
        assert len(audio_np) == 3

    def test_extracts_from_model_outputs_key(self):
        import numpy as np

        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        mm = {"model_outputs": np.array([0.5, -0.5], dtype=np.float32), "sr": 16000}
        audio_np, sr = f._extract_audio_tensor(mm)
        assert sr == 16000
        assert len(audio_np) == 2

    def test_missing_audio_raises(self):
        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        with pytest.raises(ValueError, match="No audio data"):
            f._extract_audio_tensor({"sr": 24000})

    def test_squeezes_extra_dims(self):
        import numpy as np

        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        mm = {"audio": np.array([[0.1, 0.2, 0.3]], dtype=np.float32), "sr": 24000}
        audio_np, _ = f._extract_audio_tensor(mm)
        assert audio_np.ndim == 1


class TestAudioFormatterEncode:
    def test_wav_encoding(self):
        import numpy as np

        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        audio_bytes, media_type = f._encode_audio(
            np.zeros(2400, dtype=np.float32), 24000, "wav"
        )
        assert media_type == "audio/wav"
        assert audio_bytes[:4] == b"RIFF"

    def test_unsupported_format_falls_back_to_wav(self):
        import numpy as np

        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        _, media_type = f._encode_audio(np.zeros(100, dtype=np.float32), 24000, "xyz")
        assert media_type == "audio/wav"

    def test_default_format_is_wav(self):
        import numpy as np

        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        _, media_type = f._encode_audio(np.zeros(100, dtype=np.float32), 24000)
        assert media_type == "audio/wav"


class TestAudioFormatterFormat:
    @pytest.mark.asyncio
    async def test_empty_returns_error(self):
        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        result = await f.format({}, "req-1")
        assert result["status"] == "failed"
        assert "No audio generated" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        import numpy as np

        from dynamo.vllm.omni.output_formatter import AudioFormatter

        f = AudioFormatter(model_name="test", media_fs=None, media_http_url=None)
        mm = {"audio": np.random.randn(4800).astype(np.float32), "sr": 24000}
        result = await f.format(mm, "req-1")
        assert result["status"] == "completed"
        assert result["object"] == "audio.speech"
        assert len(result["data"]) == 1
        assert result["data"][0]["b64_json"] is not None


# ── OutputFormatter dispatcher ─────────────────────────────


class TestOutputFormatter:
    """Tests pass the full ctx that _generate_openai_mode actually sends
    (request_type, fps, response_format, previous_text, speed) to catch
    signature mismatches in individual formatters early."""

    # Full ctx matching _generate_openai_mode's call signature
    _FULL_CTX = dict(fps=16, response_format=None, previous_text="", speed=1.0)

    @pytest.mark.asyncio
    async def test_routes_text(self):
        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.output_formatter import OutputFormatter

        f = OutputFormatter(model_name="test-model")
        stage = MagicMock()
        stage.final_output_type = "text"
        stage.request_output = _make_request_output("hello world")
        chunk = await f.format(
            stage, "req-1", request_type=RequestType.CHAT_COMPLETION, **self._FULL_CTX
        )
        assert chunk["choices"][0]["delta"]["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_routes_image(self):
        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.output_formatter import OutputFormatter

        f = OutputFormatter(model_name="test-model")
        stage = MagicMock()
        stage.final_output_type = "image"
        img = MagicMock()
        img.save = lambda b, format: b.write(b"px")
        stage.images = [img]
        chunk = await f.format(
            stage, "req-1", request_type=RequestType.CHAT_COMPLETION, **self._FULL_CTX
        )
        assert chunk["choices"][0]["delta"]["content"][0]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_routes_audio(self):
        import numpy as np

        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.output_formatter import OutputFormatter

        f = OutputFormatter(model_name="test-model")
        stage = MagicMock()
        stage.final_output_type = "audio"
        stage.multimodal_output = {
            "audio": np.random.randn(2400).astype(np.float32),
            "sr": 24000,
        }
        chunk = await f.format(
            stage, "req-1", request_type=RequestType.AUDIO_GENERATION, **self._FULL_CTX
        )
        assert chunk["status"] == "completed"

    @pytest.mark.asyncio
    async def test_unknown_type_returns_none(self):
        from dynamo.vllm.omni.output_formatter import OutputFormatter

        f = OutputFormatter(model_name="test-model")
        stage = MagicMock()
        stage.final_output_type = "unknown_modality"
        result = await f.format(stage, "req-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_text_without_request_output_returns_none(self):
        from dynamo.vllm.omni.output_formatter import OutputFormatter

        f = OutputFormatter(model_name="test-model")
        stage = MagicMock()
        stage.final_output_type = "text"
        stage.request_output = None
        result = await f.format(stage, "req-1")
        assert result is None
