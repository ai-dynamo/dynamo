# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for output_formatter.py — modality-specific formatters."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

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
    pytest.mark.gpu_0,
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
]


# ── TextFormatter ──────────────────────────────────────────


def _make_request_output(
    text="hello world", finish_reason=None, num_cached_tokens=None
):
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
    ro.num_cached_tokens = num_cached_tokens
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
    async def test_empty_frames_returns_video_error(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        stage = MagicMock()
        stage.images = []
        result = await f.format(
            stage, "req-1", request_type=RequestType.VIDEO_GENERATION
        )
        assert result["object"] == "video"
        assert result["status"] == "failed"
        assert "No video outputs" in result["error"]

    @pytest.mark.asyncio
    async def test_error_returns_failed_status(self):
        from unittest.mock import patch

        f = _make_diffusion_formatter()
        with patch(
            "dynamo.vllm.omni.output_formatter.normalize_video_frames",
            side_effect=RuntimeError("boom"),
        ):
            chunk = await f._encode_video([MagicMock()], "req-1", fps=16)
        assert chunk["status"] == "failed"
        assert "boom" in chunk["error"]

    @pytest.mark.asyncio
    async def test_muxes_video_audio_and_reports_metadata(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        stage = SimpleNamespace(
            images=[np.zeros((2, 4, 4, 3), dtype=np.float32)],
            multimodal_output={
                "audio": np.zeros((1, 2, 128), dtype=np.float32),
                "fps": 24,
                "audio_sample_rate": 32000,
            },
        )
        with patch(
            "dynamo.vllm.omni.output_formatter.mux_video_audio_bytes",
            return_value=b"muxed-mp4",
        ) as mux:
            result = await f.format(
                stage,
                "req-video-audio",
                request_type=RequestType.VIDEO_GENERATION,
                response_format="b64_json",
            )

        assert result["status"] == "completed"
        assert result["data"][0]["fps"] == 24
        assert result["data"][0]["audio_sample_rate"] == 32000
        assert result["data"][0]["b64_json"] is not None
        assert mux.call_count == 1
        assert mux.call_args.kwargs["fps"] == 24.0
        assert mux.call_args.kwargs["audio_sample_rate"] == 32000

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("sample_rate_key", "sample_rate"),
        [("sample_rate", 22050), ("sampling_rate", 44100), ("sr", 16000)],
    )
    async def test_uses_audio_sample_rate_aliases(self, sample_rate_key, sample_rate):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        stage = SimpleNamespace(
            images=[np.zeros((2, 4, 4, 3), dtype=np.float32)],
            multimodal_output={
                "audio": np.zeros((1, 2, 128), dtype=np.float32),
                sample_rate_key: sample_rate,
            },
        )
        with patch(
            "dynamo.vllm.omni.output_formatter.mux_video_audio_bytes",
            return_value=b"muxed-mp4",
        ) as mux:
            result = await f.format(
                stage,
                "req-video-audio",
                request_type=RequestType.VIDEO_GENERATION,
                response_format="b64_json",
            )

        assert result["data"][0]["audio_sample_rate"] == sample_rate
        assert mux.call_args.kwargs["audio_sample_rate"] == sample_rate

    @pytest.mark.asyncio
    async def test_uses_model_audio_sample_rate_then_generic_fallback(self):
        configured = DiffusionFormatter(
            model_name="test-model",
            media_fs=None,
            media_http_url=None,
            model_config=SimpleNamespace(
                audio_vae=SimpleNamespace(config=SimpleNamespace(sampling_rate=48000))
            ),
        )
        audio = {"audio": np.zeros((1, 2, 128), dtype=np.float32)}
        video = [np.zeros((2, 4, 4, 3), dtype=np.float32)]

        with patch(
            "dynamo.vllm.omni.output_formatter.mux_video_audio_bytes",
            return_value=b"muxed-mp4",
        ) as mux:
            configured_result = await configured._encode_video(
                video,
                "req-configured-rate",
                fps=24,
                multimodal_output=audio,
                response_format="b64_json",
            )
            fallback_result = await _make_diffusion_formatter()._encode_video(
                video,
                "req-fallback-rate",
                fps=24,
                multimodal_output=audio,
                response_format="b64_json",
            )

        assert configured_result["data"][0]["audio_sample_rate"] == 48000
        assert fallback_result["data"][0]["audio_sample_rate"] == 24000
        assert [call.kwargs["audio_sample_rate"] for call in mux.call_args_list] == [
            48000,
            24000,
        ]

    @pytest.mark.asyncio
    async def test_returns_every_generated_video(self):
        from dynamo.common.utils.output_modalities import RequestType

        f = _make_diffusion_formatter()
        videos = [np.zeros((2, 4, 4, 3), dtype=np.float32) for _ in range(2)]
        stage = SimpleNamespace(
            images=videos,
            multimodal_output={
                "audio": np.zeros((2, 2, 128), dtype=np.float32),
                "metadata": {
                    "video": {"fps": 24},
                    "audio": {"sample_rate": 32000},
                },
            },
        )
        with patch(
            "dynamo.vllm.omni.output_formatter.mux_video_audio_bytes",
            return_value=b"muxed-mp4",
        ) as mux:
            result = await f.format(
                stage,
                "req-video-audio",
                request_type=RequestType.VIDEO_GENERATION,
                response_format="b64_json",
            )

        assert len(result["data"]) == 2
        assert mux.call_count == 2

    @pytest.mark.asyncio
    async def test_silent_video_keeps_vp9_encoder(self):
        f = _make_diffusion_formatter()
        with (
            patch(
                "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
                return_value=b"vp9-mp4",
            ) as vp9,
            patch("dynamo.vllm.omni.output_formatter.mux_video_audio_bytes") as mux,
        ):
            result = await f._encode_video(
                [np.zeros((4, 4, 3), dtype=np.float32)],
                "req-silent",
                fps=16,
                response_format="b64_json",
            )

        assert result["status"] == "completed"
        assert result["data"][0]["audio_sample_rate"] is None
        vp9.assert_called_once()
        mux.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsupported_output_format_hint_falls_back_to_mp4(self):
        f = _make_diffusion_formatter()
        with patch(
            "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
            return_value=b"vp9-mp4",
        ) as encode:
            result = await f._encode_video(
                [np.zeros((4, 4, 3), dtype=np.float32)],
                "req-output-hint",
                fps=16,
                response_format="b64_json",
                output_format="webm",
            )

        assert result["status"] == "completed"
        assert result["data"][0]["output_format"] == "mp4"
        assert encode.call_args.kwargs["output_format"] == "mp4"

    @pytest.mark.asyncio
    async def test_normalizes_direct_fchw_torch_video(self):
        f = _make_diffusion_formatter()
        video = torch.zeros((2, 3, 8, 10), dtype=torch.float32)

        with patch(
            "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
            return_value=b"vp9-mp4",
        ) as encode:
            result = await f._encode_video(
                video,
                "req-tensor",
                fps=16,
                response_format="b64_json",
            )

        assert result["status"] == "completed"
        frames = encode.call_args.args[0]
        assert frames.shape == (2, 8, 10, 3)
        assert frames.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_splits_direct_bfchw_torch_videos(self):
        f = _make_diffusion_formatter()
        videos = torch.zeros((2, 2, 3, 8, 10), dtype=torch.float32)

        with patch(
            "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
            return_value=b"vp9-mp4",
        ) as encode:
            result = await f._encode_video(
                videos,
                "req-tensor-batch",
                fps=16,
                response_format="b64_json",
            )

        assert result["status"] == "completed"
        assert len(result["data"]) == 2
        assert encode.call_count == 2
        assert all(
            call.args[0].shape == (2, 8, 10, 3) for call in encode.call_args_list
        )

    @pytest.mark.asyncio
    async def test_splits_list_wrapped_bfhwc_numpy_videos(self):
        f = _make_diffusion_formatter()
        videos = [np.zeros((2, 2, 8, 10, 3), dtype=np.float32)]

        with patch(
            "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
            return_value=b"vp9-mp4",
        ) as encode:
            result = await f._encode_video(
                videos,
                "req-numpy-batch",
                fps=16,
                response_format="b64_json",
            )

        assert result["status"] == "completed"
        assert len(result["data"]) == 2
        assert encode.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("as_tensor", [False, True])
    @pytest.mark.parametrize("num_frames", [1, 3, 4, 8])
    async def test_normalizes_cfhw_negative_one_to_one_video(
        self, as_tensor, num_frames
    ):
        f = _make_diffusion_formatter()
        video = np.linspace(-1.0, 1.0, num=3 * num_frames * 16 * 16, dtype=np.float32)
        video = video.reshape(3, num_frames, 16, 16)
        if as_tensor:
            video = torch.from_numpy(video)

        with patch(
            "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
            return_value=b"vp9-mp4",
        ) as encode:
            result = await f._encode_video(
                video,
                "req-cfhw",
                fps=16,
                response_format="b64_json",
            )

        assert result["status"] == "completed"
        frames = encode.call_args.args[0]
        assert frames.shape == (num_frames, 16, 16, 3)
        assert frames.dtype == np.uint8
        assert frames.min() == 0
        assert frames.max() == 255


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

    @pytest.mark.parametrize(
        ("num_cached_tokens", "expected_prompt_tokens_details"),
        [
            (None, None),
            (0, {"cached_tokens": 0}),
            (3, {"cached_tokens": 3}),
        ],
    )
    def test_cached_token_details(
        self, num_cached_tokens, expected_prompt_tokens_details
    ):
        ro = _make_request_output(num_cached_tokens=num_cached_tokens)

        usage = _build_completion_usage(ro)

        assert usage["prompt_tokens_details"] == expected_prompt_tokens_details


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


# ── AudioFormatter — output_format field (new branch behavior) ──────────────


class TestAudioFormatterOutputFormat:
    """output_format context kwarg drives codec selection; AudioData carries it."""

    def _make_formatter(self):
        from dynamo.vllm.omni.output_formatter import AudioFormatter

        return AudioFormatter(model_name="test", media_fs=None, media_http_url=None)

    def _make_mm_output(self):
        import numpy as np

        return {"audio": np.zeros(100, dtype=np.float32), "sr": 24000}

    @pytest.mark.asyncio
    async def test_output_format_mp3_passed_as_codec(self):
        f = self._make_formatter()
        mm = self._make_mm_output()
        with patch.object(
            f, "_encode_audio", return_value=(b"bytes", "audio/mpeg")
        ) as mock_enc:
            await f.format(mm, "r1", response_format="b64_json", output_format="mp3")
        _, args, _ = mock_enc.mock_calls[0]
        assert args[2] == "mp3"

    @pytest.mark.asyncio
    async def test_output_format_none_defaults_to_wav(self):
        f = self._make_formatter()
        mm = self._make_mm_output()
        with patch.object(
            f, "_encode_audio", return_value=(b"bytes", "audio/wav")
        ) as mock_enc:
            await f.format(mm, "r2", response_format="b64_json", output_format=None)
        _, args, _ = mock_enc.mock_calls[0]
        assert args[2] == "wav"

    @pytest.mark.asyncio
    async def test_audio_data_carries_output_format_b64_path(self):
        f = self._make_formatter()
        mm = self._make_mm_output()
        with patch.object(f, "_encode_audio", return_value=(b"bytes", "audio/flac")):
            result = await f.format(
                mm, "r3", response_format="b64_json", output_format="flac"
            )
        assert result["data"][0]["output_format"] == "flac"

    @pytest.mark.asyncio
    async def test_audio_data_carries_output_format_url_path(self):
        from unittest.mock import patch as _patch

        f = self._make_formatter()
        mm = self._make_mm_output()
        with (
            patch.object(f, "_encode_audio", return_value=(b"bytes", "audio/ogg")),
            _patch(
                "dynamo.vllm.omni.output_formatter.upload_to_fs",
                return_value="http://x/a.ogg",
            ),
        ):
            result = await f.format(
                mm, "r4", response_format="url", output_format="opus"
            )
        assert result["data"][0]["output_format"] == "opus"
        assert result["data"][0]["url"] is not None


# ── DiffusionFormatter — VideoData.output_format (new branch behavior) ──────


class TestDiffusionFormatterVideoOutputFormat:
    """_encode_video always sets VideoData.output_format='mp4'."""

    def _patches(self):
        from unittest.mock import patch as _patch

        return (
            _patch(
                "dynamo.vllm.omni.output_formatter.normalize_video_frames",
                return_value=[MagicMock()],
            ),
            _patch(
                "dynamo.vllm.omni.output_formatter.frames_to_numpy",
                return_value=MagicMock(),
            ),
            _patch(
                "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
                return_value=b"video-bytes",
            ),
            _patch(
                "dynamo.vllm.omni.output_formatter.upload_to_fs",
                return_value="http://x/v.mp4",
            ),
            _patch(
                "dynamo.vllm.omni.output_formatter.asyncio.to_thread",
                side_effect=lambda fn, *a, **kw: fn(*a, **kw),
            ),
        )

    @pytest.mark.asyncio
    async def test_video_url_response_format(self):
        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.output_formatter import DiffusionFormatter

        f = DiffusionFormatter(model_name="test", media_fs=None, media_http_url=None)
        stage = MagicMock()
        stage.images = [MagicMock()]

        p1, p2, p3, p4, p5 = self._patches()
        with p1, p2, p3, p4 as mock_upload, p5:
            result = await f.format(
                stage,
                "r5",
                request_type=RequestType.VIDEO_GENERATION,
                fps=16,
                response_format="url",
            )

        assert result is not None
        assert result["data"][0]["output_format"] == "mp4"
        assert result["data"][0]["url"] == "http://x/v.mp4"
        assert result["data"][0].get("b64_json") is None
        mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_video_b64_response_format(self):
        import base64

        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.output_formatter import DiffusionFormatter

        f = DiffusionFormatter(model_name="test", media_fs=None, media_http_url=None)
        stage = MagicMock()
        stage.images = [MagicMock()]

        p1, p2, p3, p4, p5 = self._patches()
        with p1, p2, p3, p4 as mock_upload, p5:
            result = await f.format(
                stage,
                "r6",
                request_type=RequestType.VIDEO_GENERATION,
                fps=16,
                response_format="b64_json",
            )

        assert result is not None
        assert result["data"][0]["output_format"] == "mp4"
        assert result["data"][0].get("url") is None
        assert result["data"][0]["b64_json"] is not None
        base64.b64decode(result["data"][0]["b64_json"])  # must be valid base64
        mock_upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_video_default_response_format_is_url(self):
        """Omitting response_format defaults to url."""
        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.output_formatter import DiffusionFormatter

        f = DiffusionFormatter(model_name="test", media_fs=None, media_http_url=None)
        stage = MagicMock()
        stage.images = [MagicMock()]

        p1, p2, p3, p4, p5 = self._patches()
        with p1, p2, p3, p4 as mock_upload, p5:
            result = await f.format(
                stage, "r7", request_type=RequestType.VIDEO_GENERATION, fps=16
            )

        assert result is not None
        assert result["data"][0]["url"] == "http://x/v.mp4"
        mock_upload.assert_called_once()
