# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from dynamo.common.protocols.image_protocol import NvCreateImageRequest
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest
from dynamo.common.utils.output_modalities import RequestType

try:
    from dynamo.vllm.omni.omni_handler import (
        EngineInputs,
        OmniHandler,
        prepare_image_output,
    )
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_handler():
    with patch(
        "dynamo.vllm.omni.omni_handler.BaseOmniHandler.__init__", return_value=None
    ):
        handler = OmniHandler.__new__(OmniHandler)

    config = MagicMock()
    config.model = "test-model"
    config.served_model_name = None
    config.output_modalities = ["text"]
    handler.config = config
    return handler


class TestEngineInputs:
    def test_defaults(self):
        """EngineInputs uses CHAT_COMPLETION, fps=0, and None optionals by default."""
        ei = EngineInputs(prompt={"prompt": "hello"})
        assert ei.request_type == RequestType.CHAT_COMPLETION
        assert ei.fps == 0
        assert ei.sampling_params_list is None
        assert ei.response_format is None


class TestPrepareImageOutput:
    def test_b64_json(self):
        """b64_json format returns data URI with base64 prefix."""
        img = MagicMock()
        img.save = lambda b, format: b.write(b"fake_png_data")
        results = prepare_image_output([img], "b64_json")
        assert len(results) == 1
        assert results[0].startswith("data:image/png;base64,")

    def test_b64_default_when_none(self):
        """None response_format defaults to base64 encoding."""
        img = MagicMock()
        img.save = lambda b, format: b.write(b"data")
        results = prepare_image_output([img], None)
        assert results[0].startswith("data:image/png;base64,")

    def test_invalid_format(self):
        """Unsupported response_format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid response format"):
            prepare_image_output([MagicMock()], "invalid")

    def test_multiple_images(self):
        """Multiple input images produce one output entry each."""
        imgs = [MagicMock() for _ in range(3)]
        for img in imgs:
            img.save = lambda b, format: b.write(b"px")
        results = prepare_image_output(imgs, "b64_json")
        assert len(results) == 3


class TestBuildEngineInputs:
    def test_chat_completion(self):
        """Chat request extracts text prompt with no sampling params."""
        handler = _make_handler()
        raw = {"messages": [{"role": "user", "content": "hello"}]}
        inputs = handler.build_engine_inputs(raw, RequestType.CHAT_COMPLETION)
        assert inputs.request_type == RequestType.CHAT_COMPLETION
        assert inputs.prompt["prompt"] == "hello"
        assert inputs.sampling_params_list is None

    def test_image_generation(self):
        """Image request parses prompt, size, and creates diffusion sampling params."""
        handler = _make_handler()
        req = NvCreateImageRequest(prompt="a cat", size="512x512")
        inputs = handler.build_engine_inputs(req, RequestType.IMAGE_GENERATION)
        assert inputs.request_type == RequestType.IMAGE_GENERATION
        assert inputs.prompt["prompt"] == "a cat"
        assert len(inputs.sampling_params_list) == 1
        sp = inputs.sampling_params_list[0]
        assert sp.height == 512
        assert sp.width == 512

    def test_video_generation(self):
        """Video request parses prompt, size, seconds, and sets fps."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone", model="test", size="832x480", seconds=2
        )
        inputs = handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        assert inputs.request_type == RequestType.VIDEO_GENERATION
        assert inputs.prompt["prompt"] == "a drone"
        assert inputs.fps > 0

    def test_audio_not_implemented(self):
        """Audio generation raises NotImplementedError."""
        handler = _make_handler()
        with pytest.raises(NotImplementedError):
            handler.build_engine_inputs({}, RequestType.AUDIO_GENERATION)


class TestFormatTextChunk:
    def _make_output(self, text="hello world", finish_reason=None):
        output = MagicMock()
        output.text = text
        output.finish_reason = finish_reason
        request_output = MagicMock()
        request_output.outputs = [output]
        request_output.prompt_token_ids = [1, 2, 3]
        return request_output

    def test_delta_text(self):
        """Delta content is the diff between current and previous text."""
        handler = _make_handler()
        ro = self._make_output("hello world")
        chunk = handler._format_text_chunk(ro, "req-1", "hello ")
        assert chunk["choices"][0]["delta"]["content"] == "world"

    def test_no_outputs_returns_error(self):
        """Empty engine outputs produce an error chunk."""
        handler = _make_handler()
        ro = MagicMock()
        ro.outputs = []
        chunk = handler._format_text_chunk(ro, "req-1", "")
        assert "Error" in chunk["choices"][0]["delta"]["content"]

    def test_finish_reason_included(self):
        """Final chunk includes finish_reason and usage stats."""
        handler = _make_handler()
        handler._normalize_finish_reason = lambda r: r
        handler._build_completion_usage = lambda ro: {
            "prompt_tokens": 3,
            "completion_tokens": 1,
        }
        ro = self._make_output("done", finish_reason="stop")
        chunk = handler._format_text_chunk(ro, "req-1", "")
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in chunk


class TestFormatImageChunk:
    def test_chat_completion_format(self):
        """Chat completion route returns image_url content parts."""
        handler = _make_handler()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"px")
        chunk = handler._format_image_chunk(
            [img], "req-1", request_type=RequestType.CHAT_COMPLETION
        )
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["choices"][0]["delta"]["content"][0]["type"] == "image_url"

    def test_image_generation_b64_format(self):
        """Image generation with b64_json format returns base64 data."""
        handler = _make_handler()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"px")
        chunk = handler._format_image_chunk(
            [img],
            "req-1",
            response_format="b64_json",
            request_type=RequestType.IMAGE_GENERATION,
        )
        assert chunk["data"][0]["b64_json"] is not None

    def test_image_generation_default_format_returns_b64(self):
        """Image generation with response_format=None defaults to b64_json."""
        handler = _make_handler()
        img = MagicMock()
        img.save = lambda b, format: b.write(b"px")
        chunk = handler._format_image_chunk(
            [img],
            "req-1",
            response_format=None,
            request_type=RequestType.IMAGE_GENERATION,
        )
        assert chunk["data"][0]["b64_json"] is not None

    def test_empty_images_returns_error(self):
        """Empty image list produces an error chunk."""
        handler = _make_handler()
        chunk = handler._format_image_chunk([], "req-1")
        assert "Error" in chunk["choices"][0]["delta"]["content"]


class TestFormatVideoChunk:
    @pytest.mark.asyncio
    async def test_empty_frames_returns_none(self):
        """Empty frame list returns None."""
        handler = _make_handler()
        result = await handler._format_video_chunk([], "req-1", fps=16)
        assert result is None

    @pytest.mark.asyncio
    async def test_error_returns_failed_status(self):
        """Encoding failure returns NvVideosResponse with failed status and error."""
        handler = _make_handler()
        with patch(
            "dynamo.vllm.omni.omni_handler.normalize_video_frames",
            side_effect=RuntimeError("boom"),
        ):
            chunk = await handler._format_video_chunk([MagicMock()], "req-1", fps=16)
        assert chunk["status"] == "failed"
        assert "boom" in chunk["error"]
