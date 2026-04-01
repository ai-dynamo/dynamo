# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

try:
    from PIL import Image

    from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
    from dynamo.common.protocols.image_protocol import NvCreateImageRequest
    from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
    from dynamo.common.utils.output_modalities import RequestType
    from dynamo.vllm.omni.omni_handler import EngineInputs, OmniHandler
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


class TestBuildEngineInputs:
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        """Chat request extracts text prompt with no sampling params."""
        handler = _make_handler()
        raw = {"messages": [{"role": "user", "content": "hello"}]}
        inputs = await handler.build_engine_inputs(raw, RequestType.CHAT_COMPLETION)
        assert inputs.request_type == RequestType.CHAT_COMPLETION
        assert inputs.prompt["prompt"] == "hello"
        assert inputs.sampling_params_list is None

    @pytest.mark.asyncio
    async def test_image_generation(self):
        """Image request parses prompt, size, and creates diffusion sampling params."""
        handler = _make_handler()
        req = NvCreateImageRequest(prompt="a cat", size="512x512")
        inputs = await handler.build_engine_inputs(req, RequestType.IMAGE_GENERATION)
        assert inputs.request_type == RequestType.IMAGE_GENERATION
        assert inputs.prompt["prompt"] == "a cat"
        assert len(inputs.sampling_params_list) == 1
        sp = inputs.sampling_params_list[0]
        assert sp.height == 512
        assert sp.width == 512

    @pytest.mark.asyncio
    async def test_video_generation(self):
        """Video request parses prompt, size, seconds, and sets fps."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone", model="test", size="832x480", seconds=2
        )
        inputs = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        assert inputs.request_type == RequestType.VIDEO_GENERATION
        assert inputs.prompt["prompt"] == "a drone"
        assert inputs.fps > 0

    @pytest.mark.asyncio
    async def test_audio_generation_delegates_toaudio(self):
        """Audio request delegates to audio."""
        handler = _make_handler()
        expected = EngineInputs(
            prompt={"prompt": "Hello world"},
            request_type=RequestType.AUDIO_GENERATION,
        )

        async def mock_engine_inputs(req):
            return expected

        handler.audio = MagicMock()
        handler.audio.build_engine_inputs = mock_engine_inputs
        inputs = await handler.build_engine_inputs(
            NvCreateAudioSpeechRequest(input="Hello world"),
            RequestType.AUDIO_GENERATION,
        )
        assert inputs.request_type == RequestType.AUDIO_GENERATION
        assert inputs.prompt["prompt"] == "Hello world"


class TestI2VEngineInputs:
    """Tests for image-to-video: multi_modal_data attachment, I2V nvext params, and protocol fields."""

    @pytest.mark.asyncio
    async def test_t2v_no_multi_modal_data_and_i2v_attaches_image(self):
        """T2V has no multi_modal_data; I2V attaches image to prompt."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone", model="test", size="832x480", seconds=2
        )

        # T2V: no image
        t2v = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        assert "multi_modal_data" not in t2v.prompt

        # I2V: image attached
        img = Image.new("RGB", (64, 64), color="red")
        i2v = await handler.build_engine_inputs(
            req, RequestType.VIDEO_GENERATION, image=img
        )
        assert i2v.prompt["multi_modal_data"]["image"] is img

    @pytest.mark.asyncio
    async def test_i2v_nvext_params_on_sampling_params(self):
        """boundary_ratio and guidance_scale_2 are forwarded to sampling params."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="bear",
            model="test",
            size="832x480",
            nvext=VideoNvExt(
                boundary_ratio=0.875, guidance_scale_2=1.0, num_inference_steps=40
            ),
        )
        result = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        sp = result.sampling_params_list[0]
        assert sp.boundary_ratio == 0.875
        assert sp.guidance_scale_2 == 1.0
        assert sp.num_inference_steps == 40

    def test_i2v_protocol_roundtrip(self):
        """VideoNvExt and NvCreateVideoRequest serialize/deserialize I2V fields correctly."""
        req = NvCreateVideoRequest(
            prompt="bear playing",
            model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            input_reference="/tmp/bear.png",
            size="832x480",
            nvext=VideoNvExt(boundary_ratio=0.9, guidance_scale_2=2.0, seed=42),
        )
        data = req.model_dump()
        assert data["input_reference"] == "/tmp/bear.png"
        assert data["nvext"]["boundary_ratio"] == 0.9
        assert data["nvext"]["guidance_scale_2"] == 2.0

        # Defaults are None
        empty = VideoNvExt()
        assert empty.boundary_ratio is None
        assert empty.guidance_scale_2 is None
