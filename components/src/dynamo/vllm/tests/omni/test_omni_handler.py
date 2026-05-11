# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

try:
    from PIL import Image
    from vllm.sampling_params import SamplingParams
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
    from dynamo.common.protocols.image_protocol import NvCreateImageRequest
    from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
    from dynamo.common.utils.output_modalities import RequestType
    from dynamo.vllm.omni.audio_handler import AudioGenerationHandler
    from dynamo.vllm.omni.omni_handler import EngineInputs, OmniHandler
    from dynamo.vllm.omni.utils import build_original_prompt, parse_omni_request
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_handler(stage_types=("diffusion",)):
    with patch(
        "dynamo.vllm.omni.omni_handler.BaseOmniHandler.__init__", return_value=None
    ):
        handler = OmniHandler.__new__(OmniHandler)

    config = MagicMock()
    config.model = "test-model"
    config.served_model_name = None
    config.output_modalities = ["text"]
    handler.config = config

    defaults = []
    for st in stage_types:
        if st == "diffusion":
            defaults.append(OmniDiffusionSamplingParams())
        else:
            llm_default = MagicMock(spec=SamplingParams)
            llm_default.clone.return_value = SamplingParams()
            defaults.append(llm_default)

    engine_client = MagicMock()
    engine_client.default_sampling_params_list = defaults
    engine_client.engine.get_stage_metadata.side_effect = lambda i: {
        "stage_type": stage_types[i]
    }
    handler.engine_client = engine_client
    return handler


class TestEngineInputs:
    def test_defaults(self):
        """EngineInputs uses CHAT_COMPLETION, fps=0, and None optionals by default."""
        ei = EngineInputs(prompt={"prompt": "hello"})
        assert ei.request_type == RequestType.CHAT_COMPLETION
        assert ei.fps == 0
        assert ei.sampling_params_list is None
        assert ei.response_format is None
        assert ei.output_format is None


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


class TestBuildSamplingParamsList:
    def test_single_diffusion_stage(self):
        handler = _make_handler(stage_types=("diffusion",))
        sp = OmniDiffusionSamplingParams(height=512, width=512)
        result = handler._build_sampling_params_list(sp)
        assert len(result) == 1
        assert result[0] is sp

    def test_llm_then_diffusion(self):
        handler = _make_handler(stage_types=("llm", "diffusion"))
        sp = OmniDiffusionSamplingParams(height=512, width=512)
        result = handler._build_sampling_params_list(sp)
        assert len(result) == 2
        assert isinstance(result[0], SamplingParams)
        assert result[1] is sp

    def test_fallback_when_defaults_empty(self):
        handler = _make_handler()
        handler.engine_client.default_sampling_params_list = []
        sp = OmniDiffusionSamplingParams(height=512, width=512)
        result = handler._build_sampling_params_list(sp)
        assert result == [sp]

    def test_llm_default_is_cloned(self):
        handler = _make_handler(stage_types=("llm", "diffusion"))
        sp = OmniDiffusionSamplingParams()
        handler._build_sampling_params_list(sp)
        handler.engine_client.default_sampling_params_list[0].clone.assert_called_once()


class TestBuildOriginalPrompt:
    """build_original_prompt only carries prompt/negative_prompt/multi_modal_data.

    height/width/num_inference_steps live in OmniDiffusionSamplingParams, not the prompt.
    """

    def test_basic_fields(self):
        result = build_original_prompt(
            {"prompt": "a cat"}, nvext={}, height=512, width=512
        )
        assert result["prompt"] == "a cat"
        assert result.get("negative_prompt") is None
        assert "height" not in result
        assert "width" not in result

    def test_negative_prompt_from_request(self):
        result = build_original_prompt(
            {"prompt": "a cat", "negative_prompt": "blurry"},
            nvext={"negative_prompt": "ignored"},
            height=1024,
            width=1024,
        )
        assert result["negative_prompt"] == "blurry"

    def test_multi_modal_data_forwarded(self):
        img = object()
        result = build_original_prompt(
            {"prompt": "x", "multi_modal_data": {"image": img}},
            nvext={},
            height=512,
            width=512,
        )
        assert result["multi_modal_data"]["image"] is img

    def test_no_inference_steps_or_guidance(self):
        result = build_original_prompt(
            {"prompt": "x"},
            nvext={"num_inference_steps": 50, "guidance_scale": 7.5},
            height=512,
            width=512,
        )
        assert "num_inference_steps" not in result
        assert "guidance_scale" not in result


class TestParseOmniRequest:
    """parse_omni_request: original_prompt only has prompt/negative_prompt,
    geometry goes into sampling_params_list dict."""

    @pytest.mark.asyncio
    async def test_image_sampling_params_has_geometry(self):
        request = {
            "prompt": "a sunset",
            "size": "512x512",
            "output_modalities": ["image"],
        }
        result = await parse_omni_request(request, ["image"])
        sp = result["sampling_params_list"]
        assert sp["height"] == 512
        assert sp["width"] == 512

    @pytest.mark.asyncio
    async def test_image_original_prompt_no_geometry(self):
        request = {
            "prompt": "a sunset",
            "size": "512x512",
            "output_modalities": ["image"],
        }
        result = await parse_omni_request(request, ["image"])
        op = result["original_prompt"]
        assert op["prompt"] == "a sunset"
        assert "height" not in op
        assert "width" not in op

    @pytest.mark.asyncio
    async def test_nvext_params_go_into_sampling_params_not_prompt(self):
        request = {
            "prompt": "x",
            "size": "512x512",
            "nvext": {"num_inference_steps": 30, "guidance_scale": 4.0},
        }
        result = await parse_omni_request(request, ["image"])
        sp = result["sampling_params_list"]
        assert sp["num_inference_steps"] == 30
        assert sp["guidance_scale"] == 4.0
        op = result["original_prompt"]
        assert "num_inference_steps" not in op
        assert "guidance_scale" not in op


# ---------------------------------------------------------------------------
# AudioGenerationHandler — data_source / response_format field mapping
# ---------------------------------------------------------------------------


def _make_audio_handler():
    config = MagicMock()
    config.tts_max_instructions_length = 200
    config.tts_max_new_tokens_min = 1
    config.tts_max_new_tokens_max = 4096
    config.tts_ref_audio_timeout = 10
    config.tts_ref_audio_max_bytes = 1024 * 1024
    engine_client = MagicMock()
    engine_client.model_config.hf_config.talker_config = None
    return AudioGenerationHandler(config, engine_client, None, None)


class TestAudioHandlerFieldMapping:
    """AudioGenerationHandler maps data_source→response_format and response_format→output_format."""

    @pytest.mark.asyncio
    async def test_generic_path_maps_data_source_to_response_format(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        req = NvCreateAudioSpeechRequest(
            input="hello", data_source="url", response_format="mp3"
        )
        result = await handler.build_engine_inputs(req)

        assert result.response_format == "url"  # data_source → response_format
        assert result.output_format == "mp3"  # response_format → output_format

    @pytest.mark.asyncio
    async def test_generic_path_maps_data_source_b64_json(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        req = NvCreateAudioSpeechRequest(
            input="hello", data_source="b64_json", response_format="opus"
        )
        result = await handler.build_engine_inputs(req)

        assert result.response_format == "b64_json"
        assert result.output_format == "opus"

    @pytest.mark.asyncio
    async def test_generic_path_no_data_source_passes_none(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        # No data_source → response_format in EngineInputs will be None
        req = NvCreateAudioSpeechRequest(input="hello", response_format="wav")
        result = await handler.build_engine_inputs(req)

        assert result.response_format is None
        assert result.output_format == "wav"

    @pytest.mark.asyncio
    async def test_tts_path_applies_same_field_mapping(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=True)
        handler._validate_tts_request = MagicMock()
        handler._estimate_tts_prompt_len = MagicMock(return_value=10)

        req = NvCreateAudioSpeechRequest(
            input="hi", data_source="url", response_format="flac"
        )
        result = await handler.build_engine_inputs(req)

        assert result.response_format == "url"
        assert result.output_format == "flac"

    @pytest.mark.asyncio
    async def test_request_type_is_audio_generation(self):
        handler = _make_audio_handler()
        handler._is_tts_model = MagicMock(return_value=False)

        result = await handler.build_engine_inputs(
            NvCreateAudioSpeechRequest(input="hi")
        )
        assert result.request_type == RequestType.AUDIO_GENERATION
