# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dynamo.common.lora.manager import LoRAInfo

try:
    from PIL import Image
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
    from dynamo.common.protocols.image_protocol import NvCreateImageRequest
    from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
    from dynamo.common.utils.output_modalities import RequestType
    from dynamo.llm.exceptions import InvalidArgument
    from dynamo.vllm.lora_state import LoRAState
    from dynamo.vllm.omni.audio_handler import AudioGenerationHandler
    from dynamo.vllm.omni.main import _register_lora_engine_routes
    from dynamo.vllm.omni.omni_handler import EngineInputs, OmniHandler
    from dynamo.vllm.omni.utils import (
        MAX_IMAGE_DIMENSION,
        build_original_prompt,
        image_generation_size_from_request,
        parse_omni_request,
        streaming_sampling_params,
    )
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
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
    config.enable_lora = False  # Disable LoRA for tests unless explicitly set
    config.engine_args = SimpleNamespace(enable_lora=False)
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
    engine_client.engine.get_stage_metadata.side_effect = lambda i: SimpleNamespace(
        stage_type=stage_types[i]
    )
    handler.engine_client = engine_client

    # BaseOmniHandler.__init__ is mocked out in tests; recreate LoRA state attrs
    # expected by BaseWorkerHandler helpers called by OmniHandler.
    handler._lora_state = LoRAState()
    handler.loaded_loras = handler._lora_state.loaded_loras
    handler._lora_load_locks = handler._lora_state.lora_load_locks
    handler._lora_load_locks_guard = handler._lora_state.lora_load_locks_guard
    handler._lora_capacity = None
    handler._lora_capacity_guard = (
        asyncio.Lock()
    )  # Shared capacity guard for concurrent loads
    handler._engine_loaded_loras = set()

    # Add attributes required by _resolve_lora_request() (called by _resolve_and_apply_lora)
    handler._served_model_name = config.served_model_name or config.model
    handler._served_model_aliases = tuple(
        getattr(config, "served_model_aliases", ()) or ()
    )
    handler.engine_args = SimpleNamespace(model=config.model)

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
        assert ei.stream_audio is False


def test_streaming_sampling_params_preserves_request_overrides():
    engine_client = SimpleNamespace(
        default_sampling_params_list=[SamplingParams(temperature=0.1, max_tokens=8)]
    )
    requested = SamplingParams(temperature=0.7, max_tokens=42, top_p=0.8)

    result = streaming_sampling_params(engine_client, [requested])

    assert result[0].temperature == 0.7
    assert result[0].max_tokens == 42
    assert result[0].top_p == 0.8
    assert result[0].output_kind == RequestOutputKind.DELTA
    assert requested.output_kind == RequestOutputKind.CUMULATIVE


def test_streaming_sampling_params_preserves_explicit_empty_list():
    engine_client = SimpleNamespace(
        default_sampling_params_list=[SamplingParams(temperature=0.1)]
    )

    result = streaming_sampling_params(engine_client, [])

    assert result == []


def test_streaming_sampling_params_preserves_empty_engine_defaults():
    engine_client = SimpleNamespace(default_sampling_params_list=[])

    result = streaming_sampling_params(engine_client)

    assert result == []


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
        assert inputs.prompt["modalities"] == ["image"]
        assert inputs.prompt["mm_processor_kwargs"] == {
            "target_h": 512,
            "target_w": 512,
        }
        assert len(inputs.sampling_params_list) == 1
        sp = inputs.sampling_params_list[0]
        assert sp.height == 512
        assert sp.width == 512

    @pytest.mark.asyncio
    async def test_image_chat_completion_uses_multimodal_prompt(self):
        """Image chat requests must use vLLM-Omni multimodal preprocessing."""
        handler = _make_handler(stage_types=("llm", "diffusion"))
        handler.config.output_modalities = ["image"]
        raw = {
            "messages": [{"role": "user", "content": "a glass teapot"}],
            "extra_body": {"height": 768, "width": 512, "seed": 123},
        }

        inputs = await handler.build_engine_inputs(raw, RequestType.CHAT_COMPLETION)

        assert inputs.request_type == RequestType.CHAT_COMPLETION
        assert inputs.prompt["prompt"] == "a glass teapot"
        assert inputs.prompt["modalities"] == ["image"]
        assert inputs.prompt["mm_processor_kwargs"] == {
            "target_h": 768,
            "target_w": 512,
        }
        assert len(inputs.sampling_params_list) == 2
        sp = inputs.sampling_params_list[1]
        assert sp.height == 768
        assert sp.width == 512

    @pytest.mark.asyncio
    async def test_video_generation(self):
        """Video request parses prompt, size, seconds, and sets fps."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a drone", model="test-model", size="832x480", seconds=2
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
            prompt="a drone", model="test-model", size="832x480", seconds=2
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
            model="test-model",
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

    async def test_media_passthrough_reaches_sampling_params(self):
        """A top-level SDK extra_body field, nested by the frontend under
        extra_args["media_passthrough"], rides sampling params extra_args to
        the engine. Nothing is set on the sampling params by attribute name."""
        handler = _make_handler()
        req = NvCreateVideoRequest(
            prompt="a cat by the sea",
            model="test-model",
            size="832x480",
            extra_args={
                "media_passthrough": {
                    "backend_custom_knob": 0.5,
                    "denoise_strength": 0.8,
                }
            },
        )
        result = await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)
        sp = result.sampling_params_list[0]
        assert sp.extra_args["backend_custom_knob"] == 0.5
        assert sp.extra_args["denoise_strength"] == 0.8

    @pytest.mark.parametrize(
        "bad_knob",
        [
            {"frame_interpolation_model_path": "attacker/repository"},
            {"guardrails": False},
            {"safety_checker": None},
            {"vae_checkpoint_url": "http://evil/x.pkl"},
        ],
    )
    async def test_media_passthrough_rejects_load_and_policy_knobs(self, bad_knob):
        """A path/checkpoint field or a policy control is refused while the
        request is being built, before any engine call, so it cannot reach a
        model load or a guardrail switch."""
        handler = _make_handler()
        handler.engine_client.generate = MagicMock(
            side_effect=AssertionError("engine must not run for a rejected request")
        )
        req = NvCreateVideoRequest(
            prompt="a cat",
            model="test-model",
            size="832x480",
            extra_args={"media_passthrough": bad_knob},
        )
        with pytest.raises(ValueError):
            await handler.build_engine_inputs(req, RequestType.VIDEO_GENERATION)

    def test_media_passthrough_absent_is_a_no_op(self):
        req = NvCreateVideoRequest(prompt="a cat", model="test-model")
        assert req.extra_args is None
        handler = _make_handler()
        inputs = handler._engine_inputs_from_video(req)
        assert inputs.sampling_params_list is not None

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


class TestLoraEngineRouteRegistration:
    @pytest.mark.asyncio
    async def test_register_lora_engine_routes_dispatches_each_handler(self):
        runtime = MagicMock()
        handler = SimpleNamespace(
            load_lora=MagicMock(),
            unload_lora=MagicMock(),
            list_loras=MagicMock(),
        )

        async def _yield_once(payload):
            yield {"status": "ok", "payload": payload}

        handler.load_lora.side_effect = _yield_once
        handler.unload_lora.side_effect = _yield_once
        handler.list_loras.side_effect = _yield_once

        _register_lora_engine_routes(runtime, handler)

        registered = {
            call.args[0]: call.args[1]
            for call in runtime.register_engine_route.call_args_list
        }

        # Routes use the update/ engine-management prefix.
        assert set(registered) == {
            "update/load_lora",
            "update/unload_lora",
            "update/list_loras",
        }

        body = {"lora_name": "adapterA"}
        assert await registered["update/load_lora"](body) == {
            "status": "ok",
            "payload": body,
        }
        assert await registered["update/unload_lora"](body) == {
            "status": "ok",
            "payload": body,
        }
        assert await registered["update/list_loras"](body) == {
            "status": "ok",
            "payload": body,
        }


class TestLoraRequestParsing:
    def test_extract_lora_name_accepts_delete_route_alias_shape(self):
        handler = _make_handler()

        assert (
            handler._extract_lora_name_from_request({"name": "adapterA"}) == "adapterA"
        )
        assert (
            handler._extract_lora_name_from_request({"adapter_name": "adapterA"})
            == "adapterA"
        )
        assert (
            handler._extract_lora_name_from_request({"model": "adapterA"}) == "adapterA"
        )


class TestLoraEnablement:
    def test_resolve_lora_request_unknown_adapter_raises_when_enabled(self):
        handler = _make_handler()
        handler.config.engine_args.enable_lora = True

        with patch(
            "dynamo.vllm.omni.omni_handler.get_lora_manager",
            return_value=MagicMock(),
        ):
            with pytest.raises(ValueError, match="unknown model or LoRA adapter"):
                handler._resolve_lora_request("ghost-adapter")

    def test_resolve_lora_request_unknown_adapter_is_none_when_manager_missing(self):
        handler = _make_handler()
        handler.config.engine_args.enable_lora = True

        with patch("dynamo.vllm.omni.omni_handler.get_lora_manager", return_value=None):
            assert handler._resolve_lora_request("ghost-adapter") is None

    def test_resolve_lora_request_served_alias_is_treated_as_base_model(self):
        handler = _make_handler()
        handler.config.engine_args.enable_lora = True
        handler.config.served_model_aliases = ["test-model-alias"]
        handler._served_model_aliases = tuple(handler.config.served_model_aliases)

        with patch(
            "dynamo.vllm.omni.omni_handler.get_lora_manager",
            return_value=MagicMock(),
        ):
            assert handler._resolve_lora_request("test-model-alias") is None


class TestLoraCapacity:
    def test_resolve_lora_capacity_uses_configured_max_loras(self):
        """Omni Base handler should defer LoRA capacity to configured max_loras."""
        from dynamo.vllm.omni.base_handler import BaseOmniHandler

        handler = BaseOmniHandler.__new__(BaseOmniHandler)
        config = SimpleNamespace(
            engine_args=SimpleNamespace(enable_lora=True, max_loras=4)
        )

        assert handler._resolve_lora_capacity(config) == 4

    @pytest.mark.asyncio
    async def test_second_distinct_adapter_load_is_rejected_at_capacity_one(self):
        handler = _make_handler()
        handler._lora_capacity = 1
        handler._lora_state.loaded_loras = {
            "adapterA": LoRAInfo(id=123, path="/cache/adapterA")
        }
        handler.loaded_loras = handler._lora_state.loaded_loras

        results = [
            result
            async for result in handler.load_lora(
                {"lora_name": "adapterB", "source": {"uri": "file:///adapter-b"}}
            )
        ]

        assert results[-1]["status"] == "error"
        assert "LoRA capacity exceeded" in results[-1]["message"]
        handler.engine_client.add_lora.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_adapter_still_rejected_at_capacity_when_hot_swap_enabled(self):
        handler = _make_handler()
        handler._lora_capacity = 1
        handler._lora_state.loaded_loras = {
            "adapterA": LoRAInfo(id=123, path="/cache/adapterA")
        }
        handler.loaded_loras = handler._lora_state.loaded_loras

        with patch.dict("os.environ", {"DYN_LORA_HOTSWAP_ENABLED": "true"}):
            results = [
                result
                async for result in handler.load_lora(
                    {"lora_name": "adapterB", "source": {"uri": "file:///adapter-b"}}
                )
            ]

        assert results[-1]["status"] == "error"
        assert "LoRA capacity exceeded" in results[-1]["message"]
        handler.engine_client.add_lora.assert_not_called()


class TestDiffusionLoraAttachment:
    def test_apply_lora_to_diffusion_sampling_params_sets_lora_request(self):
        diffusion_sp = OmniDiffusionSamplingParams()
        llm_sp = SamplingParams()
        lora_request = MagicMock()

        OmniHandler._apply_lora_to_sampling_params(
            [llm_sp, diffusion_sp],
            lora_request,
        )

        assert diffusion_sp.lora_request is lora_request

    def test_apply_lora_to_diffusion_sampling_params_raises_when_attr_missing(self):
        class _NoLoraRequestSamplingParams(OmniDiffusionSamplingParams):
            def __setattr__(self, name, value):
                if name == "lora_request" and value is not None:
                    raise AttributeError("lora_request is not supported")
                super().__setattr__(name, value)

        diffusion_sp = _NoLoraRequestSamplingParams()
        lora_request = MagicMock()

        with pytest.raises(
            RuntimeError,
            match="OmniDiffusionSamplingParams no longer exposes 'lora_request'",
        ):
            OmniHandler._apply_lora_to_sampling_params([diffusion_sp], lora_request)


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
    """parse_omni_request: image geometry goes into sampling params and processor kwargs."""

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
    async def test_image_prompt_uses_multimodal_preprocessor_kwargs(self):
        request = {
            "prompt": "a sunset",
            "size": "512x512",
            "output_modalities": ["image"],
        }
        result = await parse_omni_request(request, ["image"])
        prompt = result["engine_inputs"]
        assert prompt["prompt"] == "a sunset"
        assert prompt["modalities"] == ["image"]
        assert prompt["mm_processor_kwargs"] == {"target_h": 512, "target_w": 512}

        op = result["original_prompt"]
        assert op["prompt"] == "a sunset"
        assert "height" not in op
        assert "width" not in op
        assert op["modalities"] == ["image"]
        assert op["mm_processor_kwargs"] == {"target_h": 512, "target_w": 512}

    def test_image_request_uses_nvext_negative_prompt(self):
        request = {
            "prompt": "a red apple",
            "size": "1024x1024",
            "nvext": {"negative_prompt": "blurry, low quality"},
        }

        result = asyncio.run(parse_omni_request(request, ["image"]))

        assert result["engine_inputs"]["negative_prompt"] == "blurry, low quality"
        assert result["original_prompt"]["negative_prompt"] == "blurry, low quality"

    def test_image_request_uses_nvext_dimensions_consistently(self):
        request = {
            "prompt": "a red apple",
            "size": "512x512",
            "nvext": {"height": 640, "width": 768},
        }

        result = asyncio.run(parse_omni_request(request, ["image"]))

        assert result["sampling_params_list"]["height"] == 640
        assert result["sampling_params_list"]["width"] == 768
        assert result["engine_inputs"]["mm_processor_kwargs"] == {
            "target_h": 640,
            "target_w": 768,
        }
        assert result["original_prompt"]["mm_processor_kwargs"] == {
            "target_h": 640,
            "target_w": 768,
        }

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

    @pytest.mark.asyncio
    async def test_image_chat_request_uses_multimodal_preprocessor_kwargs(self):
        request = {
            "messages": [{"role": "user", "content": "a glass teapot"}],
            "extra_body": {"height": 768, "width": 512, "guidance_scale": 1.5},
        }

        result = await parse_omni_request(request, ["image"])

        prompt = result["engine_inputs"]
        assert prompt["prompt"] == "a glass teapot"
        assert prompt["modalities"] == ["image"]
        assert prompt["mm_processor_kwargs"] == {"target_h": 768, "target_w": 512}
        assert result["original_prompt"] == prompt
        assert result["sampling_params_list"] == {
            "height": 768,
            "width": 512,
            "guidance_scale": 1.5,
        }

    @pytest.mark.parametrize("bad", ["abc", [1, 2], {"w": 1}, 1.5, True])
    def test_nvext_dimensions_reject_non_integers(self, bad):
        # nvext is applied after image_generation_size_from_request and wins, so
        # it needs its own bound or it reopens every case that helper rejects.
        request = {"prompt": "x", "size": "512x512", "nvext": {"width": bad}}
        with pytest.raises(ValueError, match=r"nvext\.width must be an integer"):
            asyncio.run(parse_omni_request(request, ["image"]))

    @pytest.mark.parametrize("bad", [0, -1, MAX_IMAGE_DIMENSION + 1])
    def test_nvext_dimensions_reject_out_of_range(self, bad):
        request = {"prompt": "x", "size": "512x512", "nvext": {"height": bad}}
        with pytest.raises(ValueError, match=r"nvext\.height must be between"):
            asyncio.run(parse_omni_request(request, ["image"]))

    def test_nvext_overrides_an_out_of_range_size(self):
        # nvext is the highest-priority source, so the size it replaces never
        # reaches the engine and must not be validated on the way past.
        request = {
            "prompt": "x",
            "size": "99999x99999",
            "nvext": {"width": 512, "height": 512},
        }
        result = asyncio.run(parse_omni_request(request, ["image"]))
        sp = result["sampling_params_list"]
        assert (sp["width"], sp["height"]) == (512, 512)
        assert result["engine_inputs"]["mm_processor_kwargs"] == {
            "target_h": 512,
            "target_w": 512,
        }

    def test_nvext_partial_override_still_validates_the_surviving_size(self):
        # Only width is replaced, so the height that survives from `size` is
        # still the value that reaches the engine, and still has to be bounded.
        request = {"prompt": "x", "size": "512x99999", "nvext": {"width": 512}}
        with pytest.raises(ValueError, match=r"height in size='512x99999'"):
            asyncio.run(parse_omni_request(request, ["image"]))


class TestImageGenerationSizeValidation:
    """Client-supplied image dimensions are bounded wherever they enter."""

    @pytest.mark.parametrize("bad", ["not-a-number", [1], {"w": 1}, 1.5, True])
    def test_rejects_non_integer_width(self, bad):
        with pytest.raises(ValueError, match="width must be an integer"):
            image_generation_size_from_request({"width": bad})

    @pytest.mark.parametrize("bad", [0, -1, MAX_IMAGE_DIMENSION + 1])
    def test_rejects_out_of_range_width(self, bad):
        with pytest.raises(ValueError, match="width must be between"):
            image_generation_size_from_request({"width": bad})

    @pytest.mark.parametrize("size", ["0x0", "-1x-1", "8192x8192"])
    def test_rejects_out_of_range_size(self, size):
        # The message must name ``size``: the client never sent ``width`` and
        # would have no field to correct.
        with pytest.raises(ValueError, match=r"width in size='"):
            image_generation_size_from_request({"size": size})

    def test_unparseable_size_still_falls_back_to_defaults(self):
        # parse_size's documented contract: only what it does parse is bounded.
        assert image_generation_size_from_request({"size": "not-a-size"}) == (
            1024,
            1024,
        )

    def test_explicit_width_overrides_an_out_of_range_size(self):
        # The size value is discarded, so it must not be validated on its way out.
        request = {"size": "99999x99999", "width": 512, "height": 512}
        assert image_generation_size_from_request(request) == (512, 512)

    def test_a_discarded_extra_body_width_is_not_validated(self):
        # Same rule one level down: the top-level field wins over extra_body, so
        # the extra_body value never reaches the engine and must not fail the
        # request. Only the value that survives the precedence chain is checked.
        request = {"extra_body": {"width": "abc"}, "width": 512}
        assert image_generation_size_from_request(request) == (512, 1024)

    def test_extra_body_width_still_applies_when_not_overridden(self):
        request = {"extra_body": {"width": 100, "height": 100}}
        assert image_generation_size_from_request(request) == (100, 100)

    def test_accepts_the_maximum(self):
        maximum = f"{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
        assert image_generation_size_from_request({"size": maximum}) == (
            MAX_IMAGE_DIMENSION,
            MAX_IMAGE_DIMENSION,
        )

    def test_a_long_size_is_truncated_in_the_error(self):
        # size is unbounded client input and this message reaches both the
        # caller and the handler's log line, so it must not echo all of it.
        long_size = "9" * 100 + "x1"
        with pytest.raises(ValueError) as excinfo:
            image_generation_size_from_request({"size": long_size})
        message = str(excinfo.value)
        assert long_size not in message
        assert "..." in message and len(message) < 120

    def test_non_string_size_falls_back_to_defaults(self):
        # parse_size tolerates a non-string size; the error label must too.
        assert image_generation_size_from_request({"size": 1024}) == (1024, 1024)


class TestImageEndpointSizeValidation:
    """/v1/images/generations takes the same bound as the chat path."""

    def test_rejects_out_of_range_size(self):
        handler = _make_handler()
        req = NvCreateImageRequest(prompt="x", size="99999x99999")
        with pytest.raises(ValueError, match=r"width in size='99999x99999'"):
            handler._engine_inputs_from_image(req)

    def test_accepts_a_supported_size(self):
        handler = _make_handler()
        req = NvCreateImageRequest(prompt="x", size="1024x768")
        inputs = handler._engine_inputs_from_image(req)
        assert inputs.prompt["mm_processor_kwargs"] == {
            "target_h": 768,
            "target_w": 1024,
        }

    @pytest.mark.asyncio
    async def test_rejection_propagates_instead_of_yielding_a_chat_chunk(self):
        """The images route has no failure shape, so a rejection must not be
        yielded as a chat.completion.chunk. It leaves the handler as
        InvalidArgument, the registered binding exception the HTTP layer answers
        with a 400."""
        handler = _make_handler()
        handler.config.output_modalities = ["image"]
        request = {"prompt": "x", "size": "99999x99999"}

        with pytest.raises(InvalidArgument) as excinfo:
            async for _ in handler._generate_openai_mode(request, None, "req-1"):
                pass

        # The client-facing message is the reason alone: errors.rs reads a
        # registered exception's .value(py).str(), so no "ValueError: " prefix.
        assert str(excinfo.value) == (
            "width in size='99999x99999' must be between 1 and 4096"
        )


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
