# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AudioGenerationHandler."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from dynamo.common.protocols.audio_protocol import (
        AudioNvExt,
        NvCreateAudioSpeechRequest,
    )
    from dynamo.common.utils.output_modalities import RequestType
    from dynamo.vllm.omni import audio_handler as audio_handler_module
    from dynamo.vllm.omni.audio_handler import AudioGenerationHandler
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

try:
    import vllm_omni.model_executor.models.audex.prompt  # noqa: F401
    import vllm_omni.model_executor.models.audex.tta  # noqa: F401

    _AUDEX_AVAILABLE = True
except ImportError:
    _AUDEX_AVAILABLE = False

# The Audex tests exercise vLLM-Omni's model-owned prompt and RVQ builders,
# which only ship in Audex-capable builds. Skip rather than fail on a build
# without them, matching the optional-dependency guard above.
requires_audex = pytest.mark.skipif(
    not _AUDEX_AVAILABLE,
    reason="vLLM-Omni build has no Audex support (model_executor.models.audex)",
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_audio_handler(**config_overrides):
    """Create an AudioGenerationHandler with mocked dependencies."""
    config = MagicMock()
    config.model = "test-tts-model"
    config.served_model_name = None
    config.tts_max_instructions_length = 500
    config.tts_max_new_tokens_min = 1
    config.tts_max_new_tokens_max = 4096
    config.tts_ref_audio_timeout = 15
    config.tts_ref_audio_max_bytes = 50 * 1024 * 1024
    for k, v in config_overrides.items():
        setattr(config, k, v)

    engine_client = MagicMock()
    engine_client.model_config.hf_config = MagicMock(spec=[])

    handler = AudioGenerationHandler(
        config=config,
        engine_client=engine_client,
        media_output_fs=None,
        media_output_http_url=None,
    )
    return handler


class TestValidateTtsRequest:
    """Tests for _validate_tts_request."""

    @pytest.mark.asyncio
    async def test_empty_input_rejected(self):
        handler = _make_audio_handler()
        req = NvCreateAudioSpeechRequest(input="   ")
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await handler.build_engine_inputs(req)

    def test_invalid_task_type_rejected_by_pydantic(self):
        """Pydantic Literal validation rejects invalid task_type at construction."""
        with pytest.raises(Exception):
            NvCreateAudioSpeechRequest(input="hello", task_type="Banana")

    def test_valid_task_types_accepted(self):
        handler = _make_audio_handler()
        for task in ("CustomVoice", "VoiceDesign", "Base"):
            req = NvCreateAudioSpeechRequest(input="hello", task_type=task)
            if task == "VoiceDesign":
                req.instructions = "cheerful"
            elif task == "Base":
                req.ref_audio = "data:audio/wav;base64,AAAA"
            handler._validate_tts_request(req)

    def test_voice_design_requires_instructions(self):
        handler = _make_audio_handler()
        req = NvCreateAudioSpeechRequest(input="hello", task_type="VoiceDesign")
        with pytest.raises(ValueError, match="instructions"):
            handler._validate_tts_request(req)

    def test_base_requires_ref_audio(self):
        handler = _make_audio_handler()
        req = NvCreateAudioSpeechRequest(input="hello", task_type="Base")
        with pytest.raises(ValueError, match="ref_audio"):
            handler._validate_tts_request(req)

    def test_ref_text_only_for_base(self):
        handler = _make_audio_handler()
        req = NvCreateAudioSpeechRequest(
            input="hello", task_type="CustomVoice", ref_text="foo"
        )
        with pytest.raises(ValueError, match="only valid for Base"):
            handler._validate_tts_request(req)

    def test_instructions_length_enforced(self):
        handler = _make_audio_handler(tts_max_instructions_length=10)
        req = NvCreateAudioSpeechRequest(input="hello", instructions="x" * 11)
        with pytest.raises(ValueError, match="Instructions too long"):
            handler._validate_tts_request(req)

    def test_max_new_tokens_range(self):
        handler = _make_audio_handler()
        req = NvCreateAudioSpeechRequest(input="hello", max_new_tokens=0)
        with pytest.raises(ValueError, match="at least"):
            handler._validate_tts_request(req)

        req = NvCreateAudioSpeechRequest(input="hello", max_new_tokens=99999)
        with pytest.raises(ValueError, match="cannot exceed"):
            handler._validate_tts_request(req)

    def test_invalid_voice_rejected_when_speakers_loaded(self):
        handler = _make_audio_handler()
        handler._tts_supported_speakers = {"vivian", "ryan"}
        req = NvCreateAudioSpeechRequest(input="hello", voice="nonexistent")
        with pytest.raises(ValueError, match="Invalid voice"):
            handler._validate_tts_request(req)

    def test_valid_voice_accepted(self):
        handler = _make_audio_handler()
        handler._tts_supported_speakers = {"vivian", "ryan"}
        req = NvCreateAudioSpeechRequest(input="hello", voice="Vivian")
        handler._validate_tts_request(req)  # Should not raise

    def test_invalid_language_rejected_when_languages_loaded(self):
        handler = _make_audio_handler()
        handler._tts_supported_languages = {"english", "chinese"}
        req = NvCreateAudioSpeechRequest(input="hello", language="Klingon")
        with pytest.raises(ValueError, match="Invalid language"):
            handler._validate_tts_request(req)

    def test_auto_language_always_accepted(self):
        handler = _make_audio_handler()
        handler._tts_supported_languages = {"english"}
        req = NvCreateAudioSpeechRequest(input="hello", language="Auto")
        handler._validate_tts_request(req)  # Should not raise


class TestIsTtsModel:
    """Tests for _is_tts_model detection."""

    def test_qwen3_tts_detected(self):
        handler = _make_audio_handler()
        stage = MagicMock()
        stage.model_stage = "qwen3_tts"
        handler.engine_client.stage_list = [stage]
        assert handler._is_tts_model() is True

    def test_non_tts_model(self):
        handler = _make_audio_handler()
        stage = MagicMock()
        stage.model_stage = "diffusion"
        handler.engine_client.stage_list = [stage]
        assert handler._is_tts_model() is False

    def test_no_stage_list(self):
        handler = _make_audio_handler()
        handler.engine_client.stage_list = None
        assert handler._is_tts_model() is False


def test_tts_prompt_len_uses_prompt_embeds_builder(monkeypatch):
    estimator = MagicMock(return_value=37)
    monkeypatch.setattr(
        audio_handler_module,
        "Qwen3TTSPromptEmbedsBuilder",
        SimpleNamespace(estimate_prompt_len_from_additional_information=estimator),
    )

    handler = _make_audio_handler()
    tokenizer = MagicMock(return_value={"input_ids": [1, 2]})
    handler._tts_tokenizer = tokenizer
    codec_language_id = {"english": 1}
    spk_is_dialect = {"vivian": "english"}
    handler.engine_client.model_config.hf_config = SimpleNamespace(
        talker_config=SimpleNamespace(
            codec_language_id=codec_language_id,
            spk_is_dialect=spk_is_dialect,
        )
    )
    tts_params = {"task_type": ["CustomVoice"], "input": "hello"}

    assert handler._estimate_tts_prompt_len(tts_params) == 37

    kwargs = estimator.call_args.kwargs
    assert kwargs["additional_information"] is tts_params
    assert kwargs["task_type"] == "CustomVoice"
    assert kwargs["codec_language_id"] is codec_language_id
    assert kwargs["spk_is_dialect"] is spk_is_dialect
    assert kwargs["tokenize_prompt"]("hello") == [1, 2]
    tokenizer.assert_called_once_with("hello", padding=False)


def test_tts_prompt_len_falls_back_when_builder_is_unavailable(monkeypatch):
    monkeypatch.setattr(audio_handler_module, "Qwen3TTSPromptEmbedsBuilder", None)

    assert _make_audio_handler()._estimate_tts_prompt_len({}) == 2048


def test_audex_request_fails_cleanly_without_audex_support(monkeypatch):
    """A build without the Audex modules must report a request error.

    RuntimeError is one of the types the handler turns into an error response,
    so the request fails instead of the exception escaping the generator.
    """
    monkeypatch.setattr(audio_handler_module, "audex_prompt", None)

    with pytest.raises(RuntimeError, match="no Audex support"):
        audio_handler_module.AudioGenerationHandler._audex_prompt_builders("audex_tts")


def test_tts_prompt_len_propagates_estimator_errors(monkeypatch):
    estimator = MagicMock(side_effect=RuntimeError("estimator failed"))
    monkeypatch.setattr(
        audio_handler_module,
        "Qwen3TTSPromptEmbedsBuilder",
        SimpleNamespace(estimate_prompt_len_from_additional_information=estimator),
    )
    handler = _make_audio_handler()
    handler._tts_tokenizer = MagicMock(return_value={"input_ids": [1, 2]})

    with pytest.raises(RuntimeError, match="estimator failed"):
        handler._estimate_tts_prompt_len({})


class TestEngineInputsFromAudio:
    """Tests for build_engine_inputs."""

    @pytest.mark.asyncio
    async def test_generic_path_for_non_tts(self):
        """Non-TTS model gets plain text prompt."""
        handler = _make_audio_handler()
        stage = MagicMock()
        stage.model_stage = "diffusion"
        handler.engine_client.stage_list = [stage]

        req = NvCreateAudioSpeechRequest(
            input="Hello world",
            nvext=AudioNvExt(frontend_accepts_audio_chunks=True),
        )
        inputs = await handler.build_engine_inputs(req)
        assert inputs.request_type == RequestType.AUDIO_GENERATION
        assert inputs.prompt["prompt"] == "Hello world"
        assert inputs.sampling_params_list is None
        assert inputs.stream_audio is True

    @pytest.mark.asyncio
    async def test_legacy_frontend_gets_complete_response(self):
        """Workers aggregate audio unless the frontend advertises that it accepts chunks."""
        handler = _make_audio_handler()
        handler.engine_client.stage_list = None

        inputs = await handler.build_engine_inputs(
            NvCreateAudioSpeechRequest(input="hello")
        )

        assert inputs.stream_audio is False

    @pytest.mark.asyncio
    async def test_empty_input_rejected(self):
        handler = _make_audio_handler()
        req = NvCreateAudioSpeechRequest(input="  ")
        with pytest.raises(ValueError, match="empty"):
            await handler.build_engine_inputs(req)

    @pytest.mark.asyncio
    async def test_speed_propagated(self):
        """Speed from request is stored in EngineInputs."""
        handler = _make_audio_handler()
        handler.engine_client.stage_list = None  # non-TTS path
        req = NvCreateAudioSpeechRequest(
            input="hello",
            speed=2.0,
            nvext=AudioNvExt(frontend_accepts_audio_chunks=True),
        )
        inputs = await handler.build_engine_inputs(req)
        assert inputs.speed == 2.0
        assert inputs.stream_audio is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_args",
        [
            {"response_format": "mp3"},
            {"response_format": "pcm", "data_source": "url"},
        ],
    )
    async def test_non_streaming_eligibility(self, request_args):
        handler = _make_audio_handler()
        handler.engine_client.stage_list = None

        inputs = await handler.build_engine_inputs(
            NvCreateAudioSpeechRequest(
                input="hello",
                nvext=AudioNvExt(frontend_accepts_audio_chunks=True),
                **request_args,
            )
        )

        assert inputs.stream_audio is False


def _make_audex_handler(*stages, **config_overrides):
    """Audio handler whose engine reports the given Audex ``model_stage`` names."""
    handler = _make_audio_handler(**config_overrides)
    handler.engine_client.stage_list = [
        SimpleNamespace(model_stage=stage) for stage in stages
    ]
    handler.engine_client.stage_configs = []
    handler.engine_client.default_sampling_params_list = [
        SimpleNamespace(max_tokens=2048, temperature=0.1, extra_args=None),
        SimpleNamespace(max_tokens=8192, temperature=0.0, extra_args=None),
    ]
    return handler


class TestAudexModelDetection:
    """Tests for _audex_model_type."""

    def test_tts_pipeline_detected(self):
        """thinker + code2wav is the speech pipeline."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        assert handler._audex_model_type() == "audex"

    def test_tta_pipeline_detected(self):
        """tta_thinker + xcodec is the text-to-audio pipeline."""
        handler = _make_audex_handler("audex_tta_thinker", "audex_xcodec")
        assert handler._audex_model_type() == "audex_tta"

    def test_s2s_pipeline_detected(self):
        """audex_omni is speech-capable only alongside the code2wav decoder."""
        handler = _make_audex_handler("audex_omni", "audex_code2wav")
        assert handler._audex_model_type() == "audex"

    def test_thinker_only_pipeline_is_not_speech(self):
        """The thinker-only deployment is text-final: no speech path."""
        handler = _make_audex_handler("audex_omni")
        assert handler._audex_model_type() is None

    def test_non_audex_pipeline(self):
        """A non-Audex stage name must not take the Audex path."""
        handler = _make_audex_handler("qwen3_tts")
        assert handler._audex_model_type() is None

    def test_stage_configs_shapes(self):
        """model_stage is read from nested engine_args and from dict configs."""
        handler = _make_audio_handler()
        handler.engine_client.stage_list = []
        handler.engine_client.stage_configs = [
            SimpleNamespace(engine_args={"model_stage": "audex_thinker"}),
            {"engine_args": {"model_stage": "audex_code2wav"}},
        ]
        assert handler._audex_model_type() == "audex"


@requires_audex
class TestAudexEngineInputs:
    """Tests for the Audex prompt/param contract."""

    @pytest.mark.asyncio
    async def test_tts_prompt_primes_codec_generation(self):
        """The ChatML prompt must prime <speechgen_start>, not pass raw text.

        A plain text prompt makes the thinker emit a text continuation with
        zero codec tokens, which fails the request downstream.
        """
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="Hello world")
        inputs = await handler.build_engine_inputs(req, request_id="r1")

        prompt = inputs.prompt["prompt"]
        assert prompt != "Hello world"
        assert prompt.endswith("<think></think><speechgen_start>")
        assert "Hello world" in prompt
        assert "<|text to speech|>" in prompt
        assert inputs.request_type == RequestType.AUDIO_GENERATION

    @pytest.mark.asyncio
    async def test_tts_unguided_by_default(self):
        """No cfg_scale means no CFG plumbing (the official TTS baseline)."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="hello")
        inputs = await handler.build_engine_inputs(req, request_id="r1")

        stage0 = inputs.sampling_params_list[0]
        assert "cfg_scale" not in stage0.extra_args
        assert "cfg_pair_id" not in stage0.extra_args
        assert stage0.temperature == 0.1

    @pytest.mark.asyncio
    async def test_guided_request_fails_without_engine_sampling_defaults(self):
        """The defaults are the only channel for the CFG contract.

        Attaching nothing would answer with unguided audio — a different-sounding
        result rather than a reported failure.
        """
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        handler.engine_client.default_sampling_params_list = []
        req = NvCreateAudioSpeechRequest(input="hello", cfg_scale=1.5)

        with pytest.raises(RuntimeError, match="no default_sampling_params_list"):
            await handler.build_engine_inputs(req, request_id="r1")

    @pytest.mark.asyncio
    async def test_unguided_request_needs_no_engine_sampling_defaults(self):
        """Nothing to override, so the engine keeps its own defaults."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        handler.engine_client.default_sampling_params_list = []
        req = NvCreateAudioSpeechRequest(input="hello")

        inputs = await handler.build_engine_inputs(req, request_id="r1")
        assert inputs.sampling_params_list is None

    @pytest.mark.asyncio
    async def test_tts_cfg_scale_one_stays_unguided(self):
        """cfg_scale=1.0 is a no-op scale, so it must not start a CFG pair."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="hello", cfg_scale=1.0)
        inputs = await handler.build_engine_inputs(req, request_id="r1")
        assert "cfg_scale" not in inputs.sampling_params_list[0].extra_args

    @pytest.mark.asyncio
    async def test_tts_cfg_attaches_pair_contract(self, monkeypatch):
        """Guided requests carry the pair id and a length-matched null prompt."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        monkeypatch.setattr(
            handler, "_get_audex_tokenizer", lambda model_type: MagicMock()
        )
        import vllm_omni.model_executor.models.audex.prompt as audex_prompt

        monkeypatch.setattr(audex_prompt, "build_null_prompt", lambda cond, tok: "NULL")

        req = NvCreateAudioSpeechRequest(input="hello", cfg_scale=1.5)
        inputs = await handler.build_engine_inputs(req, request_id="r1")

        stage0 = inputs.sampling_params_list[0]
        assert stage0.extra_args["cfg_scale"] == 1.5
        assert stage0.extra_args["cfg_role"] == "cond"
        assert stage0.extra_args["cfg_pair_id"] == "r1"
        assert stage0.extra_args["cfg_null_prompt"] == "NULL"
        # Guidance sharpens the distribution, so temperature drops.
        assert stage0.temperature == 0.05

    @pytest.mark.asyncio
    async def test_cfg_ignored_without_request_id(self):
        """Without a pair id, decode unguided rather than corrupt a pair."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="hello", cfg_scale=1.5)
        inputs = await handler.build_engine_inputs(req)
        assert "cfg_pair_id" not in inputs.sampling_params_list[0].extra_args

    @pytest.mark.asyncio
    async def test_shared_engine_defaults_not_mutated(self):
        """Per-request CFG state must not leak into the engine's shared defaults."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        shared = handler.engine_client.default_sampling_params_list

        req = NvCreateAudioSpeechRequest(input="hello", max_new_tokens=64)
        inputs = await handler.build_engine_inputs(req, request_id="r1")

        assert inputs.sampling_params_list[0].max_tokens == 64
        assert shared[0].max_tokens == 2048
        assert shared[0].extra_args is None

    @pytest.mark.asyncio
    async def test_tta_prompt_and_rvq_contract(self, monkeypatch):
        """TTA primes <audiogen_start> and always attaches the RVQ phase mask."""
        handler = _make_audex_handler("audex_tta_thinker", "audex_xcodec")
        monkeypatch.setattr(
            handler, "_get_audex_tokenizer", lambda model_type: MagicMock()
        )
        import vllm_omni.model_executor.models.audex.prompt as audex_prompt
        import vllm_omni.model_executor.models.audex.tta as audex_tta

        monkeypatch.setattr(
            audex_tta, "build_tta_phase_token_ids", lambda tok: ([[1], [2]], 10, 11)
        )
        monkeypatch.setattr(
            audex_prompt, "build_tta_null_prompt", lambda cond, tok: "NULL"
        )

        req = NvCreateAudioSpeechRequest(input="a dog barking")
        inputs = await handler.build_engine_inputs(req, request_id="r1")

        assert inputs.prompt["prompt"].endswith("<think></think><audiogen_start>")
        assert "<|text to audio|>" in inputs.prompt["prompt"]

        extra = inputs.sampling_params_list[0].extra_args
        assert extra["tta_rvq"]["start_tid"] == 10
        assert extra["tta_rvq"]["start_in_prompt"] is True
        # TTA guidance is effectively mandatory; the official default is 3.0.
        assert extra["cfg_scale"] == 3.0
        assert extra["cfg_pair_id"] == "r1"


class TestAudexValidation:
    """Audex rejects parameters it cannot honor instead of ignoring them."""

    @requires_audex
    @pytest.mark.asyncio
    @pytest.mark.parametrize("voice", [None, "", "default", "DEFAULT"])
    async def test_default_voice_accepted(self, voice):
        """An omitted or ``default`` voice names the one built-in voice."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="hello", voice=voice)
        inputs = await handler.build_engine_inputs(req, request_id="r1")
        assert inputs.prompt["prompt"].endswith("<speechgen_start>")

    @pytest.mark.asyncio
    async def test_named_voice_rejected(self):
        """Audex has one built-in voice; a named voice must not be silently ignored."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="hello", voice="vivian")
        with pytest.raises(ValueError, match="single built-in voice"):
            await handler.build_engine_inputs(req, request_id="r1")

    @pytest.mark.asyncio
    async def test_tta_voice_rejected(self):
        """Text-to-audio has no voices at all, not even ``default``."""
        handler = _make_audex_handler("audex_tta_thinker", "audex_xcodec")
        req = NvCreateAudioSpeechRequest(input="rain", voice="vivian")
        with pytest.raises(ValueError, match="no voices"):
            await handler.build_engine_inputs(req, request_id="r1")

    @pytest.mark.asyncio
    async def test_ref_audio_rejected(self):
        """Audex cannot clone a reference voice, so ref_audio is an error."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(
            input="hello", ref_audio="data:audio/wav;base64,AAAA"
        )
        with pytest.raises(ValueError, match="reference audio"):
            await handler.build_engine_inputs(req, request_id="r1")

    @pytest.mark.asyncio
    async def test_cfg_scale_out_of_range_rejected(self):
        """cfg_scale outside the supported range fails fast."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="hello", cfg_scale=50.0)
        with pytest.raises(ValueError, match="cfg_scale"):
            await handler.build_engine_inputs(req, request_id="r1")

    @pytest.mark.asyncio
    async def test_max_new_tokens_out_of_range_rejected(self):
        """max_new_tokens above the codec cap fails fast."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="hello", max_new_tokens=99999)
        with pytest.raises(ValueError, match="max_new_tokens"):
            await handler.build_engine_inputs(req, request_id="r1")

    @pytest.mark.asyncio
    async def test_empty_input_rejected(self):
        """Whitespace-only input is rejected before the prompt is built."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(input="   ")
        with pytest.raises(ValueError, match="empty"):
            await handler.build_engine_inputs(req, request_id="r1")

    @requires_audex
    @pytest.mark.asyncio
    async def test_audex_never_streams_chunks(self):
        """Cumulative snapshots must be aggregated, never streamed per payload."""
        handler = _make_audex_handler("audex_thinker", "audex_code2wav")
        req = NvCreateAudioSpeechRequest(
            input="hello",
            nvext=AudioNvExt(frontend_accepts_audio_chunks=True),
        )
        inputs = await handler.build_engine_inputs(req, request_id="r1")
        assert inputs.stream_audio is False
