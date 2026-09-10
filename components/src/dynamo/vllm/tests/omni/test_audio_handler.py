# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AudioGenerationHandler."""

import asyncio
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


class TestResolveRefAudio:
    """ref_audio is client-supplied, so every failure must be a clean rejection."""

    @staticmethod
    def _wav_bytes(samples=1600, rate=16000):
        import io

        import numpy as np
        import soundfile as sf

        buf = io.BytesIO()
        # Random content so the base64 payload really contains '+' and '/'.
        rng = np.random.default_rng(0)
        sf.write(
            buf, rng.standard_normal(samples).astype("float32"), rate, format="WAV"
        )
        return buf.getvalue()

    @staticmethod
    def _data_uri(payload: bytes) -> str:
        import base64

        return "data:audio/wav;base64," + base64.b64encode(payload).decode()

    def test_decodes_a_valid_data_uri(self):
        handler = _make_audio_handler()
        wav = self._wav_bytes()
        data, rate = asyncio.run(handler._resolve_ref_audio(self._data_uri(wav)))
        assert len(data) == 1600 and rate == 16000

    def test_decodes_a_percent_encoded_payload(self):
        # A data URI that travelled through a URL has '+' and '/' escaped.
        # Permissive base64 silently dropped the '%' and produced wrong bytes.
        import base64
        import urllib.parse

        handler = _make_audio_handler()
        wav = self._wav_bytes()
        encoded = base64.b64encode(wav).decode()
        assert "+" in encoded or "/" in encoded
        quoted = urllib.parse.quote(encoded, safe="=")
        data, rate = asyncio.run(
            handler._resolve_ref_audio(f"data:audio/wav;base64,{quoted}")
        )
        assert len(data) == 1600 and rate == 16000

    @pytest.mark.parametrize(
        "uri, expected",
        [
            ("data:audio/wav", "missing ',' separator"),
            ("data:audio/wav,RIFFraw", "expected base64 payload"),
            ("data:audio/wav;base64,!!not-base64!!", "Malformed base64"),
        ],
    )
    def test_rejects_malformed_data_uris(self, uri, expected):
        handler = _make_audio_handler()
        with pytest.raises(ValueError, match=expected):
            asyncio.run(handler._resolve_ref_audio(uri))

    @pytest.mark.parametrize("payload", [b"", b"abcd"])
    def test_rejects_valid_base64_that_is_not_audio(self, payload):
        # Reaches soundfile, which raises LibsndfileError (a RuntimeError);
        # unguarded that surfaces as a 500 rather than a bad-request error.
        handler = _make_audio_handler()
        with pytest.raises(ValueError, match="not readable audio"):
            asyncio.run(handler._resolve_ref_audio(self._data_uri(payload)))

    def test_rejects_an_oversized_data_uri_before_decoding(self):
        handler = _make_audio_handler(tts_ref_audio_max_bytes=16)
        with pytest.raises(ValueError, match="too large"):
            asyncio.run(handler._resolve_ref_audio(self._data_uri(self._wav_bytes())))

    def test_accepts_a_payload_exactly_at_the_limit(self):
        # The encoded guard must not consume any of the decoded budget: base64
        # expansion and the URI header are not audio bytes.
        wav = self._wav_bytes()
        handler = _make_audio_handler(tts_ref_audio_max_bytes=len(wav))
        data, _ = asyncio.run(handler._resolve_ref_audio(self._data_uri(wav)))
        assert len(data) == 1600

    def test_percent_encoding_does_not_consume_the_decoded_budget(self):
        # Percent escapes cost 3 URI characters per base64 character, so a
        # naive length estimate rejects a payload whose decoded size fits.
        import base64
        import urllib.parse

        wav = self._wav_bytes()
        quoted = urllib.parse.quote(base64.b64encode(wav).decode(), safe="=")
        assert len(quoted) > len(wav) * 4 // 3  # really is inflated
        handler = _make_audio_handler(tts_ref_audio_max_bytes=len(wav))
        data, _ = asyncio.run(
            handler._resolve_ref_audio(f"data:audio/wav;base64,{quoted}")
        )
        assert len(data) == 1600

    def test_rejects_an_unsupported_scheme(self):
        handler = _make_audio_handler()
        with pytest.raises(ValueError, match="must be a URL"):
            asyncio.run(handler._resolve_ref_audio("ftp://example.com/a.wav"))
