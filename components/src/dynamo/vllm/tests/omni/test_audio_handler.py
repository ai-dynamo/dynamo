# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AudioGenerationHandler."""

from unittest.mock import MagicMock

import pytest

try:
    from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
    from dynamo.common.utils.output_modalities import RequestType
    from dynamo.vllm.omni.audio_handler import AudioGenerationHandler
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
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


class TestEngineInputsFromAudio:
    """Tests for build_engine_inputs."""

    @pytest.mark.asyncio
    async def test_generic_path_for_non_tts(self):
        """Non-TTS model gets plain text prompt."""
        handler = _make_audio_handler()
        stage = MagicMock()
        stage.model_stage = "diffusion"
        handler.engine_client.stage_list = [stage]

        req = NvCreateAudioSpeechRequest(input="Hello world")
        inputs = await handler.build_engine_inputs(req)
        assert inputs.request_type == RequestType.AUDIO_GENERATION
        assert inputs.prompt["prompt"] == "Hello world"
        assert inputs.sampling_params_list is None

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
        req = NvCreateAudioSpeechRequest(input="hello", speed=2.0)
        inputs = await handler.build_engine_inputs(req)
        assert inputs.speed == 2.0


class TestExtractAudioTensor:
    """Tests for _extract_audio_tensor."""

    def test_extracts_from_audio_key(self):
        import numpy as np

        handler = _make_audio_handler()
        mm = {"audio": np.array([0.1, -0.2, 0.3], dtype=np.float32), "sr": 24000}
        audio_np, sr = handler._extract_audio_tensor(mm)
        assert sr == 24000
        assert len(audio_np) == 3

    def test_extracts_from_model_outputs_key(self):
        import numpy as np

        handler = _make_audio_handler()
        mm = {
            "model_outputs": np.array([0.5, -0.5], dtype=np.float32),
            "sr": 16000,
        }
        audio_np, sr = handler._extract_audio_tensor(mm)
        assert sr == 16000
        assert len(audio_np) == 2

    def test_missing_audio_raises(self):
        handler = _make_audio_handler()
        with pytest.raises(ValueError, match="No audio data"):
            handler._extract_audio_tensor({"sr": 24000})

    def test_squeezes_extra_dims(self):
        import numpy as np

        handler = _make_audio_handler()
        mm = {"audio": np.array([[0.1, 0.2, 0.3]], dtype=np.float32), "sr": 24000}
        audio_np, _ = handler._extract_audio_tensor(mm)
        assert audio_np.ndim == 1


class TestEncodeAudio:
    """Tests for _encode_audio."""

    def test_wav_encoding(self):
        import numpy as np

        handler = _make_audio_handler()
        audio = np.zeros(2400, dtype=np.float32)
        audio_bytes, media_type = handler._encode_audio(audio, 24000, "wav")
        assert media_type == "audio/wav"
        assert len(audio_bytes) > 0
        assert audio_bytes[:4] == b"RIFF"  # WAV header

    def test_unsupported_format_falls_back_to_wav(self):
        import numpy as np

        handler = _make_audio_handler()
        audio = np.zeros(100, dtype=np.float32)
        _, media_type = handler._encode_audio(audio, 24000, "xyz")
        assert media_type == "audio/wav"

    def test_default_format_is_wav(self):
        import numpy as np

        handler = _make_audio_handler()
        audio = np.zeros(100, dtype=np.float32)
        _, media_type = handler._encode_audio(audio, 24000)
        assert media_type == "audio/wav"


class TestFormatAudioChunk:
    """Tests for format_output."""

    @pytest.mark.asyncio
    async def test_empty_mm_output_returns_error(self):
        handler = _make_audio_handler()
        result = await handler.format_output({}, "req-1")
        assert result["status"] == "failed"
        assert "No audio generated" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        import numpy as np

        handler = _make_audio_handler()
        mm = {"audio": np.random.randn(4800).astype(np.float32), "sr": 24000}
        result = await handler.format_output(mm, "req-1")
        assert result["status"] == "completed"
        assert result["object"] == "audio.speech"
        assert len(result["data"]) == 1
        assert result["data"][0]["b64_json"] is not None
