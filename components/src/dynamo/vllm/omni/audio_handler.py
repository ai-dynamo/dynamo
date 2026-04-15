# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio/TTS handler utilities for the vLLM-Omni backend.

Extracted from omni_handler.py to keep modality-specific logic separate.
OmniHandler holds an instance as ``self.audio`` (composition).
"""

import base64
import logging
from typing import Any, Dict

from vllm_omni.inputs.data import OmniTextPrompt

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.utils.output_modalities import RequestType

logger = logging.getLogger(__name__)

# model_stage names that receive Qwen3-TTS-specific prompt format
# (prompt_token_ids + additional_information). Other audio models
# (MiMo-Audio, Qwen3-Omni, Stable Audio, etc.) use a plain text prompt.
# Mirrors vLLM-Omni's _TTS_MODEL_STAGES in serving_speech.py.
_TTS_MODEL_STAGES: set = {"qwen3_tts"}

# Fallback language set used when model config is unavailable.
_TTS_LANGUAGES_FALLBACK = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}


class AudioGenerationHandler:
    """Handles audio/TTS request processing for the vLLM-Omni backend.

    Instantiated by OmniHandler during initialization and held as a
    composition attribute (``self._audio_handler``).  This keeps
    audio-specific logic (validation, prompt building, encoding) out
    of the orchestrator.
    """

    def __init__(self, config, engine_client, media_output_fs, media_output_http_url):
        self.config = config
        self.engine_client = engine_client
        self.media_output_fs = media_output_fs
        self.media_output_http_url = media_output_http_url
        self._tts_tokenizer: Any = None

        # Cache TTS capabilities from model config at init.
        self._tts_supported_speakers: set = self._load_supported_speakers()
        self._tts_supported_languages: set = self._load_supported_languages()
        if self._tts_supported_speakers:
            logger.info(
                "Loaded %d TTS speakers: %s",
                len(self._tts_supported_speakers),
                sorted(self._tts_supported_speakers),
            )
        if self._tts_supported_languages:
            logger.info(
                "Loaded %d TTS languages: %s",
                len(self._tts_supported_languages),
                sorted(self._tts_supported_languages),
            )

    # -- TTS capability loading from model config -----------------------------

    def _load_supported_speakers(self) -> set:
        """Load supported speakers from model config (case-insensitive).

        Reads ``hf_config.talker_config.spk_id`` or ``speaker_id``,
        matching vLLM-Omni's ``_load_supported_speakers()``.
        """
        try:
            hf_config = self.engine_client.model_config.hf_config
            talker_config = getattr(hf_config, "talker_config", None)
            if talker_config is None:
                return set()
            for attr_name in ("spk_id", "speaker_id"):
                speakers_dict = getattr(talker_config, attr_name, None)
                if speakers_dict and isinstance(speakers_dict, dict):
                    return {s.lower() for s in speakers_dict.keys()}
        except Exception as e:
            logger.warning("Could not load speakers from model config: %s", e)
        return set()

    def _load_supported_languages(self) -> set:
        """Load supported languages from model config.

        Reads ``hf_config.talker_config.codec_language_id``.
        """
        try:
            hf_config = self.engine_client.model_config.hf_config
            talker_config = getattr(hf_config, "talker_config", None)
            if talker_config is None:
                return set()
            lang_dict = getattr(talker_config, "codec_language_id", None)
            if lang_dict and isinstance(lang_dict, dict):
                return {lang.lower() for lang in lang_dict.keys()}
        except Exception as e:
            logger.warning("Could not load languages from model config: %s", e)
        return set()

    # -- TTS model detection --------------------------------------------------

    def _is_tts_model(self) -> bool:
        """Check if the loaded model is a Qwen3-TTS-style model.

        Searches for a TTS model_stage in the engine's stage list,
        stage configs, or model config. Supports multiple vLLM-Omni versions.
        """
        # Try stage_list
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                ms = getattr(stage, "model_stage", None)
                logger.debug("_is_tts_model: stage=%s model_stage=%s", stage, ms)
                if ms in _TTS_MODEL_STAGES:
                    return True

        # Try stage_configs
        stage_configs = getattr(self.engine_client, "stage_configs", None)
        if stage_configs:
            for cfg in stage_configs:
                engine_args = (
                    cfg.get("engine_args", {})
                    if isinstance(cfg, dict)
                    else getattr(cfg, "engine_args", {})
                )
                ms = (
                    engine_args.get("model_stage")
                    if isinstance(engine_args, dict)
                    else getattr(engine_args, "model_stage", None)
                )
                logger.debug("_is_tts_model: stage_config model_stage=%s", ms)
                if ms in _TTS_MODEL_STAGES:
                    return True

        # Try model_config.hf_config.model_type (universal fallback)
        try:
            model_type = self.engine_client.model_config.hf_config.model_type
            logger.debug("_is_tts_model: hf_config.model_type=%s", model_type)
            if model_type in _TTS_MODEL_STAGES:
                return True
        except (AttributeError, TypeError) as e:
            logger.debug("_is_tts_model: hf_config fallback failed: %s", e)

        logger.warning(
            "_is_tts_model: could not detect TTS model. "
            "stage_list=%s, stage_configs=%s",
            stage_list is not None,
            stage_configs is not None,
        )
        return False

    # -- Audio engine input construction --------------------------------------

    async def build_engine_inputs(self, req: NvCreateAudioSpeechRequest):
        """Build engine inputs for an audio/TTS request.

        Two code paths (matching vLLM-Omni serving_speech.py):

        * **TTS path** (Qwen3-TTS): ``prompt_token_ids`` +
          ``additional_information``.
        * **Generic audio path** (MiMo-Audio, etc.): plain text prompt.
        """
        # Import here to avoid circular dependency
        from dynamo.vllm.omni.omni_handler import EngineInputs

        if not req.input or not req.input.strip():
            raise ValueError("Input text cannot be empty")

        if self._is_tts_model():
            return await self._engine_inputs_tts(req)

        # Generic audio model – plain text prompt (same as image/video)
        prompt = OmniTextPrompt(prompt=req.input)
        logger.info(f"Audio request (generic): input='{req.input[:50]}...'")
        return EngineInputs(
            prompt=prompt,
            sampling_params_list=None,
            request_type=RequestType.AUDIO_GENERATION,
            response_format=req.response_format,
            speed=req.speed or 1.0,
        )

    # -- Qwen3-TTS-specific helpers -------------------------------------------

    async def _engine_inputs_tts(self, req: NvCreateAudioSpeechRequest):
        """Build engine inputs for Qwen3-TTS models."""
        from dynamo.vllm.omni.omni_handler import EngineInputs

        self._validate_tts_request(req)

        if req.voice is not None:
            req.voice = req.voice.lower()

        task_type = req.task_type or "CustomVoice"

        tts_params: Dict[str, Any] = {
            "text": [req.input],
            "task_type": [task_type],
            "language": [req.language or "Auto"],
            "instruct": [req.instructions or ""],
            "max_new_tokens": [req.max_new_tokens or 2048],
        }

        if req.voice is not None:
            tts_params["speaker"] = [req.voice]
        elif task_type == "CustomVoice":
            tts_params["speaker"] = ["Vivian"]

        if req.ref_audio is not None:
            wav_list, sr = await self._resolve_ref_audio(req.ref_audio)
            tts_params["ref_audio"] = [[wav_list, sr]]
        if req.ref_text is not None:
            tts_params["ref_text"] = [req.ref_text]

        if task_type == "VoiceDesign":
            tts_params["non_streaming_mode"] = [True]

        estimated_len = self._estimate_tts_prompt_len(tts_params)

        prompt = {
            "prompt_token_ids": [1] * estimated_len,
            "additional_information": tts_params,
        }

        logger.info(
            f"Audio TTS request: input='{req.input[:50]}...', "
            f"voice={tts_params.get('speaker', ['N/A'])[0]}, "
            f"task_type={task_type}, prompt_len={estimated_len}"
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=None,
            request_type=RequestType.AUDIO_GENERATION,
            response_format=req.response_format,
            speed=req.speed or 1.0,
        )

    def _validate_tts_request(self, req: NvCreateAudioSpeechRequest) -> None:
        """Validate Qwen3-TTS-specific request parameters."""
        task_type = req.task_type or "CustomVoice"

        _ALLOWED_TASK_TYPES = {"CustomVoice", "VoiceDesign", "Base"}
        if task_type not in _ALLOWED_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}'. "
                f"Supported: {', '.join(sorted(_ALLOWED_TASK_TYPES))}"
            )

        if req.language is not None:
            supported_langs = self._tts_supported_languages or {
                lang.lower() for lang in _TTS_LANGUAGES_FALLBACK
            }
            if req.language.lower() not in supported_langs and req.language != "Auto":
                raise ValueError(
                    f"Invalid language '{req.language}'. "
                    f"Supported: Auto, {', '.join(sorted(supported_langs))}"
                )

        if task_type == "CustomVoice" and req.voice is not None:
            if self._tts_supported_speakers:
                if req.voice.lower() not in self._tts_supported_speakers:
                    raise ValueError(
                        f"Invalid voice '{req.voice}'. "
                        f"Supported: {', '.join(self._tts_supported_speakers)}"
                    )

        if task_type == "Base" and req.ref_audio is None:
            raise ValueError("Base task requires 'ref_audio' for voice cloning")

        if task_type != "Base":
            if req.ref_text is not None:
                raise ValueError("'ref_text' is only valid for Base task")

        if task_type == "VoiceDesign" and not req.instructions:
            raise ValueError(
                "VoiceDesign task requires 'instructions' to describe the voice"
            )

        if (
            req.instructions
            and len(req.instructions) > self.config.tts_max_instructions_length
        ):
            raise ValueError(
                f"Instructions too long "
                f"(max {self.config.tts_max_instructions_length} characters)"
            )

        if req.max_new_tokens is not None:
            if req.max_new_tokens < self.config.tts_max_new_tokens_min:
                raise ValueError(
                    f"max_new_tokens must be at least "
                    f"{self.config.tts_max_new_tokens_min}"
                )
            if req.max_new_tokens > self.config.tts_max_new_tokens_max:
                raise ValueError(
                    f"max_new_tokens cannot exceed "
                    f"{self.config.tts_max_new_tokens_max}"
                )

    async def _resolve_ref_audio(self, ref_audio_str: str) -> tuple:
        """Download or decode reference audio for voice cloning (Base task)."""
        import io

        import soundfile as sf

        if ref_audio_str.startswith(("http://", "https://")):
            import ipaddress
            import socket
            from urllib.parse import urlparse

            import aiohttp

            parsed = urlparse(ref_audio_str)
            if not parsed.hostname:
                raise ValueError("Invalid ref_audio URL")
            for info in socket.getaddrinfo(
                parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
            ):
                ip_str = str(info[4][0]).split("%", 1)[0]
                addr = ipaddress.ip_address(ip_str)
                if addr.is_private or addr.is_loopback:
                    raise ValueError(
                        f"ref_audio URL resolves to blocked address: {addr}"
                    )

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    ref_audio_str,
                    timeout=aiohttp.ClientTimeout(
                        total=self.config.tts_ref_audio_timeout
                    ),
                ) as resp:
                    if resp.status != 200:
                        raise ValueError(
                            f"Failed to download ref_audio: HTTP {resp.status}"
                        )
                    audio_bytes = await resp.read()
                    if len(audio_bytes) > self.config.tts_ref_audio_max_bytes:
                        raise ValueError(
                            f"ref_audio too large "
                            f"({len(audio_bytes)} bytes, "
                            f"max {self.config.tts_ref_audio_max_bytes})"
                        )
        elif ref_audio_str.startswith("data:"):
            _, encoded = ref_audio_str.split(",", 1)
            audio_bytes = base64.b64decode(encoded)
            if len(audio_bytes) > self.config.tts_ref_audio_max_bytes:
                raise ValueError(
                    f"ref_audio data URI too large "
                    f"({len(audio_bytes)} bytes, "
                    f"max {self.config.tts_ref_audio_max_bytes})"
                )
        else:
            raise ValueError(
                "ref_audio must be a URL (http/https) or base64 data URI (data:...)"
            )

        wav_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return wav_data, int(sr)

    def _estimate_tts_prompt_len(self, tts_params: Dict[str, Any]) -> int:
        """Estimate Qwen3-TTS prompt length using its tokenizer.

        Falls back to 2048 if the model-specific estimator is unavailable.
        """
        try:
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )

            if not hasattr(self, "_tts_tokenizer") or self._tts_tokenizer is None:
                from transformers import AutoTokenizer

                self._tts_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model,
                    trust_remote_code=True,
                    padding_side="left",
                )

            hf_config = self.engine_client.model_config.hf_config
            talker_config = getattr(hf_config, "talker_config", None)
            task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]

            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
                additional_information=tts_params,
                task_type=task_type,
                tokenize_prompt=lambda t: self._tts_tokenizer(t, padding=False)[
                    "input_ids"
                ],
                codec_language_id=(
                    getattr(talker_config, "codec_language_id", None)
                    if talker_config
                    else None
                ),
                spk_is_dialect=(
                    getattr(talker_config, "spk_is_dialect", None)
                    if talker_config
                    else None
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to estimate TTS prompt length, using fallback 2048: %s", e
            )
            return 2048
