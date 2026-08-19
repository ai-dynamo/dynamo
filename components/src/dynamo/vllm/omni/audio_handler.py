# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio/TTS handler utilities for the vLLM-Omni backend.

Extracted from omni_handler.py to keep modality-specific logic separate.
OmniHandler holds an instance as ``self.audio`` (composition).
"""

import base64
import copy
import logging
import os
from typing import Any, Dict, Optional

from transformers import AutoTokenizer
from vllm_omni.inputs.data import OmniTextPrompt

try:
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
        Qwen3TTSPromptEmbedsBuilder,
    )
except ImportError:
    Qwen3TTSPromptEmbedsBuilder = None  # type: ignore[assignment, misc]

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.utils.output_modalities import RequestType

logger = logging.getLogger(__name__)

# model_stage names that receive Qwen3-TTS-specific prompt format
# (prompt_token_ids + additional_information). Other audio models
# (MiMo-Audio, Qwen3-Omni, Stable Audio, etc.) use a plain text prompt.
# Mirrors vLLM-Omni's _TTS_MODEL_STAGES in serving_speech.py.
_TTS_MODEL_STAGES: set = {"qwen3_tts"}

# Audex (nvidia/Nemotron-Labs-Audex-2B) needs its own prompt format: the
# thinker consumes a literal ChatML prompt whose assistant turn is primed with
# ``<think></think><speechgen_start>`` (TTS) or ``<audiogen_start>`` (TTA).
# A plain text prompt makes the thinker emit a text/thinking continuation with
# zero ``<speechcodec_N>`` tokens, so stage 0 ships an empty codec payload and
# the request fails with "Audex thinker produced no codec tokens".
# Mirrors vLLM-Omni's _AUDEX_*_MODEL_STAGES in serving_speech.py.
_AUDEX_TTS_MODEL_STAGES: set = {"audex_thinker", "audex_omni"}
_AUDEX_TTA_MODEL_STAGES: set = {"audex_tta_thinker"}
# The audio-capable ``audex_omni`` thinker is only speech-capable when
# deployed WITH the speech decoder; the thinker-only deployment is text-final.
_AUDEX_CODE2WAV_STAGE = "audex_code2wav"

# Classifier-free guidance bounds, mirroring the vLLM-Omni Audex adapters.
# TTS guidance is optional (unguided is the official baseline); TTA guidance is
# effectively mandatory for quality, hence the 3.0 default.
_AUDEX_CFG_SCALE_MIN = 1.0
_AUDEX_CFG_SCALE_MAX = 10.0
_AUDEX_TTA_DEFAULT_CFG_SCALE = 3.0
# Official TTA generation cap, in codec tokens: 4000 tokens is 1000 frames at
# XCodec1's 4 RVQ codebooks per frame. Decode caps the result again — the
# XCodec1 stage keeps at most ``max_tta_frames`` (default 500, roughly 10 s),
# so this bound governs generation length, not output duration.
_AUDEX_TTA_CODEC_CAP = 4000

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
        # Audex lazily-loaded, process-wide caches (see _get_audex_tokenizer).
        self._audex_tokenizer: Any = None
        self._audex_tta_rvq: Optional[Dict[str, Any]] = None

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

    def _engine_model_stages(self) -> set:
        """Collect every ``model_stage`` name the engine exposes.

        Reads the engine's stage list and stage configs, tolerating the several
        shapes vLLM-Omni versions use (objects or dicts, ``engine_args`` nested
        or flat).
        """
        stages: set = set()

        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                ms = getattr(stage, "model_stage", None)
                if ms:
                    stages.add(ms)

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
                if ms:
                    stages.add(ms)

        logger.debug("engine model stages: %s", sorted(stages))
        return stages

    def _audex_model_type(self) -> str | None:
        """Return ``"audex"``/``"audex_tta"`` for an Audex deployment, else None.

        Mirrors vLLM-Omni's ``_detect_tts_model_type`` for the Audex stages:
        ``audex_omni`` only serves speech when the code2wav decoder is also
        deployed (the thinker-only pipeline is text-final).
        """
        stages = self._engine_model_stages()
        if stages & _AUDEX_TTA_MODEL_STAGES:
            return "audex_tta"
        for stage in stages & _AUDEX_TTS_MODEL_STAGES:
            if stage == "audex_omni" and _AUDEX_CODE2WAV_STAGE not in stages:
                continue
            return "audex"
        return None

    def emits_cumulative_waveforms(self) -> bool:
        """True when the decoder streams cumulative waveform snapshots.

        Audex's code2wav/XCodec1 stages re-emit the whole waveform decoded so
        far on every yield, so the snapshots must be de-duplicated to the
        longest one rather than concatenated. Other audio models served through
        this handler (Qwen3-TTS, MiMo-Audio) are left on the per-payload path
        they already used: concatenating cumulative snapshots multiplies the
        duration, but de-duplicating incremental frames would silently drop
        audio, so the buffering is scoped to the pipeline known to need it.
        """
        return self._audex_model_type() is not None

    def _is_tts_model(self) -> bool:
        """Check if the loaded model is a Qwen3-TTS-style model.

        Searches for a TTS model_stage in the engine's stage list,
        stage configs, or model config. Supports multiple vLLM-Omni versions.
        """
        stages = self._engine_model_stages()
        if stages & _TTS_MODEL_STAGES:
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
            "_is_tts_model: could not detect TTS model. engine model stages=%s",
            sorted(stages),
        )
        return False

    # -- Audio engine input construction --------------------------------------

    async def build_engine_inputs(
        self, req: NvCreateAudioSpeechRequest, request_id: str | None = None
    ):
        """Build engine inputs for an audio/TTS request.

        Three code paths (matching vLLM-Omni serving_speech.py):

        * **TTS path** (Qwen3-TTS): ``prompt_token_ids`` +
          ``additional_information``.
        * **Audex path**: literal ChatML prompt priming codec generation,
          plus the CFG / RVQ-phase contracts on stage-0 sampling params.
        * **Generic audio path** (MiMo-Audio, etc.): plain text prompt.

        ``request_id`` is the final Dynamo request id; Audex uses it as the CFG
        pair id that binds a guided request to its unconditional companion.
        """
        # Import here to avoid circular dependency
        from dynamo.vllm.omni.omni_handler import EngineInputs

        if not req.input or not req.input.strip():
            raise ValueError("Input text cannot be empty")

        audex_model_type = self._audex_model_type()
        if audex_model_type is not None:
            return self._engine_inputs_audex(req, audex_model_type, request_id)

        if self._is_tts_model():
            return await self._engine_inputs_tts(req)

        # Generic audio model – plain text prompt (same as image/video)
        prompt = OmniTextPrompt(prompt=req.input)
        logger.info(f"Audio request (generic): input='{req.input[:50]}...'")
        return EngineInputs(
            prompt=prompt,
            sampling_params_list=None,
            request_type=RequestType.AUDIO_GENERATION,
            response_format=req.data_source,
            output_format=req.response_format,
            speed=req.speed or 1.0,
        )

    # -- Audex-specific helpers -----------------------------------------------

    @staticmethod
    def _audex_prompt_builders(model_type: str) -> tuple:
        """Return the (conditional, null) ChatML prompt builders for a task.

        TTS and TTA prime different codec spaces, so each has its own pair of
        model-owned builders. Imported lazily: vllm_omni only ships them in
        Audex-capable builds.
        """
        from vllm_omni.model_executor.models.audex import prompt

        if model_type == "audex_tta":
            return prompt.build_tta_cond_prompt, prompt.build_tta_null_prompt
        return prompt.build_cond_prompt, prompt.build_null_prompt

    def _engine_inputs_audex(
        self,
        req: NvCreateAudioSpeechRequest,
        model_type: str,
        request_id: str | None,
    ):
        """Build engine inputs for Audex TTS/TTA.

        The thinker only emits ``<speechcodec_N>``/``<audiocodec_N>`` tokens
        when its assistant turn is primed by the exact ChatML prompt the
        checkpoint was trained on, so the prompt is built with vLLM-Omni's
        model-owned builders rather than passed through as plain text.
        """
        from dynamo.vllm.omni.omni_handler import EngineInputs

        self._validate_audex_request(req, model_type)

        build_cond, _ = self._audex_prompt_builders(model_type)
        cond_prompt = build_cond(req.input)

        sampling_params_list = self._audex_sampling_params_list(
            req, model_type, cond_prompt, request_id
        )

        logger.info(
            "Audex %s request: input='%s...', request_id=%s",
            model_type,
            req.input[:50],
            request_id,
        )
        return EngineInputs(
            prompt=OmniTextPrompt(prompt=cond_prompt),
            sampling_params_list=sampling_params_list,
            request_type=RequestType.AUDIO_GENERATION,
            response_format=req.data_source,
            output_format=req.response_format,
            speed=req.speed or 1.0,
        )

    def _validate_audex_request(
        self, req: NvCreateAudioSpeechRequest, model_type: str
    ) -> None:
        """Reject unsupported Audex parameters (mirrors the vLLM-Omni adapters).

        Audex has a single built-in voice and no voice cloning, so a caller
        asking for a named voice or reference audio must get an explicit error
        instead of silently synthesized different-sounding audio.
        """
        voice = (req.voice or "").strip().lower()
        if voice not in ("", "default"):
            if model_type == "audex_tta":
                raise ValueError(
                    f"Audex TTA generates general audio and has no voices; "
                    f"got voice={req.voice!r}. Omit 'voice' or pass 'default'."
                )
            raise ValueError(
                f"Audex has a single built-in voice and no voice cloning; "
                f"got voice={req.voice!r}. Omit 'voice' or pass 'default'."
            )
        if req.ref_audio is not None or req.ref_text is not None:
            raise ValueError(
                "Audex does not support reference audio (no voice cloning)."
            )

        cfg_scale = req.cfg_scale
        if cfg_scale is not None and not (
            _AUDEX_CFG_SCALE_MIN <= cfg_scale <= _AUDEX_CFG_SCALE_MAX
        ):
            raise ValueError(
                f"cfg_scale must be within "
                f"[{_AUDEX_CFG_SCALE_MIN}, {_AUDEX_CFG_SCALE_MAX}]; got {cfg_scale}. "
                f"1.0 disables guidance."
            )

        self._validate_max_new_tokens(req)

    def _audex_sampling_params_list(
        self,
        req: NvCreateAudioSpeechRequest,
        model_type: str,
        cond_prompt: str,
        request_id: str | None,
    ) -> list | None:
        """Clone the engine's stage defaults and attach the Audex contracts.

        The engine's ``default_sampling_params_list`` entries are SHARED across
        requests, so per-request CFG pair state is written onto deep copies —
        mutating the shared defaults would leak one request's pair id into the
        next. Returns ``None`` when there is nothing to override, which leaves
        the engine on its own defaults.
        """
        defaults = list(
            getattr(self.engine_client, "default_sampling_params_list", None) or []
        )
        if not defaults:
            logger.warning(
                "Audex: engine exposed no default_sampling_params_list; "
                "per-request CFG/RVQ overrides are unavailable"
            )
            return None

        params_list = copy.deepcopy(defaults)
        stage0 = params_list[0]

        if req.max_new_tokens is not None:
            stage0.max_tokens = req.max_new_tokens

        extra_args = getattr(stage0, "extra_args", None)
        if extra_args is None:
            extra_args = {}
            stage0.extra_args = extra_args

        cfg_scale = req.cfg_scale
        if model_type == "audex_tta":
            # The RVQ phase contract gates which codec ids are sampleable per
            # position; without it the stream is phase-invalid and decode is
            # rejected. TTA guidance defaults on (official setting 3.0).
            extra_args["tta_rvq"] = self._audex_tta_rvq_contract()
            if cfg_scale is None:
                cfg_scale = _AUDEX_TTA_DEFAULT_CFG_SCALE

        if cfg_scale is None or cfg_scale <= 1.0:
            extra_args.pop("cfg_scale", None)
            return params_list

        if request_id is None:
            # The pair id must be unique per request; without it the guided and
            # unconditional sequences cannot be matched, so fall back to
            # unguided decoding rather than corrupting a concurrent pair.
            logger.warning(
                "Audex: no request id available; ignoring cfg_scale=%s "
                "and decoding unguided",
                cfg_scale,
            )
            extra_args.pop("cfg_scale", None)
            return params_list

        _, build_null = self._audex_prompt_builders(model_type)
        null_prompt = build_null(cond_prompt, self._get_audex_tokenizer(model_type))
        if model_type != "audex_tta":
            # Guidance sharpens the distribution, so the unguided default
            # temperature adds excess sampling noise (vLLM-Omni measures a CER
            # win for 0.05 over 0.1 at cfg 1.5).
            #
            # That measurement is 2B-tuned: vLLM-Omni's audex_tts_30b.yaml
            # records 0.1 as the 30B-A3B's best guided temperature. Applying
            # 0.05 to both sizes matches vLLM-Omni's own behavior, which treats
            # a size-aware value as a known follow-up rather than a defect.
            stage0.temperature = 0.05

        extra_args.update(
            {
                "cfg_scale": cfg_scale,
                "cfg_role": "cond",
                "cfg_pair_id": request_id,
                "cfg_null_prompt": null_prompt,
            }
        )
        return params_list

    def _audex_tta_rvq_contract(self) -> Dict[str, Any]:
        """Build (once per process) the TTA RVQ phase-mask contract.

        The same dict is handed to every request, unlike the surrounding
        sampling params which are deep-copied. That is safe because the
        contract is read-only downstream: vLLM-Omni's
        ``TTARVQPhaseMaskLogitsProcessor`` copies the scalars into its own
        per-sequence state and only reads ``phase_token_ids``. vLLM-Omni's own
        adapter shares one cached dict the same way.
        """
        if self._audex_tta_rvq is None:
            from vllm_omni.model_executor.models.audex.tta import (
                build_tta_phase_token_ids,
            )

            tokenizer = self._get_audex_tokenizer("audex_tta")
            phase_token_ids, start_tid, end_tid = build_tta_phase_token_ids(tokenizer)
            self._audex_tta_rvq = {
                "phase_token_ids": phase_token_ids,
                "start_tid": start_tid,
                "end_tid": end_tid,
                "codec_cap": _AUDEX_TTA_CODEC_CAP,
                # The TTA prompt already ends with <audiogen_start>.
                "start_in_prompt": True,
            }
        return self._audex_tta_rvq

    def _get_audex_tokenizer(self, model_type: str):
        """Load (once per process) the Audex thinker tokenizer.

        Mirrors vLLM-Omni's ``_get_audex_tokenizer``: the checkpoint is a repo
        of per-stage subfolders, and the stage's model_config may already point
        at the resolved subfolder (joining again would yield a missing path that
        transformers then mistakes for a repo id).

        The TTS and TTA thinkers both tokenize with
        ``checkpoint_folder_audiogen`` — only the full ``audex_omni`` pipeline
        uses ``checkpoint_folder_full``. The snapshot *profile* still differs
        between TTS and TTA (it decides which stage weights are fetched), which
        is why the two are not interchangeable even though the tokenizer folder
        is shared.
        """
        if self._audex_tokenizer is None:
            from vllm_omni.model_executor.models.audex.checkpoint import (
                ensure_audex_snapshot,
            )

            # The snapshot profile decides which subset is fetched: "tts" also
            # pulls the speech decoder, so it is not interchangeable with "tta".
            if "audex_omni" in self._engine_model_stages():
                profile, folder = "full", "checkpoint_folder_full"
            else:
                profile = "tta" if model_type == "audex_tta" else "tts"
                folder = "checkpoint_folder_audiogen"

            model_path = os.path.normpath(self.engine_client.model_config.model)
            if os.path.basename(model_path).startswith("checkpoint_folder"):
                root = os.path.dirname(model_path)
            else:
                root = ensure_audex_snapshot(model_path, profile=profile)
            # The checkpoint ships custom code; without trust_remote_code
            # transformers prompts on stdin, which would hang the worker.
            self._audex_tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(root, folder),
                trust_remote_code=self.config.engine_args.trust_remote_code,
            )
        return self._audex_tokenizer

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
            response_format=req.data_source,
            output_format=req.response_format,
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

        self._validate_max_new_tokens(req)

    def _validate_max_new_tokens(self, req: NvCreateAudioSpeechRequest) -> None:
        """Bound the caller's generation length (shared by the TTS/Audex paths)."""
        if req.max_new_tokens is None:
            return
        if req.max_new_tokens < self.config.tts_max_new_tokens_min:
            raise ValueError(
                f"max_new_tokens must be at least {self.config.tts_max_new_tokens_min}"
            )
        if req.max_new_tokens > self.config.tts_max_new_tokens_max:
            raise ValueError(
                f"max_new_tokens cannot exceed {self.config.tts_max_new_tokens_max}"
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
        if Qwen3TTSPromptEmbedsBuilder is None:
            logger.warning(
                "Qwen3-TTS prompt estimator is unavailable, using fallback 2048"
            )
            return 2048

        if not hasattr(self, "_tts_tokenizer") or self._tts_tokenizer is None:
            self._tts_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=self.config.engine_args.trust_remote_code,
                padding_side="left",
            )

        hf_config = self.engine_client.model_config.hf_config
        talker_config = getattr(hf_config, "talker_config", None)
        task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]

        return (
            Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
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
        )
