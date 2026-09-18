# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audex (nvidia/Nemotron-Labs-Audex-*) request preparation.

Owns everything model-specific about an Audex ``/v1/audio/speech`` request so
that ``AudioGenerationHandler`` stays a modality dispatcher: stage detection,
parameter validation, the ChatML prompt that primes codec generation, and the
per-request CFG / RVQ-phase contracts written onto the stage-0 sampling params.

vLLM-Omni does ship ``AudexAdapter`` / ``AudexTTAAdapter`` under
``entrypoints.openai.tts_adapters``, but they cover only prompt building plus
validation of an ``OpenAICreateSpeechRequest``, and they are constructed with a
``SpeechServingContext`` holding a back-reference to the running
``OmniOpenAIServingSpeech``. The CFG and RVQ contracts are not in the adapters
at all -- they live in ``serving_speech.py`` next to that server. Reusing them
would therefore couple Dynamo to upstream's HTTP protocol and serving objects
for the smaller half of the work. The parts that actually drift with the
checkpoint -- the prompt text, the RVQ phase table, and the snapshot layout --
are model-owned code under ``model_executor.models.audex``, and this adapter
calls that code directly (see ``_prompt_builders``, ``_tta_rvq_contract``,
``tokenizer``). Revisit if upstream moves the contracts into the adapters.
"""

import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import AutoTokenizer

try:
    # Imported as modules, not symbols: the Audex helpers are looked up as
    # attributes at call time so a test can substitute one.
    from vllm_omni.model_executor.models.audex import checkpoint as audex_checkpoint
    from vllm_omni.model_executor.models.audex import prompt as audex_prompt
    from vllm_omni.model_executor.models.audex import tta as audex_tta
except ImportError:
    # vllm_omni only ships these in Audex-capable builds. Missing support fails
    # the Audex request path (see _require_audex) rather than the import, so
    # every other audio model still serves on such a build.
    audex_checkpoint = None  # type: ignore[assignment]
    audex_prompt = None  # type: ignore[assignment]
    audex_tta = None  # type: ignore[assignment]

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.vllm.omni.utils import engine_model_stages, validate_audio_max_new_tokens

logger = logging.getLogger(__name__)

# Audex needs its own prompt format: the thinker consumes a literal ChatML
# prompt whose assistant turn is primed with ``<think></think><speechgen_start>``
# (TTS) or ``<audiogen_start>`` (TTA). A plain text prompt makes the thinker
# emit a text/thinking continuation with zero ``<speechcodec_N>`` tokens, so
# stage 0 ships an empty codec payload and the request fails with "Audex thinker
# produced no codec tokens".
# Mirrors vLLM-Omni's Audex adapter ``stage_keys`` in tts_adapters/audex*.py.
TTS_MODEL_STAGES: frozenset = frozenset({"audex_thinker", "audex_omni"})
TTA_MODEL_STAGES: frozenset = frozenset({"audex_tta_thinker"})
# The audio-capable ``audex_omni`` thinker is only speech-capable when
# deployed WITH the speech decoder; the thinker-only deployment is text-final.
CODE2WAV_STAGE = "audex_code2wav"

# The two Audex tasks, and the only values ``model_type()`` returns. Every
# branch below reads "not TTA" as TTS, so these are named rather than spelled
# inline: a typo'd literal would silently select the TTS prompt and codec space.
MODEL_TYPE_TTS = "audex"
MODEL_TYPE_TTA = "audex_tta"
MODEL_TYPES: frozenset = frozenset({MODEL_TYPE_TTS, MODEL_TYPE_TTA})

# Classifier-free guidance bounds, mirroring the vLLM-Omni Audex adapters.
# TTS guidance is optional (unguided is the official baseline); TTA guidance is
# effectively mandatory for quality, hence the 3.0 default.
CFG_SCALE_MIN = 1.0
CFG_SCALE_MAX = 10.0
TTA_DEFAULT_CFG_SCALE = 3.0
# Official TTA generation cap, in codec tokens: 4000 tokens is 1000 frames at
# XCodec1's 4 RVQ codebooks per frame. Decode caps the result again — the
# XCodec1 stage keeps at most ``max_tta_frames`` (default 500, roughly 10 s),
# so this bound governs generation length, not output duration.
TTA_CODEC_CAP = 4000

# Guidance sharpens the distribution, so the unguided default temperature adds
# excess sampling noise (vLLM-Omni measures a CER win for 0.05 over 0.1 at
# cfg 1.5).
#
# That measurement is 2B-tuned: vLLM-Omni's audex_tts_30b.yaml records 0.1 as
# the 30B-A3B's best guided temperature. Applying 0.05 to both sizes matches
# vLLM-Omni's own behavior, which treats a size-aware value as a known
# follow-up rather than a defect.
GUIDED_TTS_TEMPERATURE = 0.05


def _requested_cfg_scale(req: NvCreateAudioSpeechRequest) -> Optional[float]:
    """Return the caller's guidance scale, or None when it was not requested.

    CFG is Audex-only, so it rides in ``nvext`` rather than as a top-level
    OpenAI field (vLLM-Omni likewise takes it under ``extra_params``).
    """
    return req.nvext.cfg_scale if req.nvext is not None else None


def _require_audex(module: Any, name: str) -> Any:
    """Return an Audex submodule, or fail the request if the build lacks it.

    RuntimeError (not ImportError) so the handler reports it as a request
    error instead of letting it escape as an unhandled exception.
    """
    if module is None:
        raise RuntimeError(
            "This vLLM-Omni build has no Audex support: "
            f"vllm_omni.model_executor.models.audex.{name} is unavailable"
        )
    return module


@dataclass
class AudexPreparedRequest:
    """What the handler needs to build ``EngineInputs`` for an Audex request.

    Plays the role of vLLM-Omni's ``tts_adapters.base.PreparedRequest`` without
    importing it, and carries only what the handler actually consumes: the
    adapter owns the prompt and the stage params, the handler owns the
    response-format plumbing that is identical for every audio model.
    """

    prompt: str
    sampling_params_list: Optional[list]


class AudexRequestAdapter:
    """Prepares Audex TTS/TTA requests for the vLLM-Omni engine.

    Held by ``AudioGenerationHandler`` as ``self.audex``. ``model_type()``
    returns None on a non-Audex deployment, which is how the handler decides
    whether this adapter owns the request at all.
    """

    def __init__(self, config: Any, engine_client: Any) -> None:
        self.config = config
        self.engine_client = engine_client
        # Lazily-loaded, process-wide caches (see tokenizer/_tta_rvq_contract).
        self._tokenizer: Any = None
        self._tta_rvq: Optional[Dict[str, Any]] = None

    # -- deployment detection -------------------------------------------------

    def model_type(self) -> str | None:
        """Return ``"audex"``/``"audex_tta"`` for an Audex deployment, else None.

        Mirrors vLLM-Omni's Audex ``stage_serves_speech``: ``audex_omni`` only
        serves speech when the code2wav decoder is also deployed (the
        thinker-only pipeline is text-final).

        Recomputed per call rather than cached at construction: the engine's
        stage list is the source of truth and this is a set build over a
        handful of names.
        """
        stages = engine_model_stages(self.engine_client)
        if stages & TTA_MODEL_STAGES:
            return MODEL_TYPE_TTA
        for stage in stages & TTS_MODEL_STAGES:
            if stage == "audex_omni" and CODE2WAV_STAGE not in stages:
                continue
            return MODEL_TYPE_TTS
        return None

    # -- request preparation --------------------------------------------------

    def prepare(
        self,
        req: NvCreateAudioSpeechRequest,
        request_id: str | None,
        model_type: str,
    ) -> AudexPreparedRequest:
        """Validate ``req`` and build the Audex prompt plus stage params.

        ``model_type`` is a non-None ``model_type()`` result, passed in by the
        caller that already resolved it to decide this adapter owns the request.
        It is re-checked against the known task types because every downstream
        branch treats "not TTA" as TTS: an unrecognized value would otherwise be
        served as TTS, answering with a phase-invalid or wrong-prompt waveform
        rather than an error.

        ``request_id`` is the final Dynamo request id; Audex uses it as the CFG
        pair id that binds a guided request to its unconditional companion.
        """
        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"unknown Audex model type {model_type!r}; "
                f"expected one of {sorted(MODEL_TYPES)}"
            )

        self.validate(req, model_type)

        build_cond, _ = self._prompt_builders(model_type)
        cond_prompt = build_cond(req.input)

        logger.info(
            "Audex %s request: input='%s...', request_id=%s",
            model_type,
            req.input[:50],
            request_id,
        )
        return AudexPreparedRequest(
            prompt=cond_prompt,
            sampling_params_list=self.sampling_params_list(
                req, model_type, cond_prompt, request_id
            ),
        )

    def validate(self, req: NvCreateAudioSpeechRequest, model_type: str) -> None:
        """Reject unsupported Audex parameters (mirrors the vLLM-Omni adapters).

        Audex has a single built-in voice and no voice cloning, so a caller
        asking for a named voice or reference audio must get an explicit error
        instead of silently synthesized different-sounding audio.
        """
        voice = (req.voice or "").strip().lower()
        if voice not in ("", "default"):
            if model_type == MODEL_TYPE_TTA:
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

        cfg_scale = _requested_cfg_scale(req)
        if cfg_scale is not None and not (CFG_SCALE_MIN <= cfg_scale <= CFG_SCALE_MAX):
            raise ValueError(
                f"nvext.cfg_scale must be within [{CFG_SCALE_MIN}, {CFG_SCALE_MAX}]; "
                f"got {cfg_scale}. 1.0 disables guidance."
            )

        validate_audio_max_new_tokens(req.max_new_tokens, self.config)

    def sampling_params_list(
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
        the engine on its own defaults; a request that does need an override
        fails instead, since the defaults are the only channel for it.
        """
        requested_cfg = _requested_cfg_scale(req)
        defaults = list(self.engine_client.default_sampling_params_list or [])
        if not defaults:
            if (
                model_type == MODEL_TYPE_TTA
                or req.max_new_tokens is not None
                or (requested_cfg is not None and requested_cfg > CFG_SCALE_MIN)
            ):
                # Dropping the contract here would answer with different-sounding
                # audio (unguided, or for TTA a phase-invalid codec stream that
                # decode rejects) instead of reporting the broken engine contract.
                raise RuntimeError(
                    "Audex: engine exposed no default_sampling_params_list, so "
                    "the per-request CFG/RVQ contract cannot be attached"
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

        cfg_scale = requested_cfg
        if model_type == MODEL_TYPE_TTA:
            # The RVQ phase contract gates which codec ids are sampleable per
            # position; without it the stream is phase-invalid and decode is
            # rejected. TTA guidance defaults on (official setting 3.0).
            extra_args["tta_rvq"] = self._tta_rvq_contract()
            if cfg_scale is None:
                cfg_scale = TTA_DEFAULT_CFG_SCALE

        if cfg_scale is None or cfg_scale <= CFG_SCALE_MIN:
            extra_args.pop("cfg_scale", None)
            return params_list

        if request_id is None:
            # The pair id must be unique per request, and it is the final Dynamo
            # request id, which every serving path has. Decoding unguided instead
            # would answer a guided request with different-sounding audio, so a
            # caller that reached here without an id gets the broken contract
            # reported rather than silently downgraded output.
            raise RuntimeError(
                f"Audex: cfg_scale={cfg_scale} needs a request id to pair the "
                "guided and unconditional sequences, but none was supplied"
            )

        _, build_null = self._prompt_builders(model_type)
        null_prompt = build_null(cond_prompt, self.tokenizer(model_type))
        if model_type != MODEL_TYPE_TTA:
            stage0.temperature = GUIDED_TTS_TEMPERATURE

        extra_args.update(
            {
                "cfg_scale": cfg_scale,
                "cfg_role": "cond",
                "cfg_pair_id": request_id,
                "cfg_null_prompt": null_prompt,
            }
        )
        return params_list

    # -- model-owned vLLM-Omni helpers ----------------------------------------

    @staticmethod
    def _prompt_builders(model_type: str) -> tuple:
        """Return the (conditional, null) ChatML prompt builders for a task.

        TTS and TTA prime different codec spaces, so each has its own pair of
        model-owned builders.
        """
        prompt = _require_audex(audex_prompt, "prompt")

        if model_type == MODEL_TYPE_TTA:
            return prompt.build_tta_cond_prompt, prompt.build_tta_null_prompt
        return prompt.build_cond_prompt, prompt.build_null_prompt

    def _tta_rvq_contract(self) -> Dict[str, Any]:
        """Build (once per process) the TTA RVQ phase-mask contract.

        The same dict is handed to every request, unlike the surrounding
        sampling params which are deep-copied. That is safe because the
        contract is read-only downstream: vLLM-Omni's
        ``TTARVQPhaseMaskLogitsProcessor`` copies the scalars into its own
        per-sequence state and only reads ``phase_token_ids``. vLLM-Omni's own
        adapter shares one cached dict the same way.
        """
        if self._tta_rvq is None:
            tta = _require_audex(audex_tta, "tta")
            phase_token_ids, start_tid, end_tid = tta.build_tta_phase_token_ids(
                self.tokenizer(MODEL_TYPE_TTA)
            )
            self._tta_rvq = {
                "phase_token_ids": phase_token_ids,
                "start_tid": start_tid,
                "end_tid": end_tid,
                "codec_cap": TTA_CODEC_CAP,
                # The TTA prompt already ends with <audiogen_start>.
                "start_in_prompt": True,
            }
        return self._tta_rvq

    def tokenizer(self, model_type: str):
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
        if self._tokenizer is None:
            checkpoint = _require_audex(audex_checkpoint, "checkpoint")

            # The snapshot profile decides which subset is fetched: "tts" also
            # pulls the speech decoder, so it is not interchangeable with "tta".
            if "audex_omni" in engine_model_stages(self.engine_client):
                profile, folder = "full", "checkpoint_folder_full"
            else:
                profile = "tta" if model_type == MODEL_TYPE_TTA else "tts"
                folder = "checkpoint_folder_audiogen"

            model_path = os.path.normpath(self.engine_client.model_config.model)
            if os.path.basename(model_path).startswith("checkpoint_folder"):
                root = os.path.dirname(model_path)
            else:
                root = checkpoint.ensure_audex_snapshot(model_path, profile=profile)
            # The checkpoint ships custom code; without trust_remote_code
            # transformers prompts on stdin, which would hang the worker.
            self._tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(root, folder),
                trust_remote_code=self.config.engine_args.trust_remote_code,
            )
        return self._tokenizer
