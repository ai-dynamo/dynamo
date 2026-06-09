# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni RawEngine — aggregated omni on the unified backend.

Thin glue: wraps the aggregated omni handler (``OmniHandler`` over vLLM-Omni's
``AsyncOmni`` orchestrator) whose ``generate(request: dict, context)`` already
matches the ``RawEngine`` contract (raw OpenAI request dict in, response dict
out). Same ABC + raw adapter as the diffusion engines — the abstraction
generalizes. Aggregated only: the multi-stage disaggregated form
(``--stage-id`` / ``--omni-router``) is a multi-worker pipeline that the
single-Worker unified model can't express, so it stays on
``python -m dynamo.vllm.omni``.

The heavier vLLM-Omni handler imports (``OmniHandler``, the health-check
payload) stay lazy inside ``start``/``health_check_payload`` so they only load
when the engine actually runs — same pattern as ``sglang/unified_diffusion.py``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any, Optional

from dynamo._core import Context
from dynamo.common.backend.engine import EngineConfig, RawEngine
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode as CommonDisaggregationMode
from dynamo.common.storage import get_fs
from dynamo.llm import ModelInput
from dynamo.vllm.omni.args import parse_omni_args

logger = logging.getLogger(__name__)

# vLLM-Omni declares OUTPUT modalities (CLI ``--output-modalities``); the
# unified Worker registers ENDPOINT types. Map one to the other so the frontend
# lights up the right routes (/v1/images, /v1/videos, /v1/audio/speech, chat).
_ENDPOINT_TYPES_BY_MODALITY: dict[str, str] = {
    "text": "chat,completions",
    "image": "images",
    "video": "videos",
    "audio": "audios",
}


class VllmOmniEngine(RawEngine):
    """RawEngine that wraps vLLM-Omni's aggregated ``OmniHandler``.

    Aggregated-only: one worker owns the whole multi-stage pipeline internally
    (AsyncOmni), so there is no prefill/decode split and no KV cache to route on.
    """

    def __init__(self, config: Any):
        self._config = config  # OmniConfig (vLLM-Omni args + runtime config)
        self._handler: Any = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple["VllmOmniEngine", WorkerConfig]:
        # parse_omni_args() reads sys.argv; honor an explicit argv when given.
        if argv is not None:
            saved, sys.argv = sys.argv, [sys.argv[0], *argv]
            try:
                config = parse_omni_args()
            finally:
                sys.argv = saved
        else:
            config = parse_omni_args()

        if config.stage_id is not None or config.omni_router:
            raise ValueError(
                "VllmOmniEngine supports aggregated omni only; multi-stage "
                "(--stage-id / --omni-router) stays on `python -m dynamo.vllm.omni`."
            )
        if not config.served_model_name:
            config.served_model_name = config.model

        modalities = config.output_modalities or ["image"]
        try:
            endpoint_types = ",".join(
                _ENDPOINT_TYPES_BY_MODALITY[m] for m in modalities
            )
        except KeyError as e:
            raise ValueError(f"unsupported omni output modality: {e}") from e

        engine = cls(config)
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            # Raw media pipeline: the frontend forwards the request verbatim
            # (no tokenizer); omni self-parses any multimodal input.
            model_input=ModelInput.Text,
            endpoint_types=endpoint_types,
            # One endpoint for the whole pipeline; no KV cache, no disagg split.
            enable_kv_routing=False,
            disaggregation_mode=CommonDisaggregationMode.AGGREGATED,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id  # omni needs no cluster-wide per-worker key
        from dynamo.vllm.omni import OmniHandler

        media_fs = (
            get_fs(self._config.media_output_fs_url)
            if self._config.media_output_fs_url
            else None
        )
        # OmniHandler only stashes ``runtime`` (unused in its generate path); the
        # unified Worker owns registration/serving, so pass None.
        self._handler = OmniHandler(
            runtime=None,
            config=self._config,
            default_sampling_params={},
            shutdown_event=None,
            media_output_fs=media_fs,
            media_output_http_url=self._config.media_output_http_url,
        )
        logger.info(
            "VllmOmniEngine ready (model=%s, output_modalities=%s)",
            self._config.model,
            self._config.output_modalities,
        )
        return EngineConfig(
            model=self._config.model,
            served_model_name=self._config.served_model_name or self._config.model,
        )

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self._handler is None:
            raise RuntimeError("generate() called before start() completed")
        # OmniHandler.generate already matches the RawEngine contract
        # (raw request dict in, response dict out).
        async for chunk in self._handler.generate(request, context):
            yield chunk

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        from dynamo.vllm.health_check import VllmOmniHealthCheckPayload

        payload = await VllmOmniHealthCheckPayload.create(self._handler.engine_client)
        return payload.to_dict()

    async def cleanup(self) -> None:
        # Null-safe against a partial start() (the ABC contract).
        if self._handler is not None:
            self._handler.cleanup()
            self._handler = None
