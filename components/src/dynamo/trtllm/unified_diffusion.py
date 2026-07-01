# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM DiffusionEngine — the raw-media sibling of trtllm/llm_engine.py.

Thin glue: reuses the VisualGen wrapper (engines/diffusion_engine.py) and the
existing per-modality handlers (request_handlers/diffusion), whose
``generate(request: dict, context)`` already matches the ``DiffusionEngine``
ABC — pick the handler for the modality and delegate.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, Optional

from dynamo._core import Context
from dynamo.common.backend.engine import DiffusionEngine, EngineConfig
from dynamo.common.backend.health_check import build_raw_health_check_payload
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode as CommonDisaggregationMode
from dynamo.llm import ModelInput
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.configs.diffusion_config import DiffusionConfig
from dynamo.trtllm.constants import Modality

logger = logging.getLogger(__name__)

# Modality -> registered endpoint type (Rust parses it into ModelType). Adding
# audio = one entry here + a handler in `_make_handler`; no framework changes.
_ENDPOINT_TYPE_BY_MODALITY: dict[Modality, str] = {
    Modality.IMAGE_DIFFUSION: "images",
    Modality.VIDEO_DIFFUSION: "videos",
    # Modality.AUDIO_DIFFUSION: "audios",  # future
}


def _make_handler(modality: Modality, engine: Any, config: DiffusionConfig) -> Any:
    """Build the handler for ``modality`` (the output_kind->encoder dispatch:
    image->PNG, video->MP4, audio->future). Imported lazily — the handlers pull
    in torch / VisualGen, which must not load at module import."""
    if modality == Modality.IMAGE_DIFFUSION:
        from dynamo.trtllm.request_handlers.diffusion import ImageGenerationHandler

        return ImageGenerationHandler(engine, config)
    if modality == Modality.VIDEO_DIFFUSION:
        from dynamo.trtllm.request_handlers.diffusion import VideoGenerationHandler

        return VideoGenerationHandler(engine, config)
    raise NotImplementedError(
        f"diffusion modality {modality} is not supported yet; "
        "image_diffusion and video_diffusion are available (audio is reserved)"
    )


def _diffusion_health_payload(modality: Modality) -> Optional[dict[str, Any]]:
    """A cheap real-inference canary (tiny size, one step) per modality,
    mirroring the SGLang diffusion health checks. ``None`` disables probing."""
    if modality == Modality.IMAGE_DIFFUSION:
        return build_raw_health_check_payload(
            {
                "prompt": "ping",
                "n": 1,
                "size": "64x64",
                "response_format": "b64_json",
                "nvext": {"num_inference_steps": 1},
            }
        )
    if modality == Modality.VIDEO_DIFFUSION:
        return build_raw_health_check_payload(
            {
                "prompt": "ping",
                "size": "64x64",
                "response_format": "b64_json",
                "nvext": {"num_inference_steps": 1, "num_frames": 8},
            }
        )
    return None


class TrtllmDiffusionEngine(DiffusionEngine):
    """DiffusionEngine that wraps TensorRT-LLM's VisualGen pipeline.

    Aggregated-only: a single worker owns the whole denoising pipeline, so
    there is no prefill/decode split and no KV cache to route on.
    """

    def __init__(
        self,
        diffusion_config: DiffusionConfig,
        modality: Modality,
        model_name: str,
        served_model_name: Optional[str] = None,
    ):
        self.diffusion_config = diffusion_config
        self.modality = modality
        self.model_name = model_name
        self.served_model_name = served_model_name
        # Built in start(): the VisualGen wrapper + the modality's handler.
        self._engine: Any = None
        self._handler: Any = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[TrtllmDiffusionEngine, WorkerConfig]:
        config = parse_args(argv)

        if not Modality.is_diffusion(config.modality):
            raise ValueError(
                f"TrtllmDiffusionEngine requires a diffusion modality; got "
                f"'{config.modality.value}'. Pass --modality image_diffusion or "
                f"--modality video_diffusion (use dynamo.trtllm.unified_main for "
                f"text/multimodal LLMs)."
            )

        diffusion_config = DiffusionConfig.from_config(config)
        endpoint_types = _ENDPOINT_TYPE_BY_MODALITY[config.modality]

        engine = cls(
            diffusion_config=diffusion_config,
            modality=config.modality,
            model_name=config.model,
            served_model_name=config.served_model_name,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            # Raw media pipeline: request forwarded verbatim, no tokenizer.
            model_input=ModelInput.Text,
            endpoint_types=endpoint_types,
            # Diffusion has no KV cache and no prefill/decode split.
            enable_kv_routing=False,
            disaggregation_mode=CommonDisaggregationMode.AGGREGATED,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id  # diffusion needs no cluster-wide per-worker key
        # Lazy import: VisualGen pulls in heavy optional deps.
        from dynamo.trtllm.engines.diffusion_engine import (
            DiffusionEngine as VisualGenDiffusionEngine,
        )

        logger.info(
            "Starting TrtllmDiffusionEngine: modality=%s, model=%s",
            self.modality.value,
            self.model_name,
        )
        self._engine = VisualGenDiffusionEngine(self.diffusion_config)
        await self._engine.initialize()
        self._handler = _make_handler(
            self.modality, self._engine, self.diffusion_config
        )
        logger.info("TrtllmDiffusionEngine ready (serving %s)", self.modality.value)

        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name or self.model_name,
        )

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self._handler is None:
            raise RuntimeError("generate() called before start() completed")
        # The handler's generate() already matches the DiffusionEngine
        # contract (raw request dict in, response dict out).
        async for chunk in self._handler.generate(request, context):
            yield chunk

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        return _diffusion_health_payload(self.modality)

    async def cleanup(self) -> None:
        # Null-safe against a partial start() (the ABC contract). Handlers
        # and the VisualGen engine expose synchronous cleanup().
        if self._handler is not None:
            self._handler.cleanup()
            self._handler = None
        if self._engine is not None:
            self._engine.cleanup()
            self._engine = None
