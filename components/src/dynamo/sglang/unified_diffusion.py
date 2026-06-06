# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang DiffusionEngine — the raw-media sibling of sglang/llm_engine.py.

Thin glue: reuses SGLang's ``DiffGenerator`` and the existing per-modality
handlers (request_handlers/{image_diffusion,video_generation}), whose
``generate(request: dict, context)`` already matches the ``DiffusionEngine``
ABC. Same ABC + raw adapter as TRT-LLM — the abstraction generalizes across
engines.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any, Optional

from dynamo._core import Context
from dynamo.common.backend.engine import DiffusionEngine, EngineConfig
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode as CommonDisaggregationMode
from dynamo.common.storage import get_fs
from dynamo.llm import ModelInput
from dynamo.sglang.args import parse_args

logger = logging.getLogger(__name__)

# Internal modality tags (SGLang selects diffusion mode via boolean CLI flags,
# not an enum). Adding audio = a tag here + a handler in `_make_handler` + a
# flag in args; no framework changes.
_IMAGE = "image"
_VIDEO = "video"

_ENDPOINT_TYPE_BY_MODALITY: dict[str, str] = {
    _IMAGE: "images",
    _VIDEO: "videos",
}


def _make_handler(modality: str, generator: Any, config: Any, fs: Any) -> Any:
    """Build the handler for ``modality`` (the output_kind->encoder dispatch:
    image->PNG, video->MP4; audio reserved). Imported lazily — the handlers
    pull in torch / SGLang internals."""
    if modality == _IMAGE:
        from dynamo.sglang.request_handlers import ImageDiffusionWorkerHandler

        return ImageDiffusionWorkerHandler(generator, config, publisher=None, fs=fs)
    if modality == _VIDEO:
        from dynamo.sglang.request_handlers import VideoGenerationWorkerHandler

        return VideoGenerationWorkerHandler(generator, config, publisher=None, fs=fs)
    raise NotImplementedError(
        f"diffusion modality {modality!r} is not supported yet; "
        "image and video are available (audio is reserved)"
    )


class SglangDiffusionEngine(DiffusionEngine):
    """DiffusionEngine that wraps SGLang's ``DiffGenerator``.

    Aggregated-only: a single worker owns the whole pipeline, so there is no
    prefill/decode split and no KV cache to route on.
    """

    def __init__(self, config: Any, modality: str):
        self.config = config  # SGLang Config (server_args + dynamo_args)
        self.modality = modality
        self._generator: Any = None
        self._handler: Any = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SglangDiffusionEngine, WorkerConfig]:
        config = await parse_args(argv if argv is not None else sys.argv[1:])
        dynamo_args = config.dynamo_args
        server_args = config.server_args

        if dynamo_args.image_diffusion_worker and dynamo_args.video_generation_worker:
            raise ValueError(
                "SglangDiffusionEngine accepts only one of --image-diffusion-worker "
                "or --video-generation-worker, not both."
            )
        if dynamo_args.image_diffusion_worker:
            modality = _IMAGE
        elif dynamo_args.video_generation_worker:
            modality = _VIDEO
        else:
            raise ValueError(
                "SglangDiffusionEngine requires --image-diffusion-worker or "
                "--video-generation-worker (use dynamo.sglang.unified_main for "
                "text/diffusion-LLM workers)."
            )

        engine = cls(config, modality)
        worker_config = WorkerConfig.from_runtime_config(
            dynamo_args,
            model_name=server_args.model_path,
            served_model_name=server_args.served_model_name,
            # Raw media pipeline: request forwarded verbatim, no tokenizer.
            model_input=ModelInput.Text,
            endpoint_types=_ENDPOINT_TYPE_BY_MODALITY[modality],
            # Diffusion has no KV cache and no prefill/decode split.
            enable_kv_routing=False,
            disaggregation_mode=CommonDisaggregationMode.AGGREGATED,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id  # diffusion needs no cluster-wide per-worker key
        from sglang.multimodal_gen import DiffGenerator

        server_args = self.config.server_args
        tp_size = getattr(server_args, "tp_size", 1)
        dp_size = getattr(server_args, "dp_size", 1)

        logger.info(
            "Starting SglangDiffusionEngine: modality=%s, model=%s",
            self.modality,
            server_args.model_path,
        )
        self._generator = DiffGenerator.from_pretrained(
            model_path=server_args.model_path,
            num_gpus=tp_size * dp_size,
            tp_size=tp_size,
            dp_size=dp_size,
            dist_timeout=getattr(server_args, "dist_timeout", None),
        )
        fs = get_fs(self.config.dynamo_args.media_output_fs_url)
        self._handler = _make_handler(self.modality, self._generator, self.config, fs)
        logger.info("SglangDiffusionEngine ready (serving %s)", self.modality)

        return EngineConfig(
            model=server_args.model_path,
            served_model_name=server_args.served_model_name or server_args.model_path,
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
        from dynamo.sglang.health_check import (
            ImageDiffusionHealthCheckPayload,
            VideoGenerationHealthCheckPayload,
        )

        model_path = self.config.server_args.model_path
        if self.modality == _IMAGE:
            return ImageDiffusionHealthCheckPayload(model_path=model_path).to_dict()
        return VideoGenerationHealthCheckPayload(model_path=model_path).to_dict()

    async def cleanup(self) -> None:
        # Null-safe against a partial start() (the ABC contract).
        if self._handler is not None:
            # Handler.cleanup() drops the generator + empties the CUDA cache.
            self._handler.cleanup()
            self._handler = None
        elif self._generator is not None:
            # Partial start: generator built but _make_handler failed, so the
            # handler never took ownership. Free the model directly rather than
            # leak its weights (and scheduler child).
            import torch

            del self._generator
            torch.cuda.empty_cache()
        self._generator = None
