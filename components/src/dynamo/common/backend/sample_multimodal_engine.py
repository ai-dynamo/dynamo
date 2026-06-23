# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference CPU-only multimodal engine (VLM / Omni model, no GPU required).

Shows how a VLM implements the two worker roles in the encode-disaggregated
(E/Agg) topology.  Both paths use synthetic logic — no GPU, no real weights.
Use ``sample_engine.py`` for the full P/D disaggregated text-only path.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode

from .engine import EngineConfig, GenerateChunk, GenerateRequest, LLMEngine
from .multimodal import encoder_terminal_chunk, extract_multimodal_inputs
from .worker import WorkerConfig


class SampleMultimodalEngine(LLMEngine):
    """Reference CPU-only VLM engine.

    Override ``_run_encoder`` for the encode role.  ``vision_tokens``
    controls synthetic encoding latency.
    """

    def __init__(
        self,
        model_name: str = "sample-multimodal-model",
        max_tokens: int = 16,
        vision_tokens: int = 64,
        delay: float = 0.01,
        disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.vision_tokens = vision_tokens
        self.delay = delay
        self.disaggregation_mode = disaggregation_mode

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SampleMultimodalEngine, WorkerConfig]:
        parser = argparse.ArgumentParser(
            description="Sample Dynamo multimodal (VLM) backend"
        )
        parser.add_argument("--model-name", default="sample-multimodal-model")
        parser.add_argument("--namespace", default="dynamo")
        parser.add_argument("--component", default="sample-multimodal")
        parser.add_argument("--endpoint", default="generate")
        parser.add_argument("--max-tokens", type=int, default=16)
        parser.add_argument(
            "--vision-tokens",
            type=int,
            default=64,
            help="Number of synthetic image-embedding tokens produced per image.",
        )
        parser.add_argument("--delay", type=float, default=0.01)
        parser.add_argument("--endpoint-types", default="chat,completions")
        parser.add_argument("--discovery-backend", default="etcd")
        parser.add_argument("--request-plane", default="tcp")
        parser.add_argument("--event-plane", default=None)
        parser.add_argument(
            "--disaggregation-mode",
            choices=[
                DisaggregationMode.AGGREGATED.value,
                DisaggregationMode.ENCODE.value,
            ],
            default=DisaggregationMode.AGGREGATED.value,
            help="Worker role: 'agg' (default) or 'encode'.",
        )
        args = parser.parse_args(argv)

        mode = DisaggregationMode(args.disaggregation_mode)
        engine = cls(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            vision_tokens=args.vision_tokens,
            delay=args.delay,
            disaggregation_mode=mode,
        )
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.model_name,
            served_model_name=args.model_name,
            endpoint_types=args.endpoint_types,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
            disaggregation_mode=mode,
            enable_kv_routing=False,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.model_name,
            context_length=4096,
        )

    async def _run_encoder(self, request: GenerateRequest, context: Context) -> Any:
        """Synthetic vision-encoder forward pass.

        Handle format: ``enc_<prompt_len>_<vision_tokens>_<uuid8>``
        """
        await asyncio.sleep(self.delay * self.vision_tokens / 64)
        prompt_len = len(request.get("token_ids", []))  # type: ignore[arg-type]
        return {
            "handle": f"enc_{prompt_len}_{self.vision_tokens}_{uuid.uuid4().hex[:8]}"
        }

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.disaggregation_mode == DisaggregationMode.ENCODE:
            result = await self._run_encoder(request, context)
            prompt_len = len(request.get("token_ids", []))
            yield encoder_terminal_chunk(result, prompt_len)
            return

        mm_data = extract_multimodal_inputs(request)
        encoder_result = request.get("encoder_result")

        if mm_data is not None and encoder_result is not None:
            # Validate the handle forwarded by the EncodeRouter.
            handle = (
                encoder_result.get("handle", "")
                if isinstance(encoder_result, dict)
                else ""
            )
            if not isinstance(handle, str) or not handle.startswith("enc_"):
                raise ValueError(
                    f"encoder_result.handle has unexpected format: {handle!r}; "
                    "expected a string starting with 'enc_'"
                )
        elif mm_data is not None:
            await asyncio.sleep(self.delay * self.vision_tokens / 64)

        token_ids: list[int] = request.get("token_ids", [])  # type: ignore[assignment]
        prompt_len = len(token_ids)
        stop_conditions = request.get("stop_conditions") or {}
        max_new = stop_conditions.get("max_tokens") or self.max_tokens

        for i in range(max_new):
            if context.is_stopped():
                yield GenerateChunk(
                    token_ids=[],
                    index=0,
                    finish_reason="cancelled",
                    completion_usage={
                        "prompt_tokens": prompt_len,
                        "completion_tokens": i,
                        "total_tokens": prompt_len + i,
                    },
                )
                return
            await asyncio.sleep(self.delay)
            token_id = (i + 1) % 32000
            chunk = GenerateChunk(token_ids=[token_id], index=0)
            if i == max_new - 1:
                chunk["finish_reason"] = "length"
                chunk["completion_usage"] = {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": max_new,
                    "total_tokens": prompt_len + max_new,
                }
            yield chunk

    async def cleanup(self) -> None:
        pass
