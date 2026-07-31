# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import base64
import time
from collections.abc import AsyncGenerator
from typing import Any

from dynamo._core import Context
from dynamo.llm import ModelInput

from .engine import DiffusionEngine, EngineConfig
from .health_check import build_raw_health_check_payload, is_probe
from .worker import WorkerConfig

# A 1x1 transparent PNG — the smallest valid image, used so the sample engine
# returns a real (decodable) image without a GPU or model.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class SampleDiffusionEngine(DiffusionEngine):
    """Reference CPU-only DiffusionEngine (no model). Serves ``/v1/images`` by
    returning a fixed tiny PNG after a delay — exercises the raw media pipeline
    (RawEngineAdapter + PyRawEngine) end-to-end without a GPU, and is a template
    for real engines. ``generate`` takes the OpenAI image-request dict and
    yields one ``NvImagesResponse``-shaped dict.
    """

    def __init__(
        self,
        model_name: str = "sample-diffusion-model",
        delay: float = 0.05,
    ):
        self.model_name = model_name
        self.delay = delay

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SampleDiffusionEngine, WorkerConfig]:
        parser = argparse.ArgumentParser(description="Sample Dynamo diffusion backend")
        parser.add_argument("--model-name", default="sample-diffusion-model")
        parser.add_argument("--namespace", default="dynamo")
        parser.add_argument("--component", default="sample")
        parser.add_argument("--endpoint", default="generate")
        parser.add_argument("--delay", type=float, default=0.05)
        parser.add_argument("--endpoint-types", default="images")
        parser.add_argument("--discovery-backend", default="etcd")
        parser.add_argument("--request-plane", default="tcp")
        parser.add_argument("--event-plane", default=None)
        args = parser.parse_args(argv)

        engine = cls(model_name=args.model_name, delay=args.delay)
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.model_name,
            served_model_name=args.model_name,
            # Raw media pipeline: the frontend forwards the request verbatim,
            # so there is no tokenizer stage.
            model_input=ModelInput.Text,
            endpoint_types=args.endpoint_types,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
            # Diffusion has no KV cache to route on.
            enable_kv_routing=False,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.model_name,
        )

    async def health_check_payload(self) -> dict[str, Any]:
        # A minimal image request the canary can send through generate().
        return build_raw_health_check_payload(
            {"prompt": "ping", "n": 1, "response_format": "b64_json"}
        )

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        del context  # sample engine ignores cancellation (single short step)

        raw_n = request.get("n", 1)
        try:
            n = int(raw_n)
        except (TypeError, ValueError) as exc:
            raise ValueError("`n` must be an integer") from exc
        if n < 1 or n > 10:
            raise ValueError("`n` must be between 1 and 10")
        response_format = request.get("response_format") or "url"

        # Simulate a single denoising step.
        if not is_probe(request):
            await asyncio.sleep(self.delay)

        image_bytes = base64.b64decode(_TINY_PNG_B64)
        data: list[dict[str, Any]] = []
        for _ in range(n):
            if response_format == "b64_json":
                data.append({"b64_json": base64.b64encode(image_bytes).decode("utf-8")})
            else:
                # No object store in the sample engine; return a data URL so
                # "url" responses are still self-contained.
                data.append({"url": f"data:image/png;base64,{_TINY_PNG_B64}"})

        yield {"created": int(time.time()), "data": data}

    async def cleanup(self) -> None:
        pass
