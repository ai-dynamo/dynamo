# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Squeeze-Evolve Dynamo component (experimental).

Usage (tiers cheapest-first; see README.md for the full per-tier schema):
    python -m dynamo.squeeze_evolve \\
        --tiers '[{"endpoint":"dynamo.cheap.generate","model":"Qwen/cheap"},
                  {"endpoint":"dynamo.expensive.generate","model":"Qwen/big"}]' \\
        --model-name squeeze-evolve/aime25 --confidence-percentiles 50 --loops 5

Registers as a chat model (``ModelInput.Text`` + ``ModelType.Chat``) so the
frontend routes ``/v1/chat/completions`` for ``--model-name`` to this service.
Each request becomes one Squeeze-Evolve problem; the evolutionary loop runs,
fanning generation across the tiers (each a per-tier ``KvRouter``), and the final
evolved candidate is streamed back as the assistant message.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator

import uvloop

from dynamo.llm import (
    AicPerfConfig,
    KvRouterConfig,
    ModelInput,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.squeeze_evolve.args import parse_args
from dynamo.squeeze_evolve.orchestrator import SqueezeEvolveOrchestrator

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI message content (string or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def extract_messages(request: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize an OpenAI chat request to role/text messages.

    Preserves the full conversation (system, user, assistant, ...) so the tier
    tokenizer's chat template sees it intact instead of only the user turns;
    multimodal content parts are flattened to their text. A non-list ``messages``
    payload yields an empty list rather than raising.
    """
    raw = request.get("messages")
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            continue
        text = _content_to_text(msg.get("content"))
        if text:
            messages.append({"role": role, "content": text})
    return messages


def chat_chunk(
    model: str,
    req_id: str,
    *,
    content: str | None = None,
    finish: str | None = None,
) -> dict[str, Any]:
    """Build one OpenAI ``chat.completion.chunk``."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta = {"role": "assistant", "content": content}
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class SqueezeEvolveHandler:
    """One Squeeze-Evolve problem per chat request against a shared orchestrator."""

    def __init__(
        self, model_name: str, orchestrator: SqueezeEvolveOrchestrator
    ) -> None:
        self._model_name = model_name
        self._orchestrator = orchestrator

    async def generate(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        model = self._model_name
        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        messages = extract_messages(request)
        if not messages:
            yield chat_chunk(model, req_id, content="")
            yield chat_chunk(model, req_id, finish="stop")
            return

        answer = await self._orchestrator.run(messages)
        yield chat_chunk(model, req_id, content=answer)
        yield chat_chunk(model, req_id, finish="stop")


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    tiers = config.tiers
    model_name = config.model_name
    if tiers is None or model_name is None:
        # config.validate() already rejects both; recheck to narrow the Optionals.
        raise ValueError("--tiers and --model-name are required")
    logger.info(
        "Squeeze-Evolve starting (model=%s, tiers=%d, loops=%d)",
        model_name,
        len(tiers),
        config.loops,
    )

    orchestrator = SqueezeEvolveOrchestrator(
        cfg=config.to_algo_config(),
        tiers=tiers,
        runtime=runtime,
        # Parity with dynamo.router.args.build_kv_router_config /
        # build_aic_perf_config, which are typed for DynamoRouterConfig
        # (single --endpoint); this config inherits the shared bases directly.
        kv_router_config=KvRouterConfig(**config.kv_router_kwargs()),
        aic_perf_config=(
            AicPerfConfig(**config.aic_perf_kwargs())
            if config.router_prefill_load_model == "aic"
            else None
        ),
        default_block_size=config.default_block_size,
    )
    handler = SqueezeEvolveHandler(model_name, orchestrator)

    generate_endpoint = runtime.endpoint(f"{config.namespace}.squeeze_evolve.generate")
    await register_model(
        model_input=ModelInput.Text,
        model_type=ModelType.Chat,
        endpoint=generate_endpoint,
        model_path=config.model_path or tiers[-1].model,
        model_name=model_name,
        worker_type=WorkerType.Aggregated,
    )

    logger.info(
        "Serving %s at %s.squeeze_evolve.generate", model_name, config.namespace
    )
    await generate_endpoint.serve_endpoint(
        handler.generate,
        graceful_shutdown=True,
        metrics_labels=[("service", "squeeze_evolve")],
    )


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
