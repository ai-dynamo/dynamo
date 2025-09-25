# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock backend engine for Dynamo.

This worker streams pre-defined responses at a configurable speed. It is useful for
integration tests or demos where you want deterministic behaviour without a real
model backend.
"""

import argparse
import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import uvloop

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_RESPONSE_TEXT = """<|start|>assistant<|channel|>analysis<|message|>We need to call get_current_weather with location "New York" and format "fahrenheit".<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather<|constrain|>json<|message|>{"format":"fahrenheit","location":"New York"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_current_weather<|constrain|>json<|message|>{"temperature": 72,"unit":"fahrenheit","description":"Partly cloudy"}<|call|><|start|>assistant<|channel|>final<|message|>Here's the current weather in New York (US units):- **Temperature:** 72 °F- **Conditions:** Partly cloudyLet me know if you'd like more details (e.g., humidity, wind, forecast).<|end|>"""
DEFAULT_FINISH_REASON = "stop"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


@dataclass
class Config:
    namespace: str
    component: str
    endpoint: str
    model: str
    served_model_name: Optional[str]
    migration_limit: int
    tool_call_parser: Optional[str]
    reasoning_parser: Optional[str]
    response_text: str
    chunk_size: int
    chunk_delay: float
    chars_per_second: float
    finish_reason: str


class MockEngine:
    """Utility that slices a static response into streamed chunks."""

    def __init__(
        self,
        response_text: str,
        chunk_size: int,
        chunk_delay: float,
        chars_per_second: float,
        finish_reason: str,
    ) -> None:
        self.response_text = response_text
        self.chunk_size = max(1, chunk_size)
        self.chunk_delay = max(0.0, chunk_delay)
        self.chars_per_second = max(0.0, chars_per_second)
        self.finish_reason = finish_reason or DEFAULT_FINISH_REASON

    def delay_for_chunk(self, chunk_length: int) -> float:
        if self.chars_per_second > 0 and chunk_length > 0:
            return chunk_length / self.chars_per_second
        return self.chunk_delay

    async def wait_between_chunks(self, chunk_length: int) -> None:
        delay = self.delay_for_chunk(chunk_length)
        if delay > 0:
            await asyncio.sleep(delay)

    def usage(self) -> dict:
        completion_tokens = len(self.response_text)
        return {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        }


class RequestHandler:
    """Handler bridged into Dynamo runtime."""

    def __init__(self, engine: MockEngine, reported_model_name: str) -> None:
        self.engine = engine
        self.reported_model_name = reported_model_name

    async def generate(self, request: dict, context=None):
        model_name = request.get("model", self.reported_model_name)
        stream = request.get("stream", True)
        request_id = f"mock-{uuid.uuid4().hex}"
        created = int(time.time())
        logger.debug("Mock request %s received (stream=%s)", request_id, stream)

        if not stream:
            # Non-streaming response: mimic OpenAI's chat completion payload.
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": self.engine.response_text,
                        },
                        "finish_reason": self.engine.finish_reason,
                    }
                ],
                "usage": self.engine.usage(),
            }
            yield response
            return

        # Streaming response mode.
        role_sent = False
        text = self.engine.response_text
        text_length = len(text)
        index = 0

        while index < text_length:
            if _should_stop(context):
                logger.debug("Mock request %s cancelled mid-stream", request_id)
                raise GeneratorExit("Mock engine request cancelled")

            chunk = text[index : index + self.engine.chunk_size]
            delta = {}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            if chunk:
                delta["content"] = chunk

            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            logger.debug("Mock request %s sent chunk (%d:%d)", request_id, index, index + len(chunk))
            yield payload

            index += len(chunk)
            if index < text_length:
                await self.engine.wait_between_chunks(len(chunk))

        if _should_stop(context):
            logger.debug("Mock request %s cancelled before final chunk", request_id)
            raise GeneratorExit("Mock engine request cancelled")

        final_payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": self.engine.finish_reason,
                }
            ],
            "usage": self.engine.usage(),
        }
        logger.debug("Mock request %s finished", request_id)
        yield final_payload


def _should_stop(context: Any) -> bool:
    if context is None:
        return False
    for attr in ("is_stopped", "is_killed"):
        check = getattr(context, attr, None)
        if callable(check) and check():
            return True
    return False


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    config = parse_args()

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    endpoint = component.endpoint(config.endpoint)

    runtime_config = ModelRuntimeConfig()
    runtime_config.tool_call_parser = config.tool_call_parser
    runtime_config.reasoning_parser = config.reasoning_parser

    await register_llm(
        ModelInput.Text,
        ModelType.Chat | ModelType.Completions,
        endpoint,
        config.model,
        config.served_model_name,
        migration_limit=config.migration_limit,
        runtime_config=runtime_config,
    )

    engine = MockEngine(
        response_text=config.response_text,
        chunk_size=config.chunk_size,
        chunk_delay=config.chunk_delay,
        chars_per_second=config.chars_per_second,
        finish_reason=config.finish_reason,
    )
    reported_model = config.served_model_name or config.model
    handler = RequestHandler(engine, reported_model)

    effective_delay = engine.delay_for_chunk(engine.chunk_size)
    logger.info(
        "Mock engine started for model '%s' (chunk_size=%d, base_delay=%.3fs, cps=%.2f, effective_delay=%.3fs)",
        reported_model,
        engine.chunk_size,
        engine.chunk_delay,
        engine.chars_per_second,
        effective_delay,
    )

    await endpoint.serve_endpoint(handler.generate)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Mock Dynamo backend worker.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=(
            "Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. "
            f"Default: {DEFAULT_ENDPOINT}"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Identifier reported as the backing model (model_path argument to register_llm).",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="",
        help="Optional name exposed to clients. Defaults to the model identifier.",
    )
    parser.add_argument(
        "--migration-limit",
        type=int,
        default=0,
        help="Maximum number of times a request may migrate to another worker.",
    )
    parser.add_argument(
        "--response-text",
        type=str,
        default=DEFAULT_RESPONSE_TEXT,
        help="Static response returned by the mock engine.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of characters streamed per chunk.",
    )
    parser.add_argument(
        "--chunk-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between streamed chunks when chars-per-second is not set.",
    )
    parser.add_argument(
        "--chars-per-second",
        type=float,
        default=0.0,
        help="Target output speed. Overrides chunk-delay when greater than zero.",
    )
    parser.add_argument(
        "--finish-reason",
        type=str,
        default=DEFAULT_FINISH_REASON,
        help="Finish reason reported in the final chunk.",
    )

    parser.add_argument(
        "--dyn-tool-call-parser",
        type=str,
        default="",
        help="Name of the tool call parser to advertise via runtime metadata.",
    )
    parser.add_argument(
        "--dyn-reasoning-parser",
        type=str,
        default="",
        help="Name of the reasoning parser to advertise via runtime metadata.",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    if args.chunk_delay < 0:
        parser.error("--chunk-delay cannot be negative")
    if args.chars_per_second < 0:
        parser.error("--chars-per-second cannot be negative")

    endpoint_str = args.endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        parser.error(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint'."
        )

    namespace, component, endpoint = endpoint_parts

    served_model_name = args.served_model_name or None
    tool_call_parser = args.dyn_tool_call_parser or None
    reasoning_parser = args.dyn_reasoning_parser or None

    return Config(
        namespace=namespace,
        component=component,
        endpoint=endpoint,
        model=args.model,
        served_model_name=served_model_name,
        migration_limit=args.migration_limit,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        response_text=args.response_text,
        chunk_size=args.chunk_size,
        chunk_delay=args.chunk_delay,
        chars_per_second=args.chars_per_second,
        finish_reason=args.finish_reason,
    )


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
