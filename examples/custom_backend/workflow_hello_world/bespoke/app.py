# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the bespoke gateway or one of its HTTP stage workers."""

import argparse
import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

import aiohttp
from aiohttp import web

from examples.custom_backend.workflow_hello_world.common import (
    HelloStage,
    MergeStage,
    WorldStage,
)

SESSION_KEY = web.AppKey("stage_client", aiohttp.ClientSession)
HELLO_URL_KEY = web.AppKey("hello_url", str)
WORLD_URL_KEY = web.AppKey("world_url", str)


async def _read_object(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise web.HTTPBadRequest(text="request body must be JSON") from error
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    return payload


async def _hello(request: web.Request) -> web.Response:
    payload = await _read_object(request)
    return web.json_response({"text": await HelloStage().run(payload)})


async def _world(request: web.Request) -> web.Response:
    payload = await _read_object(request)
    return web.json_response({"text": await WorldStage().run(payload)})


async def _call_stage(
    session: aiohttp.ClientSession, url: str, payload: Mapping[str, Any]
) -> str:
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        result = await response.json()
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError(f"stage {url!r} returned an invalid response")
    return result["text"]


async def _fan_out(
    session: aiohttp.ClientSession,
    hello_url: str,
    world_url: str,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    hello_task = asyncio.create_task(_call_stage(session, hello_url, payload))
    world_task = asyncio.create_task(_call_stage(session, world_url, payload))
    tasks = (hello_task, world_task)
    try:
        hello, world = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return hello, world


def _openai_response(model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _chat_completions(request: web.Request) -> web.Response:
    payload = await _read_object(request)
    if payload.get("stream") is True:
        raise web.HTTPBadRequest(text="this comparison supports stream=false only")
    model = payload.get("model")
    messages = payload.get("messages")
    if not isinstance(model, str) or not isinstance(messages, list):
        raise web.HTTPBadRequest(text="model and messages are required")

    app = request.app
    try:
        hello, world = await _fan_out(
            app[SESSION_KEY],
            app[HELLO_URL_KEY],
            app[WORLD_URL_KEY],
            payload,
        )
    except asyncio.TimeoutError as error:
        raise web.HTTPGatewayTimeout(text="a stage timed out") from error
    except (aiohttp.ClientError, ValueError) as error:
        raise web.HTTPBadGateway(text=f"stage invocation failed: {error}") from error

    text = await MergeStage().run(hello, world)
    return web.json_response(_openai_response(model, text))


async def _client_session(app: web.Application) -> AsyncIterator[None]:
    timeout_seconds = float(os.environ.get("BESPOKE_STAGE_TIMEOUT_S", "5"))
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_seconds)
    ) as session:
        app[SESSION_KEY] = session
        yield


def _gateway_app() -> web.Application:
    app = web.Application()
    app[HELLO_URL_KEY] = os.environ.get(
        "BESPOKE_HELLO_URL", "http://127.0.0.1:8101/run"
    )
    app[WORLD_URL_KEY] = os.environ.get(
        "BESPOKE_WORLD_URL", "http://127.0.0.1:8102/run"
    )
    app.cleanup_ctx.append(_client_session)
    app.router.add_post("/v1/chat/completions", _chat_completions)
    return app


def _stage_app(role: str) -> web.Application:
    app = web.Application()
    if role == "hello":
        app.router.add_post("/run", _hello)
    elif role == "world":
        app.router.add_post("/run", _world)
    else:
        raise ValueError(f"unsupported stage role: {role}")
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("gateway", "hello", "world"))
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    app = _gateway_app() if args.role == "gateway" else _stage_app(args.role)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
