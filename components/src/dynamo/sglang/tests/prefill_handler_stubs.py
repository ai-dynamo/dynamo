# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared stubs for the SGLang prefill handler unit tests."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler


def kv_hit_prefill_handler(monkeypatch, responses, *, cancelled):
    """Prefill handler whose engine yields ``responses``; ``cancelled`` fires the monitor."""

    async def results():
        for response in responses:
            yield response

    class _Engine:
        async def async_generate(self, **kwargs):
            return results()

    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = _Engine()
    handler.config = SimpleNamespace(server_args=SimpleNamespace())
    handler.shutdown_event = None
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._generate_bootstrap_room = lambda: 17
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}

    @asynccontextmanager
    async def cancellation_monitor(*args, **kwargs):
        async def fire_or_wait():
            if not cancelled:
                await asyncio.Future()

        task = asyncio.create_task(fire_or_wait())
        if cancelled:
            await task
        try:
            yield task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    handler._cancellation_monitor = cancellation_monitor
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.llm.prefill_handler.require_reasoning_kwargs",
        lambda engine, request, **kwargs: {},
    )
    return handler


async def drain(stream):
    """Collect every item from an async stream."""
    return [item async for item in stream]
