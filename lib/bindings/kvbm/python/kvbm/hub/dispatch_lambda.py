# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the synchronous `(req_dict, event) -> None` closure the
Rust velo handler invokes for each dispatched prefill request.

The closure captures the vLLM `AsyncLLM` instance and its asyncio loop.
Each invocation schedules `_run` onto the loop via
`call_soon_threadsafe`, returning immediately so the Rust caller can
drop the GIL and await the `CompletionEvent` on a tokio oneshot.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable


def make_dispatch_lambda(
    engine: Any, loop: asyncio.AbstractEventLoop
) -> Callable[[dict, Any], None]:
    """Build the `(req_dict, event) -> None` callable.

    `engine` is a vLLM `AsyncLLM` (or anything with the same
    `async def generate(prompt, sampling_params, request_id)` shape).
    `loop` is the asyncio loop that owns the engine — typically the one
    running when the caller constructs the handler.

    The returned closure is invoked under the GIL from a Rust velo
    handler. It schedules `_run` onto `loop` via `call_soon_threadsafe`
    (thread-safe by contract) and returns instantly.
    """
    # Local imports so this module can be imported without vLLM installed —
    # the lambda is only called when a worker is actually serving prefill.
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    async def _run(req: dict, event: Any) -> None:
        try:
            prompt = TokensPrompt(prompt_token_ids=req["token_ids"])
            sp = SamplingParams(temperature=0.0, max_tokens=1)
            transfer_params = req.get("kv_transfer_params")
            if transfer_params is not None:
                if sp.extra_args is None:
                    sp.extra_args = {}
                sp.extra_args["kv_transfer_params"] = transfer_params
            req_id = req.get("request_id") or str(uuid.uuid4())
            async for _ in engine.generate(prompt, sp, req_id):
                pass
            event.ok()
        except BaseException as e:
            try:
                event.err(repr(e))
            except Exception:
                pass

    def _lambda(req: dict, event: Any) -> None:
        loop.call_soon_threadsafe(lambda: loop.create_task(_run(req, event)))

    return _lambda
