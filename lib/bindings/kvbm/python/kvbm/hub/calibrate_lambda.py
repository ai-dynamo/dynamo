# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the `(resolved_dict, event) -> None` closure the Rust
calibrate handler invokes when triggered.

The Rust side resolves the user's `CalibrationRequest` against the
framework-captured `CalibrationDefaults` first, so this lambda only sees
the final, clamped knobs (`seq`, `osl`, `seed`, `body_vocab`,
`buster_vocab`, `warmup`). It does not run any analysis — it issues
single-stream requests against the vLLM `AsyncLLM`, captures per-token
timings, and emits a raw JSON payload back through
`event.ok_with_payload(json_str)`. All regression / model fitting
happens in Rust on the receiving side.

Each request's **first token is a globally-unique cache-buster value**
drawn from `buster_vocab`, with the buster counter seeded by
`time.time_ns()`. Without this, vLLM's prefix cache and the hub's G2 KV
index would serve calibration prompts from cache and skew the TTFT
measurements toward the cached path.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
import uuid
from typing import Any, Callable


def capture_calibration_defaults(engine: Any, model_id: str) -> dict:
    """Snapshot the engine's max-sequence / vocab caps as a plain dict
    the Rust handler can deserialize into `CalibrationDefaults`.

    Best-effort accessor walk over vLLM's `AsyncLLM` shape — falls back
    to `max_model_len` when finer-grained `max_input_len` /
    `max_output_len` are not exposed.
    """
    cfg = getattr(engine, "model_config", None) or getattr(engine, "vllm_config", None)
    if cfg is None:
        raise RuntimeError(
            "capture_calibration_defaults: engine has no model_config / vllm_config"
        )

    max_model_len = int(getattr(cfg, "max_model_len", 0)) or int(
        getattr(cfg, "max_seq_len", 0)
    )
    if max_model_len <= 0:
        raise RuntimeError(
            "capture_calibration_defaults: could not read max_model_len from engine"
        )

    max_input_len = int(getattr(cfg, "max_input_len", 0) or max_model_len)
    max_output_len = int(getattr(cfg, "max_output_len", 0) or max_model_len)

    tok = getattr(engine, "tokenizer", None)
    if tok is None:
        raise RuntimeError(
            "capture_calibration_defaults: engine has no tokenizer attribute"
        )
    vocab_size = int(getattr(tok, "vocab_size", 0)) or int(
        getattr(getattr(tok, "tokenizer", None), "vocab_size", 0)
    )
    if vocab_size <= 0:
        raise RuntimeError(
            "capture_calibration_defaults: could not read vocab_size from tokenizer"
        )

    return {
        "model_id": model_id,
        "max_seq_len": max_model_len,
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
        "vocab_size": vocab_size,
        "safe_vocab_lo": 1000,
    }


def make_calibrate_lambda(
    engine: Any, loop: asyncio.AbstractEventLoop
) -> Callable[[dict, Any], None]:
    """Build the `(resolved_dict, event) -> None` callable.

    `engine` is a vLLM `AsyncLLM` (or anything with the same
    `async def generate(prompt, sampling_params, request_id)` shape).
    `loop` is the asyncio loop that owns the engine.

    The returned closure is invoked under the GIL from a Rust velo
    handler. It schedules `_run` onto `loop` via `call_soon_threadsafe`
    and returns instantly.
    """
    # Local imports so this module can be imported without vLLM installed.
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    # Process-monotonic cache-buster counter. Seeded by time so distinct
    # calibration invocations within the same process never reuse first
    # tokens (which would let vLLM's prefix cache serve the prompt).
    _bust_cnt = itertools.count(start=int(time.time_ns()))

    def _bust(buster_lo: int, buster_hi: int) -> int:
        span = max(1, buster_hi - buster_lo)
        return buster_lo + (next(_bust_cnt) % span)

    async def _one(
        isl: int,
        osl: int,
        rng: random.Random,
        body_lo: int,
        body_hi: int,
        buster_lo: int,
        buster_hi: int,
    ) -> dict:
        assert isl >= 2, "isl must be >= 2 to fit cache-buster + body"
        first_token = _bust(buster_lo, buster_hi)
        token_ids = [first_token] + [
            rng.randint(body_lo, body_hi - 1) for _ in range(isl - 1)
        ]
        sp = SamplingParams(
            max_tokens=osl,
            min_tokens=osl,
            ignore_eos=True,
            temperature=0.0,
        )
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        per_token_us: list[int] = []
        ttft_us: int | None = None
        prev_us: int | None = None
        t0 = time.monotonic_ns()
        req_id = str(uuid.uuid4())
        async for _ in engine.generate(prompt, sp, req_id):
            now_us = (time.monotonic_ns() - t0) // 1000
            if ttft_us is None:
                ttft_us = now_us
            else:
                assert prev_us is not None
                per_token_us.append(now_us - prev_us)
            prev_us = now_us
        if ttft_us is None:
            raise RuntimeError(f"engine.generate returned no tokens for isl={isl}")
        return {
            "isl": isl,
            "osl": osl,
            "ttft_us": ttft_us,
            "itl_us": per_token_us,
            "first_token": first_token,
        }

    async def _run(resolved: dict, event: Any) -> None:
        try:
            seq: list[int] = list(resolved["seq"])
            osl: int = int(resolved["osl"])
            body_lo, body_hi = resolved["body_vocab"]
            buster_lo, buster_hi = resolved["buster_vocab"]
            seed = int(resolved["seed"])
            do_warmup = bool(resolved.get("warmup", True))
            rng = random.Random(seed)

            if do_warmup:
                # Tiny request just to ensure the engine is hot; result discarded.
                await _one(128, 16, rng, body_lo, body_hi, buster_lo, buster_hi)

            traces = []
            for isl in seq:
                traces.append(
                    await _one(isl, osl, rng, body_lo, body_hi, buster_lo, buster_hi)
                )
            event.ok_with_payload(json.dumps({"traces": traces}))
        except BaseException as e:
            try:
                event.err(repr(e))
            except Exception:
                pass

    def _lambda(resolved: dict, event: Any) -> None:
        loop.call_soon_threadsafe(lambda: loop.create_task(_run(resolved, event)))

    return _lambda
