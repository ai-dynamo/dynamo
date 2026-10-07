# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
import uuid
from pathlib import Path

import requests

from dynamo.runtime import Context
from tests.router.helper import managed_runtime, poll_for_worker_instances
from tests.utils.client import send_request
from tests.utils.engine_metrics import EngineMetrics
from tests.utils.payloads import ChatPayload
from tests.utils.prometheus import find_metric_samples


def _metrics(port: int) -> str:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=2)
    response.raise_for_status()
    return response.text


def _assert_native_completion(
    outputs: list[dict],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "length",
) -> None:
    assert outputs, "Sidecar produced no response"
    assert sum(len(output["token_ids"]) for output in outputs) == completion_tokens
    assert sum(output.get("finish_reason") is not None for output in outputs) == 1
    terminal = outputs[-1]
    assert terminal["finish_reason"] == finish_reason, terminal
    usage = terminal["completion_usage"]
    assert usage["prompt_tokens"] == prompt_tokens, usage
    assert usage["completion_tokens"] == completion_tokens, usage
    assert usage["total_tokens"] == prompt_tokens + completion_tokens, usage


def assert_native_cancellation_and_recovery(
    *,
    metrics: EngineMetrics,
    model: str,
    namespace: str,
    discovery_backend: str = "etcd",
) -> None:
    """Check explicit native stop and consumer drop on the existing deployment."""
    max_tokens = 2048
    completion_progress = metrics.completion_progress(max_tokens)

    async def native_checks() -> None:
        with managed_runtime(discovery_backend, "tcp") as runtime:
            endpoint = runtime.endpoint(f"{namespace}.backend.generate")
            worker_ids = await poll_for_worker_instances(endpoint, 1, max_wait_time=10)
            assert len(worker_ids) == 1, worker_ids
            client = await endpoint.client()
            payload = {
                "model": model,
                "token_ids": [11] * 128,
                "stop_conditions": {"max_tokens": max_tokens, "ignore_eos": True},
                "sampling_options": {"temperature": 0.0},
            }

            async def drain_cancelled(stream) -> None:
                async for output in stream:
                    assert output.get("finish_reason") in (None, "cancelled"), output

            async def recover() -> None:
                before = await asyncio.to_thread(metrics.progress)
                stream = await client.direct(
                    {
                        **payload,
                        "stop_conditions": {"max_tokens": 4, "ignore_eos": True},
                    },
                    worker_ids[0],
                    annotated=False,
                )
                outputs = [output async for output in stream]
                _assert_native_completion(
                    outputs, prompt_tokens=128, completion_tokens=4
                )
                await asyncio.to_thread(
                    metrics.assert_recovered, before=before, max_tokens=4
                )

            for is_explicit_stop in (True, False):
                await asyncio.to_thread(metrics.wait_for_scheduler)
                before = await asyncio.to_thread(metrics.progress)
                context = Context(f"cancel-{is_explicit_stop}-{uuid.uuid4()}")
                try:
                    stream = await asyncio.wait_for(
                        client.direct(
                            payload, worker_ids[0], annotated=False, context=context
                        ),
                        timeout=10,
                    )
                    output = await asyncio.wait_for(anext(stream), timeout=10)
                    assert (
                        output["token_ids"] and output.get("finish_reason") is None
                    ), output
                    await asyncio.to_thread(metrics.wait_for_scheduler, active=True)
                    if is_explicit_stop:
                        context.stop_generating()
                        await asyncio.wait_for(drain_cancelled(stream), timeout=10)
                    del stream
                    await asyncio.to_thread(
                        metrics.assert_cancelled,
                        before=before,
                        completion_progress=completion_progress,
                    )
                    if not is_explicit_stop:
                        assert (
                            not context.is_stopped()
                        ), "Dropping the consumer stopped its context"
                finally:
                    context.stop_generating()

                await asyncio.wait_for(recover(), timeout=30 + metrics.settle_timeout)
                await asyncio.to_thread(metrics.wait_for_scheduler)

    asyncio.run(native_checks())


def _transferred(
    backend: str, prefill_http_port: int, probe_path: Path | None = None
) -> float:
    if backend == "sglang":
        return sum(
            find_metric_samples(
                _metrics(prefill_http_port), "sglang:kv_transfer_total_mb_sum"
            )
        )
    assert probe_path is not None
    return sum(
        json.loads(line)["bytes"] for line in probe_path.read_text().splitlines()
    )


def assert_kv_transfer(
    *,
    backend: str,
    payload: ChatPayload,
    prefill_http_port: int,
    prefill_metrics: EngineMetrics,
    decode_metrics: EngineMetrics,
    probe_path: Path | None = None,
) -> None:
    """Require a fresh completed transfer as well as a successful response."""

    before = _transferred(backend, prefill_http_port, probe_path)
    payload.body["messages"][0]["content"] = (
        f"Request {uuid.uuid4()}. " + payload.body["messages"][0]["content"]
    )
    with send_request(payload.url(), payload.body) as response:
        payload.process_response(response)
    deadline = time.monotonic() + 10
    while _transferred(backend, prefill_http_port, probe_path) <= before:
        assert time.monotonic() < deadline, f"{backend}: no completed KV transfer"
        time.sleep(0.05)
    decode_metrics.wait_for_scheduler()
    prefill_metrics.wait_for_scheduler()
