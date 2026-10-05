# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
import uuid
from itertools import product
from pathlib import Path

import aiohttp
import requests

from dynamo.runtime import Context
from tests.fault_tolerance.cancellation.utils import (
    CancellableRequest,
    read_streaming_responses,
)
from tests.router.helper import managed_runtime, poll_for_worker_instances
from tests.serve.sidecar_native_checks import (
    assert_native_completion,
    assert_native_logprobs,
)
from tests.utils.client import send_request
from tests.utils.payloads import ChatPayload, StreamingChatPayload
from tests.utils.prometheus import find_metric_samples


def _metrics(port: int) -> str:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=2)
    response.raise_for_status()
    return response.text


def _wait_for_scheduler(backend: str, port: int, *, is_active: bool = False) -> None:
    names = {
        "vllm": ("vllm:num_requests_running", "vllm:num_requests_waiting"),
        "sglang": ("sglang:num_running_reqs", "sglang:num_queue_reqs"),
    }[backend]
    deadline = time.monotonic() + 10
    while True:
        body = _metrics(port)
        samples = [find_metric_samples(body, name) for name in names]
        assert all(samples), f"Missing scheduler metrics {names}: {body}"
        running, waiting = map(sum, samples)
        if running > 0 if is_active else running == waiting == 0:
            return
        assert (
            time.monotonic() < deadline
        ), f"{backend} scheduler is_active={is_active}: {body}"
        time.sleep(0.05)


def assert_cancellation_and_recovery(
    *,
    backend: str,
    model: str,
    namespace: str,
    frontend_port: int,
    engine_http_port: int,
    discovery_backend: str = "etcd",
) -> None:
    """Check native metadata, explicit stop, consumer drop, and HTTP disconnect."""

    async def native_checks() -> None:
        with managed_runtime(discovery_backend, "tcp") as runtime:
            endpoint = runtime.endpoint(f"{namespace}.backend.generate")
            worker_ids = await poll_for_worker_instances(endpoint, 1, max_wait_time=10)
            assert len(worker_ids) == 1, worker_ids
            client = await endpoint.client()
            await assert_native_logprobs(
                backend=backend, model=model, client=client, worker_id=worker_ids[0]
            )

            def payload(max_tokens: int, is_native_http: bool) -> dict:
                result = {
                    "model": model,
                    "token_ids": [11] * 128,
                    "stop_conditions": {"max_tokens": max_tokens, "ignore_eos": True},
                    "sampling_options": {"temperature": 0.0},
                }
                if is_native_http:
                    result["extra_args"] = {
                        "sglang_tito": {
                            "sampling_params": {
                                "max_new_tokens": max_tokens,
                                "ignore_eos": True,
                                "temperature": 0.0,
                            },
                        }
                    }
                return result

            async def drain_cancelled(stream, is_native_http: bool) -> None:
                try:
                    async for output in stream:
                        assert output.get("finish_reason") in (
                            None,
                            "cancelled",
                        ), output
                except ValueError as error:
                    if not is_native_http or not str(error).startswith("Cancelled:"):
                        raise

            async def recover(is_native_http: bool) -> None:
                stream = await client.direct(
                    payload(4, is_native_http),
                    worker_ids[0],
                    annotated=False,
                )
                outputs = [output async for output in stream]
                if is_native_http:
                    assert outputs, "Sidecar produced no native HTTP response"
                    assert all(not output["token_ids"] for output in outputs), outputs
                    raw = [
                        output["engine_data"]["sglang_response"] for output in outputs
                    ]
                    assert sum(len(item["output_ids"]) for item in raw) == 4, raw
                    assert all(
                        output.get("finish_reason") is None for output in outputs[:-1]
                    ), outputs
                    assert outputs[-1]["finish_reason"] == "stop", outputs[-1]
                    usage = raw[-1]["meta_info"]
                    assert usage["finish_reason"]["type"] == "length", usage
                    assert usage["prompt_tokens"] == 128, usage
                    assert usage["completion_tokens"] == 4, usage
                else:
                    assert_native_completion(
                        outputs, prompt_tokens=128, completion_tokens=4
                    )

            for is_native_http, is_explicit_stop in product(
                (False, True) if backend == "sglang" else (False,), (True, False)
            ):
                await asyncio.to_thread(_wait_for_scheduler, backend, engine_http_port)
                context = Context(
                    f"cancel-{is_native_http}-{is_explicit_stop}-{uuid.uuid4()}"
                )
                try:
                    stream = await asyncio.wait_for(
                        client.direct(
                            payload(2048, is_native_http),
                            worker_ids[0],
                            annotated=False,
                            context=context,
                        ),
                        timeout=10,
                    )
                    output = await asyncio.wait_for(anext(stream), timeout=10)
                    tokens = (
                        output["engine_data"]["sglang_response"]["output_ids"]
                        if is_native_http
                        else output["token_ids"]
                    )
                    assert tokens and output.get("finish_reason") is None, output
                    await asyncio.to_thread(
                        _wait_for_scheduler, backend, engine_http_port, is_active=True
                    )
                    if is_explicit_stop:
                        context.stop_generating()
                        await asyncio.wait_for(
                            drain_cancelled(stream, is_native_http), timeout=10
                        )
                    del stream
                    await asyncio.to_thread(
                        _wait_for_scheduler, backend, engine_http_port
                    )
                    if not is_explicit_stop:
                        assert (
                            not context.is_stopped()
                        ), "Dropping the consumer stopped its context"
                finally:
                    context.stop_generating()

                await asyncio.wait_for(recover(is_native_http), timeout=30)
                await asyncio.to_thread(_wait_for_scheduler, backend, engine_http_port)

    asyncio.run(native_checks())
    _wait_for_scheduler(backend, engine_http_port)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to a thousand."}],
        "max_tokens": 2048,
        "ignore_eos": True,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    request = CancellableRequest()
    try:
        request.post(
            f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
            json=body,
            stream=True,
            timeout=30,
        )
        deadline = time.monotonic() + 10
        while request.get_response() is None:
            request.raise_for_early_failure()
            assert (
                time.monotonic() < deadline
            ), "No streaming response before cancellation"
            time.sleep(0.05)
        read_streaming_responses(
            request, expected_count=1, deadline_s=10, require_content=True
        )
        _wait_for_scheduler(backend, engine_http_port, is_active=True)
    finally:
        request.cancel()
    _wait_for_scheduler(backend, engine_http_port)
    payload = StreamingChatPayload(
        body={**body, "max_tokens": 4, "stream_options": {"include_usage": True}},
        expected_response=[],
        expected_log=[],
        expected_finish_reason="length",
        expected_completion_tokens=4,
        port=frontend_port,
    )
    with send_request(payload.url(), payload.body, stream=True) as response:
        payload.process_response(response)
    _wait_for_scheduler(backend, engine_http_port)


def kv_transfer_total(
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
    decode_http_port: int,
    probe_path: Path | None = None,
) -> None:
    """Require a fresh completed transfer as well as a successful response."""

    before = kv_transfer_total(backend, prefill_http_port, probe_path)
    payload.body["messages"][0]["content"] = (
        f"Request {uuid.uuid4()}. " + payload.body["messages"][0]["content"]
    )
    with send_request(payload.url(), payload.body) as response:
        payload.process_response(response)
    deadline = time.monotonic() + 10
    while kv_transfer_total(backend, prefill_http_port, probe_path) <= before:
        assert time.monotonic() < deadline, f"{backend}: no completed KV transfer"
        time.sleep(0.05)
    _wait_for_scheduler(backend, decode_http_port)
    _wait_for_scheduler(backend, prefill_http_port)


def assert_sglang_transfer_wait_cancelled(
    *,
    namespace: str,
    model: str,
    decode_http_port: int,
    bootstrap_port: int,
    discovery_backend: str = "etcd",
) -> None:
    """Cancel unmatched decode work through the existing sidecar deployment."""

    async def run() -> None:
        with managed_runtime(discovery_backend, "tcp") as runtime:
            endpoint = runtime.endpoint(f"{namespace}.backend.generate")
            worker_ids = await poll_for_worker_instances(endpoint, 1, max_wait_time=10)
            assert len(worker_ids) == 1, worker_ids
            client = await endpoint.client()
            context = Context(f"transfer-wait-{uuid.uuid4()}")
            payload = {
                "model": model,
                "token_ids": [11] * 128,
                "stop_conditions": {"max_tokens": 8, "ignore_eos": True},
                "sampling_options": {"temperature": 0.0},
                "bootstrap_info": {
                    "bootstrap_host": "127.0.0.1",
                    "bootstrap_port": bootstrap_port,
                    "bootstrap_room": uuid.uuid4().int & ((1 << 63) - 1),
                },
            }

            async def generate() -> None:
                stream = await client.direct(
                    payload, worker_ids[0], annotated=False, context=context
                )
                async for output in stream:
                    assert (
                        context.is_stopped()
                    ), f"Decode produced output without its prefill peer: {output}"
                    assert_native_completion(
                        [output],
                        prompt_tokens=128,
                        completion_tokens=0,
                        finish_reason="cancelled",
                    )

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2)
            ) as session:

                async def wait_for_transfer_queue(has_work: bool) -> None:
                    deadline = asyncio.get_running_loop().time() + 10
                    while True:
                        if has_work and generation.done():
                            await generation
                            raise AssertionError("Decode ended before KV transfer")
                        async with session.get(
                            f"http://127.0.0.1:{decode_http_port}/v1/loads?include=core"
                        ) as response:
                            response.raise_for_status()
                            body = await response.json()
                        loads = body["loads"]
                        assert loads, body
                        running = sum(load["num_running_reqs"] for load in loads)
                        waiting = sum(load["num_waiting_reqs"] for load in loads)
                        is_awaiting_kv = sum(
                            load["num_total_tokens"] for load in loads
                        ) > sum(load["num_active_tokens"] for load in loads)
                        if running == 0 and (
                            waiting > 0 and is_awaiting_kv if has_work else waiting == 0
                        ):
                            return
                        assert (
                            asyncio.get_running_loop().time() < deadline
                        ), f"SGLang transfer queue has_work={has_work}: {body}"
                        await asyncio.sleep(0.05)

                await wait_for_transfer_queue(False)
                generation = asyncio.create_task(generate())
                try:
                    await wait_for_transfer_queue(True)
                finally:
                    context.stop_generating()
                    # The transport may close before forwarding the cancelled terminal.
                    await asyncio.wait_for(generation, timeout=10)
                await wait_for_transfer_queue(False)

    asyncio.run(run())
