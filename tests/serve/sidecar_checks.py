# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
import uuid
from pathlib import Path

import requests

from dynamo.runtime import Context
from tests.fault_tolerance.cancellation.utils import (
    CancellableRequest,
    read_streaming_responses,
)
from tests.router.helper import managed_runtime, poll_for_worker_instances
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


def _engine_progress(backend: str, port: int) -> float:
    name, labels = {
        "vllm": ("vllm:iteration_tokens_total_count", {}),
        "sglang": ("sglang:realtime_tokens_total", {"mode": "decode"}),
    }[backend]
    body = _metrics(port)
    samples = find_metric_samples(body, name, labels)
    assert samples, f"Missing engine progress counter {name}: {body}"
    return sum(samples)


def _wait_for_cleanup(
    backend: str, port: int, *, before: float, max_tokens: int
) -> None:
    _wait_for_scheduler(backend, port)
    # Read progress after idle; a single metrics response is not an atomic snapshot.
    delta = _engine_progress(backend, port) - before
    assert delta >= 0, f"{backend} engine progress counter reset: {delta}"
    # Non-speculative vLLM emits one batch per token; SGLang prefill supplies one.
    completed = max_tokens if backend == "vllm" else max_tokens - 1
    assert delta < completed, f"{backend} completed generation: {delta=}"


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


def assert_cancellation_and_recovery(
    *,
    backend: str,
    model: str,
    namespace: str,
    frontend_port: int,
    engine_http_port: int,
    discovery_backend: str = "etcd",
) -> None:
    """Check explicit stop, consumer drop, and HTTP disconnect cleanup and recovery."""
    max_tokens = 2048
    if backend == "sglang":
        body = _metrics(engine_http_port)
        for name, minimum in (
            ("sglang:context_len", 4096),
            ("sglang:max_total_num_tokens", 8192),
        ):
            samples = find_metric_samples(body, name)
            assert samples and min(samples) >= minimum, (
                f"Cancellation request may be shortened: {name} must be >= {minimum}: "
                f"{body}"
            )

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
                before = await asyncio.to_thread(
                    _engine_progress, backend, engine_http_port
                )
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
                if backend == "vllm":
                    deadline = time.monotonic() + 10
                    while True:
                        delta = (
                            await asyncio.to_thread(
                                _engine_progress, backend, engine_http_port
                            )
                            - before
                        )
                        if delta == 4:
                            break
                        assert 0 <= delta < 4 and time.monotonic() < deadline, (
                            "Expected one vLLM engine batch per recovery token: "
                            f"{delta=}"
                        )
                        await asyncio.sleep(0.05)

            for is_explicit_stop in (True, False):
                await asyncio.to_thread(_wait_for_scheduler, backend, engine_http_port)
                before = await asyncio.to_thread(
                    _engine_progress, backend, engine_http_port
                )
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
                    await asyncio.to_thread(
                        _wait_for_scheduler, backend, engine_http_port, is_active=True
                    )
                    if is_explicit_stop:
                        context.stop_generating()
                        await asyncio.wait_for(drain_cancelled(stream), timeout=10)
                    del stream
                    await asyncio.to_thread(
                        _wait_for_cleanup,
                        backend,
                        engine_http_port,
                        before=before,
                        max_tokens=max_tokens,
                    )
                    if not is_explicit_stop:
                        assert (
                            not context.is_stopped()
                        ), "Dropping the consumer stopped its context"
                finally:
                    context.stop_generating()

                await asyncio.wait_for(recover(), timeout=30)
                await asyncio.to_thread(_wait_for_scheduler, backend, engine_http_port)

    asyncio.run(native_checks())
    _wait_for_scheduler(backend, engine_http_port)
    before = _engine_progress(backend, engine_http_port)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to a thousand."}],
        "max_tokens": max_tokens,
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
    _wait_for_cleanup(backend, engine_http_port, before=before, max_tokens=max_tokens)
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
    decode_http_port: int,
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
    _wait_for_scheduler(backend, decode_http_port)
    _wait_for_scheduler(backend, prefill_http_port)
