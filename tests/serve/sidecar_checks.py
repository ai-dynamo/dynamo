# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
import time
import uuid
from pathlib import Path

import grpc
import requests

from dynamo.runtime import Context
from tests.fault_tolerance.cancellation.utils import CancellableRequest
from tests.router.helper import managed_runtime, poll_for_worker_instances
from tests.utils.client import send_request
from tests.utils.payloads import ChatPayload, StreamingChatPayload
from tests.utils.prometheus import find_metric_samples


def _metrics(port: int) -> str:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=2)
    response.raise_for_status()
    return response.text


def _trtllm_active_requests(port: int) -> int:
    # The launcher installs the optional OpenEngine bindings before validation.
    # Loading them here keeps collection working in the other backend images.
    server_pb2 = importlib.import_module("openengine.v1.server_pb2")
    services = importlib.import_module("openengine.v1.openengine_pb2_grpc")
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        load = services.ControlStub(channel).GetLoad(
            server_pb2.GetLoadRequest(), timeout=2
        )
    assert load.HasField(
        "running_requests"
    ), f"Missing OpenEngine active-request count: {load}"
    return load.running_requests


def _wait_for_scheduler(backend: str, port: int, *, is_active: bool = False) -> None:
    if backend == "trtllm":
        # GetLoad counts native in-flight requests, not internal scheduler slots.
        # Recovery below also requires the same engine to accept and finish work.
        deadline = time.monotonic() + 10
        while True:
            active = _trtllm_active_requests(port)
            if (active > 0) if is_active else (active == 0):
                return
            assert time.monotonic() < deadline, f"TRT-LLM active requests: {active}"
            time.sleep(0.05)
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
    engine_port: int,
    discovery_backend: str = "etcd",
    probe_path: Path | None = None,
) -> None:
    """Check explicit stop, consumer drop, and HTTP disconnect cleanup and recovery."""

    async def native_checks() -> None:
        with managed_runtime(discovery_backend, "tcp") as runtime:
            endpoint = runtime.endpoint(f"{namespace}.backend.generate")
            worker_ids = await poll_for_worker_instances(endpoint, 1, max_wait_time=10)
            assert len(worker_ids) == 1, worker_ids
            client = await endpoint.client()
            payload = {
                "model": model,
                "token_ids": [11] * 128,
                "stop_conditions": {"max_tokens": 2048, "ignore_eos": True},
                "sampling_options": {"temperature": 0.0},
            }

            async def drain_cancelled(stream) -> None:
                async for output in stream:
                    assert output.get("finish_reason") in (None, "cancelled"), output

            async def recover() -> None:
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

            for is_explicit_stop in (True, False):
                await asyncio.to_thread(_wait_for_scheduler, backend, engine_port)
                request_id = f"cancel-{is_explicit_stop}-{uuid.uuid4()}"
                context = Context(request_id)
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
                        _wait_for_scheduler, backend, engine_port, is_active=True
                    )
                    if is_explicit_stop:
                        context.stop_generating()
                        await asyncio.wait_for(drain_cancelled(stream), timeout=10)
                    del stream
                    await asyncio.to_thread(_wait_for_scheduler, backend, engine_port)
                    if probe_path is not None:
                        await asyncio.to_thread(
                            assert_engine_released,
                            probe_path,
                            engine_port,
                            request_id,
                            2048,
                        )
                    if not is_explicit_stop:
                        assert (
                            not context.is_stopped()
                        ), "Dropping the consumer stopped its context"
                finally:
                    context.stop_generating()

                await asyncio.wait_for(recover(), timeout=30)
                await asyncio.to_thread(_wait_for_scheduler, backend, engine_port)

    asyncio.run(native_checks())
    _wait_for_scheduler(backend, engine_port)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to a thousand."}],
        "max_tokens": 2048,
        "ignore_eos": True,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    before_requests = probe_events(probe_path) if probe_path is not None else []
    request = CancellableRequest()
    try:
        request.post(
            f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
            json=body,
            stream=True,
            timeout=10,
        )
        deadline = time.monotonic() + 10
        while request.get_response() is None:
            request.raise_for_early_failure()
            assert (
                time.monotonic() < deadline
            ), "No streaming response before cancellation"
            time.sleep(0.05)
        response = request.get_response()
        response.raise_for_status()
        # Retain the iterator until explicit cancellation: dropping iter_lines()
        # after its first event closes the underlying response on some clients.
        lines = response.iter_lines(chunk_size=1)
        for line in lines:
            assert time.monotonic() < deadline, "No token before cancellation"
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line.removeprefix(b"data:").strip())
            assert "error" not in event, event
            choices = event.get("choices", [])
            assert all(choice.get("finish_reason") is None for choice in choices), event
            if any((choice.get("delta") or {}).get("content") for choice in choices):
                break
        else:
            raise AssertionError("Stream ended before producing a token")
        _wait_for_scheduler(backend, engine_port, is_active=True)
    finally:
        request.cancel()
    _wait_for_scheduler(backend, engine_port)
    if probe_path is not None:
        previous_ids = {
            e.get("request_id") for e in before_requests if e["kind"] == "submitted"
        }
        requests = [
            e
            for e in probe_events(probe_path)
            if e["kind"] == "submitted"
            and e["port"] == engine_port
            and e["request_id"] not in previous_ids
        ]
        assert len(requests) == 1, requests
        assert_engine_released(probe_path, engine_port, requests[0]["request_id"], 2048)
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
    _wait_for_scheduler(backend, engine_port)


def _transferred(
    backend: str, prefill_port: int, probe_path: Path | None = None
) -> float:
    if backend == "sglang":
        return sum(
            find_metric_samples(
                _metrics(prefill_port), "sglang:kv_transfer_total_mb_sum"
            )
        )
    assert probe_path is not None
    return sum(
        event["bytes"]
        for event in probe_events(probe_path)
        if event["kind"] == "transfer"
    )


def assert_kv_transfer(
    *,
    backend: str,
    payload: ChatPayload,
    prefill_port: int,
    decode_port: int,
    probe_path: Path | None = None,
) -> dict:
    """Require a fresh completed transfer as well as a successful response."""

    before = _transferred(backend, prefill_port, probe_path)
    payload.body["messages"][0]["content"] = (
        f"Request {uuid.uuid4()}. " + payload.body["messages"][0]["content"]
    )
    with send_request(payload.url(), payload.body) as response:
        payload.process_response(response)
        result = response.json()
    deadline = time.monotonic() + 10
    while _transferred(backend, prefill_port, probe_path) <= before:
        assert time.monotonic() < deadline, f"{backend}: no completed KV transfer"
        time.sleep(0.05)
    _wait_for_scheduler(backend, decode_port)
    _wait_for_scheduler(backend, prefill_port)
    return result


def probe_events(path: Path) -> list[dict]:
    # A writer may be appending the last event; only parse complete JSONL rows.
    complete, _, _ = path.read_text().rpartition("\n")
    return [json.loads(line) for line in complete.splitlines()]


def assert_engine_released(
    path: Path,
    port: int,
    request_id: str,
    budget: int,
    *,
    require_active: bool = True,
) -> None:
    """Require executor completion before exhaustion and return of all KV blocks.

    Block reuse is disabled by the test probe so zero used blocks means actual
    request-resource release, independently of OpenEngine's registration map.
    """
    deadline = time.monotonic() + 10
    while True:
        events = [e for e in probe_events(path) if e["port"] == port]
        assert not [e for e in events if e["kind"] == "stats_error"], events[-3:]
        submissions = [
            e
            for e in events
            if e["kind"] == "submitted" and e["request_id"] == request_id
        ]
        if submissions:
            assert len(submissions) == 1, submissions
            submitted_at = submissions[0]["submitted_at"]
            # Public result IDs and scheduler IDs differ. These probes admit one
            # request at a time per engine; bind it to batches constructed after
            # submission, not to the gRPC map or asynchronously delivered old stats.
            assert not any(
                e["kind"] == "submitted" and e["submitted_at"] > submitted_at
                for e in events
            ), "Cleanup must be checked before admitting another request"
            stats = [
                e["stats"]
                for e in events
                if e["kind"] == "stats" and e["batch_started_at"] >= submitted_at
            ]
            request_ids = {
                r["id"] for stat in stats for r in stat.get("requestStats", [])
            }
            assert (
                len(request_ids) <= 1
            ), f"Concurrent executor work prevents attribution: {request_ids}"
            completed = [
                (stat, r)
                for stat in stats
                for r in stat.get("requestStats", [])
                if r["stage"] == "GENERATION_COMPLETE"
            ]
            active_ids = {
                r["id"]
                for stat in stats
                for r in stat.get("requestStats", [])
                if r["stage"] == "GENERATION_IN_PROGRESS"
            }
            if completed and (
                not require_active or completed[-1][1]["id"] in active_ids
            ):
                last, result = completed[-1]
                assert (
                    result["numGeneratedTokens"] < budget
                ), f"Cancelled request exhausted its budget: {result}"
                if any(
                    s["iter"] >= last["iter"]
                    and s["numQueuedRequests"] == 0
                    and s["kvCacheStats"]["usedNumBlocks"] == 0
                    for s in stats
                ):
                    return
        assert (
            time.monotonic() < deadline
        ), f"Executor did not release {request_id}: {events[-3:]}"
        time.sleep(0.05)
