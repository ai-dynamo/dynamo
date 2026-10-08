# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TRT-LLM handoff parity and cancellation assertions."""

import asyncio
import importlib
import time
import uuid
from pathlib import Path

import grpc

from dynamo.runtime import Context
from tests.router.helper import managed_runtime, poll_for_worker_instances
from tests.serve.sidecar_checks import (
    _assert_native_completion,
    _trtllm_active_requests,
    assert_engine_released,
    probe_events,
)
from tests.utils.router_logs import has_upstream_cancellation


def assert_handoff_parity_and_cancellation(
    *,
    model: str,
    namespace: str,
    discovery_backend: str,
    prefill_port: int,
    decode_port: int,
    probe_path: Path,
    result: dict,
    worker_log: Path,
) -> None:
    async def check():
        with managed_runtime(discovery_backend, "tcp") as runtime:
            prefill_endpoint = runtime.endpoint(f"{namespace}.prefill.generate")
            decode_endpoint = runtime.endpoint(f"{namespace}.backend.generate")
            prefill_ids = await poll_for_worker_instances(
                prefill_endpoint, 1, max_wait_time=10
            )
            decode_ids = await poll_for_worker_instances(
                decode_endpoint, 1, max_wait_time=10
            )
            assert len(prefill_ids) == len(decode_ids) == 1
            prefill = await prefill_endpoint.client()
            decode = await decode_endpoint.client()
            prompt = result["nvext"]["prompt_token_ids"]
            payload = {
                "model": model,
                "token_ids": prompt,
                "sampling_options": {"temperature": 0.0},
                "stop_conditions": {"max_tokens": 8, "ignore_eos": True},
            }
            # A decode worker without a handoff runs prefill locally: same model,
            # prompt and sampling, without any transferred cache contents.
            stream = await decode.direct(payload, decode_ids[0], annotated=False)
            outputs = [output async for output in stream]
            _assert_native_completion(
                outputs, prompt_tokens=len(prompt), completion_tokens=8
            )
            tokens = [token for output in outputs for token in output["token_ids"]]
            assert (
                tokens == result["nvext"]["completion_token_ids"]
            ), "Transferred KV changed greedy token output"

            prefill_id = f"prefill-cancel-{uuid.uuid4()}"
            stream = await prefill.direct(
                payload, prefill_ids[0], annotated=False, context=Context(prefill_id)
            )
            outputs = [output async for output in stream]
            handoff = outputs[-1]["disaggregated_params"]
            # Runtime trace propagation accepts UUID request IDs.
            request_id = str(uuid.uuid4())
            context = Context(request_id)
            release = Path(str(probe_path) + ".release")
            release.unlink(missing_ok=True)
            gate_request = Path(str(probe_path) + ".request")
            gate_request.write_text(request_id)

            async def run_decode():
                stream = await decode.direct(
                    {
                        **payload,
                        "stop_conditions": {"max_tokens": 1024, "ignore_eos": True},
                        "prefill_result": {"disaggregated_params": handoff},
                    },
                    decode_ids[0],
                    annotated=False,
                    context=context,
                )
                return [output async for output in stream]

            task = asyncio.create_task(run_decode())
            try:
                async with asyncio.timeout(10):
                    while not any(
                        e["kind"] == "decode_gate" and e.get("request_id") == request_id
                        for e in probe_events(probe_path)
                    ):
                        if task.done():
                            raise AssertionError(
                                f"Decode ended before gated handoff: {task.result()}"
                            )
                        await asyncio.sleep(0.01)
                context.stop_generating()
                # Stop travels asynchronously. Keep the token gated until the
                # receiving sidecar runtime records Stop for this exact request.
                async with asyncio.timeout(10):
                    while not has_upstream_cancellation(
                        worker_log.read_text(), request_id
                    ):
                        await asyncio.sleep(0.01)
                received_at = time.time()
                # Give the cancelled sidecar a scheduling window while checking
                # that its native stream stays open and executor keeps running.
                # Releasing the token must be what allows cleanup to start.
                minimum_hold = time.monotonic() + 0.25
                deadline = time.monotonic() + 3
                while True:
                    events = probe_events(probe_path)
                    assert not any(
                        e["kind"] == "stream_closed"
                        and e.get("request_id") == request_id
                        for e in events
                    ), "Sidecar abandoned decode before receiving its first token"
                    assert (
                        await asyncio.to_thread(_trtllm_active_requests, decode_port)
                        == 1
                    ), "Native request ended while cancellation should still be deferred"
                    if time.monotonic() >= minimum_hold and any(
                        e["kind"] == "stats"
                        and e["port"] == decode_port
                        and e["batch_started_at"] >= received_at
                        and any(
                            r["stage"] == "GENERATION_IN_PROGRESS"
                            for r in e["stats"]["requestStats"]
                        )
                        for e in events
                    ):
                        break
                    assert (
                        time.monotonic() < deadline
                    ), "No native execution observed after deferred cancellation"
                    await asyncio.sleep(0.01)
                release.touch()
                cancelled = await asyncio.wait_for(task, timeout=10)
                # Runtime cancellation may close the client stream before forwarding
                # the sidecar's cancelled trailer. Executor evidence below is required.
                assert all(
                    output.get("finish_reason") in (None, "cancelled")
                    for output in cancelled
                ), cancelled
                await asyncio.to_thread(
                    assert_engine_released, probe_path, decode_port, request_id, 1024
                )
                # Context-only requests leave no terminal requestStats, and idle
                # transfer reaping emits no fresh stats. One local token forces a
                # new snapshot: zero used blocks then proves *all* prefill KV was
                # freed, including the cancelled handoff's retained blocks.
                generation = importlib.import_module("openengine.v1.generation_pb2")
                services = importlib.import_module("openengine.v1.openengine_pb2_grpc")
                sentinel_id = f"prefill-cleanup-{uuid.uuid4()}"
                async with grpc.aio.insecure_channel(
                    f"127.0.0.1:{prefill_port}"
                ) as channel:
                    responses = [
                        response
                        async for response in services.InferenceStub(channel).Generate(
                            generation.GenerateRequest(
                                request_id=sentinel_id,
                                model=model,
                                token_ids=generation.TokenIds(ids=[11]),
                                sampling=generation.SamplingParams(temperature=0),
                                stopping=generation.StoppingOptions(
                                    max_tokens=1, ignore_eos=True
                                ),
                            ),
                            timeout=10,
                        )
                    ]
                terminal = responses[-1]
                assert terminal.WhichOneof("event") == "finished", responses
                assert terminal.finished.reason == generation.FINISH_REASON_LENGTH
                assert terminal.usage.completion_tokens == 1, terminal
                await asyncio.to_thread(
                    assert_engine_released,
                    probe_path,
                    prefill_port,
                    sentinel_id,
                    2,
                    require_active=False,
                )
            finally:
                context.stop_generating()
                release.touch()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                gate_request.unlink(missing_ok=True)

    asyncio.run(asyncio.wait_for(check(), timeout=60))
