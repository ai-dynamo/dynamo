# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import uuid
from pathlib import Path

from dynamo.runtime import Context
from tests.router.helper import managed_runtime, poll_for_worker_instances
from tests.serve.sidecar_checks import _transferred, _wait_for_scheduler
from tests.serve.sidecar_native_checks import assert_native_completion


def assert_native_handoff(
    *,
    backend: str,
    namespace: str,
    model: str,
    prefill_http_port: int,
    decode_http_port: int,
    bootstrap_port: int | None = None,
    probe_path: Path | None = None,
    discovery_backend: str = "etcd",
) -> None:
    """Check raw prefill/decode handoff on the existing E2E workers."""
    before = _transferred(backend, prefill_http_port, probe_path)

    async def run() -> None:
        with managed_runtime(discovery_backend, "tcp") as runtime:
            prefill_endpoint = runtime.endpoint(f"{namespace}.prefill.generate")
            decode_endpoint = runtime.endpoint(f"{namespace}.backend.generate")
            prefill_ids, decode_ids = await asyncio.gather(
                poll_for_worker_instances(prefill_endpoint, 1, max_wait_time=10),
                poll_for_worker_instances(decode_endpoint, 1, max_wait_time=10),
            )
            assert len(prefill_ids) == len(decode_ids) == 1
            assert prefill_ids != decode_ids
            prefill_client, decode_client = await asyncio.gather(
                prefill_endpoint.client(), decode_endpoint.client()
            )
            payload = {
                "model": model,
                "token_ids": list(uuid.uuid4().bytes) * 8,
                "stop_conditions": {"max_tokens": 8, "ignore_eos": True},
                "sampling_options": {"temperature": 0.0},
            }
            prefill_context = Context(f"prefill-{uuid.uuid4()}")
            decode_context = Context(f"decode-{uuid.uuid4()}")

            async def collect(stream) -> list[dict]:
                return [output async for output in stream]

            async def handoff() -> None:
                prefill_stream = await prefill_client.direct(
                    payload, prefill_ids[0], annotated=False, context=prefill_context
                )
                prefill_outputs = []
                async for output in prefill_stream:
                    prefill_outputs.append(output)
                    if output.get("disaggregated_params") is not None:
                        params = output["disaggregated_params"]
                        break
                else:
                    raise AssertionError(f"No prefill handoff: {prefill_outputs}")

                if backend == "vllm":
                    assert (
                        isinstance(params["remote_engine_id"], str)
                        and params["remote_engine_id"]
                    ), params
                    assert (
                        isinstance(params["remote_block_ids"], list)
                        and params["remote_block_ids"]
                    ), params
                    assert output["finish_reason"] == "length", output
                else:
                    assert set(params) == {
                        "bootstrap_host",
                        "bootstrap_port",
                        "bootstrap_room",
                    }, params
                    assert params["bootstrap_host"] == "127.0.0.1", params
                    assert params["bootstrap_port"] == bootstrap_port, params
                    assert isinstance(params["bootstrap_room"], int) and (
                        0 <= params["bootstrap_room"] <= (1 << 63) - 1
                    ), params
                    assert output.get("finish_reason") is None, output

                # SGLang needs decode to rendezvous before prefill can finish.
                decode_stream = await decode_client.direct(
                    {**payload, "prefill_result": {"disaggregated_params": params}},
                    decode_ids[0],
                    annotated=False,
                    context=decode_context,
                )
                prefill_tail, decode_outputs = await asyncio.gather(
                    collect(prefill_stream), collect(decode_stream)
                )
                prefill_outputs.extend(prefill_tail)
                assert [
                    output["disaggregated_params"]
                    for output in prefill_outputs
                    if output.get("disaggregated_params") is not None
                ] == [params]
                assert_native_completion(
                    prefill_outputs, prompt_tokens=128, completion_tokens=0
                )
                assert_native_completion(
                    decode_outputs, prompt_tokens=128, completion_tokens=8
                )

            try:
                await asyncio.wait_for(handoff(), timeout=30)
            finally:
                prefill_context.stop_generating()
                decode_context.stop_generating()

    asyncio.run(run())
    deadline = time.monotonic() + 10
    while _transferred(backend, prefill_http_port, probe_path) <= before:
        assert (
            time.monotonic() < deadline
        ), f"{backend}: no completed native KV transfer"
        time.sleep(0.05)
    _wait_for_scheduler(backend, decode_http_port)
    _wait_for_scheduler(backend, prefill_http_port)
