# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid
from dataclasses import dataclass, field

import aiohttp

from dynamo.runtime import Context
from tests.router.helper import managed_runtime, poll_for_worker_instances
from tests.utils.engine_metrics import EngineMetrics
from tests.utils.payloads import KvTransferPayload


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


@dataclass
class NativeCancellationPayload:
    model: str

    def request(self, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "token_ids": [11] * 128,
            "stop_conditions": {"max_tokens": max_tokens, "ignore_eos": True},
            "sampling_options": {"temperature": 0.0},
        }

    def tokens(self, output: dict) -> list[int]:
        return output["token_ids"]

    def validate_recovery(self, outputs: list[dict]) -> None:
        _assert_native_completion(outputs, prompt_tokens=128, completion_tokens=4)

    async def drain_cancelled(self, stream) -> None:
        async for output in stream:
            assert output.get("finish_reason") in (None, "cancelled"), output


class SGLangHttpCancellationPayload(NativeCancellationPayload):
    def request(self, max_tokens: int) -> dict:
        return {
            **super().request(max_tokens),
            "extra_args": {
                "sglang_tito": {
                    "sampling_params": {
                        "max_new_tokens": max_tokens,
                        "ignore_eos": True,
                        "temperature": 0.0,
                    },
                }
            },
        }

    def tokens(self, output: dict) -> list[int]:
        return output["engine_data"]["sglang_response"]["output_ids"]

    def validate_recovery(self, outputs: list[dict]) -> None:
        assert outputs, "Sidecar produced no native HTTP response"
        assert all(not output["token_ids"] for output in outputs), outputs
        raw = [output["engine_data"]["sglang_response"] for output in outputs]
        assert sum(len(item["output_ids"]) for item in raw) == 4, raw
        assert all(
            output.get("finish_reason") is None for output in outputs[:-1]
        ), outputs
        assert outputs[-1]["finish_reason"] == "stop", outputs[-1]
        usage = raw[-1]["meta_info"]
        assert usage["finish_reason"]["type"] == "length", usage
        assert usage["prompt_tokens"] == 128, usage
        assert usage["completion_tokens"] == 4, usage

    async def drain_cancelled(self, stream) -> None:
        try:
            await super().drain_cancelled(stream)
        except ValueError as error:
            if not str(error).startswith("Cancelled:"):
                raise


def assert_native_cancellation_and_recovery(
    *,
    metrics: EngineMetrics,
    payloads: list[NativeCancellationPayload],
    namespace: str,
    discovery_backend: str = "etcd",
) -> None:
    """Check explicit native stop and consumer drop on the existing deployment."""
    max_tokens = 2048
    completion_progress = metrics.completion_progress(max_tokens)

    async def native_checks(payload: NativeCancellationPayload) -> None:
        with managed_runtime(discovery_backend, "tcp") as runtime:
            endpoint = runtime.endpoint(f"{namespace}.backend.generate")
            worker_ids = await poll_for_worker_instances(endpoint, 1, max_wait_time=10)
            assert len(worker_ids) == 1, worker_ids
            client = await endpoint.client()

            async def recover() -> None:
                before = await asyncio.to_thread(metrics.progress)
                stream = await client.direct(
                    payload.request(4),
                    worker_ids[0],
                    annotated=False,
                )
                outputs = [output async for output in stream]
                payload.validate_recovery(outputs)
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
                            payload.request(max_tokens),
                            worker_ids[0],
                            annotated=False,
                            context=context,
                        ),
                        timeout=10,
                    )
                    output = await asyncio.wait_for(anext(stream), timeout=10)
                    assert (
                        payload.tokens(output) and output.get("finish_reason") is None
                    ), output
                    await asyncio.to_thread(metrics.wait_for_scheduler, is_active=True)
                    if is_explicit_stop:
                        context.stop_generating()
                        await asyncio.wait_for(
                            payload.drain_cancelled(stream), timeout=10
                        )
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

    for payload in payloads:
        asyncio.run(native_checks(payload))


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
                    _assert_native_completion(
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


@dataclass
class SGLangTransferRecoveryPayload(KvTransferPayload):
    """Cancel an unmatched native transfer, then verify normal frontend recovery."""

    namespace: str = field(kw_only=True)
    decode_http_port: int = field(kw_only=True)
    bootstrap_port: int = field(kw_only=True)
    discovery_backend: str = field(default="etcd", kw_only=True)

    def before_request(self) -> None:
        assert_sglang_transfer_wait_cancelled(
            namespace=self.namespace,
            model=self.body["model"],
            decode_http_port=self.decode_http_port,
            bootstrap_port=self.bootstrap_port,
            discovery_backend=self.discovery_backend,
        )
        super().before_request()
