# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend admission gate's waiting boundary for the real vLLM handlers.

A model backend's request holds waiting capacity from admission until its
first generated token or completion response, whichever comes first, on every
worker role. These tests serve the real
``DecodeWorkerHandler.generate`` and ``PrefillWorkerHandler.generate`` from an
in-process runtime endpoint, then register its model with that worker's role
through the same public ``register_model`` API, so requests pass through the
real ingress, the Python binding's response classification and the real gate,
with an event-controlled stand-in for vLLM's ``AsyncLLM``. Registering once
the endpoint is serving deliberately exercises a role recorded late; it does
not reproduce the vLLM worker's own startup order. No engine starts and no GPU
is used; the gate's own gauges, scraped from the process's metrics endpoint,
are the oracle.

They establish what the handlers hand the gate:

- Obtaining the stream releases nothing.
- A healthy token-mode decode forwards only completion outputs that carry
  tokens or a finish or stop reason, so an output with neither never reaches
  the gate. Metadata that does reach it — here an item forwarded ahead of the
  handler's own — leaves the request waiting too.
- The decode handler's first item carrying a generated token releases waiting
  capacity, while request capacity stays held until the stream ends.
- An error the decode handler yields before ever calling the engine is an
  ordinary response whose finish reason reports the error, so it completes the
  request: it releases waiting capacity, while request capacity stays held
  until the stream ends. An annotated error frame, unlike it, is interim.
- The prefill handler yields one result, after the engine's prefill output:
  its token, if the engine produced one, its KV-transfer parameters and its
  usage, and no finish reason. That result ends the wait, by its token or,
  without one, as its completion response; it releases waiting capacity as it
  reaches the gate, while the stream is still held open.

They cannot establish when a real engine produces its first output relative to
a remote-prefill KV transfer. In the installed vLLM, a decode request whose KV
is loaded asynchronously waits in ``WAITING_FOR_REMOTE_KVS`` until its connector
reports the transfer received (or failed, after which it recomputes locally),
so its first token follows that point; but the gate neither observes nor
guarantees it. Likewise a prefill's result does not mean that the decode worker
has received its KV: the transfer it hands off follows.
"""

import asyncio
import contextlib
import os
import time
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from vllm.outputs import CompletionOutput, RequestOutput

import dynamo.vllm.handlers as mod
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.constants import DisaggregationMode, EmbeddingTransferMode
from dynamo.vllm.multimodal_utils.protocol import PatchedTokensPrompt
from dynamo.vllm.multimodal_utils.request_processor import (
    PreparedMultimodalInput,
    VllmMultimodalRequestProcessor,
)
from tests.utils.constants import DefaultPort
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.fault_tolerance,
]

PREFIX = "dynamo_backend_admission"
ENGINE_REQUEST_COUNT = f"{PREFIX}_engine_request_count"
ENGINE_REQUEST_WAITING_COUNT = f"{PREFIX}_engine_request_waiting_count"
# These request bodies name no DP rank (`routing.dp_rank`).
UNSPECIFIED = 'dp_rank="unspecified"'

SETTLE_TIMEOUT_S = 10
# How long a count must hold still before the test accepts that it did not move.
HOLD_S = 0.5


class ControlledEngine:
    """Stands in for vLLM's ``AsyncLLM``: each ``generate`` call yields exactly
    the outputs the test releases, in order, and ends on ``None``."""

    def __init__(self) -> None:
        self.tokenizer = None
        # Read only to select a KV connector protocol, which tests replace.
        self.vllm_config = None
        self.calls = 0
        self.yielded = 0
        self.outputs: asyncio.Queue = asyncio.Queue()

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.calls += 1
        while (output := await self.outputs.get()) is not None:
            self.yielded += 1
            yield output

    async def abort(self, request_id) -> None:
        pass


def _output(
    token_ids: list[int] | None,
    finish_reason: str | None,
    kv_transfer_params: dict | None = None,
    finished: bool | None = None,
) -> RequestOutput:
    """An engine output: one completion output with `token_ids`, or none at all
    when `token_ids` is None. It is finished when it names a finish reason,
    unless `finished` says otherwise."""
    outputs = []
    if token_ids is not None:
        outputs.append(
            CompletionOutput(
                index=0,
                text="",
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        )
    return RequestOutput(
        request_id="admission-first-response",
        prompt=None,
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=outputs,
        finished=finish_reason is not None if finished is None else finished,
        kv_transfer_params=kv_transfer_params,
    )


def _decode_handler(engine: ControlledEngine) -> mod.DecodeWorkerHandler:
    """The real decode handler, as a remote-prefill decode worker, with only
    what needs a real model or engine replaced: input preparation, prompt
    building, LoRA lookup and the engine-abort monitor."""
    config = MagicMock()
    config.model = "test-model"
    config.disaggregation_mode = DisaggregationMode.DECODE
    config.embedding_transfer_mode = EmbeddingTransferMode.LOCAL
    config.enable_multimodal = False
    config.enable_rl = False
    model_config = MagicMock(enable_prompt_embeds=False)
    with patch.object(mod.BaseWorkerHandler, "__init__", return_value=None):
        handler = mod.DecodeWorkerHandler(
            runtime=MagicMock(),
            config=config,
            engine=engine,
            default_sampling_params={},
            model_config=model_config,
        )
    handler.config = config
    handler.model_config = model_config
    handler.engine_client = engine
    handler.runtime = MagicMock()
    handler.use_vllm_tokenizer = False
    handler.model_max_len = 4096
    handler.default_sampling_params = {}
    handler._custom_encoder = None
    handler._deferred_aborts = {}
    handler._multimodal_request_processor = VllmMultimodalRequestProcessor(
        model=config.model,
        enable_multimodal=False,
        use_unified_vision_chunk=False,
    )
    handler._multimodal_request_processor.prepare_input = AsyncMock(
        side_effect=lambda request, *args, **kwargs: PreparedMultimodalInput(
            request=request, multi_modal_data=None, mm_processor_kwargs=None
        )
    )
    handler._build_prompt_from_request = MagicMock(
        return_value=PatchedTokensPrompt(prompt_token_ids=[1, 2, 3])
    )
    handler._resolve_lora_request = MagicMock(return_value=None)
    handler._abort_monitor = MagicMock(return_value=nullcontext())
    return handler


class _KvProtocol:
    """Stands in for the KV connector protocol a real vLLM config selects."""

    def prefill_request_kv_transfer_params(self) -> dict:
        return {"do_remote_decode": True}

    def decode_request_kv_transfer_params(self, output: RequestOutput):
        return output.kv_transfer_params


def _prefill_handler(engine: ControlledEngine) -> mod.PrefillWorkerHandler:
    """The real prefill handler, with only what needs a real model or engine
    replaced: input preparation and its handoff, prompt building, LoRA lookup,
    the KV connector protocol and the engine-abort monitor."""
    config = MagicMock()
    config.model = "test-model"
    config.disaggregation_mode = DisaggregationMode.PREFILL
    config.enable_multimodal = False
    config.enable_rl = False
    handler = mod.PrefillWorkerHandler.__new__(mod.PrefillWorkerHandler)
    handler.config = config
    handler.engine_client = engine
    handler.runtime = MagicMock()
    handler.default_sampling_params = {}
    handler.model_max_len = 4096
    handler.dp_range = (0, 1)
    processor = MagicMock()
    processor.prepare_input = AsyncMock(
        side_effect=lambda request, *args, **kwargs: PreparedMultimodalInput(
            request=request, multi_modal_data=None, mm_processor_kwargs=None
        )
    )
    processor.build_prefill_handoff = MagicMock(return_value=None)
    handler._multimodal_request_processor = processor
    handler._build_prompt_from_request = MagicMock(
        return_value=PatchedTokensPrompt(prompt_token_ids=[1, 2, 3])
    )
    handler._resolve_lora_request = MagicMock(return_value=None)
    handler._abort_monitor = MagicMock(return_value=nullcontext())
    return handler


def _decode_request(**extra) -> dict:
    """A token-mode decode request, as after a remote prefill."""
    return {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 4},
        "output_options": {},
        "prefill_result": {
            "disaggregated_params": {
                "kv_transfer_params": {"do_remote_prefill": True},
            },
        },
        **extra,
    }


class GateGauges:
    """The gate's occupancy for unspecified-rank work, scraped from the
    process's own metrics endpoint."""

    def __init__(self, session: aiohttp.ClientSession, port: int) -> None:
        self._session = session
        self._url = f"http://127.0.0.1:{port}/metrics"

    async def read(self) -> tuple[float, float]:
        """(engine requests, engine-waiting requests). Both series must be
        exposed: a missing one is a failure, never a zero."""
        async with self._session.get(self._url) as response:
            text = await response.text()
            assert response.status == 200, text
        values = {}
        for line in text.splitlines():
            for name in (ENGINE_REQUEST_COUNT, ENGINE_REQUEST_WAITING_COUNT):
                if line.startswith(f"{name}{{") and UNSPECIFIED in line:
                    values[name] = float(line.rpartition(" ")[2])
        missing = {ENGINE_REQUEST_COUNT, ENGINE_REQUEST_WAITING_COUNT} - set(values)
        assert not missing, f"series {missing} with {UNSPECIFIED} are not exposed"
        return values[ENGINE_REQUEST_COUNT], values[ENGINE_REQUEST_WAITING_COUNT]

    async def ready(self) -> None:
        """Wait out the metrics server starting and the gate's family being
        registered, then require an idle gate."""
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            try:
                observed = await self.read()
                break
            except (aiohttp.ClientConnectionError, AssertionError):
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.05)
        assert observed == (0, 0), f"the gate must start idle, got {observed}"

    async def settle(self, expected: tuple[float, float], what: str) -> None:
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while (observed := await self.read()) != expected:
            assert time.monotonic() < deadline, (
                f"timed out waiting for {what}: (engine requests, waiting) is "
                f"{observed}, expected {expected}"
            )
            await asyncio.sleep(0.02)

    async def hold(self, expected: tuple[float, float], what: str) -> None:
        """Assert the gauges stay at `expected` for a while: they must not move."""
        deadline = time.monotonic() + HOLD_S
        while time.monotonic() < deadline:
            assert await self.read() == expected, what
            await asyncio.sleep(0.02)


async def _until(predicate, what: str) -> None:
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


@contextlib.asynccontextmanager
async def _served(generate, tmp_path, worker_type=WorkerType.Decode):
    """Serve `generate` as a model backend of `worker_type`, through the real
    ingress and gate, and yield a client for it with the gate's gauges."""
    # The system port must fit the runtime's port range, which the shared
    # allocator respects; an arbitrary ephemeral port may not.
    with reserved_ports(1, DefaultPort.SYSTEM1.value) as (port,):
        os.environ["DYN_SYSTEM_PORT"] = str(port)
        os.environ["DYN_FILE_KV"] = str(tmp_path)
        runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
        endpoint = runtime.endpoint("admission-first-response.backend.generate")
        server = asyncio.ensure_future(endpoint.serve_endpoint(generate))
        try:
            client = await endpoint.client()
            await asyncio.wait_for(client.wait_for_instances(), SETTLE_TIMEOUT_S)
            # Registering after serving starts records the role late, through
            # the public API: the endpoint is a model backend, whose requests
            # the gate admits, from then on. A prefill worker exposes no OpenAI
            # surface; each disaggregated role needs the other.
            prefill = worker_type == WorkerType.Prefill
            await register_model(
                ModelInput.Tokens,
                ModelType.Prefill if prefill else ModelType.Chat,
                endpoint,
                "test-model",
                worker_type=worker_type,
                needs=[[WorkerType.Decode if prefill else WorkerType.Prefill]],
                skip_model_assets=True,
            )
            async with aiohttp.ClientSession() as session:
                gauges = GateGauges(session, port)
                await gauges.ready()
                yield client, gauges
        finally:
            server.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server
            runtime.shutdown()


async def _next(stream):
    return await asyncio.wait_for(stream.__anext__(), SETTLE_TIMEOUT_S)


def _forwarded(generate, *, first=None, end: asyncio.Event | None = None):
    """Forward every item the real handler yields, unchanged: after `first`, an
    item the handler itself never sends, when given; and then hold the stream
    open until `end` is set, when given, so the gate can be observed between a
    response and the end of the stream."""

    async def forwarded(request, context):
        if first is not None:
            yield first
        async for item in generate(request, context):
            yield item
        if end is not None:
            await end.wait()

    return forwarded


METADATA = {"token_ids": [], "engine_data": {"note": "metadata only"}}


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_waiting_capacity_is_released_at_the_decode_handlers_first_token(
    tmp_path,
):
    engine = ControlledEngine()
    handler = _decode_handler(engine)
    async with _served(_forwarded(handler.generate, first=METADATA), tmp_path) as (
        client,
        gauges,
    ):
        stream = await client.generate(_decode_request())
        await gauges.hold((1, 1), "obtaining the stream is not a response")

        # Metadata that reaches the gate is no token: the request keeps waiting.
        assert (await _next(stream)).data() == METADATA
        await _until(lambda: engine.calls == 1, "the handler to call the engine")
        await gauges.hold((1, 1), "delivered metadata is not a first token")

        # A completion output with neither tokens nor a finish or stop reason is
        # filtered out by the handler, so nothing reaches the gate and nothing
        # is released.
        engine.outputs.put_nowait(_output([], None))
        await _until(lambda: engine.yielded == 1, "the empty output to be produced")
        await gauges.hold((1, 1), "a filtered empty output is not a first token")

        engine.outputs.put_nowait(_output([11], None))
        first = await _next(stream)
        assert first.data()["token_ids"] == [11]
        await gauges.settle((1, 0), "the first token to release waiting capacity")
        await gauges.hold((1, 0), "request capacity stays held while it streams")

        engine.outputs.put_nowait(_output([12], "stop"))
        engine.outputs.put_nowait(None)
        rest = [item.data() async for item in stream]
        assert [item["token_ids"] for item in rest] == [[12]]
        assert rest[-1]["finish_reason"] == "stop"
        await gauges.settle((0, 0), "the end of the stream to release the request")


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_decode_handler_error_before_the_engine_completes_its_wait(
    tmp_path,
):
    """External encoder results need an aggregated worker, so this decode
    handler yields an error before ever calling the engine. It is an ordinary
    response whose finish reason reports the error, not an annotated error
    frame, so it completes the request: it releases waiting capacity, and the
    end of its stream releases request capacity, each exactly once."""
    engine = ControlledEngine()
    end = asyncio.Event()
    handler = _decode_handler(engine)
    async with _served(_forwarded(handler.generate, end=end), tmp_path) as (
        client,
        gauges,
    ):
        stream = await client.generate(_decode_request(encoder_result={}))
        error = (await _next(stream)).data()
        assert error["finish_reason"].startswith("error:"), error
        assert error["token_ids"] == []
        assert engine.calls == 0, "the engine was never called"
        await gauges.settle((1, 0), "the error's completion to end the wait")
        await gauges.hold((1, 0), "the open stream keeps its request capacity")

        end.set()
        assert [item async for item in stream] == []
        await gauges.settle((0, 0), "the end of the stream to release the request")
        await gauges.hold((0, 0), "and nothing is released twice")
        assert engine.calls == 0


def _prefill_request() -> dict:
    """A token-mode prefill request, as the prefill router sends it."""
    return {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 4},
        "output_options": {},
    }


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "token_ids", [[11], None], ids=["with its token", "without a token"]
)
async def test_the_prefill_handlers_result_completes_its_wait_while_held_open(
    tmp_path, monkeypatch, token_ids
):
    monkeypatch.setattr(mod, "make_kv_connector_protocol", lambda _: _KvProtocol())
    engine = ControlledEngine()
    end = asyncio.Event()
    handler = _prefill_handler(engine)
    async with _served(
        _forwarded(handler.generate, end=end), tmp_path, WorkerType.Prefill
    ) as (client, gauges):
        stream = await client.generate(_prefill_request())
        await _until(lambda: engine.calls == 1, "the handler to call the engine")
        await gauges.hold((1, 1), "the prefill has not produced its result")

        handoff = {"remote_block_ids": [7], "remote_engine_id": "prefill"}
        # The engine's one, final prefill output, which may carry no completion
        # output at all.
        engine.outputs.put_nowait(
            _output(token_ids, "length" if token_ids else None, handoff, finished=True)
        )
        result = (await _next(stream)).data()
        assert result["token_ids"] == (token_ids or [])
        assert result["disaggregated_params"] == {"kv_transfer_params": handoff}
        assert "completion_usage" in result
        assert "finish_reason" not in result, "the prefill result names none"
        await gauges.settle((1, 0), "the prefill's result to complete its wait")

        # The engine is done; the stream stays open, holding request capacity.
        engine.outputs.put_nowait(None)
        await gauges.hold((1, 0), "the open stream keeps its request capacity")
        end.set()
        assert [item async for item in stream] == []
        await gauges.settle((0, 0), "the end of the stream to release the request")
