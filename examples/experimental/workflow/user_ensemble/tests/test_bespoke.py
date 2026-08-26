# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from examples.experimental.workflow.user_ensemble.bespoke.orchestrator_worker import (
    BespokeEnsembleHandler,
    _collect_generation,
)
from examples.experimental.workflow.user_ensemble.common.stages import (
    EnsembleResponseStage,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class _Runner:
    def __init__(self, output: Mapping[str, Any]) -> None:
        self.output = output

    async def run(self, inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        del inputs, context
        return self.output


class _RequestAdapter:
    async def run(self, inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        del context
        request = dict(inputs["request"])
        request["encoder_result"] = {"test": "encoded"}
        return {"request": request}


async def _stream() -> AsyncIterator[dict[str, Any]]:
    yield {"token_ids": [4], "index": 0}
    yield {"token_ids": [2], "index": 0, "finish_reason": "stop"}


class _Client:
    request: dict[str, Any] | None = None

    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        assert annotated is False
        self.request = dict(request)
        return _stream()


class _DetachedContext:
    def __init__(self) -> None:
        self.stopped = False

    def stop_generating(self) -> None:
        self.stopped = True


class _Context:
    def __init__(self) -> None:
        self.detached_context = _DetachedContext()
        self.cancelled = asyncio.Event()

    def id(self) -> str:
        return "request-1"

    def detached(self, request_id: str) -> _DetachedContext:
        assert request_id == "request-1:generator"
        return self.detached_context

    async def async_killed_or_stopped(self) -> None:
        await self.cancelled.wait()


def _handler(
    client: Any,
    *,
    classifier: Any | None = None,
) -> BespokeEnsembleHandler:
    return BespokeEnsembleHandler(
        encoder=_Runner(
            {
                "encoder_features": "features",
                "encoder_metadata": {"rows": [0, 1]},
            }
        ),
        classifier=classifier or _Runner({"scores": {"dummy-positive": 1.0}}),
        request_adapter=_RequestAdapter(),
        response=EnsembleResponseStage(),
        generator_client=client,
    )


async def test_bespoke_handler_manually_coordinates_same_result() -> None:
    client = _Client()
    context = _Context()

    chunks = [
        chunk async for chunk in _handler(client).generate({"token_ids": [1]}, context)
    ]

    assert client.request == {
        "token_ids": [1],
        "encoder_result": {"test": "encoded"},
    }
    assert chunks[0]["token_ids"] == [4, 2]
    assert chunks[0]["engine_data"]["user_ensemble"] == {
        "classifier_scores": {"dummy-positive": 1.0}
    }
    assert context.detached_context.stopped is False


class _StalledRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def run(self, inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        del inputs, context
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _StalledClient(_Client):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        del request, annotated, context

        async def stalled() -> AsyncIterator[dict[str, Any]]:
            self.started.set()
            await asyncio.Event().wait()
            yield {}

        return stalled()


async def test_bespoke_handler_propagates_caller_cancellation() -> None:
    client = _StalledClient()
    classifier = _StalledRunner()
    context = _Context()
    stream = _handler(client, classifier=classifier).generate(
        {"token_ids": [1]}, context
    )
    response = asyncio.create_task(anext(stream))
    await client.started.wait()

    context.cancelled.set()

    with pytest.raises(StopAsyncIteration):
        await response
    assert context.detached_context.stopped is True
    assert classifier.cancelled is True


@pytest.mark.parametrize(
    "request,match",
    [
        ({"sampling_options": {"n": 2}}, "requires n=1"),
        ({"output_options": {"logprobs": 1}}, "does not support logprobs"),
        (
            {"output_options": {"prompt_logprobs": 1}},
            "does not support logprobs",
        ),
    ],
)
async def test_bespoke_handler_matches_generate_option_restrictions(
    request: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        await anext(_handler(_Client()).generate(request, _Context()))


@pytest.mark.parametrize(
    "chunks,match",
    [
        ([{"token_ids": [1], "index": 0}], "no terminal"),
        (
            [
                {"token_ids": [1], "index": 0, "finish_reason": "stop"},
                {"token_ids": [2], "index": 0},
            ],
            "after its terminal",
        ),
        (
            [
                {
                    "token_ids": [1],
                    "index": 0,
                    "finish_reason": "stop",
                    "log_probs": [-0.1],
                }
            ],
            "generator logprobs",
        ),
        (
            [{"token_ids": [1], "index": 0, "finish_reason": ""}],
            "invalid finish_reason",
        ),
    ],
)
async def test_bespoke_stream_validation_parity(
    chunks: list[dict[str, Any]],
    match: str,
) -> None:
    async def values() -> AsyncIterator[dict[str, Any]]:
        for chunk in chunks:
            yield chunk

    with pytest.raises((RuntimeError, TypeError), match=match):
        await _collect_generation(values())
