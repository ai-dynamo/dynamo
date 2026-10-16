# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import asynccontextmanager

import pytest

from dynamo.llm.exceptions import DynamoException, WorkerOverloaded
from dynamo.sglang.errors import classified_stream, raise_if_abort_error
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "status,expected,reason",
    [
        (400, "InvalidRequest", "backend.invalid_argument"),
        (422, "InvalidRequest", "backend.invalid_argument"),
        (429, "CapacityExhausted", "capacity.worker_overloaded"),
        (503, "Unavailable", "backend.unavailable"),
        (500, "Internal", "backend.unknown"),
        (True, "Internal", "backend.unknown"),
        (None, "Internal", "backend.unknown"),
    ],
)
def test_structured_abort_classifies_cause_without_exposing_diagnostic(
    status, expected, reason
):
    with pytest.raises(DynamoException) as raised:
        raise_if_abort_error(
            {
                "type": "abort",
                "status_code": status,
                "err_type": "native-error",
                "message": "private path /engine/internal",
            }
        )
    assert raised.value.error_class == expected
    assert raised.value.reason == reason
    assert "private path" not in str(raised.value.public_details)


@pytest.mark.parametrize(
    "finish",
    [
        {"type": "abort"},
        {"type": "abort", "status_code": None, "err_type": None},
        {"type": "stop", "status_code": 429},
    ],
)
def test_nonerror_finish_and_cancellation_are_unchanged(finish):
    raise_if_abort_error(finish)


@pytest.mark.asyncio
async def test_engine_iteration_value_error_is_internal():
    async def failed():
        raise ValueError("internal executor invariant")
        yield

    with pytest.raises(DynamoException) as raised:
        await anext(classified_stream(failed()))
    assert raised.value.error_class == "Internal"


@pytest.mark.asyncio
async def test_existing_semantics_and_cancellation_propagate_unchanged():
    for failure in [WorkerOverloaded("full"), asyncio.CancelledError()]:

        async def failed():
            raise failure
            yield

        with pytest.raises(type(failure)) as raised:
            await anext(classified_stream(failed()))
        assert raised.value is failure


@pytest.mark.asyncio
async def test_prefill_consumption_propagates_capacity_abort():
    handler = object.__new__(PrefillWorkerHandler)

    @asynccontextmanager
    async def monitor(*_):
        task = asyncio.create_task(asyncio.Event().wait())
        try:
            yield task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    handler._cancellation_monitor = monitor

    async def results():
        yield {
            "meta_info": {
                "id": "prefill-id",
                "finish_reason": {
                    "type": "abort",
                    "status_code": 429,
                    "message": "capacity",
                },
            }
        }

    with pytest.raises(DynamoException) as raised:
        await handler._consume_results(results(), None, None)
    assert raised.value.error_class == "CapacityExhausted"
    assert raised.value.reason == "capacity.worker_overloaded"


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
async def test_submission_failure_keeps_known_identity_and_private_unknown(typed):
    from dynamo.sglang.errors import submit_generation

    failure = (
        WorkerOverloaded("admission full")
        if typed
        else ValueError("private tokenizer invariant")
    )

    async def submit():
        raise failure

    with pytest.raises(DynamoException) as raised:
        await submit_generation(submit())
    if typed:
        assert raised.value is failure
    else:
        assert raised.value.error_class == "Internal"
        assert raised.value.diagnostic == str(failure)
        assert raised.value.public_details is None
