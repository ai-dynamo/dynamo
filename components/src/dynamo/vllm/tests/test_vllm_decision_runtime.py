# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.llm.exceptions import EngineShutdown, HttpError, WorkerOverloaded
from dynamo.vllm.decision_native_lifecycle import translate_native_errors

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.mark.asyncio
async def test_native_overload_preserves_capacity_classification():
    class NativeOverload(Exception):
        pass

    native = SimpleNamespace(
        overload_errors=(NativeOverload,), overload_error=WorkerOverloaded
    )

    async def overloaded():
        raise NativeOverload("private engine details")

    with pytest.raises(WorkerOverloaded) as caught:
        await translate_native_errors(overloaded(), native, None)
    assert str(caught.value) == "Native decision engine capacity is exhausted"


@pytest.mark.asyncio
async def test_unknown_engine_error_is_not_capacity_rejection():
    native = SimpleNamespace(
        overload_errors=(),
        engine_generate_error=(),
        engine_dead_error=(),
        retryable_request_error=(),
    )

    async def failed():
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        await translate_native_errors(failed(), native, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("dead", [False, True])
async def test_pinned_engine_failures_without_a_shutdown_callback(
    pinned_engine_errors, dead
):
    native = SimpleNamespace(
        overload_errors=(),
        engine_generate_error=pinned_engine_errors.EngineGenerateError,
        engine_dead_error=pinned_engine_errors.EngineDeadError,
        retryable_request_error=pinned_engine_errors.RetryableRequestError,
    )
    cause = (native.engine_dead_error if dead else native.engine_generate_error)()

    async def failed():
        raise cause

    with pytest.raises(EngineShutdown if dead else HttpError) as failure:
        await translate_native_errors(failed(), native, None)
    assert failure.value.__cause__ is cause
    if not dead:
        assert failure.value.code == 503
