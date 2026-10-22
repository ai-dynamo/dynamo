# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned native overloads retain HTTP 503 without exposing engine messages."""

import asyncio
from importlib import import_module
from unittest.mock import Mock

import pytest

from dynamo.vllm import decision_heads as heads
from dynamo.vllm.tests.test_vllm_decision_heads import Engine as HeadEngine
from dynamo.vllm.tests.test_vllm_decision_heads import NativeApi as HeadNativeApi
from dynamo.vllm.tests.test_vllm_decision_heads import adapter as head_adapter
from dynamo.vllm.tests.test_vllm_decision_heads import question as head_question
from dynamo.vllm.tests.test_vllm_decision_heads import wire as head_wire
from dynamo.vllm.tests.test_vllm_decision_reads import adapter as read_adapter
from dynamo.vllm.tests.test_vllm_decision_reads import question as read_question
from dynamo.vllm.tests.test_vllm_decision_reads import wire as read_wire
from dynamo.vllm.tests.test_vllm_decision_reads_native import Engine

pytest_plugins = ["dynamo.vllm.tests.test_vllm_decision_reads_native"]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.fixture
def native_errors():
    errors = pytest.importorskip("vllm.exceptions")
    if not all(
        hasattr(errors, name) for name in ("QueueOverflowError", "MaxQueuedTokensError")
    ):
        pytest.skip("Exact native overload error APIs are unavailable")
    return (errors.QueueOverflowError, errors.MaxQueuedTokensError), import_module(
        "dynamo.llm.exceptions"
    ).HttpError


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize("phase", ["admission", "engine"])
@pytest.mark.parametrize("error_index", [0, 1])
async def test_actual_native_overload_is_a_bounded_503_and_drains_children(
    native_reads, native_errors, monkeypatch, kind, phase, error_index
):
    error_types, http_error = native_errors
    if kind == "head":
        api, engine = HeadNativeApi(), HeadEngine()
        monkeypatch.setattr(heads, "_load_native", lambda _: api)
        instance = await head_adapter(engine)
        request = head_wire(questions=[head_question(), head_question(1)])
        method = "encode"
    else:
        api, engine = native_reads, Engine()
        instance = await read_adapter(engine)
        request = read_wire(questions=[read_question(), read_question(1)])
        method = "generate"
    api.overload_errors, api.http_error = error_types, http_error
    error = error_types[error_index]()
    error.message = "private engine diagnostic must not escape"
    if phase == "admission":
        engine.check_admission = Mock(side_effect=error)
    else:
        engine.block = asyncio.Event()
        original = getattr(engine, method)

        async def fail_one_child(*args, **kwargs):
            request_id = kwargs["request_id"] if "request_id" in kwargs else args[2]
            if request_id.endswith("-0"):
                await engine.started.wait()
                raise error
            async for output in original(*args, **kwargs):
                yield output

        monkeypatch.setattr(engine, method, fail_one_child)
    with pytest.raises(http_error) as caught:
        await instance.execute(
            request, [], request_id="overloaded", cancelled=lambda: False
        )
    assert caught.value.code == 503
    assert caught.value.message == "Native decision engine is temporarily unavailable"
    assert engine.active == 0
    if phase == "admission":
        assert engine.calls == []
    else:
        assert len(engine.aborted) == 2


@pytest.mark.asyncio
async def test_untrusted_status_like_errors_are_not_reclassified(
    native_reads, native_errors
):
    class UnknownError(RuntimeError):
        http_status = 503
        message = "unknown"

    error_types, http_error = native_errors
    native_reads.overload_errors, native_reads.http_error = error_types, http_error
    engine = Engine()
    engine.check_admission = Mock(side_effect=UnknownError("original failure"))
    instance = await read_adapter(engine)
    with pytest.raises(UnknownError, match="original failure"):
        await instance.execute(
            read_wire(), [], request_id="unknown", cancelled=lambda: False
        )
