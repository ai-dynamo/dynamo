# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.vllm.handlers import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        EngineGenerateError("private generation failure"),
        ValueError("private engine bug"),
    ],
)
async def test_runtime_generation_fault_is_not_client_validation(failure):
    handler = object.__new__(DecodeWorkerHandler)
    handler._log_with_lora_context = lambda *_args, **_kwargs: None
    handler.engine_client = SimpleNamespace()

    async def failed(*_args):
        raise failure
        yield

    handler._generate_with_lora_admission_lock = failed
    with pytest.raises(Exception) as raised:
        await anext(handler.generate_tokens({}, None, "runtime-fault"))
    assert raised.value.error_class == "Internal"
    assert raised.value.reason == "backend.unknown"
    assert raised.value.diagnostic == str(failure)
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    "error_type,expected",
    [
        ("VLLMValidationError", "InvalidRequest"),
        ("VLLMNotFoundError", "NotFound"),
        ("VLLMUnprocessableEntityError", "InvalidRequest"),
        ("QueueOverflowError", "CapacityExhausted"),
    ],
)
def test_native_exception_identity_controls_classification(error_type, expected):
    from vllm import exceptions

    from dynamo.vllm.errors import engine_error

    error_class = getattr(exceptions, error_type)
    failure = (
        error_class()
        if error_type == "QueueOverflowError"
        else error_class("private engine details")
    )
    mapped = engine_error(failure)
    assert mapped.error_class == expected
    assert "private engine details" not in str(mapped.public_details)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["explicit_http_validation", "request_builder_validation"]
)
async def test_outer_handler_preserves_existing_validation(kind):
    from dynamo.llm.exceptions import HttpError
    from dynamo.vllm.handlers import _translate_vllm_client_errors

    failure = (
        HttpError(400, "invalid sampling bounds")
        if kind == "explicit_http_validation"
        else ValueError("invalid request shape")
    )

    async def request_handler():
        raise failure
        yield

    with pytest.raises(type(failure)) as raised:
        await anext(_translate_vllm_client_errors(request_handler()))
    assert raised.value is failure


@pytest.mark.asyncio
async def test_empty_native_output_is_a_protocol_failure():
    from dynamo.llm.exceptions import DynamoException

    handler = object.__new__(DecodeWorkerHandler)
    handler._log_with_lora_context = lambda *_args, **_kwargs: None

    async def empty(*_args):
        yield SimpleNamespace(outputs=[])

    handler._generate_with_lora_admission_lock = empty
    with pytest.raises(DynamoException) as raised:
        await anext(handler.generate_tokens({}, None, "empty-output"))
    assert raised.value.error_class == "BackendProtocol"
    assert raised.value.reason == "backend.protocol"
    assert raised.value.diagnostic == "No outputs from vLLM engine"


@pytest.mark.asyncio
async def test_fatal_native_engine_death_keeps_shutdown_and_exit(monkeypatch):
    from unittest.mock import Mock

    from vllm.v1.engine.exceptions import EngineDeadError

    import dynamo.vllm.handlers as handlers

    handler = object.__new__(DecodeWorkerHandler)
    handler._log_with_lora_context = lambda *_args, **_kwargs: None
    handler.runtime = SimpleNamespace(shutdown=Mock())

    async def failed(*_args):
        raise EngineDeadError()
        yield

    handler._generate_with_lora_admission_lock = failed
    exit_process = Mock(side_effect=SystemExit(1))
    monkeypatch.setattr(handlers.os, "_exit", exit_process)
    with pytest.raises(SystemExit) as raised:
        await anext(handler.generate_tokens({}, None, "dead-engine"))
    assert raised.value.code == 1
    handler.runtime.shutdown.assert_called_once_with()
    exit_process.assert_called_once_with(1)
