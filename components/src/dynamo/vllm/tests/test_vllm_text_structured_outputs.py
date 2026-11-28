# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-output constraints on the text-in/text-out worker path."""

import copy
import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dynamo.vllm.tests.test_structured_output_validation import (
    native_validation_config as native_validation_config,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]


@pytest.fixture(scope="module")
def handlers():
    # Handler SDK dependencies are optional in marker-report collection.
    return importlib.import_module("dynamo.vllm.handlers")


def _schema(value):
    return {
        "type": "object",
        "properties": {"value": value},
        "required": ["value"],
        "additionalProperties": False,
    }


def _request(schema):
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "Respond with JSON"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        },
    }


@pytest.mark.parametrize("response_format", [None, {"type": "text"}])
def test_text_or_absent_format_preserves_default_constraint(handlers, response_format):
    from vllm.sampling_params import StructuredOutputsParams

    default_outputs = StructuredOutputsParams(regex="ok")
    params = handlers.build_sampling_params_openai(
        {"response_format": response_format},
        {"structured_outputs": default_outputs},
    )
    assert params.structured_outputs is not default_outputs
    assert params.structured_outputs.regex == "ok"


def test_json_schema_overrides_json_default_without_mutating_it(handlers):
    from vllm.sampling_params import StructuredOutputsParams

    default_outputs = StructuredOutputsParams(json={"type": "integer"})
    schema = _schema({"type": "string", "enum": ["ok"]})
    request = _request(schema)
    before = copy.deepcopy(request)
    first = handlers.build_sampling_params_openai(
        request, {"structured_outputs": default_outputs}
    )
    second = handlers.build_sampling_params_openai(request, {})
    assert first.structured_outputs.json == schema
    assert second.structured_outputs.json == schema
    assert first.structured_outputs is not second.structured_outputs
    assert default_outputs.json == {"type": "integer"}
    assert default_outputs._backend is None
    assert first.structured_outputs._backend is None
    assert second.structured_outputs._backend is None
    assert request == before


def test_json_object_format_reaches_sampling_params(handlers):
    params = handlers.build_sampling_params_openai(
        {"response_format": {"type": "json_object"}}, {}
    )
    assert params.structured_outputs.json_object is True


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "unknown"},
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {"name": "answer"}},
    ],
)
def test_malformed_response_format_is_invalid_argument(handlers, response_format):
    from dynamo.llm.exceptions import InvalidArgument

    with pytest.raises(InvalidArgument, match="^Invalid response_format$"):
        handlers.build_sampling_params_openai({"response_format": response_format}, {})


@pytest.mark.parametrize("format_type", ["json_schema", "json_object"])
def test_legacy_json_keeps_precedence_over_response_format(handlers, format_type):
    request = _request({"type": "integer"})
    if format_type == "json_object":
        request["response_format"] = {"type": "json_object"}
    request["guided_json"] = {"type": "string", "enum": ["legacy"]}
    request["guided_whitespace_pattern"] = "[ ]?"
    params = handlers.build_sampling_params_openai(request, {})
    assert params.structured_outputs.json == request["guided_json"]
    assert params.structured_outputs.json_object is None
    assert params.structured_outputs.whitespace_pattern == "[ ]?"


@pytest.mark.parametrize("legacy", [{"guided_regex": "ok"}, {"guided_choice": ["ok"]}])
def test_response_format_rejects_independent_legacy_constraints(handlers, legacy):
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError):
        handlers.build_sampling_params_openai(
            _request({"type": "integer"}) | legacy, {}
        )


def test_response_format_rejects_conflicting_default_constraint(handlers):
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import StructuredOutputsParams

    with pytest.raises(VLLMValidationError):
        handlers.build_sampling_params_openai(
            _request({"type": "integer"}),
            {"structured_outputs": StructuredOutputsParams(regex="ok")},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("regex", "ok"), ("grammar", 'root ::= "ok"'), ("choice", ["ok"])],
)
def test_legacy_constraint_reaches_sampling_params(handlers, field, value):
    params = handlers.build_sampling_params_openai({f"guided_{field}": value}, {})
    assert getattr(params.structured_outputs, field) == value


def test_json_ref_cycle_uses_same_guard_as_token_worker(handlers):
    from dynamo.llm.exceptions import HttpError

    with pytest.raises(HttpError) as caught:
        handlers.build_sampling_params_openai(_request({"$ref": "#"}), {})
    assert caught.value.code == 400


@pytest.mark.parametrize(
    "request_data",
    [{}, {"guided_choice": []}, {"guided_whitespace_pattern": "[ ]?"}],
)
def test_no_constraint_does_not_enable_structured_outputs(handlers, request_data):
    assert (
        handlers.build_sampling_params_openai(request_data, {}).structured_outputs
        is None
    )


def test_native_backend_selection_does_not_mutate_defaults(
    handlers, native_validation_config
):
    from vllm.config import StructuredOutputsConfig
    from vllm.sampling_params import StructuredOutputsParams

    defaults = {"structured_outputs": StructuredOutputsParams(json={"type": "object"})}
    before = copy.deepcopy(defaults)
    model, tokenizer = native_validation_config
    first = handlers.build_sampling_params_openai({}, defaults)
    first.verify(model, None, StructuredOutputsConfig(), tokenizer=tokenizer)
    assert first.structured_outputs._backend == "xgrammar"
    second = handlers.build_sampling_params_openai({}, defaults)
    assert second.structured_outputs._backend is None
    assert not second.structured_outputs._backend_was_auto
    assert defaults == before


@pytest.mark.parametrize("backend", ["auto", "xgrammar"])
@pytest.mark.parametrize("kind", ["valid", "required-empty", "oneof-empty-null"])
def test_text_schema_matches_token_worker_validation(
    handlers, native_validation_config, backend, kind
):
    from vllm.config import StructuredOutputsConfig
    from vllm.exceptions import VLLMValidationError

    from dynamo.vllm.errors import vllm_client_error_to_http_error

    value = {"type": "string", "enum": ["ok"] if kind == "valid" else []}
    if kind == "oneof-empty-null":
        value = {"oneOf": [value, {"type": "null"}]}
    schema = _schema(value)
    model, tokenizer = native_validation_config
    text = handlers.build_sampling_params_openai(_request(copy.deepcopy(schema)), {})
    token = handlers.build_sampling_params(
        {"sampling_options": {"guided_decoding": {"json": copy.deepcopy(schema)}}}, {}
    )
    expected_backend = (
        "xgrammar"
        if kind == "valid"
        else "guidance"
        if kind == "oneof-empty-null" and backend == "auto"
        else None
    )
    for params in (text, token):
        assert params.structured_outputs._backend is None
        config = StructuredOutputsConfig(backend=backend)
        if expected_backend is not None:
            params.verify(model, None, config, tokenizer=tokenizer)
            assert params.structured_outputs._backend == expected_backend
            assert params.structured_outputs._backend_was_auto is (backend == "auto")
        else:
            with pytest.raises(VLLMValidationError) as caught:
                params.verify(model, None, config, tokenizer=tokenizer)
            error = vllm_client_error_to_http_error(caught.value)
            expected = (
                "Invalid JSON schema: enum array must not be empty"
                if backend == "xgrammar"
                else "Invalid JSON schema: a required property is unsatisfiable"
            )
            assert str(error) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_text_handler_delivers_schema_and_translates_native_rejection(
    handlers, native_validation_config, valid
):
    from vllm.config import StructuredOutputsConfig

    from dynamo.llm.exceptions import InvalidArgument

    model, tokenizer = native_validation_config
    schema = _schema({"type": "string", "enum": ["ok"] if valid else []})
    received = []

    async def generate(prompt, sampling_params, request_id, **kwargs):
        received.append(sampling_params.structured_outputs.json)
        sampling_params.verify(
            model,
            None,
            StructuredOutputsConfig(backend="xgrammar"),
            tokenizer=tokenizer,
        )
        yield SimpleNamespace(
            outputs=[SimpleNamespace(index=0, text="ok", finish_reason=None)]
        )

    handler = SimpleNamespace(
        input_param_manager=SimpleNamespace(get_input_param=lambda *a, **kw: "prompt"),
        default_sampling_params={},
        engine_client=SimpleNamespace(generate=generate),
        config=SimpleNamespace(
            disaggregation_mode=handlers.DisaggregationMode.AGGREGATED
        ),
        _to_local_dp_rank=lambda rank: None,
        _deferred_aborts={},
        _shutdown_on_engine_dead=lambda exc: None,
        _abort_monitor=lambda *a, **kw: nullcontext(),
    )
    context = SimpleNamespace(trace_headers=lambda: {})
    source = handlers.DecodeWorkerHandler._generate_text_mode(
        handler, _request(schema), context, "text-schema"
    )
    translated = handlers._translate_vllm_client_errors(source)
    if valid:
        chunks = [chunk async for chunk in translated]
        assert chunks[0]["choices"][0]["delta"]["content"] == "ok"
    else:
        with pytest.raises(
            InvalidArgument, match="^Invalid JSON schema: enum array must not be empty$"
        ):
            _ = [chunk async for chunk in translated]
    assert received == [schema]
