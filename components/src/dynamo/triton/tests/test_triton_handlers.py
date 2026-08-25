# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.triton.handlers.RequestHandler: Dynamo <-> Triton
tensor conversion, streaming responses, request validation, and dtype caching."""

import asyncio
import types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from dynamo.triton import handlers

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _MockModel:
    """Records the request it builds and replays a fixed response stream."""

    def __init__(
        self, output_metadata: list[dict[str, Any]], responses: list[Any]
    ) -> None:
        self._output_metadata = output_metadata
        self._responses = responses
        self.last_request: types.SimpleNamespace | None = None

    def create_request(self) -> types.SimpleNamespace:
        self.last_request = types.SimpleNamespace(inputs={})
        return self.last_request

    def metadata(self) -> dict[str, Any]:
        return {"outputs": self._output_metadata}

    def async_infer(self, _inference_request: Any) -> AsyncIterator[Any]:
        async def _stream() -> AsyncIterator[Any]:
            for response in self._responses:
                yield response

        return _stream()


def build_dynamo_request(
    *tensor_specs: tuple[str, str, list[int], list[Any]]
) -> dict[str, Any]:
    """Build a Dynamo request envelope from (name, data_type, shape, values) tensor specs."""
    return {
        "tensors": [
            {
                "metadata": {"name": name, "shape": shape, "data_type": data_type},
                "data": {"data_type": data_type, "values": values},
            }
            for name, data_type, shape, values in tensor_specs
        ]
    }


def build_triton_response(
    request_id: str, model_name: str, output_tensors: dict[str, Any]
) -> types.SimpleNamespace:
    """A fake tritonserver inference response."""
    return types.SimpleNamespace(
        request_id=request_id,
        model=types.SimpleNamespace(name=model_name),
        outputs=output_tensors,
    )


def run_handler_generate(
    triton_output_metadata: list[dict[str, Any]],
    triton_responses: list[Any],
    dynamo_request: dict[str, Any],
) -> tuple[_MockModel, list[dict[str, Any]]]:
    """Build a RequestHandler over a _MockModel and drive generate to completion."""
    model = _MockModel(triton_output_metadata, triton_responses)
    handler = handlers.RequestHandler(MagicMock(), model)

    async def _collect() -> list[dict[str, Any]]:
        return [
            dynamo_response
            async for dynamo_response in handler.generate(dynamo_request)
        ]

    return model, asyncio.run(_collect())


def assert_dynamo_response(
    dynamo_response: dict[str, Any],
    response_id: str,
    model_name: str,
    outputs: dict[str, tuple[str, list[int], list[Any]]],
) -> None:
    """Assert a Dynamo response's id, model, and output tensors."""
    assert dynamo_response["id"] == response_id
    assert dynamo_response["model"] == model_name

    tensors = {t["metadata"]["name"]: t for t in dynamo_response["tensors"]}
    assert set(tensors) == set(outputs)
    for name, (data_type, shape, values) in outputs.items():
        metadata, data = tensors[name]["metadata"], tensors[name]["data"]
        assert metadata["data_type"] == data_type
        assert metadata["shape"] == shape
        assert data["data_type"] == data_type
        assert data["values"] == values


def _mock_bytes_tensor(array: np.ndarray) -> types.SimpleNamespace:
    return types.SimpleNamespace(to_bytes_array=lambda: array)


# --- Tests ------------------------------------------------------------------


@pytest.mark.parametrize(
    "dynamo_request, expected_triton_inputs",
    [
        (
            build_dynamo_request(
                ("INPUT0", "Int32", [2], [1, 2]),
                ("INPUT1", "Int32", [2], [4, 5]),
            ),
            {
                "INPUT0": np.array([1, 2], np.int32),
                "INPUT1": np.array([4, 5], np.int32),
            },
        ),
        (
            build_dynamo_request(("IN", "Bytes", [1], [list(b"hello world 1234")])),
            {"IN": np.array([b"hello world 1234"], dtype=object)},
        ),
        (build_dynamo_request(), {}),
    ],
    ids=["int32-multi", "bytes", "empty"],
)
def test_generate_converts_dynamo_request_to_triton(
    dynamo_request, expected_triton_inputs
):
    model, _ = run_handler_generate(
        [{"name": "OUT", "datatype": "INT32"}],
        [
            build_triton_response(
                "req-id", "mock-model", {"OUT": np.array([0], np.int32)}
            )
        ],
        dynamo_request,
    )

    assert model.last_request is not None
    triton_inputs = model.last_request.inputs
    assert set(expected_triton_inputs.keys()) == set(triton_inputs.keys())
    for name, expected in expected_triton_inputs.items():
        np.testing.assert_array_equal(triton_inputs[name], expected)
        assert triton_inputs[name].dtype == expected.dtype


@pytest.mark.parametrize(
    "triton_dtype, triton_output_tensor, dynamo_dtype, dynamo_shape, dynamo_output_tensor",
    [
        (
            "FP32",
            np.array([1.0, 2.0, 3.0], np.float32),
            "Float32",
            [3],
            [1.0, 2.0, 3.0],
        ),
        ("INT64", np.array([9, 10], np.int64), "Int64", [2], [9, 10]),
        ("INT32", np.array([[1, 2], [3, 4]], np.int32), "Int32", [2, 2], [1, 2, 3, 4]),
        (
            "BYTES",
            _mock_bytes_tensor(np.array([b"hello world 1234"], dtype=object)),
            "Bytes",
            [1],
            [list(b"hello world 1234")],
        ),
        # A dtype absent from the mapping is forwarded verbatim.
        ("FP8", np.array([0.25, 0.5], np.float32), "FP8", [2], [0.25, 0.5]),
    ],
    ids=[
        "fp32",
        "int64",
        "2d-shape",
        "unknown-dtype-passthrough",
        "bytes",
    ],
)
def test_generate_converts_triton_response_to_dynamo(
    triton_dtype, triton_output_tensor, dynamo_dtype, dynamo_shape, dynamo_output_tensor
):
    _, dynamo_responses = run_handler_generate(
        [{"name": "OUT", "datatype": triton_dtype}],
        [build_triton_response("req-id", "identity", {"OUT": triton_output_tensor})],
        build_dynamo_request(("IN", "Float32", [1], [0.0])),
    )

    assert len(dynamo_responses) == 1
    assert_dynamo_response(
        dynamo_responses[0],
        "req-id",
        "identity",
        {"OUT": (dynamo_dtype, dynamo_shape, dynamo_output_tensor)},
    )


def test_generate_triton_response_multiple_outputs():
    """Every output named in the model metadata appears in the response tensors."""
    _, dynamo_responses = run_handler_generate(
        [
            {"name": "OUTPUT0", "datatype": "FP32"},
            {"name": "OUTPUT1", "datatype": "INT64"},
        ],
        [
            build_triton_response(
                "req-id",
                "multi",
                {
                    "OUTPUT0": np.array([1.5], dtype=np.float32),
                    "OUTPUT1": np.array([9, 10], dtype=np.int64),
                },
            )
        ],
        build_dynamo_request(("IN", "Float32", [1], [0.0])),
    )

    assert_dynamo_response(
        dynamo_responses[0],
        "req-id",
        "multi",
        {
            "OUTPUT0": ("Float32", [1], [1.5]),
            "OUTPUT1": ("Int64", [2], [9, 10]),
        },
    )


def test_generate_streams_multiple_triton_responses():
    """A multi-response inference stream yields one Dynamo response per element."""
    _, dynamo_responses = run_handler_generate(
        [{"name": "OUTPUT0", "datatype": "INT32"}],
        [
            build_triton_response(
                "req-id", "decoupled", {"OUTPUT0": np.array([idx], dtype=np.int32)}
            )
            for idx in range(3)
        ],
        build_dynamo_request(("IN", "Int32", [1], [0])),
    )

    assert len(dynamo_responses) == 3
    for idx, dynamo_response in enumerate(dynamo_responses):
        assert_dynamo_response(
            dynamo_response, "req-id", "decoupled", {"OUTPUT0": ("Int32", [1], [idx])}
        )


# --- Handler behavior: validation, dtype cache ----------------------


def _make_handler(
    *,
    server_ready: bool | Exception = True,
    model_ready: bool = True,
    model_name: str = "test-model",
    output_metadata: list[dict[str, str]] | None = None,
) -> handlers.RequestHandler:
    """Build a RequestHandler over configurable MagicMocks.

    ``server_ready`` accepts a bool or an Exception; when an Exception is
    passed, ``Server.ready()`` raises it. ``model_ready`` accepts only a
    bool. ``async_infer`` is wired to fail loudly so accidental inference
    is caught immediately.
    """
    if output_metadata is None:
        output_metadata = [{"name": "OUT", "datatype": "FP32"}]

    def _readiness(value: bool | Exception) -> MagicMock:
        m = MagicMock()
        if isinstance(value, Exception):
            m.side_effect = value
        else:
            m.return_value = value
        return m

    server = MagicMock()
    server.ready = _readiness(server_ready)

    model = MagicMock()
    model.ready = _readiness(model_ready)
    model.name = model_name
    model.metadata = MagicMock(return_value={"outputs": output_metadata})
    model.async_infer = MagicMock(
        side_effect=AssertionError("async_infer must not be called")
    )

    return handlers.RequestHandler(server, model)


def _run_generate(
    handler: handlers.RequestHandler, request: dict[str, Any]
) -> list[dict[str, Any]]:
    async def _collect() -> list[dict[str, Any]]:
        return [response async for response in handler.generate(request)]

    return asyncio.run(_collect())


# --- Request validation ---


def test_generate_rejects_request_missing_tensors_key():
    """A non-tensor request raises ValueError with a clear message, not KeyError."""
    handler = _make_handler()

    with pytest.raises(ValueError, match="missing 'tensors' key"):
        _run_generate(handler, {"prompt": "hello"})


@pytest.mark.parametrize("bad_dtype", ["Float16", "BFloat16"])
def test_generate_rejects_unsupported_input_dtype(bad_dtype):
    """Half-precision input tensors are rejected upfront."""
    handler = _make_handler()
    request = build_dynamo_request(("IN", bad_dtype, [1], [0.0]))

    with pytest.raises(ValueError, match=f"does not support {bad_dtype}"):
        _run_generate(handler, request)


def test_generate_error_message_lists_all_unsupported_dtypes():
    """When multiple unsupported dtypes are present they're all named in the error."""
    handler = _make_handler()
    request = build_dynamo_request(
        ("IN0", "Float16", [1], [0.0]),
        ("IN1", "BFloat16", [1], [0.0]),
    )

    with pytest.raises(ValueError, match="BFloat16 / Float16"):
        _run_generate(handler, request)


# --- Dtype cache ---


def test_init_populates_output_dtype_cache():
    """_output_dtypes is built from model.metadata()['outputs'] once at construction."""
    handler = _make_handler(
        output_metadata=[
            {"name": "OUT0", "datatype": "FP32"},
            {"name": "OUT1", "datatype": "INT32"},
        ],
    )
    assert handler._output_dtypes == {"OUT0": "FP32", "OUT1": "INT32"}


def test_generate_does_not_call_model_metadata_per_response():
    """Streaming responses reuse the cached dtype map instead of re-querying metadata."""
    model = _MockModel(
        [{"name": "OUT", "datatype": "INT32"}],
        [
            build_triton_response("req", "m", {"OUT": np.array([idx], np.int32)})
            for idx in range(3)
        ],
    )
    handler = handlers.RequestHandler(MagicMock(), model)

    # Wrap metadata *after* __init__ so we only observe generate()-time calls.
    metadata_calls_during_generate = 0
    original_metadata = model.metadata

    def _counting_metadata() -> dict[str, Any]:
        nonlocal metadata_calls_during_generate
        metadata_calls_during_generate += 1
        return original_metadata()

    model.metadata = _counting_metadata

    _run_generate(handler, build_dynamo_request(("IN", "Int32", [1], [0])))

    assert metadata_calls_during_generate == 0
