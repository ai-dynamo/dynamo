# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.triton.handlers.RequestHandler: Dynamo <-> Triton
tensor conversion, streaming responses, request validation, and dtype caching."""

import asyncio
import ctypes
import types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import tritonclient.grpc.model_config_pb2 as TritonModelConfigPb
from tritonserver import DataType, MemoryType, Tensor

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
        self,
        output_metadata: list[dict[str, Any]],
        responses: list[Any],
        max_batch_size: int = 0,
        name: str = "mock-model",
    ) -> None:
        self._output_metadata = output_metadata
        self._responses = responses
        self._max_batch_size = max_batch_size
        self.name = name
        self.last_request: types.SimpleNamespace | None = None

    def create_request(self) -> types.SimpleNamespace:
        self.last_request = types.SimpleNamespace(inputs={})
        return self.last_request

    def metadata(self) -> dict[str, Any]:
        return {"outputs": self._output_metadata}

    def config(self) -> dict[str, Any]:
        return {"max_batch_size": self._max_batch_size}

    def async_infer(self, _inference_request: Any) -> AsyncIterator[Any]:
        async def _stream() -> AsyncIterator[Any]:
            for response in self._responses:
                yield response

        return _stream()


def build_dynamo_request(
    *tensor_specs: tuple[str, str, list[int], list[Any]],
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
    max_batch_size: int = 0,
) -> tuple[_MockModel, list[dict[str, Any]]]:
    """Build a RequestHandler over a _MockModel and drive generate to completion."""
    model = _MockModel(triton_output_metadata, triton_responses, max_batch_size)
    handler = handlers.RequestHandler(
        MagicMock(),
        model,
        TritonModelConfigPb.ModelConfig(max_batch_size=max_batch_size),
    )

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
    ],
    ids=[
        "fp32",
        "int64",
        "bytes",
        "2d-shape",
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
    model.config = MagicMock(return_value={})
    model.async_infer = MagicMock(
        side_effect=AssertionError("async_infer must not be called")
    )

    return handlers.RequestHandler(
        server, model, TritonModelConfigPb.ModelConfig(max_batch_size=0)
    )


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


def _run_half_precision_input(
    dynamo_dtype: str, shape: list[int], values: list[int]
) -> Any:
    """Run one half-precision input through generate and return what Triton got."""
    # Use an int32 output so the response path (np.from_dlpack) stays out of
    # scope: these tests cover the input side only.
    model, _ = run_handler_generate(
        [{"name": "OUT", "datatype": "INT32"}],
        [build_triton_response("req-id", "identity", {"OUT": np.array([0], np.int32)})],
        build_dynamo_request(("IN", dynamo_dtype, shape, values)),
    )
    assert model.last_request is not None
    return model.last_request.inputs["IN"]


def test_generate_passes_fp16_input_as_numpy():
    """FP16 inputs reach tritonserver as a float16 ndarray, which its
    NUMPY_TO_TRITON_DTYPE mapping accepts."""
    # Rust half values arrive as u16 bit patterns, not numeric floats.
    bits = [0x3E00, 0xC080, 0x4240]
    triton_input = _run_half_precision_input("Float16", [3], bits)

    assert isinstance(triton_input, np.ndarray)
    assert triton_input.dtype == np.float16
    np.testing.assert_array_equal(
        triton_input, np.array([1.5, -2.25, 3.125], dtype=np.float16)
    )


# Little-endian BF16 bytes for [1.5, -2.25, 3.125, -4.0].
_BF16_ROW_BYTES = bytes.fromhex("c03f10c0484080c0")


@pytest.mark.parametrize("batch", [1, 4], ids=["batch1", "batch4"])
def test_generate_passes_bf16_input_as_triton_tensor(batch):
    """BF16 inputs reach tritonserver as a Tensor(DataType.BF16) over the raw
    bytes. tritonserver cannot convert an ml_dtypes.bfloat16 ndarray itself."""
    bits = [0x3FC0, 0xC010, 0x4048, 0xC080]
    shape = [batch, len(bits)]
    triton_input = _run_half_precision_input("BFloat16", shape, bits * batch)

    assert isinstance(triton_input, Tensor)
    assert triton_input.data_type == DataType.BF16
    assert list(triton_input.shape) == shape
    buffer = triton_input.memory_buffer
    assert buffer.memory_type == MemoryType.CPU
    assert buffer.size == len(_BF16_ROW_BYTES) * batch
    assert ctypes.string_at(buffer.data_ptr, buffer.size) == _BF16_ROW_BYTES * batch


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
    handler = handlers.RequestHandler(
        MagicMock(), model, TritonModelConfigPb.ModelConfig(max_batch_size=0)
    )

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
