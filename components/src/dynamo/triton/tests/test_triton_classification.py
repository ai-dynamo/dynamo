# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KServe classification (class_count / top-K) through the Triton worker.

Covers the two halves separately:

* ``dynamo.triton.classification`` -- ``top_k_classifications``: score formatting,
  class ordering, output shape, and label lookup.
* ``RequestHandler`` -- honoring a request's requested outputs, turning a
  ``classification`` parameter into a BYTES tensor of class strings, and
  rejecting outputs the model does not produce.
"""

import asyncio
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from test_triton_handlers import (
    _MockModel,
    assert_dynamo_response,
    build_dynamo_request,
    build_triton_response,
    run_handler_generate,
)

from dynamo.triton import classification, handlers

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# --- class_count ------------------------------------------------------------


@pytest.mark.parametrize(
    "parameters, expected",
    [
        ({}, None),
        ({"other": {"bool": True}}, None),
        ({"classification": {"int64": 3}}, 3),
        # A plain integer, as an in-process caller would build it.
        ({"classification": 3}, 3),
    ],
    ids=["absent", "unrelated-parameter", "int64-parameter", "plain-int"],
)
def test_class_count_reads_the_classification_parameter(parameters, expected):
    assert classification.class_count(parameters) == expected


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"classification": {"string": "3"}}, "expected int64_param"),
        ({"classification": {"double": 3.0}}, "expected int64_param"),
        ({"classification": True}, "expected int64_param"),
        ({"classification": {"int64": 0}}, "expected >= 1"),
        ({"classification": {"int64": -1}}, "expected >= 1"),
    ],
    ids=["string", "double", "bool", "zero", "negative"],
)
def test_class_count_rejects_invalid_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        classification.class_count(parameters)


# --- top_k_classifications -------------------------------------------


@pytest.mark.parametrize(
    "array, triton_dtype, expected",
    [
        (
            np.array([0.1, 0.5, 0.3, 0.2], np.float32),
            "FP32",
            [b"0.500000:1", b"0.300000:2"],
        ),
        (np.array([1, 3, 2], np.int32), "INT32", [b"3:1", b"2:2"]),
    ],
    ids=["fp32", "int32"],
)
def test_top_k_classifications_returns_top_scores_in_descending_order(
    array, triton_dtype, expected
):
    classes = classification.top_k_classifications(
        array, 2, triton_dtype, batched=False
    )

    assert classes.shape == (2,)
    assert classes.tolist() == expected


def test_top_k_classifications_appends_labels_when_the_output_has_them():
    owner = types.SimpleNamespace(
        output_classification_label=lambda _out, class_index: f"label{class_index}"
    )
    output = types.SimpleNamespace(memory_buffer=types.SimpleNamespace(owner=owner))
    response = types.SimpleNamespace(outputs={"OUTPUT0": output})

    classes = classification.top_k_classifications(
        np.array([0.1, 0.5], np.float32),
        2,
        "FP32",
        batched=False,
        response=response,
        output_name="OUTPUT0",
    )

    assert classes.tolist() == [b"0.500000:1:label1", b"0.100000:0:label0"]


def test_top_k_classifications_takes_the_top_k_per_batch_entry():
    """A batched output is split on its leading dimension, and class indices are
    relative to the entry, as Triton reports them."""
    classes = classification.top_k_classifications(
        np.array([[1, 3, 2], [9, 7, 8]], np.int32), 2, "INT32", batched=True
    )

    assert classes.shape == (2, 2)
    assert classes.tolist() == [[b"3:1", b"2:2"], [b"9:0", b"8:2"]]


def test_top_k_classifications_indexes_an_unbatched_output_flat():
    classes = classification.top_k_classifications(
        np.array([[1, 3, 2], [9, 7, 8]], np.int32), 2, "INT32", batched=False
    )

    assert classes.shape == (2,)
    assert classes.tolist() == [b"9:3", b"8:5"]


def test_top_k_classifications_clamps_the_count_to_the_elements_available():
    classes = classification.top_k_classifications(
        np.array([0.1, 0.2], np.float32), 5, "FP32", batched=False
    )

    assert classes.shape == (2,)


def test_top_k_classifications_handles_an_empty_batch():
    classes = classification.top_k_classifications(
        np.zeros((0, 3), np.float32), 2, "FP32", batched=True
    )

    assert classes.shape == (0,)
    assert classes.tolist() == []


# Triton's TopkClassifications switch: UINT/INT/FP32/FP64 only.
_UNSUPPORTED_CLASSIFICATION_DTYPES = ("BOOL", "BYTES", "FP16", "BF16")


@pytest.mark.parametrize("triton_dtype", _UNSUPPORTED_CLASSIFICATION_DTYPES)
def test_top_k_classifications_rejects_types_without_an_ordering(triton_dtype):
    with pytest.raises(ValueError, match=f"unsupported type '{triton_dtype}'"):
        classification.top_k_classifications(
            np.array([0.0], np.float32), 1, triton_dtype, batched=False
        )


def test_top_k_classifications_rejects_an_output_too_large_to_sort():
    oversized = np.zeros(classification.MAX_CLASSIFICATION_ELEMENTS + 1, np.float32)

    with pytest.raises(ValueError, match="too large"):
        classification.top_k_classifications(oversized, 1, "FP32", batched=False)


# --- classification_label ---------------------------------------------------


def test_classification_label_looks_up_by_the_outputs_position_in_the_response():
    """The low-level lookup keys on an output's index, so the helper has to
    map the name a caller uses back to that position."""
    owner = types.SimpleNamespace(
        output_classification_label=lambda output_index, class_index: (
            f"raw-{output_index}-{class_index}"
        )
    )
    output = types.SimpleNamespace(memory_buffer=types.SimpleNamespace(owner=owner))
    response = types.SimpleNamespace(
        outputs={"OUTPUT0": output, "OUTPUT1": output},
    )

    assert classification.classification_label(response, "OUTPUT1", 3) == "raw-1-3"


def test_classification_label_returns_none_when_the_owner_has_no_lookup():
    """The C response exposes output_classification_label; a host copy
    whose owner is just the array must not raise."""
    response = types.SimpleNamespace(
        outputs={
            "OUTPUT0": types.SimpleNamespace(
                memory_buffer=types.SimpleNamespace(owner=object())
            )
        },
    )

    assert classification.classification_label(response, "OUTPUT0", 0) is None


def test_classification_label_maps_an_unlabelled_class_to_none():
    owner = types.SimpleNamespace(output_classification_label=lambda _out, _cls: "")
    response = types.SimpleNamespace(
        outputs={
            "OUTPUT0": types.SimpleNamespace(
                memory_buffer=types.SimpleNamespace(owner=owner)
            )
        },
    )

    assert classification.classification_label(response, "OUTPUT0", 0) is None


# --- RequestHandler ---------------------------------------------------------


def _classification_request(*outputs: tuple[str, int | None]) -> dict[str, Any]:
    """A one-input request asking for the given (output name, class count) pairs."""
    request = build_dynamo_request(("IN", "Float32", [1], [0.0]))
    request["outputs"] = [
        {"name": name}
        if count is None
        else {"name": name, "parameters": {"classification": {"int64": count}}}
        for name, count in outputs
    ]
    return request


class _LabelledOutput:
    """A response output that reads as DLPack and carries the low-level response
    the label lookup goes through, the way a tritonserver output tensor does."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self.memory_buffer = types.SimpleNamespace(
            owner=types.SimpleNamespace(
                output_classification_label=(
                    lambda _output_index, class_index: f"label{class_index}"
                )
            )
        )

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        return self._array.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self) -> Any:
        return self._array.__dlpack_device__()

    def to_bytes_array(self) -> np.ndarray:
        return self._array


def _labelled_response(outputs: dict[str, Any]) -> types.SimpleNamespace:
    return build_triton_response(
        "req-id",
        "classifier",
        {name: _LabelledOutput(array) for name, array in outputs.items()},
    )


def test_generate_returns_class_strings_for_a_classification_output():
    _, dynamo_responses = run_handler_generate(
        [{"name": "OUTPUT0", "datatype": "FP32"}],
        [_labelled_response({"OUTPUT0": np.array([0.1, 0.5, 0.3], np.float32)})],
        _classification_request(("OUTPUT0", 2)),
    )

    assert_dynamo_response(
        dynamo_responses[0],
        "req-id",
        "classifier",
        {
            "OUTPUT0": (
                "Bytes",
                [2],
                [list(b"0.500000:1:label1"), list(b"0.300000:2:label2")],
            )
        },
    )


def test_generate_keeps_the_batch_dimension_for_a_batchable_model():
    _, dynamo_responses = run_handler_generate(
        [{"name": "OUTPUT0", "datatype": "INT32"}],
        [_labelled_response({"OUTPUT0": np.array([[1, 3], [9, 7]], np.int32)})],
        _classification_request(("OUTPUT0", 1)),
        max_batch_size=8,
    )

    assert_dynamo_response(
        dynamo_responses[0],
        "req-id",
        "classifier",
        {"OUTPUT0": ("Bytes", [2, 1], [list(b"3:1:label1"), list(b"9:0:label0")])},
    )


def test_generate_mixes_raw_and_classification_outputs():
    _, dynamo_responses = run_handler_generate(
        [
            {"name": "OUTPUT0", "datatype": "FP32"},
            {"name": "OUTPUT1", "datatype": "FP32"},
        ],
        [
            _labelled_response(
                {
                    "OUTPUT0": np.array([0.25, 0.75], np.float32),
                    "OUTPUT1": np.array([0.5], np.float32),
                }
            )
        ],
        _classification_request(("OUTPUT0", 1), ("OUTPUT1", None)),
    )

    assert_dynamo_response(
        dynamo_responses[0],
        "req-id",
        "classifier",
        {
            "OUTPUT0": ("Bytes", [1], [list(b"0.750000:1:label1")]),
            "OUTPUT1": ("Float32", [1], [0.5]),
        },
    )


def test_generate_returns_only_the_requested_outputs():
    _, dynamo_responses = run_handler_generate(
        [
            {"name": "OUTPUT0", "datatype": "FP32"},
            {"name": "OUTPUT1", "datatype": "INT32"},
        ],
        [
            build_triton_response(
                "req-id",
                "classifier",
                {
                    "OUTPUT0": np.array([1.5], np.float32),
                    "OUTPUT1": np.array([7], np.int32),
                },
            )
        ],
        _classification_request(("OUTPUT1", None)),
    )

    assert_dynamo_response(
        dynamo_responses[0],
        "req-id",
        "classifier",
        {"OUTPUT1": ("Int32", [1], [7])},
    )


def test_generate_returns_every_output_when_outputs_is_empty():
    """An empty requested-output list is the KServe 'all outputs' default."""
    request = build_dynamo_request(("IN", "Float32", [1], [0.0]))
    request["outputs"] = []
    _, dynamo_responses = run_handler_generate(
        [
            {"name": "OUTPUT0", "datatype": "FP32"},
            {"name": "OUTPUT1", "datatype": "INT32"},
        ],
        [
            build_triton_response(
                "req-id",
                "classifier",
                {
                    "OUTPUT0": np.array([1.5], np.float32),
                    "OUTPUT1": np.array([7], np.int32),
                },
            )
        ],
        request,
    )

    assert set(
        tensor["metadata"]["name"] for tensor in dynamo_responses[0]["tensors"]
    ) == {"OUTPUT0", "OUTPUT1"}


def test_generate_rejects_an_output_the_model_does_not_produce():
    with pytest.raises(ValueError, match="unexpected inference output 'OUTPUT1'"):
        run_handler_generate(
            [{"name": "OUTPUT0", "datatype": "FP32"}],
            [build_triton_response("req-id", "classifier", {})],
            _classification_request(("OUTPUT1", 1)),
        )


def test_generate_rejects_an_invalid_classification_parameter_before_infer():
    model = _MockModel(
        [{"name": "OUTPUT0", "datatype": "FP32"}],
        [build_triton_response("req-id", "classifier", {})],
    )
    handler = handlers.RequestHandler(MagicMock(), model)

    with pytest.raises(ValueError, match="expected >= 1"):
        asyncio.run(
            handler.generate(_classification_request(("OUTPUT0", 0))).__anext__()
        )

    assert model.last_request is None


@pytest.mark.parametrize("triton_dtype", ("BYTES", "FP16"))
def test_generate_rejects_classification_on_an_unsupported_output_dtype(triton_dtype):
    array = (
        np.array([b"a"], dtype=object)
        if triton_dtype == "BYTES"
        else np.array([0.0], np.float32)
    )
    with pytest.raises(ValueError, match=f"unsupported type '{triton_dtype}'"):
        run_handler_generate(
            [{"name": "OUTPUT0", "datatype": triton_dtype}],
            [_labelled_response({"OUTPUT0": array})],
            _classification_request(("OUTPUT0", 1)),
        )
