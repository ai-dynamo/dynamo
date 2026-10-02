# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.triton.pooling_handlers.ClassifyWorkerHandler:
config-driven input/output resolution, request/response translation for the
OpenAI /v1/classify wire, id2label lookup, and validation of unsupported
inputs (token-ID variants, pooling requests, empty payloads)."""

from __future__ import annotations

import asyncio
import types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import tritonclient.grpc.model_config_pb2 as mc

from dynamo.triton.pooling_handlers import ClassifyWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config_proto(
    inputs: list[tuple[str, int]],
    outputs: list[tuple[str, int]],
    name: str = "mock-classifier",
    max_batch_size: int = 8,
) -> mc.ModelConfig:
    """Build a minimal ModelConfig with the given input/output (name, dtype) pairs."""
    config = mc.ModelConfig(name=name, max_batch_size=max_batch_size)
    for input_name, dtype in inputs:
        config.input.add(name=input_name, data_type=dtype, dims=[-1])
    for output_name, dtype in outputs:
        config.output.add(name=output_name, data_type=dtype, dims=[-1])
    return config


class _MockModel:
    """Records the InferRequest built by the handler and replays fixed responses."""

    def __init__(
        self,
        responses: list[Any],
        max_batch_size: int = 8,
        name: str = "mock-classifier",
    ) -> None:
        self._responses = responses
        self._max_batch_size = max_batch_size
        self.name = name
        self.last_request: types.SimpleNamespace | None = None

    def create_request(self) -> types.SimpleNamespace:
        self.last_request = types.SimpleNamespace(inputs={})
        return self.last_request

    def config(self) -> dict[str, Any]:
        return {"max_batch_size": self._max_batch_size}

    def ready(self) -> bool:
        return True

    def async_infer(self, _inference_request: Any) -> AsyncIterator[Any]:
        async def _stream() -> AsyncIterator[Any]:
            for response in self._responses:
                yield response

        return _stream()


class _FakeOwner:
    """Mimics the memory_buffer.owner label-lookup surface Triton exposes."""

    def __init__(self, output_labels: dict[int, dict[int, str]]) -> None:
        self._output_labels = output_labels

    def output_classification_label(self, output_idx: int, class_index: int) -> str:
        return self._output_labels.get(output_idx, {}).get(class_index, "")


def _mock_fp32_tensor(array: np.ndarray, owner: Any = None) -> Any:
    """Wrap a numpy FP32 array so it looks like a Triton response tensor.

    ``ClassifyWorkerHandler`` reads FP32 outputs with ``np.from_dlpack`` (via
    ``__dlpack__``) and reaches into ``memory_buffer.owner`` for label lookup;
    supply both from the numpy array with a small helper class.
    """

    class _Tensor:
        # Not a TritonTensor, so the handler's isinstance guard against GPU
        # tensors falls through and we skip .to_host().
        def __dlpack__(self, *_args, **_kwargs):
            return array.__dlpack__()

        def __dlpack_device__(self):
            return array.__dlpack_device__()

    tensor = _Tensor()
    tensor.memory_buffer = types.SimpleNamespace(owner=owner)
    return tensor


def _run(
    handler: ClassifyWorkerHandler,
    request: dict,
    context: Any = None,
) -> list[dict]:
    async def _collect() -> list[dict]:
        return [response async for response in handler.generate(request, context)]

    return asyncio.run(_collect())


class _FakeContext:
    """Minimal stand-in for the worker runtime context object."""

    def __init__(self, request_id: str) -> None:
        self._id = request_id

    def id(self) -> str:
        return self._id


def _make_handler(
    *,
    inputs: list[tuple[str, int]] | None = None,
    outputs: list[tuple[str, int]] | None = None,
    responses: list[Any] | None = None,
    max_batch_size: int = 8,
    classify_input_name: str | None = None,
    classify_output_name: str | None = None,
) -> tuple[_MockModel, ClassifyWorkerHandler]:
    config = _make_config_proto(
        inputs=inputs or [("TEXT", mc.DataType.TYPE_STRING)],
        outputs=outputs or [("probs", mc.DataType.TYPE_FP32)],
        max_batch_size=max_batch_size,
    )
    model = _MockModel(responses or [], max_batch_size=max_batch_size)
    handler = ClassifyWorkerHandler(
        server=MagicMock(),
        model=model,
        triton_model_config=config,
        classify_input_name=classify_input_name,
        classify_output_name=classify_output_name,
    )
    return model, handler


# ---------------------------------------------------------------------------
# Constructor / auto-resolution
# ---------------------------------------------------------------------------


class TestInitAndResolve:
    def test_explicit_overrides_win(self) -> None:
        _, handler = _make_handler(
            inputs=[
                ("TEXT_A", mc.DataType.TYPE_STRING),
                ("TEXT_B", mc.DataType.TYPE_STRING),
            ],
            outputs=[
                ("probs_a", mc.DataType.TYPE_FP32),
                ("probs_b", mc.DataType.TYPE_FP32),
            ],
            classify_input_name="TEXT_B",
            classify_output_name="probs_a",
        )
        assert handler._input_name == "TEXT_B"
        assert handler._output_name == "probs_a"

    def test_ambiguous_bytes_input_without_override_fails(self) -> None:
        with pytest.raises(ValueError, match="TYPE_STRING input tensor"):
            _make_handler(
                inputs=[
                    ("A", mc.DataType.TYPE_STRING),
                    ("B", mc.DataType.TYPE_STRING),
                ]
            )

    def test_no_bytes_input_fails(self) -> None:
        with pytest.raises(ValueError, match="TYPE_STRING input tensor"):
            _make_handler(inputs=[("tokens", mc.DataType.TYPE_INT64)])

    def test_ambiguous_fp32_output_without_override_fails(self) -> None:
        with pytest.raises(ValueError, match="TYPE_FP32 output tensor"):
            _make_handler(
                outputs=[
                    ("probs", mc.DataType.TYPE_FP32),
                    ("scores", mc.DataType.TYPE_FP32),
                ]
            )

    def test_no_fp32_output_fails(self) -> None:
        with pytest.raises(ValueError, match="TYPE_FP32 output tensor"):
            _make_handler(outputs=[("classes", mc.DataType.TYPE_INT32)])

    def test_explicit_input_name_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="no TYPE_STRING input named 'MISSING'"):
            _make_handler(classify_input_name="MISSING")

    def test_explicit_input_name_wrong_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="no TYPE_STRING input named 'tokens'"):
            _make_handler(
                inputs=[
                    ("TEXT", mc.DataType.TYPE_STRING),
                    ("tokens", mc.DataType.TYPE_INT64),
                ],
                classify_input_name="tokens",
            )

    def test_explicit_output_name_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="no TYPE_FP32 output named 'MISSING'"):
            _make_handler(classify_output_name="MISSING")

    def test_explicit_output_name_wrong_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="no TYPE_FP32 output named 'classes'"):
            _make_handler(
                outputs=[
                    ("probs", mc.DataType.TYPE_FP32),
                    ("classes", mc.DataType.TYPE_INT32),
                ],
                classify_output_name="classes",
            )

    def test_batched_reads_parsed_proto_not_runtime_config(self) -> None:
        # Regression: the disk-fallback path in main.py._read_model_config
        # leaves model.config() empty; batching must still come from the
        # parsed protobuf so batchable classifiers get the [N, 1] shape.
        class _EmptyRuntimeMockModel(_MockModel):
            def config(self) -> dict[str, Any]:
                return {}

        config = _make_config_proto(
            inputs=[("TEXT", mc.DataType.TYPE_STRING)],
            outputs=[("probs", mc.DataType.TYPE_FP32)],
            max_batch_size=8,
        )
        model = _EmptyRuntimeMockModel([])
        handler = ClassifyWorkerHandler(
            server=MagicMock(),
            model=model,
            triton_model_config=config,
        )
        assert handler._batched is True

    def _config_with_input_dims(self, dims: list[int]) -> "mc.ModelConfig":
        """Build a classify-shaped ModelConfig with explicit STRING input
        dims so dims-validation tests can probe shapes the shared fixture
        does not expose."""
        config = mc.ModelConfig(name="mock-classifier", max_batch_size=8)
        config.input.add(name="TEXT", data_type=mc.DataType.TYPE_STRING, dims=dims)
        config.output.add(name="probs", data_type=mc.DataType.TYPE_FP32, dims=[-1])
        return config

    def test_rank_2_string_input_dims_rejected(self) -> None:
        # Model declares dims=[1, 1], so Triton expects request shape
        # [N, 1, 1]. The handler always sends [N, 1], so every request
        # would fail. Reject at startup.
        with pytest.raises(ValueError, match=r"dims=\[1, 1\]"):
            ClassifyWorkerHandler(
                server=MagicMock(),
                model=_MockModel([]),
                triton_model_config=self._config_with_input_dims([1, 1]),
            )

    def test_multi_string_per_item_dims_rejected(self) -> None:
        # dims=[2] means two strings per request item; the OpenAI classify
        # request only carries one string per item, so this layout cannot
        # be served.
        with pytest.raises(ValueError, match=r"dims=\[2\]"):
            ClassifyWorkerHandler(
                server=MagicMock(),
                model=_MockModel([]),
                triton_model_config=self._config_with_input_dims([2]),
            )


# ---------------------------------------------------------------------------
# Classify happy paths
# ---------------------------------------------------------------------------


class TestClassify:
    def test_single_text_input_returns_probs_and_label(self) -> None:
        probs = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
        owner = _FakeOwner({0: {0: "neutral", 1: "positive", 2: "negative"}})
        model, handler = _make_handler(
            responses=[
                types.SimpleNamespace(
                    outputs={"probs": _mock_fp32_tensor(probs, owner=owner)}
                )
            ]
        )

        responses = _run(handler, {"model": "clf", "input": "hello world"})

        assert len(responses) == 1
        response = responses[0]
        assert response["model"] == "clf"
        assert response["object"] == "list"
        assert response["usage"]["completion_tokens"] == 0
        assert len(response["data"]) == 1
        entry = response["data"][0]
        assert entry["index"] == 0
        assert entry["num_classes"] == 3
        assert entry["label"] == "positive"  # argmax(probs) == 1
        assert entry["probs"] == pytest.approx([0.1, 0.7, 0.2])

        assert model.last_request is not None
        arr = model.last_request.inputs["TEXT"]
        assert arr.shape == (1, 1)
        assert arr[0, 0] == b"hello world"

    def test_batch_text_input_preserves_order(self) -> None:
        probs = np.array(
            [
                [0.6, 0.4],
                [0.1, 0.9],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )
        owner = _FakeOwner({0: {0: "no", 1: "yes"}})
        model, handler = _make_handler(
            responses=[
                types.SimpleNamespace(
                    outputs={"probs": _mock_fp32_tensor(probs, owner=owner)}
                )
            ]
        )

        responses = _run(handler, {"input": ["a", "b", "c"]})

        response = responses[0]
        assert [entry["index"] for entry in response["data"]] == [0, 1, 2]
        assert [entry["label"] for entry in response["data"]] == ["no", "yes", "no"]
        arr = model.last_request.inputs["TEXT"]
        assert arr.shape == (3, 1)
        assert [arr[i, 0] for i in range(3)] == [b"a", b"b", b"c"]

    def test_missing_label_owner_yields_none(self) -> None:
        probs = np.array([[0.9, 0.1]], dtype=np.float32)
        model, handler = _make_handler(
            responses=[
                types.SimpleNamespace(
                    outputs={"probs": _mock_fp32_tensor(probs, owner=None)}
                )
            ]
        )

        responses = _run(handler, {"input": "hi"})
        assert responses[0]["data"][0]["label"] is None

    def test_unbatched_output_is_normalized(self) -> None:
        probs = np.array([0.2, 0.5, 0.3], dtype=np.float32)
        owner = _FakeOwner({0: {0: "x", 1: "y", 2: "z"}})
        model, handler = _make_handler(
            responses=[
                types.SimpleNamespace(
                    outputs={"probs": _mock_fp32_tensor(probs, owner=owner)}
                )
            ],
            max_batch_size=0,
        )

        responses = _run(handler, {"input": "single"})
        assert len(responses[0]["data"]) == 1
        assert responses[0]["data"][0]["num_classes"] == 3
        assert responses[0]["data"][0]["label"] == "y"
        arr = model.last_request.inputs["TEXT"]
        assert arr.shape == (1,)
        assert arr[0] == b"single"

    def test_request_id_flows_into_response_id(self) -> None:
        probs = np.array([[0.1, 0.9]], dtype=np.float32)
        _, handler = _make_handler(
            responses=[
                types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
            ]
        )
        responses = _run(handler, {"input": "x", "request_id": "req-42"})
        assert responses[0]["id"] == "classify-req-42"

    def test_context_id_is_response_id_fallback(self) -> None:
        # When the client omits ``request_id``, the response id must fall
        # back to the runtime context id so concurrent responses stay
        # correlatable, matching the vLLM classify adapter.
        probs = np.array([[0.1, 0.9]], dtype=np.float32)
        _, handler = _make_handler(
            responses=[
                types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
            ]
        )
        responses = _run(handler, {"input": "x"}, context=_FakeContext("ctx-99"))
        assert responses[0]["id"] == "classify-ctx-99"

    def test_output_shape_flattened_from_3d(self) -> None:
        # A Triton output configured with dims=[1, classes] arrives as
        # [batch, 1, classes]. The handler must flatten non-batch dims per
        # row instead of failing at the 2-tuple unpack.
        probs = np.array(
            [
                [[0.1, 0.7, 0.2]],
                [[0.5, 0.4, 0.1]],
            ],
            dtype=np.float32,
        )
        assert probs.ndim == 3
        _, handler = _make_handler(
            responses=[
                types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
            ]
        )
        responses = _run(handler, {"input": ["a", "b"]})
        assert len(responses[0]["data"]) == 2
        assert responses[0]["data"][0]["num_classes"] == 3
        assert responses[0]["data"][0]["probs"] == pytest.approx([0.1, 0.7, 0.2])
        assert responses[0]["data"][1]["probs"] == pytest.approx([0.5, 0.4, 0.1])

    def test_row_count_mismatch_raises(self) -> None:
        # 3 inputs sent, but the model only returns 2 rows (a misconfigured
        # ensemble, an unbatched model asked to batch, or a wrong output shape).
        # The handler must not silently return a truncated classify response;
        # it must raise so the misconfiguration is visible.
        probs = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        _, handler = _make_handler(
            responses=[
                types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
            ]
        )
        with pytest.raises(RuntimeError, match="expected one row per input"):
            _run(handler, {"input": ["a", "b", "c"]})

    def test_unbatched_rank_2_output_flattens_to_single_row(self) -> None:
        # Unbatched model with dims=[2, 2] returns shape (2, 2), four
        # class scores for one input. Rank alone would read this as two
        # rows and trip the row-count check; the handler must branch on
        # the batching contract and flatten to a single row.
        probs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        _, handler = _make_handler(
            responses=[
                types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
            ],
            max_batch_size=0,
        )
        responses = _run(handler, {"input": "single"})
        assert len(responses[0]["data"]) == 1
        entry = responses[0]["data"][0]
        assert entry["num_classes"] == 4
        assert entry["probs"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_batched_declared_but_rank_1_output_raises(self) -> None:
        # A model with max_batch_size>0 must always return a leading
        # batch axis. A rank-1 response is a contract violation and must
        # raise loudly rather than be silently reshaped as one row.
        probs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        _, handler = _make_handler(
            responses=[
                types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
            ]
        )
        with pytest.raises(RuntimeError, match="expected a leading batch axis"):
            _run(handler, {"input": "x"})

    def test_unbatched_rejects_multiple_prompts(self) -> None:
        # Unbatched models cannot serve more than one prompt per request;
        # the handler must reject the shape mismatch at the front door
        # instead of letting Triton emit an opaque infer-time error.
        _, handler = _make_handler(max_batch_size=0)
        with pytest.raises(ValueError, match="received 2 prompts"):
            _run(handler, {"input": ["a", "b"]})


# ---------------------------------------------------------------------------
# Dispatch & validation
# ---------------------------------------------------------------------------


class TestValidation:
    def _handler(self) -> ClassifyWorkerHandler:
        _, handler = _make_handler()
        return handler

    def test_pooling_request_rejected_until_supported(self) -> None:
        handler = self._handler()
        with pytest.raises(ValueError, match="does not yet serve /v1/pooling"):
            _run(handler, {"input": "x", "encoding_format": "float"})

    def test_missing_input_field(self) -> None:
        with pytest.raises(ValueError, match="missing required 'input'"):
            _run(self._handler(), {"model": "clf"})

    def test_empty_string_input(self) -> None:
        with pytest.raises(ValueError, match="empty string"):
            _run(self._handler(), {"input": ""})

    def test_empty_list_input(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            _run(self._handler(), {"input": []})

    def test_list_with_empty_string_input(self) -> None:
        with pytest.raises(ValueError, match="empty strings"):
            _run(self._handler(), {"input": ["a", ""]})

    def test_token_id_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not yet accept token-ID"):
            _run(self._handler(), {"input": [1, 2, 3]})

    def test_token_batch_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not yet accept token-ID"):
            _run(self._handler(), {"input": [[1, 2], [3, 4]]})

    def test_unsupported_input_type(self) -> None:
        with pytest.raises(ValueError, match="unsupported type"):
            _run(self._handler(), {"input": 42})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("use_activation", False),
            ("use_activation", True),
            ("add_special_tokens", False),
            ("add_special_tokens", True),
            ("truncate_prompt_tokens", 128),
            ("truncation_side", "left"),
            ("mm_processor_kwargs", {"do_resize": False}),
            ("cache_salt", "salt"),
        ],
    )
    def test_unsupported_control_rejected(self, field: str, value: Any) -> None:
        # The Triton path can neither apply nor skip these; silently ignoring
        # them would let a client send ``truncate_prompt_tokens=128`` and get
        # a full-length classification back. Reject explicitly with 400
        # (ValueError → BackendError::InvalidArgument in the Rust binding).
        with pytest.raises(ValueError, match=f"does not honor '{field}'"):
            _run(self._handler(), {"input": "x", field: value})

    def test_unsupported_control_null_is_ignored(self) -> None:
        # One null-row proves the None-means-unset contract; per-field
        # rejection coverage lives in test_unsupported_control_rejected.
        probs = np.array([[0.5, 0.5]], dtype=np.float32)
        _, handler = _make_handler(
            responses=[
                types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
            ]
        )
        responses = _run(handler, {"input": "x", "use_activation": None})
        assert len(responses[0]["data"]) == 1

    def test_priority_nonzero_rejected(self) -> None:
        # `priority` is `i64` in NvCreateClassifyRequest and always
        # serialized. Only non-default (non-zero) values must be rejected.
        with pytest.raises(ValueError, match="does not honor 'priority'"):
            _run(self._handler(), {"input": "x", "priority": -2})

    def test_priority_zero_and_default_pass(self) -> None:
        # Default 0 (or explicit 0) must not trip the guard.
        probs = np.array([[0.5, 0.5]], dtype=np.float32)
        for req in [{"input": "x", "priority": 0}, {"input": "x"}]:
            _, handler = _make_handler(
                responses=[
                    types.SimpleNamespace(outputs={"probs": _mock_fp32_tensor(probs)})
                ]
            )
            assert len(_run(handler, req)[0]["data"]) == 1


class TestHealthProbe:
    def test_probe_short_circuits_before_inference(self) -> None:
        from dynamo.health_check import HEALTH_CHECK_KEY

        _, handler = _make_handler()
        responses = _run(handler, {HEALTH_CHECK_KEY: True, "model": "clf"})
        assert len(responses) == 1
        response = responses[0]
        assert response["object"] == "list"
        assert response["data"] == []
        assert response["usage"]["prompt_tokens"] == 0
