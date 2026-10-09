# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import tritonclient.grpc.model_config_pb2 as mc

from dynamo.triton.pooling_handlers import EmbeddingWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_config_proto(
    inputs: list[tuple[str, int]],
    outputs: list[tuple[str, int]],
    name: str = "mock-embedder",
    max_batch_size: int = 8,
) -> mc.ModelConfig:
    config = mc.ModelConfig(name=name, max_batch_size=max_batch_size)
    for input_name, dtype in inputs:
        config.input.add(name=input_name, data_type=dtype, dims=[-1])
    for output_name, dtype in outputs:
        config.output.add(name=output_name, data_type=dtype, dims=[-1])
    return config


class _MockModel:
    def __init__(
        self,
        responses: list[Any],
        max_batch_size: int = 8,
        name: str = "mock-embedder",
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


def _mock_fp32_tensor(array: np.ndarray) -> Any:
    """Numpy FP32 array wrapper that satisfies the handler's ``np.from_dlpack``
    read and bypasses the ``.to_host()`` branch (not a TritonTensor)."""

    class _Tensor:
        def __dlpack__(self, *_args, **_kwargs):
            return array.__dlpack__()

        def __dlpack_device__(self):
            return array.__dlpack_device__()

    return _Tensor()


def _run(
    handler: EmbeddingWorkerHandler,
    request: dict,
) -> list[dict]:
    async def _collect() -> list[dict]:
        return [response async for response in handler.generate(request)]

    return asyncio.run(_collect())


def _make_handler(
    *,
    inputs: list[tuple[str, int]] | None = None,
    outputs: list[tuple[str, int]] | None = None,
    responses: list[Any] | None = None,
    max_batch_size: int = 8,
    embed_input_name: str | None = None,
    embed_output_name: str | None = None,
) -> tuple[_MockModel, EmbeddingWorkerHandler]:
    config = _make_config_proto(
        inputs=inputs or [("TEXT", mc.DataType.TYPE_STRING)],
        outputs=outputs or [("embedding", mc.DataType.TYPE_FP32)],
        max_batch_size=max_batch_size,
    )
    model = _MockModel(responses or [], max_batch_size=max_batch_size)
    handler = EmbeddingWorkerHandler(
        server=MagicMock(),
        model=model,
        triton_model_config=config,
        embed_input_name=embed_input_name,
        embed_output_name=embed_output_name,
    )
    return model, handler


def _decode_base64_to_floats(encoded: str) -> list[float]:
    return np.frombuffer(base64.b64decode(encoded), dtype="<f4").tolist()


class TestInitAndResolve:
    def test_explicit_overrides_win(self) -> None:
        _, handler = _make_handler(
            inputs=[
                ("TEXT_A", mc.DataType.TYPE_STRING),
                ("TEXT_B", mc.DataType.TYPE_STRING),
            ],
            outputs=[
                ("emb_a", mc.DataType.TYPE_FP32),
                ("emb_b", mc.DataType.TYPE_FP32),
            ],
            embed_input_name="TEXT_B",
            embed_output_name="emb_a",
        )
        assert handler._input_name == "TEXT_B"
        assert handler._output_name == "emb_a"

    def test_ambiguous_bytes_input_without_override_fails(self) -> None:
        with pytest.raises(ValueError, match="TYPE_STRING input tensor"):
            _make_handler(
                inputs=[
                    ("A", mc.DataType.TYPE_STRING),
                    ("B", mc.DataType.TYPE_STRING),
                ]
            )

    def test_ambiguous_fp32_output_without_override_fails(self) -> None:
        with pytest.raises(ValueError, match="TYPE_FP32 output tensor"):
            _make_handler(
                outputs=[
                    ("emb", mc.DataType.TYPE_FP32),
                    ("scores", mc.DataType.TYPE_FP32),
                ]
            )

    def test_rank_2_string_input_dims_rejected(self) -> None:
        # The handler builds [N, 1] request tensors; a declared [1, 1]
        # input would produce shape [N, 1, 1] and fail every request.
        config = mc.ModelConfig(name="rank2", max_batch_size=4)
        config.input.add(name="TEXT", data_type=mc.DataType.TYPE_STRING, dims=[1, 1])
        config.output.add(name="embedding", data_type=mc.DataType.TYPE_FP32, dims=[-1])
        with pytest.raises(ValueError, match=r"dims=\[1, 1\]"):
            EmbeddingWorkerHandler(
                server=MagicMock(),
                model=_MockModel([]),
                triton_model_config=config,
            )

    def test_endpoint_label_in_dims_error(self) -> None:
        config = mc.ModelConfig(name="rank2", max_batch_size=4)
        config.input.add(name="TEXT", data_type=mc.DataType.TYPE_STRING, dims=[2])
        config.output.add(name="embedding", data_type=mc.DataType.TYPE_FP32, dims=[-1])
        with pytest.raises(ValueError, match="/v1/embeddings path"):
            EmbeddingWorkerHandler(
                server=MagicMock(),
                model=_MockModel([]),
                triton_model_config=config,
            )


class TestEmbedding:
    def test_single_text_input_returns_base64_embedding(self) -> None:
        embedding = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        response = types.SimpleNamespace(
            outputs={"embedding": _mock_fp32_tensor(embedding)}
        )
        _, handler = _make_handler(responses=[response])

        [body] = _run(handler, {"input": "hello", "model": "mock-embedder"})

        assert body["object"] == "list"
        assert body["model"] == "mock-embedder"
        assert len(body["data"]) == 1
        assert body["data"][0]["object"] == "embedding"
        assert body["data"][0]["index"] == 0
        assert _decode_base64_to_floats(body["data"][0]["embedding"]) == pytest.approx(
            [0.1, 0.2, 0.3, 0.4]
        )
        assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}

    def test_batch_text_input_preserves_order(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        response = types.SimpleNamespace(
            outputs={"embedding": _mock_fp32_tensor(embeddings)}
        )
        _, handler = _make_handler(responses=[response])

        [body] = _run(handler, {"input": ["a", "b", "c"]})

        assert [row["index"] for row in body["data"]] == [0, 1, 2]
        decoded = [_decode_base64_to_floats(row["embedding"]) for row in body["data"]]
        assert decoded == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    def test_base64_round_trip_is_bit_exact(self) -> None:
        # Carefully chosen fractional values that have exact FP32 representations
        # so the base64 round-trip is bit-exact rather than approximate.
        vec = np.array([[1.5, -2.25, 0.125, -0.0625]], dtype=np.float32)
        response = types.SimpleNamespace(outputs={"embedding": _mock_fp32_tensor(vec)})
        _, handler = _make_handler(responses=[response])

        [body] = _run(handler, {"input": "x"})

        decoded = _decode_base64_to_floats(body["data"][0]["embedding"])
        assert decoded == [1.5, -2.25, 0.125, -0.0625]

    def test_unbatched_output_is_normalized(self) -> None:
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        response = types.SimpleNamespace(outputs={"embedding": _mock_fp32_tensor(vec)})
        _, handler = _make_handler(responses=[response], max_batch_size=0)

        [body] = _run(handler, {"input": "only"})

        assert len(body["data"]) == 1
        assert _decode_base64_to_floats(body["data"][0]["embedding"]) == pytest.approx(
            [0.1, 0.2, 0.3]
        )

    def test_unbatched_rank_2_output_flattens_to_single_row(self) -> None:
        # max_batch_size=0 + dims=[2, 2] = one 4-element embedding, not
        # two 2-element embeddings; branch on batching contract, not rank.
        tensor = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        response = types.SimpleNamespace(
            outputs={"embedding": _mock_fp32_tensor(tensor)}
        )
        _, handler = _make_handler(responses=[response], max_batch_size=0)

        [body] = _run(handler, {"input": "only"})

        assert len(body["data"]) == 1
        assert _decode_base64_to_floats(body["data"][0]["embedding"]) == pytest.approx(
            [0.1, 0.2, 0.3, 0.4]
        )

    def test_batched_declared_but_rank_1_output_raises(self) -> None:
        # max_batch_size>0 but output arrived rank-1: cannot identify batch
        # axis. Raise rather than silently misinterpret as a single row.
        bad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        response = types.SimpleNamespace(outputs={"embedding": _mock_fp32_tensor(bad)})
        _, handler = _make_handler(responses=[response], max_batch_size=8)

        with pytest.raises(RuntimeError, match="expected a leading batch axis"):
            _run(handler, {"input": "x"})

    def test_row_count_mismatch_raises(self) -> None:
        # Two prompts in, one embedding row out: must raise, not silently
        # truncate to one entry with the wrong index.
        one_row = np.array([[0.1, 0.2]], dtype=np.float32)
        response = types.SimpleNamespace(
            outputs={"embedding": _mock_fp32_tensor(one_row)}
        )
        _, handler = _make_handler(responses=[response])

        with pytest.raises(RuntimeError, match="expected one row per input"):
            _run(handler, {"input": ["a", "b"]})


class TestValidation:
    def test_missing_input_field(self) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match="missing required 'input'"):
            _run(handler, {"model": "mock-embedder"})

    def test_empty_string_input(self) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match="cannot be an empty string"):
            _run(handler, {"input": ""})

    def test_empty_list_input(self) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match="cannot be an empty list"):
            _run(handler, {"input": []})

    def test_list_with_empty_string_input(self) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match="must not contain empty strings"):
            _run(handler, {"input": ["ok", ""]})

    def test_token_id_input_rejected(self) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match="does not yet accept token-ID"):
            _run(handler, {"input": [1, 2, 3]})

    def test_token_batch_input_rejected(self) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match="does not yet accept token-ID"):
            _run(handler, {"input": [[1, 2], [3, 4]]})

    def test_unbatched_rejects_multiple_prompts(self) -> None:
        _, handler = _make_handler(max_batch_size=0)
        with pytest.raises(ValueError, match="is unbatched"):
            _run(handler, {"input": ["a", "b"]})

    def test_batch_larger_than_max_batch_size_rejected(self) -> None:
        _, handler = _make_handler(max_batch_size=2)
        with pytest.raises(ValueError, match="accepts at most 2 prompts"):
            _run(handler, {"input": ["a", "b", "c"]})

    @pytest.mark.parametrize(
        "field,value",
        [
            ("dimensions", 128),
            ("add_special_tokens", True),
            ("add_special_tokens", False),
            ("truncate_prompt_tokens", 256),
        ],
    )
    def test_unsupported_control_rejected(self, field: str, value: Any) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match=f"does not honor '{field}'"):
            _run(handler, {"input": "x", field: value})

    def test_unsupported_control_null_is_ignored(self) -> None:
        # None means "unset" over the wire; must pass through.
        embedding = np.array([[0.5]], dtype=np.float32)
        response = types.SimpleNamespace(
            outputs={"embedding": _mock_fp32_tensor(embedding)}
        )
        _, handler = _make_handler(responses=[response])

        [body] = _run(
            handler,
            {
                "input": "x",
                "dimensions": None,
                "add_special_tokens": None,
                "truncate_prompt_tokens": None,
            },
        )
        assert len(body["data"]) == 1

    @pytest.mark.parametrize("value", ["float", "base64", None])
    def test_valid_encoding_format_accepted(self, value: Any) -> None:
        embedding = np.array([[0.5, 0.5]], dtype=np.float32)
        response = types.SimpleNamespace(
            outputs={"embedding": _mock_fp32_tensor(embedding)}
        )
        _, handler = _make_handler(responses=[response])

        request = {"input": "x"}
        if value is not None:
            request["encoding_format"] = value

        [body] = _run(handler, request)
        assert len(body["data"]) == 1

    def test_invalid_encoding_format_rejected(self) -> None:
        _, handler = _make_handler()
        with pytest.raises(ValueError, match="only supports encoding_format"):
            _run(handler, {"input": "x", "encoding_format": "hex"})


class TestHealthProbe:
    def test_probe_short_circuits_before_inference(self) -> None:
        from dynamo.health_check import HEALTH_CHECK_KEY

        _, handler = _make_handler()
        [body] = _run(handler, {HEALTH_CHECK_KEY: True, "model": "mock-embedder"})

        assert body["object"] == "list"
        assert body["model"] == "mock-embedder"
        assert body["data"] == []
        assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}
