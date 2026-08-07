# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import dynamo.vllm.handlers as base_handlers
import dynamo.vllm.pooling_handlers as mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


def _make_handler(
    id2label: dict[int, str] | None = None,
) -> mod.ClassifyWorkerHandler:
    model_config = MagicMock()
    model_config.hf_config = SimpleNamespace(id2label=id2label or {})
    model_config.get_pooling_task.return_value = "classify"
    with patch.object(base_handlers, "VllmEngineMonitor"):
        handler = mod.ClassifyWorkerHandler(
            runtime=MagicMock(),
            engine=MagicMock(),
            config=MagicMock(served_model_name="test-model"),
            model_config=model_config,
            shutdown_event=None,
        )
    handler.engine_client = MagicMock()
    handler.engine_client.abort = AsyncMock()
    handler.engine_client.get_supported_tasks = AsyncMock(
        return_value=("classify", "token_classify")
    )
    return handler


def _make_context() -> MagicMock:
    context = MagicMock()
    context.id.return_value = "engine-request"
    context.async_killed_or_stopped.side_effect = (
        lambda: asyncio.get_running_loop().create_future()
    )
    return context


def _pooling_output(data: Any, prompt_token_ids: list[int]) -> MagicMock:
    output = MagicMock()
    output.outputs.data = torch.tensor(data)
    output.prompt_token_ids = prompt_token_ids
    return output


class TestInputNormalization:
    @pytest.mark.parametrize(
        ("input_field", "expected"),
        [
            ("hello", ["hello"]),
            (["a", "b"], ["a", "b"]),
            ([1, 2, 3], [[1, 2, 3]]),
            ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ],
    )
    def test_supported_input_shapes(self, input_field, expected):
        assert mod._classify_pooling_input(input_field) == expected

    @pytest.mark.parametrize(
        "input_field",
        [
            ["hello", 42],
            [1, "two"],
            [[1, 2], "three"],
            [[1, 2], [3.5, 4]],
            [True, False],
        ],
    )
    def test_mixed_or_non_token_inputs_are_rejected(self, input_field):
        with pytest.raises(TypeError):
            mod._classify_pooling_input(input_field)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            mod._classify_pooling_input([])


class TestClassifyWorkerHandler:
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_classify_response_uses_model_labels(self):
        handler = _make_handler({0: "contradiction", 1: "entailment", 2: "neutral"})
        captured: dict = {}

        async def fake_encode(prompt, pooling_params, request_id):
            captured["task"] = pooling_params.task
            yield _pooling_output([0.1, 0.7, 0.2], [1, 2, 3, 4])

        handler.engine_client.encode = fake_encode
        [response] = [
            item
            async for item in handler.generate(
                {
                    "input": "premise entails hypothesis",
                    "model": "test-model",
                    "request_id": "client-request",
                },
                _make_context(),
            )
        ]

        assert captured["task"] == "classify"
        assert response["id"] == "classify-client-request"
        assert response["object"] == "list"
        assert response["data"][0] == {
            "index": 0,
            "label": "entailment",
            "probs": pytest.approx([0.1, 0.7, 0.2]),
            "num_classes": 3,
        }
        assert response["usage"] == {
            "prompt_tokens": 4,
            "total_tokens": 4,
            "completion_tokens": 0,
        }

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_classify_forwards_activation_and_text_tokenization(self):
        handler = _make_handler()
        captured: dict = {}

        async def fake_encode(prompt, pooling_params, request_id, **kwargs):
            captured["use_activation"] = pooling_params.use_activation
            captured["tokenization_kwargs"] = kwargs["tokenization_kwargs"]
            yield _pooling_output([0.4, 0.6], [1, 2])

        handler.engine_client.encode = fake_encode
        [_] = [
            item
            async for item in handler.generate(
                {
                    "input": "text",
                    "model": "test-model",
                    "use_activation": False,
                    "truncate_prompt_tokens": 64,
                    "truncation_side": "right",
                    "add_special_tokens": False,
                },
                _make_context(),
            )
        ]

        assert captured == {
            "use_activation": False,
            "tokenization_kwargs": {
                "truncate_prompt_tokens": 64,
                "truncation_side": "right",
                "add_special_tokens": False,
            },
        }

    @pytest.mark.parametrize(
        ("input_field", "is_pooling", "expected_token_ids"),
        [
            ([101, 102, 103, 104], False, [[103, 104]]),
            (
                [[101, 102, 103, 104], [201, 202, 203, 204]],
                True,
                [[103, 104], [203, 204]],
            ),
        ],
    )
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_pretokenized_input_uses_vllm_truncation(
        self, input_field, is_pooling, expected_token_ids
    ):
        from vllm.renderers import TokenizeParams

        handler = _make_handler()
        handler.engine_client.renderer.default_cmpl_tok_params = TokenizeParams(
            max_total_tokens=8
        )
        handler.engine_client.renderer.tokenizer = SimpleNamespace(
            truncation_side="left"
        )
        seen_prompts: list[dict] = []
        seen_kwargs: list[dict] = []

        async def fake_encode(prompt, pooling_params, request_id, **kwargs):
            seen_prompts.append(prompt)
            seen_kwargs.append(kwargs)
            yield _pooling_output([0.5, 0.5], prompt["prompt_token_ids"])

        handler.engine_client.encode = fake_encode
        request = {
            "input": input_field,
            "model": "test-model",
            "truncate_prompt_tokens": 2,
        }
        if is_pooling:
            request["encoding_format"] = "float"

        [response] = [item async for item in handler.generate(request, _make_context())]

        assert [prompt["prompt_token_ids"] for prompt in seen_prompts] == (
            expected_token_ids
        )
        assert all("tokenization_kwargs" not in kwargs for kwargs in seen_kwargs)
        assert response["usage"]["prompt_tokens"] == 2 * len(expected_token_ids)

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_explicit_right_truncation_keeps_first_tokens(self):
        from vllm.renderers import TokenizeParams

        handler = _make_handler()
        handler.engine_client.renderer.default_cmpl_tok_params = TokenizeParams(
            max_total_tokens=8
        )
        handler.engine_client.renderer.tokenizer = SimpleNamespace(
            truncation_side="left"
        )
        seen_prompt: dict = {}

        async def fake_encode(prompt, pooling_params, request_id):
            seen_prompt.update(prompt)
            yield _pooling_output([0.5, 0.5], prompt["prompt_token_ids"])

        handler.engine_client.encode = fake_encode
        [_] = [
            item
            async for item in handler.generate(
                {
                    "input": [101, 102, 103, 104],
                    "model": "test-model",
                    "truncate_prompt_tokens": 2,
                    "truncation_side": "right",
                },
                _make_context(),
            )
        ]

        assert seen_prompt["prompt_token_ids"] == [101, 102]


class TestPoolingWorkerHandler:
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_default_task_and_output_shape(self):
        handler = _make_handler()
        captured: dict = {}

        async def fake_encode(prompt, pooling_params, request_id):
            captured["task"] = pooling_params.task
            yield _pooling_output([[1.0, 2.0], [3.0, 4.0]], [1, 2])

        handler.engine_client.encode = fake_encode
        [response] = [
            item
            async for item in handler.generate(
                {
                    "input": "some text",
                    "model": "test-model",
                    "encoding_format": "float",
                },
                _make_context(),
            )
        ]

        assert captured["task"] == "classify"
        handler.model_config.get_pooling_task.assert_called_once_with(
            ("classify", "token_classify")
        )
        assert response["id"] == "pool-engine-request"
        assert response["data"][0] == {
            "index": 0,
            "object": "pooling",
            "data": [[1.0, 2.0], [3.0, 4.0]],
        }
        assert response["usage"] == {
            "prompt_tokens": 2,
            "total_tokens": 2,
            "completion_tokens": 0,
        }

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_requested_token_task_and_request_controls_are_forwarded(self):
        handler = _make_handler()
        captured: dict = {}

        async def fake_encode(prompt, pooling_params, request_id, **kwargs):
            captured.update(
                task=pooling_params.task,
                use_activation=pooling_params.use_activation,
                prompt=prompt,
                request_id=request_id,
                priority=kwargs["priority"],
            )
            yield _pooling_output([[0.1], [0.2]], [1, 2])

        handler.engine_client.encode = fake_encode
        [response] = [
            item
            async for item in handler.generate(
                {
                    "input": "text",
                    "model": "test-model",
                    "encoding_format": "float",
                    "task": "token_classify",
                    "use_activation": False,
                    "request_id": "client-request",
                    "priority": -3,
                    "cache_salt": "secret",
                    "mm_processor_kwargs": {"do_resize": False},
                },
                _make_context(),
            )
        ]

        assert captured["task"] == "token_classify"
        assert captured["use_activation"] is False
        assert captured["request_id"] == "engine-request-0"
        assert captured["priority"] == -3
        assert captured["prompt"]["prompt"] == "text"
        assert captured["prompt"]["cache_salt"] == "secret"
        assert captured["prompt"]["mm_processor_kwargs"] == {"do_resize": False}
        assert response["id"] == "pool-client-request"

    @pytest.mark.parametrize("encoding_format", ["base64", "bytes", "bytes_only"])
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_encoded_formats_use_vllm_tensor_encoding(self, encoding_format):
        handler = _make_handler()

        async def fake_encode(prompt, pooling_params, request_id):
            yield _pooling_output([[1.0, 2.0], [3.0, 4.0]], [1, 2])

        handler.engine_client.encode = fake_encode
        [response] = [
            item
            async for item in handler.generate(
                {
                    "input": "text",
                    "model": "test-model",
                    "encoding_format": encoding_format,
                    "embed_dtype": "float16",
                    "endianness": "big",
                },
                _make_context(),
            )
        ]

        expected = base64.b64encode(bytes.fromhex("3c00400042004400")).decode("ascii")
        item = response["data"][0]
        assert item["data"] == expected
        if encoding_format == "bytes":
            assert item["shape"] == [2, 2]
        else:
            assert "shape" not in item

    @pytest.mark.parametrize(
        ("request_field", "value", "message"),
        [
            ("dimensions", 128, "dimensions is currently not supported"),
            ("encoding_format", "hex", "Invalid 'encoding_format'"),
            ("embed_dtype", "int8", "Invalid 'embed_dtype'"),
            ("endianness", "middle", "Invalid 'endianness'"),
        ],
    )
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_unsupported_response_controls_are_rejected(
        self, request_field, value, message
    ):
        handler = _make_handler()
        request = {
            "input": "text",
            "model": "test-model",
            "encoding_format": "float",
            request_field: value,
        }

        with pytest.raises(ValueError, match=message):
            async for _ in handler.generate(request, _make_context()):
                pass


class TestPoolingBatchExecution:
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_partial_failure_cancels_in_flight_encodes(self):
        handler = _make_handler()
        cancelled: set[int] = set()
        started = {idx: asyncio.Event() for idx in range(4)}

        async def fake_encode(prompt, pooling_params, request_id):
            idx = int(request_id.rsplit("-", 1)[-1])
            started[idx].set()
            if idx == 1:
                raise RuntimeError("boom")
            try:
                await asyncio.sleep(60)
                yield MagicMock()
            except asyncio.CancelledError:
                cancelled.add(idx)
                raise

        handler.engine_client.encode = fake_encode
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in handler.generate(
                {"input": ["a", "b", "c", "d"], "model": "test-model"},
                _make_context(),
            ):
                pass

        assert all(event.is_set() for event in started.values())
        assert cancelled == {0, 2, 3}

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_fanout_is_bounded_by_scheduler_capacity(self):
        handler = _make_handler()
        handler.engine_client.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=2)
        )
        active = 0
        max_active = 0
        first_pair_started = asyncio.Event()

        async def fake_encode(prompt, pooling_params, request_id):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                first_pair_started.set()
            try:
                await first_pair_started.wait()
                yield _pooling_output([0.1, 0.2], [1])
            finally:
                active -= 1

        handler.engine_client.encode = fake_encode
        [response] = [
            item
            async for item in handler.generate(
                {
                    "input": [str(idx) for idx in range(8)],
                    "model": "test-model",
                },
                _make_context(),
            )
        ]

        assert len(response["data"]) == 8
        assert max_active == 2


class TestClassificationOutputShape:
    """`/classify` reports one probability vector of shape [num_classes].

    Flattening a token-level result instead would report num_classes as
    rows*classes and pick an argmax outside the label range, so the client
    would get a 200 carrying a null label and a nonsense class count. Native
    vLLM rejects classification output with ndim != 1.
    """

    def test_accepts_flat_vector(self):
        assert mod._classification_output_to_list(
            torch.tensor([0.1, 0.7, 0.2])
        ) == pytest.approx([0.1, 0.7, 0.2])

    def test_unwraps_singleton_batch_dim(self):
        """vLLM's pooling pipeline may return shape [1, num_classes]."""
        assert mod._classification_output_to_list(
            torch.tensor([[0.1, 0.7, 0.2]])
        ) == pytest.approx([0.1, 0.7, 0.2])
        assert mod._classification_output_to_list([[0.1, 0.7, 0.2]]) == pytest.approx(
            [0.1, 0.7, 0.2]
        )

    def test_rejects_token_level_tensor(self):
        token_level = torch.tensor(
            [[0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.05, 0.9, 0.05]]
        )
        with pytest.raises(ValueError, match=r"one probability vector"):
            mod._classification_output_to_list(token_level)

    def test_rejects_token_level_list(self):
        with pytest.raises(ValueError, match=r"4 rows"):
            mod._classification_output_to_list(
                [[0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.05, 0.9, 0.05]]
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_classify_surfaces_token_level_output_as_error(self):
        """End-to-end through the handler: a pooler emitting token-level output
        must fail visibly rather than return a mislabelled 200."""
        handler = _make_handler({0: "contradiction", 1: "entailment", 2: "neutral"})

        async def fake_encode(prompt, pooling_params, request_id):
            yield _pooling_output([[0.1, 0.2, 0.7], [0.1, 0.8, 0.1]], [1, 2])

        handler.engine_client.encode = fake_encode
        with pytest.raises(ValueError, match=r"one probability vector"):
            [
                item
                async for item in handler.generate(
                    {"input": "premise", "model": "test-model"},
                    _make_context(),
                )
            ]


class TestDefaultTokenizeParams:
    """Upstream vLLM runs every pooling/classify input through
    apply_post_tokenization, which is also where default truncation to
    max_model_len happens. Returning None with no caller overrides would skip
    that for token-ID inputs.
    """

    def test_defaults_used_when_no_overrides(self):
        defaults = object()
        engine_client = SimpleNamespace(
            renderer=SimpleNamespace(default_cmpl_tok_params=defaults)
        )
        assert mod._build_pooling_tokenize_params(engine_client, None) is defaults

    def test_overrides_layer_onto_defaults(self):
        sentinel = object()
        default_params = MagicMock()
        default_params.with_kwargs.return_value = sentinel
        engine_client = SimpleNamespace(
            renderer=SimpleNamespace(default_cmpl_tok_params=default_params)
        )

        result = mod._build_pooling_tokenize_params(
            engine_client, {"truncate_prompt_tokens": 8}
        )

        assert result is sentinel
        default_params.with_kwargs.assert_called_once_with(truncate_prompt_tokens=8)
