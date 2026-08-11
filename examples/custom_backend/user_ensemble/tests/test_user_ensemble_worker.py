# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm",
    reason="a full vLLM installation is required by the custom encoder package",
)

from dynamo.common.multimodal.embedding_transfer import TransferRequest  # noqa: E402
from dynamo.llm.exceptions import InvalidArgument  # noqa: E402
from examples.custom_backend.user_ensemble.worker import (  # noqa: E402
    UserEnsembleEngine,
    _served_model_name,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeEncoder:
    def __init__(self, artifacts: list[Any]) -> None:
        self.artifacts = artifacts
        self.calls: list[list[str]] = []
        self.shutdown_calls = 0

    async def encode(self, raws: list[str]) -> list[Any]:
        self.calls.append(raws)
        return self.artifacts

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeResponse:
    def __init__(
        self,
        data: dict[str, Any] | str | None = None,
        *,
        error: bool = False,
        comments: list[str] | None = None,
    ) -> None:
        self._data = data
        self._error = error
        self._comments = comments

    def is_error(self) -> bool:
        return self._error

    def comments(self) -> list[str] | None:
        return self._comments

    def data(self) -> dict[str, Any] | str | None:
        return self._data


class _FakeDecoderClient:
    def __init__(self, responses: list[_FakeResponse] | None = None) -> None:
        self.responses = responses or []
        self.requests: list[dict[str, Any]] = []
        self.contexts: list[Any] = []

    async def wait_for_instances(self) -> list[int]:
        return [1]

    async def generate(
        self, request: dict[str, Any], *, context: Any = None
    ) -> AsyncIterator[_FakeResponse]:
        self.requests.append(request)
        self.contexts.append(context)

        async def stream() -> AsyncIterator[_FakeResponse]:
            for response in self.responses:
                yield response

        return stream()


class _FakeRuntime:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeArtifactSender:
    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []

    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
    ):
        assert stage_embeddings is True
        self.tensors.append(embeddings)
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(None)
        return (
            TransferRequest(
                embeddings_shape=list(embeddings.shape),
                embedding_dtype_str=str(embeddings.dtype).removeprefix("torch."),
                serialized_request={"fake": len(self.tensors)},
            ),
            completion,
        )


class _RecordingClassifier:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.artifacts: Any = None

    async def classify(self, artifacts) -> str:
        self.artifacts = artifacts
        self.started.set()
        return "category-a"


def _engine(classifier=None) -> UserEnsembleEngine:
    engine = object.__new__(UserEnsembleEngine)
    engine.model_name = "test-model"
    engine.served_model_name = "test-model"
    engine._decoder_model_name = "test-model-remote-vllm"
    engine._classifier = classifier or _RecordingClassifier()
    engine._encoder = None
    engine._artifact_sender = _FakeArtifactSender()
    engine._decoder_client = None
    engine._decoder_runtime = None
    return engine


def _context(request_id: str = "request-1") -> MagicMock:
    context = MagicMock()
    context.id.return_value = request_id
    return context


def _request(image_count: int = 1) -> dict[str, Any]:
    return {
        "model": "test-model",
        "token_ids": [1, 2, 3],
        "sampling_options": {"n": 1},
        "stop_conditions": {"max_tokens": 8},
        "output_options": {},
        "multi_modal_data": {
            "image_url": [
                {"Url": f"data:image/png;base64,image-{index}"}
                for index in range(image_count)
            ]
        },
    }


async def _collect(
    engine: UserEnsembleEngine,
    request: dict[str, Any],
    context: MagicMock | None = None,
) -> list[dict[str, Any]]:
    return [chunk async for chunk in engine.generate(request, context or _context())]


async def test_local_artifacts_feed_classifier_and_remote_deltas_join_on_terminal():
    artifact = torch.ones((1, 4), dtype=torch.bfloat16)
    artifacts = [artifact]
    classifier = _RecordingClassifier()
    encoder = _FakeEncoder(artifacts)
    decoder = _FakeDecoderClient(
        [
            _FakeResponse({"token_ids": [4], "index": 0}),
            _FakeResponse(
                json.dumps(
                    {
                        "token_ids": [2],
                        "index": 0,
                        "finish_reason": "stop",
                        "completion_usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                )
            ),
        ]
    )
    engine = _engine(classifier)
    engine._encoder = encoder
    engine._decoder_client = decoder
    context = _context()
    request = _request()

    chunks = await _collect(engine, request, context)

    assert encoder.calls == [["data:image/png;base64,image-0"]]
    assert classifier.artifacts is artifacts
    assert engine._artifact_sender.tensors == artifacts
    assert decoder.contexts == [context]
    assert decoder.requests == [
        {
            "model": "test-model-remote-vllm",
            "token_ids": [1, 2, 3],
            "sampling_options": {"n": 1},
            "stop_conditions": {"max_tokens": 8},
            "output_options": {},
            "extra_args": {"dynamo_internal_final_only": True},
            "encoder_result": {
                "custom_encoder_artifacts": [
                    {
                        "kind": "tensor",
                        "transfer": {
                            "embeddings_shape": [1, 4],
                            "embedding_dtype_str": "bfloat16",
                            "serialized_request": {"fake": 1},
                        },
                    }
                ]
            },
        }
    ]
    assert chunks == [
        {
            "token_ids": [4, 2],
            "index": 0,
            "finish_reason": "stop",
            "completion_usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            "engine_data": {"ensemble": {"classifier": "category-a"}},
        }
    ]


async def test_classifier_and_remote_decoder_run_concurrently():
    classifier_started = asyncio.Event()
    decoder_started = asyncio.Event()

    class BarrierClassifier:
        async def classify(self, artifacts) -> str:
            classifier_started.set()
            await asyncio.wait_for(decoder_started.wait(), timeout=1)
            return "joined"

    class BarrierDecoder(_FakeDecoderClient):
        async def generate(
            self, request: dict[str, Any], *, context: Any = None
        ) -> AsyncIterator[_FakeResponse]:
            async def stream() -> AsyncIterator[_FakeResponse]:
                decoder_started.set()
                await asyncio.wait_for(classifier_started.wait(), timeout=1)
                yield _FakeResponse(
                    {"token_ids": [42], "index": 0, "finish_reason": "stop"}
                )

            return stream()

    engine = _engine(BarrierClassifier())
    engine._encoder = _FakeEncoder([torch.ones((1, 4))])
    engine._decoder_client = BarrierDecoder()

    [terminal] = await _collect(engine, _request())

    assert terminal["engine_data"]["ensemble"]["classifier"] == "joined"


async def test_classifier_failure_cancels_remote_decoder_context():
    decoder_started = asyncio.Event()
    decoder_cancelled = asyncio.Event()

    class FailingClassifier:
        async def classify(self, artifacts) -> str:
            await asyncio.wait_for(decoder_started.wait(), timeout=1)
            raise RuntimeError("classifier failed")

    class BlockingDecoder(_FakeDecoderClient):
        async def generate(
            self, request: dict[str, Any], *, context: Any = None
        ) -> AsyncIterator[_FakeResponse]:
            async def stream() -> AsyncIterator[_FakeResponse]:
                try:
                    decoder_started.set()
                    await asyncio.Future()
                    yield _FakeResponse()
                finally:
                    decoder_cancelled.set()

            return stream()

    context = _context()
    engine = _engine(FailingClassifier())
    engine._encoder = _FakeEncoder([torch.ones((1, 4))])
    engine._decoder_client = BlockingDecoder()

    with pytest.raises(RuntimeError, match="classifier failed"):
        await _collect(engine, _request(), context)

    assert decoder_cancelled.is_set()
    context.stop_generating.assert_called_once_with()


async def test_remote_annotated_error_stops_context():
    engine = _engine()
    engine._encoder = _FakeEncoder([torch.ones((1, 4))])
    engine._decoder_client = _FakeDecoderClient(
        [_FakeResponse(error=True, comments=["engine unavailable"])]
    )
    context = _context()

    with pytest.raises(RuntimeError, match="engine unavailable"):
        await _collect(engine, _request(), context)

    context.stop_generating.assert_called_once_with()


@pytest.mark.parametrize("image_count", [0, 2])
async def test_rejects_non_single_image_requests(image_count: int):
    engine = _engine()
    engine._encoder = _FakeEncoder([torch.ones((1, 4))])
    engine._decoder_client = _FakeDecoderClient()

    with pytest.raises(InvalidArgument, match="exactly one image"):
        await _collect(engine, _request(image_count))


@pytest.mark.parametrize(
    ("sampling_options", "output_options", "error"),
    [
        ({"n": 2}, {}, "exactly one choice"),
        ({"n": 1}, {"logprobs": 1}, "does not support logprobs"),
    ],
)
async def test_rejects_unsupported_remote_output_modes(
    sampling_options: dict[str, Any],
    output_options: dict[str, Any],
    error: str,
):
    engine = _engine()
    engine._encoder = _FakeEncoder([torch.ones((1, 4))])
    engine._decoder_client = _FakeDecoderClient()
    request = _request()
    request["sampling_options"] = sampling_options
    request["output_options"] = output_options

    with pytest.raises(InvalidArgument, match=error):
        await _collect(engine, request)


async def test_abort_and_cleanup_stop_context_and_cleanup_is_idempotent():
    encoder = _FakeEncoder([object()])
    runtime = _FakeRuntime()
    engine = _engine()
    engine._encoder = encoder
    engine._decoder_client = _FakeDecoderClient()
    engine._decoder_runtime = runtime
    context = _context("cancel-me")

    await engine.abort(context)
    await engine.cleanup()
    await engine.cleanup()

    context.stop_generating.assert_called_once_with()
    assert runtime.shutdown_calls == 1
    assert encoder.shutdown_calls == 1


@pytest.mark.parametrize(
    ("configured", "fallback", "expected"),
    [
        (None, "public/model-id", "public/model-id"),
        ("served", "public/model-id", "served"),
    ],
)
def test_served_model_name_preserves_public_cli_identity(
    configured: str | None, fallback: str, expected: str
):
    assert _served_model_name(configured, fallback) == expected
