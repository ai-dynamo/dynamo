# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm.engine.arg_utils",
    reason="a full vLLM installation is required by the ensemble worker",
)

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


class _FakeAdapter:
    def __init__(self) -> None:
        self.artifacts: Any = None

    def prepare_prompt(self, token_ids: list[int], artifacts: list[Any]) -> dict:
        self.artifacts = artifacts
        return {"prompt_token_ids": token_ids + [99]}


class _FakeDecoder:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.abort_ids: list[str] = []
        self.shutdown_calls = 0

    async def generate_final(
        self,
        request: dict[str, Any],
        prompt: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        self.started.set()
        return {
            "token_ids": [4, 2],
            "index": 0,
            "finish_reason": "stop",
            "completion_usage": {
                "prompt_tokens": len(prompt["prompt_token_ids"]),
                "completion_tokens": 2,
                "total_tokens": len(prompt["prompt_token_ids"]) + 2,
            },
        }

    async def abort(self, request_id: str) -> None:
        self.abort_ids.append(request_id)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _RecordingClassifier:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.artifacts: Any = None

    async def classify(self, artifacts) -> str:
        self.artifacts = artifacts
        self.started.set()
        return "category-a"


def _engine(classifier=None):
    engine = object.__new__(UserEnsembleEngine)
    engine.model_name = "test-model"
    engine.served_model_name = "test-model"
    engine._classifier = classifier or _RecordingClassifier()
    engine._encoder = None
    engine._adapter = None
    engine._decoder = None
    return engine


def _context(request_id: str = "request-1") -> MagicMock:
    context = MagicMock()
    context.id.return_value = request_id
    return context


def _request(image_count: int = 1) -> dict:
    return {
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


async def _collect(engine: UserEnsembleEngine, request: dict) -> list[dict]:
    return [chunk async for chunk in engine.generate(request, _context())]


async def test_encoder_artifacts_fan_out_once_and_join_on_terminal():
    artifact = object()
    artifacts = [artifact]
    classifier = _RecordingClassifier()
    encoder = _FakeEncoder(artifacts)
    adapter = _FakeAdapter()
    decoder = _FakeDecoder()
    engine = _engine(classifier)
    engine._encoder = encoder
    engine._adapter = adapter
    engine._decoder = decoder

    chunks = await _collect(engine, _request())

    assert encoder.calls == [["data:image/png;base64,image-0"]]
    assert classifier.artifacts is artifacts
    assert adapter.artifacts is artifacts
    assert chunks == [
        {
            "token_ids": [4, 2],
            "index": 0,
            "finish_reason": "stop",
            "completion_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
            "engine_data": {"ensemble": {"classifier": "category-a"}},
        }
    ]


async def test_classifier_and_decoder_run_concurrently():
    classifier_started = asyncio.Event()
    decoder_started = asyncio.Event()

    class BarrierClassifier:
        async def classify(self, artifacts) -> str:
            classifier_started.set()
            await asyncio.wait_for(decoder_started.wait(), timeout=1)
            return "joined"

    class BarrierDecoder(_FakeDecoder):
        async def generate_final(
            self,
            request: dict[str, Any],
            prompt: dict[str, Any],
            request_id: str,
        ) -> dict[str, Any]:
            decoder_started.set()
            await asyncio.wait_for(classifier_started.wait(), timeout=1)
            return {
                "token_ids": [42],
                "index": 0,
                "finish_reason": "stop",
                "completion_usage": {
                    "prompt_tokens": len(prompt["prompt_token_ids"]),
                    "completion_tokens": 1,
                    "total_tokens": len(prompt["prompt_token_ids"]) + 1,
                },
            }

    engine = _engine(BarrierClassifier())
    engine._encoder = _FakeEncoder([object()])
    engine._adapter = _FakeAdapter()
    engine._decoder = BarrierDecoder()

    [terminal] = await _collect(engine, _request())

    assert terminal["engine_data"]["ensemble"]["classifier"] == "joined"


async def test_classifier_failure_cancels_and_aborts_decoder():
    decoder_started = asyncio.Event()

    class FailingClassifier:
        async def classify(self, artifacts) -> str:
            await asyncio.wait_for(decoder_started.wait(), timeout=1)
            raise RuntimeError("classifier failed")

    class BlockingDecoder(_FakeDecoder):
        async def generate_final(
            self,
            request: dict[str, Any],
            prompt: dict[str, Any],
            request_id: str,
        ) -> dict[str, Any]:
            try:
                decoder_started.set()
                await asyncio.Future()
            finally:
                await self.abort(request_id)

    decoder = BlockingDecoder()
    engine = _engine(FailingClassifier())
    engine._encoder = _FakeEncoder([object()])
    engine._adapter = _FakeAdapter()
    engine._decoder = decoder

    with pytest.raises(Exception) as exc_info:
        await _collect(engine, _request())

    nested_errors = getattr(exc_info.value, "exceptions", ())
    assert any("classifier failed" in str(error) for error in nested_errors)
    assert decoder.abort_ids == ["request-1"]


@pytest.mark.parametrize("image_count", [0, 2])
async def test_rejects_non_single_image_requests(image_count: int):
    engine = _engine()
    engine._encoder = _FakeEncoder([object()])
    engine._adapter = _FakeAdapter()
    engine._decoder = _FakeDecoder()

    with pytest.raises(InvalidArgument, match="exactly one image"):
        await _collect(engine, _request(image_count))


async def test_abort_and_cleanup_delegate_and_cleanup_is_idempotent():
    encoder = _FakeEncoder([object()])
    decoder = _FakeDecoder()
    engine = _engine()
    engine._encoder = encoder
    engine._adapter = _FakeAdapter()
    engine._decoder = decoder

    await engine.abort(_context("cancel-me"))
    await engine.cleanup()
    await engine.cleanup()

    assert decoder.abort_ids == ["cancel-me"]
    assert decoder.shutdown_calls == 1
    assert encoder.shutdown_calls == 1


@pytest.mark.parametrize(
    ("configured", "fallback", "expected"),
    [
        (None, "public/model-id", "public/model-id"),
        ([], "public/model-id", "public/model-id"),
        (["served", "alias"], "public/model-id", "served"),
        ("served", "public/model-id", "served"),
    ],
)
def test_served_model_name_preserves_public_cli_identity(
    configured, fallback: str, expected: str
):
    assert _served_model_name(configured, fallback) == expected
