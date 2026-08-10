# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm.engine.arg_utils",
    reason="a full vLLM installation is required by the workflow example",
)

from dynamo.llm.exceptions import InvalidArgument  # noqa: E402
from dynamo.vllm.embedded_decoder import EmbeddedVllmDecoder  # noqa: E402
from dynamo.workflow import StageContext, compile_workflow  # noqa: E402
from examples.custom_backend.user_ensemble.worker import (  # noqa: E402
    DummyClassifier,
    EncoderStage,
    UserEnsembleEngine,
    _served_model_name,
    define_workflow,
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
    contract = EmbeddedVllmDecoder.contract

    def __init__(self) -> None:
        self.prompt: Any = None
        self.abort_ids: list[str] = []
        self.shutdown_calls = 0

    async def run(self, inputs, context: StageContext):
        self.prompt = inputs["prompt"]
        return {
            "chunk": {
                "token_ids": [4, 2],
                "index": 0,
                "finish_reason": "stop",
                "completion_usage": {
                    "prompt_tokens": len(self.prompt["prompt_token_ids"]),
                    "completion_tokens": 2,
                    "total_tokens": len(self.prompt["prompt_token_ids"]) + 2,
                },
            }
        }

    async def abort(self, request_id: str) -> None:
        self.abort_ids.append(request_id)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _RecordingClassifier:
    contract = DummyClassifier.contract

    def __init__(self) -> None:
        self.artifacts: Any = None

    async def run(self, inputs, context: StageContext):
        self.artifacts = inputs["artifacts"]
        return {"scores": {"category-a": 0.9, "category-b": 0.1}}


def _engine(classifier=None):
    engine = object.__new__(UserEnsembleEngine)
    engine.model_name = "test-model"
    engine.served_model_name = "test-model"
    engine._classifier = classifier or _RecordingClassifier()
    engine._encoder = None
    engine._decoder = None
    engine._plan = None
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


def _bind(engine, encoder, adapter, decoder) -> None:
    engine._encoder = encoder
    engine._decoder = decoder
    engine._plan = compile_workflow(
        define_workflow(),
        encoder=EncoderStage(encoder, adapter),
        classifier=engine._classifier,
        generator=decoder,
    )


async def _collect(engine: UserEnsembleEngine, request: dict) -> list[dict]:
    return [chunk async for chunk in engine.generate(request, _context())]


def test_workflow_is_the_readable_pipeline_and_uses_declared_ports():
    workflow = define_workflow().build()

    assert [stage.id for stage in workflow.stages] == [
        "encoder",
        "classifier",
        "generator",
    ]
    assert workflow.outputs["scores"].to_dict() == {
        "stage": "classifier",
        "output": "scores",
    }
    assert workflow.outputs["chunk"].to_dict() == {
        "stage": "generator",
        "output": "chunk",
    }


async def test_start_builds_prompt_adapter_outside_decoder() -> None:
    engine_args = MagicMock()
    backend = MagicMock()
    backend_type = MagicMock(return_value=backend)
    decoder = MagicMock()
    model_config = MagicMock()
    model_config.max_model_len = 4096
    decoder.model_config = model_config
    encoder = MagicMock()
    adapter = MagicMock()
    plan = MagicMock()
    engine = UserEnsembleEngine(
        model_name="test-model",
        served_model_name="served-model",
        engine_args=engine_args,
        encoder_backend_type=backend_type,
    )

    with (
        patch(
            "examples.custom_backend.user_ensemble.worker.EmbeddedVllmDecoder.from_engine_args",
            return_value=decoder,
        ),
        patch(
            "examples.custom_backend.user_ensemble.worker.create_custom_encoder_adapter",
            return_value=adapter,
        ) as create_adapter,
        patch(
            "examples.custom_backend.user_ensemble.worker.AsyncVisionEncoder",
            return_value=encoder,
        ),
        patch(
            "examples.custom_backend.user_ensemble.worker.compile_workflow",
            return_value=plan,
        ),
    ):
        config = await engine.start(worker_id=1)

    create_adapter.assert_called_once_with(backend, model_config, engine_args)
    encoder.load.assert_called_once_with("test-model")
    assert engine._decoder is decoder
    assert engine._encoder is encoder
    assert engine._plan is plan
    assert config.llm is not None
    assert config.llm.context_length == 4096
    assert config.llm.kv_cache_block_size is None


async def test_encoder_artifacts_fan_out_once_and_join_in_terminal_response():
    artifacts = [object()]
    classifier = _RecordingClassifier()
    encoder = _FakeEncoder(artifacts)
    adapter = _FakeAdapter()
    decoder = _FakeDecoder()
    engine = _engine(classifier)
    _bind(engine, encoder, adapter, decoder)

    chunks = await _collect(engine, _request())

    assert encoder.calls == [["data:image/png;base64,image-0"]]
    assert classifier.artifacts is artifacts
    assert adapter.artifacts is artifacts
    assert decoder.prompt == {"prompt_token_ids": [1, 2, 3, 99]}
    assert chunks[0]["engine_data"] == {
        "ensemble": {"classifier_scores": {"category-a": 0.9, "category-b": 0.1}}
    }


class _BranchBarrier:
    def __init__(self, count: int) -> None:
        self._remaining = count
        self.open = asyncio.Event()

    async def enter(self) -> None:
        self._remaining -= 1
        if self._remaining == 0:
            self.open.set()
        await self.open.wait()


class _FailingClassifier:
    contract = DummyClassifier.contract

    def __init__(self, barrier: _BranchBarrier) -> None:
        self._barrier = barrier

    async def run(self, inputs, context: StageContext):
        await self._barrier.enter()
        raise RuntimeError("classifier failed")


class _BlockingDecoder(_FakeDecoder):
    def __init__(self, barrier: _BranchBarrier) -> None:
        super().__init__()
        self._barrier = barrier

    async def run(self, inputs, context: StageContext):
        await self._barrier.enter()
        try:
            await asyncio.Event().wait()
        finally:
            await self.abort(context.attempt_id)


async def test_classifier_failure_cancels_and_aborts_decoder():
    barrier = _BranchBarrier(2)
    decoder = _BlockingDecoder(barrier)
    engine = _engine(_FailingClassifier(barrier))
    _bind(engine, _FakeEncoder([object()]), _FakeAdapter(), decoder)

    with pytest.raises(RuntimeError, match="classifier failed"):
        await _collect(engine, _request())

    assert decoder.abort_ids == ["request-1"]


@pytest.mark.parametrize("image_count", [0, 2])
async def test_rejects_non_single_image_requests(image_count: int):
    engine = _engine()
    _bind(engine, _FakeEncoder([object()]), _FakeAdapter(), _FakeDecoder())

    with pytest.raises(InvalidArgument, match="exactly one image"):
        await _collect(engine, _request(image_count))


async def test_abort_and_cleanup_delegate_and_cleanup_is_idempotent():
    encoder = _FakeEncoder([object()])
    decoder = _FakeDecoder()
    engine = _engine()
    _bind(engine, encoder, _FakeAdapter(), decoder)

    await engine.abort(_context("cancel-me"))
    await engine.cleanup()
    await engine.cleanup()

    assert decoder.abort_ids == ["cancel-me"]
    assert decoder.shutdown_calls == 1
    assert encoder.shutdown_calls == 1


def test_served_model_name_preserves_public_cli_identity():
    assert _served_model_name(None, "public/model-id") == "public/model-id"
    assert _served_model_name(["served", "alias"], "public/model-id") == "served"
