# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any, cast
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

from dynamo.common.backend import GenerateChunk, GenerateRequest  # noqa: E402
from dynamo.llm.exceptions import InvalidArgument  # noqa: E402
from dynamo.vllm.decoder_stage import VllmDecoderStage  # noqa: E402
from dynamo.workflow import StageContext, compile_workflow  # noqa: E402
from examples.custom_backend.user_ensemble.stages import (  # noqa: E402
    DummyClassifier,
    EncoderStage,
)
from examples.custom_backend.user_ensemble.worker import (  # noqa: E402
    UserEnsembleEngine,
)
from examples.custom_backend.user_ensemble.workflow import define_workflow  # noqa: E402

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
    contract = VllmDecoderStage.contract

    def __init__(self) -> None:
        self.prompt: Any = None
        self.abort_ids: list[str] = []

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

    async def abort_attempt(self, request_id: str) -> None:
        self.abort_ids.append(request_id)


class _FakeRuntime:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeTempDir:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class _RecordingClassifier:
    contract = DummyClassifier.contract

    def __init__(self) -> None:
        self.artifacts: Any = None

    async def run(self, inputs, context: StageContext):
        self.artifacts = inputs["artifacts"]
        return {"scores": {"category-a": 0.9, "category-b": 0.1}}


def _engine(classifier=None):
    engine = object.__new__(UserEnsembleEngine)
    engine._config = MagicMock()
    engine.model_name = "test-model"
    engine.served_model_name = "test-model"
    engine._engine_args = MagicMock()
    engine._classifier = classifier or _RecordingClassifier()
    engine._encoder = None
    engine._decoder_runtime = None
    engine._decoder_stage = None
    engine._prometheus_temp_dir = None
    engine._plan = None
    return engine


def _context(request_id: str = "request-1") -> MagicMock:
    context = MagicMock()
    context.id.return_value = request_id
    return context


def _request(image_count: int = 1) -> GenerateRequest:
    return cast(
        GenerateRequest,
        {
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
        },
    )


def _bind(engine, encoder, adapter, decoder) -> None:
    engine._encoder = encoder
    engine._decoder_stage = decoder
    engine._plan = compile_workflow(
        define_workflow(),
        encoder=EncoderStage(encoder, adapter),
        classifier=engine._classifier,
        generator=decoder,
    )


async def _collect(
    engine: UserEnsembleEngine, request: GenerateRequest
) -> list[GenerateChunk]:
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
    engine_client = MagicMock()
    model_config = MagicMock()
    model_config.max_model_len = 4096
    vllm_config = MagicMock()
    vllm_config.model_config = model_config
    vllm_config.shutdown_timeout = 10
    prometheus_temp_dir = MagicMock()
    encoder = MagicMock()
    adapter = MagicMock()
    plan = MagicMock()
    runtime_config = MagicMock()
    runtime_config.model = "test-model"
    runtime_config.served_model_name = "served-model"
    runtime_config.engine_args = engine_args
    engine = UserEnsembleEngine(
        config=runtime_config,
        encoder_backend_type=backend_type,
    )

    with (
        patch(
            "examples.custom_backend.user_ensemble.worker.setup_vllm_engine",
            return_value=(
                engine_client,
                vllm_config,
                {"temperature": 0.0},
                prometheus_temp_dir,
                MagicMock(),
            ),
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
    assert engine._decoder_runtime is not None
    assert engine._decoder_runtime.engine is engine_client
    assert isinstance(engine._decoder_stage, VllmDecoderStage)
    assert engine._encoder is encoder
    assert engine._prometheus_temp_dir is prometheus_temp_dir
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
            await self.abort_attempt(context.attempt_id)


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
    decoder_runtime = _FakeRuntime()
    prometheus_temp_dir = _FakeTempDir()
    engine = _engine()
    _bind(engine, encoder, _FakeAdapter(), decoder)
    engine._decoder_runtime = decoder_runtime
    engine._prometheus_temp_dir = prometheus_temp_dir

    await engine.abort(_context("cancel-me"))
    await engine.cleanup()
    await engine.cleanup()

    assert decoder.abort_ids == ["cancel-me"]
    assert decoder_runtime.shutdown_calls == 1
    assert prometheus_temp_dir.cleanup_calls == 1
    assert encoder.shutdown_calls == 1


async def test_from_args_reuses_shared_vllm_config() -> None:
    runtime_config = MagicMock()
    runtime_config.model = "public/model-id"
    runtime_config.served_model_name = None
    runtime_config.custom_encoder_class = "encoder.Backend"
    runtime_config.engine_args = MagicMock()
    runtime_config.engine_args.served_model_name = None
    backend_type = MagicMock()
    worker_config = MagicMock()

    with (
        patch(
            "examples.custom_backend.user_ensemble.worker.parse_args",
            return_value=runtime_config,
        ) as shared_parse_args,
        patch(
            "examples.custom_backend.user_ensemble.worker.configure_rl_logprobs_mode"
        ) as configure_rl,
        patch(
            "examples.custom_backend.user_ensemble.worker._load_encoder_backend",
            return_value=backend_type,
        ),
        patch(
            "examples.custom_backend.user_ensemble.worker.WorkerConfig.from_runtime_config",
            return_value=worker_config,
        ) as build_worker_config,
    ):
        engine, returned_worker_config = await UserEnsembleEngine.from_args(["--model"])

    shared_parse_args.assert_called_once_with(["--model"])
    configure_rl.assert_called_once_with(runtime_config)
    assert runtime_config.served_model_name == "public/model-id"
    assert runtime_config.engine_args.served_model_name == "public/model-id"
    assert runtime_config.engine_args.enable_prompt_embeds is True
    build_worker_config.assert_called_once_with(
        runtime_config,
        model_name="public/model-id",
        served_model_name="public/model-id",
        enable_kv_routing=False,
    )
    assert engine._config is runtime_config
    assert returned_worker_config is worker_config
