# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated encoder, classifier, and decoder workflow."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast

from dynamo._core import Context
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
    WorkerConfig,
)
from dynamo.common.backend.run import run
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.args import Config
from dynamo.vllm.decoder_runtime import VllmDecoderRuntime
from dynamo.vllm.decoder_stage import VllmDecoderStage
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
)
from dynamo.workflow import StageRunner, WorkflowExecutor, compile_workflow
from examples.custom_backend.user_ensemble.config import prepare_user_ensemble_config
from examples.custom_backend.user_ensemble.resources import (
    build_decoder_stage,
    build_encoder_stage,
    cleanup_resources,
)
from examples.custom_backend.user_ensemble.stages import DummyClassifier
from examples.custom_backend.user_ensemble.workflow import (
    adapt_workflow_result,
    define_workflow,
)


class UserEnsembleEngine(LLMEngine):
    """Expose the authored local workflow through one Dynamo endpoint."""

    def __init__(
        self,
        *,
        config: Config,
        encoder_backend_type: type[VisionEncoderBackend[Any, Any, Any]],
        classifier: StageRunner | None = None,
    ) -> None:
        self._config = config
        self.model_name = config.model
        self.served_model_name = config.served_model_name or config.model
        self._encoder_backend_type = encoder_backend_type
        self._classifier = classifier or DummyClassifier()

        self._decoder_runtime: VllmDecoderRuntime | None = None
        self._decoder_stage: VllmDecoderStage | None = None
        self._encoder: AsyncVisionEncoder[Any, Any, Any] | None = None
        self._prometheus_temp_dir: Any | None = None
        self._executor: WorkflowExecutor | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[UserEnsembleEngine, WorkerConfig]:
        config, backend_type = prepare_user_ensemble_config(argv)

        engine = cls(
            config=config,
            encoder_backend_type=backend_type,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            enable_kv_routing=False,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        decoder_stage, decoder_runtime, prometheus_temp_dir = build_decoder_stage(
            self._config
        )
        encoder: AsyncVisionEncoder[Any, Any, Any] | None = None
        try:
            encoder_stage, encoder = build_encoder_stage(
                self._config,
                self._encoder_backend_type,
                name="workflow-vision-encoder",
                model_config=decoder_runtime.model_config,
            )
            runners = {
                "encoder": encoder_stage,
                "classifier": self._classifier,
                "generator": decoder_stage,
            }
            plan = compile_workflow(define_workflow())
            executor = WorkflowExecutor(plan, runners)
        except BaseException:
            cleanup_resources(encoder, decoder_runtime, prometheus_temp_dir)
            raise

        self._decoder_runtime = decoder_runtime
        self._decoder_stage = decoder_stage
        self._encoder = encoder
        self._prometheus_temp_dir = prometheus_temp_dir
        self._executor = executor
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            llm=LlmRegistration(
                context_length=decoder_runtime.model_config.max_model_len,
            ),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        executor = self._executor
        if executor is None:
            raise RuntimeError("UserEnsembleEngine.generate() called before start()")
        request_id = context.id()
        result = await executor.run(
            {"image_url": self._single_image_url(request), "request": request},
            attempt_id=request_id,
        )

        yield cast(GenerateChunk, adapt_workflow_result(result))

    async def abort(self, context: Context) -> None:
        decoder_stage = self._decoder_stage
        if decoder_stage is not None:
            await decoder_stage.abort_attempt(context.id())

    async def cleanup(self) -> None:
        encoder = self._encoder
        decoder_runtime = self._decoder_runtime
        prometheus_temp_dir = self._prometheus_temp_dir
        self._executor = None
        self._encoder = None
        self._decoder_stage = None
        self._decoder_runtime = None
        self._prometheus_temp_dir = None

        cleanup_resources(encoder, decoder_runtime, prometheus_temp_dir)

    @staticmethod
    def _single_image_url(request: GenerateRequest) -> str:
        multimodal = request.get("multi_modal_data") or {}
        unsupported = sorted(
            key for key, value in multimodal.items() if key != IMAGE_URL_KEY and value
        )
        if unsupported:
            raise InvalidArgument(
                "UserEnsembleEngine supports image inputs only; got "
                f"unsupported multimodal data: {unsupported}"
            )

        image_items = multimodal.get(IMAGE_URL_KEY) or []
        if len(image_items) != 1:
            raise InvalidArgument(
                "UserEnsembleEngine requires exactly one image per request; "
                f"got {len(image_items)}"
            )
        image_item = image_items[0]
        if not isinstance(image_item, dict):
            raise InvalidArgument("image_url item must be an object with a 'Url' field")
        image_url = image_item.get(URL_VARIANT_KEY)
        if not isinstance(image_url, str) or not image_url:
            raise InvalidArgument(
                "image_url item must contain a non-empty 'Url' string"
            )
        return image_url


def main() -> None:
    run(UserEnsembleEngine)


if __name__ == "__main__":
    main()
