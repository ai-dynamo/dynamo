# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated encoder -> classifier + decoder workflow."""

from __future__ import annotations

import importlib
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
from dynamo.vllm.args import Config, configure_rl_logprobs_mode, parse_args
from dynamo.vllm.decoder_runtime import VllmDecoderRuntime
from dynamo.vllm.decoder_stage import VllmDecoderStage
from dynamo.vllm.main import setup_vllm_engine
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
)
from dynamo.workflow import (
    DeploymentSpec,
    StageRunner,
    WorkflowExecutor,
    compile_workflow,
)
from examples.custom_backend.user_ensemble.stages import DummyClassifier, EncoderStage
from examples.custom_backend.user_ensemble.workflow import (
    adapt_workflow_result,
    define_workflow,
)


def _load_encoder_backend(class_path: str) -> type[VisionEncoderBackend[Any, Any, Any]]:
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError(
            "--custom-encoder-class must be a dotted module.ClassName path; "
            f"got {class_path!r}"
        )
    module = importlib.import_module(module_name)
    backend_type = getattr(module, class_name)
    if not isinstance(backend_type, type) or not issubclass(
        backend_type, VisionEncoderBackend
    ):
        raise TypeError(f"{class_path} must name a VisionEncoderBackend subclass")
    return cast(type[VisionEncoderBackend[Any, Any, Any]], backend_type)


def _cleanup_resources(
    encoder: AsyncVisionEncoder[Any, Any, Any] | None,
    decoder_runtime: VllmDecoderRuntime | None,
    prometheus_temp_dir: Any | None,
) -> None:
    """Release independently owned resources even if one cleanup fails."""

    try:
        if encoder is not None:
            encoder.shutdown()
    finally:
        try:
            if prometheus_temp_dir is not None:
                prometheus_temp_dir.cleanup()
        finally:
            if decoder_runtime is not None:
                decoder_runtime.shutdown()


def prepare_user_ensemble_config(
    argv: list[str] | None = None,
) -> tuple[Config, type[VisionEncoderBackend[Any, Any, Any]]]:
    """Parse the shared decoder/encoder configuration for this example."""

    config = parse_args(argv)
    if not config.custom_encoder_class:
        raise ValueError(
            "--custom-encoder-class is required by the user ensemble example"
        )
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model
    configure_rl_logprobs_mode(config)
    config.engine_args.enable_prompt_embeds = True
    return config, _load_encoder_backend(config.custom_encoder_class)


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
        self._engine_args = config.engine_args
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
        (
            engine_client,
            vllm_config,
            default_sampling_params,
            prometheus_temp_dir,
            _component_gauges,
        ) = setup_vllm_engine(self._config)
        decoder_runtime = VllmDecoderRuntime(
            engine=engine_client,
            vllm_config=vllm_config,
            default_sampling_params=default_sampling_params,
        )
        decoder_stage = VllmDecoderStage(decoder_runtime)
        encoder: AsyncVisionEncoder[Any, Any, Any] | None = None
        try:
            backend = self._encoder_backend_type()
            adapter = create_custom_encoder_adapter(
                backend,
                decoder_runtime.model_config,
                self._engine_args,
            )
            encoder = AsyncVisionEncoder(
                backend,
                name="workflow-vision-encoder",
            )
            encoder.load(self.model_name)
            runners = {
                "encoder": EncoderStage(encoder, adapter),
                "classifier": self._classifier,
                "generator": decoder_stage,
            }
            plan = compile_workflow(
                define_workflow(),
                DeploymentSpec.local(
                    encoder="encoder",
                    classifier="classifier",
                    generator="generator",
                ),
            )
            executor = WorkflowExecutor(plan, runners)
        except BaseException:
            _cleanup_resources(encoder, decoder_runtime, prometheus_temp_dir)
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

        _cleanup_resources(encoder, decoder_runtime, prometheus_temp_dir)

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
