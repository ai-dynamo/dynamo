# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated encoder -> classifier + decoder workflow."""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast

from vllm.engine.arg_utils import AsyncEngineArgs

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

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
from dynamo.vllm.embedded_decoder import EmbeddedVllmDecoder
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.vllm.multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
)
from dynamo.workflow import (
    ExecutionPlan,
    StageContext,
    StageContract,
    StageRunner,
    ValueSpec,
    Workflow,
    compile_workflow,
)

ARTIFACTS = ValueSpec(type="object", class_id="dynamo.vllm.CustomEncoderArtifacts")
REQUEST = EmbeddedVllmDecoder.contract.inputs["request"]
PROMPT = EmbeddedVllmDecoder.contract.inputs["prompt"]


class EncoderStage:
    """Adapt the existing async encoder to the common workflow interface."""

    contract = StageContract(
        id="custom-vision-encoder",
        inputs={"image_url": ValueSpec(type="text"), "request": REQUEST},
        outputs={"artifacts": ARTIFACTS, "prompt": PROMPT},
    )

    def __init__(
        self,
        encoder: AsyncVisionEncoder[Any, Any, Any],
        adapter: CustomEncoderAdapter[Any],
    ) -> None:
        self._encoder = encoder
        self._adapter = adapter

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        del context
        request = cast(GenerateRequest, inputs["request"])
        token_ids = request.get("token_ids")
        if not isinstance(token_ids, list):
            raise InvalidArgument("request must contain token_ids")
        artifacts = await self._encoder.encode([cast(str, inputs["image_url"])])
        prompt = self._adapter.prepare_prompt(list(token_ids), artifacts)
        return {"artifacts": artifacts, "prompt": prompt}


class DummyClassifier:
    """Replaceable classification worker used by the runnable example."""

    contract = StageContract(
        id="artifact-classifier",
        inputs={"artifacts": ARTIFACTS},
        outputs={"scores": ValueSpec(type="json")},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        del inputs, context
        return {"scores": {"dummy-classification": 1.0}}


def define_workflow() -> Workflow:
    """Author the logical pipeline independently from worker construction."""

    workflow = Workflow("encoder-classifier-llm")
    image_url = workflow.input("image_url", type="text")
    request = workflow.input(
        "request", type="object", class_id="dynamo.common.backend.GenerateRequest"
    )

    encoder = workflow.stage(
        "encoder", EncoderStage, image_url=image_url, request=request
    )
    classifier = workflow.stage(
        "classifier", DummyClassifier, artifacts=encoder.artifacts
    )
    generator = workflow.stage(
        "generator", EmbeddedVllmDecoder, request=request, prompt=encoder.prompt
    )

    workflow.output("scores", classifier.scores)
    workflow.output("chunk", generator.chunk)
    return workflow


def _load_encoder_backend(class_path: str) -> type[VisionEncoderBackend[Any, Any, Any]]:
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError(
            "--encoder-class must be a dotted module.ClassName path; "
            f"got {class_path!r}"
        )
    module = importlib.import_module(module_name)
    backend_type = getattr(module, class_name)
    if not isinstance(backend_type, type) or not issubclass(
        backend_type, VisionEncoderBackend
    ):
        raise TypeError(f"{class_path} must name a VisionEncoderBackend subclass")
    return cast(type[VisionEncoderBackend[Any, Any, Any]], backend_type)


def _served_model_name(configured: str | list[str] | None, fallback: str) -> str:
    if isinstance(configured, list):
        return configured[0] if configured else fallback
    return configured or fallback


class UserEnsembleEngine(LLMEngine):
    """Expose the authored local workflow through one Dynamo endpoint."""

    def __init__(
        self,
        *,
        model_name: str,
        served_model_name: str,
        engine_args: AsyncEngineArgs,
        encoder_backend_type: type[VisionEncoderBackend[Any, Any, Any]],
        classifier: StageRunner | None = None,
    ) -> None:
        self.model_name = model_name
        self.served_model_name = served_model_name
        self._engine_args = engine_args
        self._encoder_backend_type = encoder_backend_type
        self._classifier = classifier or DummyClassifier()

        self._decoder: EmbeddedVllmDecoder | None = None
        self._encoder: AsyncVisionEncoder[Any, Any, Any] | None = None
        self._plan: ExecutionPlan | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[UserEnsembleEngine, WorkerConfig]:
        parser = FlexibleArgumentParser(
            description="Aggregated user workflow",
            allow_abbrev=False,
        )
        parser.add_argument("--namespace", default="dynamo")
        parser.add_argument("--component", default="backend")
        parser.add_argument("--endpoint", default="generate")
        parser.add_argument("--endpoint-types", default="chat,completions")
        parser.add_argument(
            "--discovery-backend",
            choices=("kubernetes", "etcd", "file", "mem"),
            default="etcd",
        )
        parser.add_argument("--request-plane", choices=("tcp", "nats"), default="tcp")
        parser.add_argument("--event-plane", choices=("nats", "zmq"), default=None)
        parser.add_argument("--custom-jinja-template", default=None)
        parser.add_argument("--encoder-class", required=True)
        parser.add_argument("--disable-kv-routing", action="store_true")
        AsyncEngineArgs.add_cli_args(parser, async_args_only=False)
        args = parser.parse_args(argv)

        requested_model = args.model
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine_args.enable_prompt_embeds = True
        served_model_name = _served_model_name(
            args.served_model_name,
            requested_model,
        )
        backend_type = _load_encoder_backend(args.encoder_class)
        custom_template = (
            os.path.abspath(os.path.expanduser(args.custom_jinja_template))
            if args.custom_jinja_template
            else None
        )
        if custom_template is not None and not os.path.isfile(custom_template):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {custom_template}"
            )

        engine = cls(
            model_name=requested_model,
            served_model_name=served_model_name,
            engine_args=engine_args,
            encoder_backend_type=backend_type,
        )
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=requested_model,
            served_model_name=served_model_name,
            endpoint_types=args.endpoint_types,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
            custom_jinja_template=custom_template,
            enable_kv_routing=not args.disable_kv_routing,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        decoder = EmbeddedVllmDecoder.from_engine_args(self._engine_args)
        try:
            backend = self._encoder_backend_type()
            adapter = create_custom_encoder_adapter(
                backend,
                decoder.model_config,
                self._engine_args,
            )
            encoder = AsyncVisionEncoder(
                backend,
                name="workflow-vision-encoder",
            )
        except BaseException:
            decoder.shutdown()
            raise
        try:
            encoder.load(self.model_name)
            plan = compile_workflow(
                define_workflow(),
                encoder=EncoderStage(encoder, adapter),
                classifier=self._classifier,
                generator=decoder,
            )
        except BaseException:
            encoder.shutdown()
            decoder.shutdown()
            raise

        self._decoder = decoder
        self._encoder = encoder
        self._plan = plan
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            llm=LlmRegistration(
                context_length=decoder.model_config.max_model_len,
            ),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        plan = self._plan
        if plan is None:
            raise RuntimeError("UserEnsembleEngine.generate() called before start()")
        request_id = context.id()
        result = await plan.run(
            {"image_url": self._single_image_url(request), "request": request},
            attempt_id=request_id,
        )

        decoded = cast(GenerateChunk, result["chunk"])
        decoded["engine_data"] = {"ensemble": {"classifier_scores": result["scores"]}}
        yield decoded

    async def abort(self, context: Context) -> None:
        decoder = self._decoder
        if decoder is not None:
            await decoder.abort(context.id())

    async def cleanup(self) -> None:
        encoder = self._encoder
        decoder = self._decoder
        self._plan = None
        self._encoder = None
        self._decoder = None

        if encoder is not None:
            encoder.shutdown()
        if decoder is not None:
            decoder.shutdown()

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
