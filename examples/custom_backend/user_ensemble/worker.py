# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated encoder -> classifier + decoder -> join worker.

The module intentionally builds on public Dynamo worker contracts instead of
specializing ``dynamo.vllm``. One encoder result is shared by two in-process
consumers. The decoder consumes an adapter-produced vLLM prompt, while the
classifier consumes the same artifact objects and contributes data to the final
``engine_data`` payload.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Protocol, cast

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
    WorkerConfig,
)
from dynamo.common.backend.run import run
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.embedded_decoder import EmbeddedVllmDecoder
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.vllm.multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
)


class Classifier(Protocol):
    """User-supplied asynchronous classifier over encoder artifacts."""

    async def classify(self, artifacts: Sequence[Any]) -> str:
        ...


class Decoder(Protocol):
    """Embedded decoder operations used by the ensemble engine."""

    async def generate_final(
        self,
        request: GenerateRequest,
        prompt: Any,
        request_id: str,
    ) -> GenerateChunk:
        ...

    async def abort(self, request_id: str) -> None:
        ...

    def shutdown(self) -> None:
        ...


class Encoder(Protocol):
    """Subset of ``AsyncVisionEncoder`` used on the request path."""

    async def encode(self, raws: list[str]) -> list[Any]:
        ...

    def shutdown(self) -> None:
        ...


class DummyClassifier:
    """Small replaceable classification branch used by the runnable smoke."""

    async def classify(self, artifacts: Sequence[Any]) -> str:
        if len(artifacts) != 1:
            raise InvalidArgument(
                f"DummyClassifier requires one encoder artifact; got {len(artifacts)}"
            )
        await asyncio.sleep(0)
        return "dummy-classification"


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
    """One-request aggregate chain with a fan-out and terminal response join."""

    def __init__(
        self,
        *,
        model_name: str,
        served_model_name: str,
        engine_args: AsyncEngineArgs,
        encoder_backend_type: type[VisionEncoderBackend[Any, Any, Any]],
        classifier: Classifier | None = None,
    ) -> None:
        self.model_name = model_name
        self.served_model_name = served_model_name
        self._engine_args = engine_args
        self._encoder_backend_type = encoder_backend_type
        self._classifier = classifier or DummyClassifier()

        self._decoder: Decoder | None = None
        self._encoder: Encoder | None = None
        self._adapter: CustomEncoderAdapter[Any] | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[UserEnsembleEngine, WorkerConfig]:
        parser = FlexibleArgumentParser(
            description="Aggregated user ensemble worker",
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
        self._decoder = decoder

        backend = self._encoder_backend_type()
        self._adapter = decoder.create_prompt_adapter(backend)
        self._encoder = AsyncVisionEncoder(
            backend,
            name="ensemble-vision-encoder",
        )
        self._encoder.load(self.model_name)

        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            llm=decoder.registration(),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        encoder, adapter, decoder = self._request_resources()
        request_id = context.id()
        image_url = self._single_image_url(request)

        artifacts = await encoder.encode([image_url])
        prompt = adapter.prepare_prompt(list(request["token_ids"]), artifacts)

        async with asyncio.TaskGroup() as group:
            classification = group.create_task(self._classifier.classify(artifacts))
            decoding = group.create_task(
                decoder.generate_final(request, prompt, request_id)
            )

        decoded = decoding.result()
        decoded["engine_data"] = {"ensemble": {"classifier": classification.result()}}
        yield decoded

    async def abort(self, context: Context) -> None:
        decoder = self._decoder
        if decoder is not None:
            await decoder.abort(context.id())

    async def cleanup(self) -> None:
        encoder = self._encoder
        decoder = self._decoder
        self._encoder = None
        self._decoder = None
        self._adapter = None

        if encoder is not None:
            encoder.shutdown()
        if decoder is not None:
            decoder.shutdown()

    def _request_resources(
        self,
    ) -> tuple[Encoder, CustomEncoderAdapter[Any], Decoder]:
        if self._encoder is None or self._adapter is None or self._decoder is None:
            raise RuntimeError("UserEnsembleEngine.generate() called before start()")
        return self._encoder, self._adapter, self._decoder

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
