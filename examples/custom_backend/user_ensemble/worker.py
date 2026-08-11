# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated encoder -> classifier + remote decoder -> join worker."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any, Protocol, cast

from dynamo._core import Context, DistributedRuntime
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
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
)


class Classifier(Protocol):
    """User-supplied asynchronous classifier over encoder artifacts."""

    async def classify(self, artifacts: Sequence[Any]) -> str:
        ...


class DecoderResponse(Protocol):
    """Annotated response returned by a Dynamo endpoint client."""

    def is_error(self) -> bool:
        ...

    def comments(self) -> list[str] | None:
        ...

    def data(self) -> Any:
        ...


class DecoderClient(Protocol):
    """Remote decoder client operations used by the ensemble engine."""

    async def wait_for_instances(self) -> list[int]:
        ...

    async def generate(
        self,
        request: GenerateRequest,
        *,
        context: Context | None = None,
    ) -> AsyncIterator[DecoderResponse]:
        ...


class Runtime(Protocol):
    """Runtime lifecycle operation used by the ensemble engine."""

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


def _served_model_name(configured: str | None, fallback: str) -> str:
    return configured or fallback


class UserEnsembleEngine(LLMEngine):
    """One-request aggregate chain with a fan-out and terminal response join."""

    def __init__(
        self,
        *,
        model_name: str,
        served_model_name: str,
        encoder_backend_type: type[VisionEncoderBackend[Any, Any, Any]],
        discovery_backend: str,
        request_plane: str,
        event_plane: str | None,
        decoder_endpoint: str,
        decoder_model_name: str,
        decoder_connect_timeout: float,
        max_model_len: int,
        classifier: Classifier | None = None,
    ) -> None:
        self.model_name = model_name
        self.served_model_name = served_model_name
        self._encoder_backend_type = encoder_backend_type
        self._discovery_backend = discovery_backend
        self._request_plane = request_plane
        self._event_plane = event_plane
        self._decoder_endpoint = decoder_endpoint
        self._decoder_model_name = decoder_model_name
        self._decoder_connect_timeout = decoder_connect_timeout
        self._max_model_len = max_model_len
        self._classifier = classifier or DummyClassifier()

        self._decoder_runtime: Runtime | None = None
        self._decoder_client: DecoderClient | None = None
        self._encoder: Encoder | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[UserEnsembleEngine, WorkerConfig]:
        parser = argparse.ArgumentParser(
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
        parser.add_argument("--model", required=True)
        parser.add_argument("--served-model-name", default=None)
        parser.add_argument("--custom-jinja-template", default=None)
        parser.add_argument("--encoder-class", required=True)
        parser.add_argument("--max-model-len", type=int, default=4096)
        parser.add_argument("--decoder-namespace", default=None)
        parser.add_argument("--decoder-component", default="remote-vllm")
        parser.add_argument("--decoder-endpoint", default="generate")
        parser.add_argument("--decoder-model-name", default=None)
        parser.add_argument("--decoder-connect-timeout", type=float, default=300.0)
        parser.add_argument("--disable-kv-routing", action="store_true")
        args = parser.parse_args(argv)

        served_model_name = _served_model_name(args.served_model_name, args.model)
        decoder_model_name = args.decoder_model_name or (
            f"{served_model_name}-remote-vllm"
        )
        decoder_namespace = args.decoder_namespace or args.namespace
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
            model_name=args.model,
            served_model_name=served_model_name,
            encoder_backend_type=backend_type,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
            decoder_endpoint=(
                f"{decoder_namespace}.{args.decoder_component}."
                f"{args.decoder_endpoint}"
            ),
            decoder_model_name=decoder_model_name,
            decoder_connect_timeout=args.decoder_connect_timeout,
            max_model_len=args.max_model_len,
        )
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.model,
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
        backend = self._encoder_backend_type()
        encoder = AsyncVisionEncoder(backend, name="ensemble-vision-encoder")
        self._encoder = encoder
        encoder.load(self.model_name)

        runtime = DistributedRuntime(
            asyncio.get_running_loop(),
            self._discovery_backend,
            self._request_plane,
            event_plane=self._event_plane,
        )
        self._decoder_runtime = runtime
        endpoint = runtime.endpoint(self._decoder_endpoint)
        client = await endpoint.client()
        self._decoder_client = cast(DecoderClient, client)
        await asyncio.wait_for(
            client.wait_for_instances(),
            timeout=self._decoder_connect_timeout,
        )

        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            llm=LlmRegistration(
                context_length=self._max_model_len,
                data_parallel_size=1,
                data_parallel_start_rank=0,
            ),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        encoder, decoder_client = self._request_resources()
        image_url = self._single_image_url(request)
        artifacts = await encoder.encode([image_url])

        classification = asyncio.create_task(self._classifier.classify(artifacts))
        decoding = asyncio.create_task(
            self._generate_remote_final(decoder_client, request, context)
        )
        try:
            await asyncio.gather(classification, decoding)
        except BaseException:
            classification.cancel()
            decoding.cancel()
            await asyncio.gather(classification, decoding, return_exceptions=True)
            raise

        decoded = decoding.result()
        engine_data = decoded.setdefault("engine_data", {})
        engine_data["ensemble"] = {"classifier": classification.result()}
        yield decoded

    async def abort(self, context: Context) -> None:
        context.stop_generating()

    async def cleanup(self) -> None:
        encoder = self._encoder
        runtime = self._decoder_runtime
        self._encoder = None
        self._decoder_client = None
        self._decoder_runtime = None

        if encoder is not None:
            encoder.shutdown()
        if runtime is not None:
            runtime.shutdown()

    async def _generate_remote_final(
        self,
        client: DecoderClient,
        request: GenerateRequest,
        context: Context,
    ) -> GenerateChunk:
        self._validate_output_options(request)
        remote_request = cast(GenerateRequest, dict(request))
        remote_request["model"] = self._decoder_model_name

        token_ids: list[int] = []
        terminal: GenerateChunk | None = None
        try:
            stream = await client.generate(remote_request, context=context)
            async for response in stream:
                if response.is_error():
                    comments = response.comments() or []
                    message = "; ".join(comments) or "unknown remote decoder error"
                    raise RuntimeError(f"Remote vLLM decoder failed: {message}")

                data = response.data()
                if isinstance(data, str):
                    data = json.loads(data)
                if data is None:
                    continue
                if not isinstance(data, dict):
                    raise RuntimeError(
                        "Remote vLLM decoder returned a non-object response"
                    )

                index = data.get("index", 0)
                if index not in (None, 0):
                    raise RuntimeError(
                        f"Remote vLLM decoder returned unexpected index {index}"
                    )
                chunk_token_ids = data.get("token_ids")
                if not isinstance(chunk_token_ids, list):
                    raise RuntimeError(
                        "Remote vLLM decoder response did not contain token_ids"
                    )
                token_ids.extend(chunk_token_ids)

                if data.get("finish_reason") is not None:
                    if terminal is not None:
                        raise RuntimeError(
                            "Remote vLLM decoder returned multiple terminal responses"
                        )
                    terminal = cast(GenerateChunk, dict(data))

            if terminal is None:
                raise RuntimeError(
                    "Remote vLLM decoder ended without a terminal response"
                )
            terminal["token_ids"] = token_ids
            terminal["index"] = 0
            return terminal
        except BaseException:
            context.stop_generating()
            raise

    def _request_resources(self) -> tuple[Encoder, DecoderClient]:
        if self._encoder is None or self._decoder_client is None:
            raise RuntimeError("UserEnsembleEngine.generate() called before start()")
        return self._encoder, self._decoder_client

    @staticmethod
    def _validate_output_options(request: GenerateRequest) -> None:
        sampling_options = request.get("sampling_options") or {}
        n = sampling_options.get("n")
        if n not in (None, 1):
            raise InvalidArgument(
                f"UserEnsembleEngine supports exactly one choice; got n={n}"
            )

        output_options = request.get("output_options") or {}
        if (
            output_options.get("logprobs") is not None
            or output_options.get("prompt_logprobs") is not None
        ):
            raise InvalidArgument("UserEnsembleEngine does not support logprobs")

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
