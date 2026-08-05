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
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

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

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    """User-supplied asynchronous classifier over encoder artifacts."""

    async def classify(self, artifacts: Sequence[Any]) -> str:
        ...


class Decoder(Protocol):
    """Subset of ``AsyncLLM`` used by the ensemble engine."""

    def generate(
        self,
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncIterator[Any]:
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


@dataclass(frozen=True)
class _DecodedResult:
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int


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
        custom_encoder_max_queue_delay_us: int = 0,
        classifier: Classifier | None = None,
    ) -> None:
        if custom_encoder_max_queue_delay_us < 0:
            raise ValueError("custom encoder queue delay must be non-negative")
        self.model_name = model_name
        self.served_model_name = served_model_name
        self._engine_args = engine_args
        self._encoder_backend_type = encoder_backend_type
        self._custom_encoder_max_queue_delay_us = custom_encoder_max_queue_delay_us
        self._classifier = classifier or DummyClassifier()

        self._decoder: Decoder | None = None
        self._encoder: Encoder | None = None
        self._adapter: CustomEncoderAdapter[Any] | None = None
        self._default_sampling_params: dict[str, Any] = {}
        self._model_max_len: int | None = None

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
        parser.add_argument("--custom-encoder-max-queue-delay-us", type=int, default=0)
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
            custom_encoder_max_queue_delay_us=(args.custom_encoder_max_queue_delay_us),
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
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = self._engine_args.create_engine_config(
            usage_context=usage_context
        )
        self._default_sampling_params = (
            vllm_config.model_config.get_diff_sampling_param()
        )
        self._model_max_len = vllm_config.model_config.max_model_len

        self._decoder = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=[],
            enable_log_requests=self._engine_args.enable_log_requests,
            disable_log_stats=self._engine_args.disable_log_stats,
        )

        backend = self._encoder_backend_type()
        self._adapter = create_custom_encoder_adapter(
            backend,
            vllm_config.model_config,
            self._engine_args,
        )
        self._encoder = AsyncVisionEncoder(
            backend,
            name="ensemble-vision-encoder",
            max_queue_delay_us=self._custom_encoder_max_queue_delay_us,
        )
        self._encoder.load(self.model_name)

        scheduler_config = vllm_config.scheduler_config
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            llm=LlmRegistration(
                context_length=self._model_max_len,
                kv_cache_block_size=vllm_config.cache_config.block_size,
                max_num_seqs=scheduler_config.max_num_seqs,
                max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
                data_parallel_size=1,
                data_parallel_start_rank=0,
            ),
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        encoder, adapter = self._request_resources()
        request_id = context.id()
        image_url = self._single_image_url(request)

        artifacts = await encoder.encode([image_url])
        prompt = adapter.prepare_prompt(list(request["token_ids"]), artifacts)
        sampling_params = self._sampling_params(
            request, len(prompt["prompt_token_ids"])
        )

        async with asyncio.TaskGroup() as group:
            classification = group.create_task(self._classifier.classify(artifacts))
            decoding = group.create_task(
                self._decode(prompt, sampling_params, request_id)
            )

        decoded = decoding.result()
        completion_tokens = len(decoded.token_ids)
        yield {
            "token_ids": decoded.token_ids,
            "index": 0,
            "finish_reason": decoded.finish_reason,
            "completion_usage": {
                "prompt_tokens": decoded.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": decoded.prompt_tokens + completion_tokens,
            },
            "engine_data": {"ensemble": {"classifier": classification.result()}},
        }

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

    def _request_resources(self) -> tuple[Encoder, CustomEncoderAdapter[Any]]:
        if self._encoder is None or self._adapter is None or self._decoder is None:
            raise RuntimeError("UserEnsembleEngine.generate() called before start()")
        return self._encoder, self._adapter

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

    def _sampling_params(
        self, request: GenerateRequest, prompt_tokens: int
    ) -> SamplingParams:
        sampling_options = request.get("sampling_options") or {}
        n = sampling_options.get("n")
        if n is None:
            n = 1
        if n != 1:
            raise InvalidArgument(
                f"UserEnsembleEngine supports exactly one choice; got n={n}"
            )

        output_options = request.get("output_options") or {}
        if (
            output_options.get("logprobs") is not None
            or output_options.get("prompt_logprobs") is not None
        ):
            raise InvalidArgument("UserEnsembleEngine does not support logprobs")

        sampling_params = SamplingParams(**self._default_sampling_params)
        for key in (
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
        ):
            value = sampling_options.get(key)
            if value is not None:
                setattr(sampling_params, key, value)

        stop_conditions = request.get("stop_conditions") or {}
        for key in ("max_tokens", "min_tokens", "ignore_eos", "stop_token_ids"):
            value = stop_conditions.get(key)
            if value is not None:
                setattr(sampling_params, key, value)
        hidden_stop_ids = stop_conditions.get("stop_token_ids_hidden")
        if hidden_stop_ids:
            sampling_params.stop_token_ids = list(
                set(sampling_params.stop_token_ids or []).union(hidden_stop_ids)
            )

        if (
            stop_conditions.get("max_tokens") is None
            and self._model_max_len is not None
        ):
            available = max(1, self._model_max_len - prompt_tokens)
            configured = self._default_sampling_params.get("max_tokens", available)
            sampling_params.max_tokens = min(configured, available)

        sampling_params.n = 1
        sampling_params.detokenize = False
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        return sampling_params

    async def _decode(
        self,
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> _DecodedResult:
        decoder = self._decoder
        if decoder is None:
            raise RuntimeError("decoder is not initialized")

        completed = False
        final_output: Any | None = None
        try:
            async for request_output in decoder.generate(
                prompt, sampling_params, request_id
            ):
                final_output = request_output
            if final_output is None or len(final_output.outputs) != 1:
                count = 0 if final_output is None else len(final_output.outputs)
                raise RuntimeError(
                    f"vLLM returned {count} outputs; exactly one is required"
                )

            output = final_output.outputs[0]
            if getattr(output, "index", 0) not in (None, 0):
                raise RuntimeError(
                    f"vLLM returned unexpected output index {output.index}"
                )
            finish_reason = getattr(output, "finish_reason", None)
            if not finish_reason:
                raise RuntimeError("vLLM final output did not include a finish reason")

            prompt_token_ids = getattr(final_output, "prompt_token_ids", None)
            prompt_tokens = (
                len(prompt_token_ids)
                if prompt_token_ids is not None
                else len(prompt["prompt_token_ids"])
            )
            result = _DecodedResult(
                token_ids=list(output.token_ids or []),
                finish_reason=str(finish_reason),
                prompt_tokens=prompt_tokens,
            )
            completed = True
            return result
        finally:
            if not completed:
                try:
                    await asyncio.shield(decoder.abort(request_id))
                except Exception:
                    logger.exception("Failed to abort vLLM request %s", request_id)


def main() -> None:
    run(UserEnsembleEngine)


if __name__ == "__main__":
    main()
