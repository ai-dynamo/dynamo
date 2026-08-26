# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous whole-pipeline control worker for CustomEncoder benchmarks."""

from __future__ import annotations

import importlib
import logging
import os
from collections import defaultdict
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Hashable, cast

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import RequestOutputKind

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
from dynamo.common.backend.publisher import KvEventSource, ZmqSource
from dynamo.common.backend.run import run
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.handlers import build_sampling_params
from dynamo.vllm.multimodal_utils.custom_encoder import (
    CustomEncoderAdapter,
    Preprocessed,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.batcher import ThreadedMicroBatcher
from dynamo.vllm.multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
)

logger = logging.getLogger(__name__)

_MISSING = object()


@dataclass(frozen=True)
class _ControlRequest:
    request: GenerateRequest
    image_url: str


def _load_encoder_backend(
    class_path: str,
) -> type[VisionEncoderBackend[Any, Any, Any]]:
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError(
            "--encoder-class must be a dotted module.ClassName path; "
            f"got {class_path!r}"
        )
    backend_type = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(backend_type, type) or not issubclass(
        backend_type, VisionEncoderBackend
    ):
        raise TypeError(f"{class_path} must name a VisionEncoderBackend subclass")
    return cast(type[VisionEncoderBackend[Any, Any, Any]], backend_type)


def _served_model_name(configured: str | list[str] | None, fallback: str) -> str:
    if isinstance(configured, list):
        return configured[0] if configured else fallback
    return configured or fallback


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _subscriber_endpoint(publisher_endpoint: str) -> str:
    if publisher_endpoint.startswith("tcp://*:"):
        return publisher_endpoint.replace("tcp://*:", "tcp://127.0.0.1:", 1)
    if publisher_endpoint.startswith("tcp://0.0.0.0:"):
        return publisher_endpoint.replace("tcp://0.0.0.0:", "tcp://127.0.0.1:", 1)
    return publisher_endpoint


class BatchedControlEngine(LLMEngine):
    """Run preprocess, vision, and one offline ``LLM.generate`` per outer batch.

    One dedicated actor owns both models. It collects at most
    ``control_max_batch_items`` requests for ``control_max_queue_delay_us``, then
    runs the complete pipeline synchronously. While vLLM generates that batch,
    later requests may queue but cannot begin image preprocessing.
    """

    def __init__(
        self,
        *,
        model_name: str,
        served_model_name: str,
        engine_args: EngineArgs,
        encoder_backend_type: type[VisionEncoderBackend[Any, Any, Any]],
        control_max_batch_items: int,
        control_max_queue_delay_us: int,
    ) -> None:
        self.model_name = model_name
        self.served_model_name = served_model_name
        self._engine_args = engine_args
        self._encoder_backend_type = encoder_backend_type
        self._control_max_batch_items = control_max_batch_items
        self._control_max_queue_delay_us = control_max_queue_delay_us

        self._batcher: ThreadedMicroBatcher[
            _ControlRequest, list[GenerateChunk]
        ] | None = None
        self._backend: VisionEncoderBackend[Any, Any, Any] | None = None
        self._adapter: CustomEncoderAdapter[Any] | None = None
        self._llm: LLM | None = None
        self._default_sampling_params: dict[str, Any] = {}
        self._model_max_len: int | None = None
        self._registration: LlmRegistration | None = None
        self._kv_event_source: ZmqSource | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[BatchedControlEngine, WorkerConfig]:
        parser = FlexibleArgumentParser(
            description="Synchronous batched CustomEncoder control worker",
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
        parser.add_argument("--control-max-batch-items", type=int, default=8)
        parser.add_argument("--control-max-queue-delay-us", type=int, default=1_000)
        parser.add_argument("--disable-kv-routing", action="store_true")
        EngineArgs.add_cli_args(parser)
        args = parser.parse_args(argv)

        if args.control_max_batch_items < 1:
            raise ValueError("--control-max-batch-items must be >= 1")
        if args.control_max_queue_delay_us < 0:
            raise ValueError("--control-max-queue-delay-us must be >= 0")

        engine_args = EngineArgs.from_cli_args(args)
        engine_args.enable_prompt_embeds = True
        model_name = str(engine_args.model)
        served_model_name = _served_model_name(
            engine_args.served_model_name,
            model_name,
        )
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
            model_name=model_name,
            served_model_name=served_model_name,
            engine_args=engine_args,
            encoder_backend_type=_load_encoder_backend(args.encoder_class),
            control_max_batch_items=args.control_max_batch_items,
            control_max_queue_delay_us=args.control_max_queue_delay_us,
        )
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=model_name,
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
        if self._batcher is not None:
            raise RuntimeError("BatchedControlEngine.start() called twice")
        batcher = ThreadedMicroBatcher(
            self._run_pipeline_batch,
            max_batch_items=self._control_max_batch_items,
            max_queue_delay_us=self._control_max_queue_delay_us,
            on_start=self._start_pipeline,
            on_stop=self._close_pipeline,
            name="custom-encoder-control",
        )
        self._batcher = batcher
        try:
            batcher.start()
        except BaseException:
            self._batcher = None
            raise
        registration = self._registration
        if registration is None:
            batcher.shutdown()
            self._batcher = None
            raise RuntimeError("control pipeline started without registration metadata")
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            llm=registration,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        batcher = self._batcher
        if batcher is None:
            raise RuntimeError("BatchedControlEngine.generate() called before start()")
        prompt_tokens = len(request.get("token_ids") or [])
        if context.is_stopped():
            yield self._cancelled_chunk(prompt_tokens)
            return

        image_url = self._single_image_url(request)
        chunks = await batcher.submit([_ControlRequest(request, image_url)])
        if context.is_stopped():
            yield self._cancelled_chunk(prompt_tokens)
            return
        for chunk in chunks[0]:
            yield chunk

    async def kv_event_sources(self) -> list[KvEventSource]:
        source = self._kv_event_source
        return [source] if source is not None else []

    async def cleanup(self) -> None:
        batcher = self._batcher
        self._batcher = None
        if batcher is not None:
            batcher.shutdown()

    def _start_pipeline(self) -> None:
        backend = self._encoder_backend_type()
        llm: LLM | None = None
        try:
            backend.build(self.model_name)
            llm = LLM.from_engine_args(self._engine_args)
            adapter = create_custom_encoder_adapter(
                backend,
                llm.model_config,
                self._engine_args,
            )
        except BaseException:
            if llm is not None:
                self._shutdown_llm(llm)
            backend.close()
            raise

        self._backend = backend
        self._adapter = adapter
        self._llm = llm
        self._default_sampling_params = llm.model_config.get_diff_sampling_param() or {}
        self._model_max_len = _positive_int_or_none(llm.model_config.max_model_len)

        vllm_config = llm.llm_engine.vllm_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        self._registration = LlmRegistration(
            context_length=self._model_max_len,
            kv_cache_block_size=_positive_int_or_none(cache_config.block_size),
            total_kv_blocks=_positive_int_or_none(cache_config.num_gpu_blocks),
            max_num_seqs=_positive_int_or_none(scheduler_config.max_num_seqs),
            max_num_batched_tokens=_positive_int_or_none(
                scheduler_config.max_num_batched_tokens
            ),
        )
        kv_events_config = vllm_config.kv_events_config
        kv_events_enabled = bool(
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        )
        self._kv_event_source = (
            ZmqSource(
                endpoint=_subscriber_endpoint(kv_events_config.endpoint),
                topic=kv_events_config.topic,
                dp_rank=0,
            )
            if kv_events_enabled
            else None
        )
        logger.info(
            "control_worker_config max_batch_items=%d queue_delay_us=%d "
            "prefix_caching=%s kv_events_enabled=%s",
            self._control_max_batch_items,
            self._control_max_queue_delay_us,
            cache_config.enable_prefix_caching,
            kv_events_enabled,
        )

    def _close_pipeline(self) -> None:
        backend = self._backend
        llm = self._llm
        self._backend = None
        self._adapter = None
        self._llm = None
        self._registration = None
        self._kv_event_source = None
        try:
            if llm is not None:
                self._shutdown_llm(llm)
        finally:
            if backend is not None:
                backend.close()

    @staticmethod
    def _shutdown_llm(llm: LLM) -> None:
        llm.llm_engine.engine_core.shutdown()

    def _run_pipeline_batch(
        self, work_items: list[_ControlRequest]
    ) -> list[list[GenerateChunk]]:
        backend, adapter, llm = self._require_pipeline()
        preprocessed = [backend.preprocess(work.image_url) for work in work_items]
        artifacts = self._encode_preprocessed(backend, preprocessed)
        prompts = [
            adapter.prepare_prompt(list(work.request["token_ids"]), [artifact])
            for work, artifact in zip(work_items, artifacts)
        ]
        sampling_params = [
            build_sampling_params(
                work.request,
                self._default_sampling_params,
                self._model_max_len,
            )
            for work in work_items
        ]
        for params in sampling_params:
            params.output_kind = RequestOutputKind.FINAL_ONLY
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        if len(outputs) != len(work_items):
            raise RuntimeError(
                f"offline LLM returned {len(outputs)} results for "
                f"{len(work_items)} requests"
            )
        return [
            self._request_output_to_chunks(work.request, output)
            for work, output in zip(work_items, outputs)
        ]

    @staticmethod
    def _encode_preprocessed(
        backend: VisionEncoderBackend[Any, Any, Any],
        preprocessed: list[Preprocessed[Any]],
    ) -> list[Any]:
        grouped: dict[
            Hashable | None, list[tuple[int, Preprocessed[Any]]]
        ] = defaultdict(list)
        for index, item in enumerate(preprocessed):
            grouped[item.bucket_key].append((index, item))

        output: list[Any] = [_MISSING] * len(preprocessed)
        for entries in grouped.values():
            pending = list(entries)
            while pending:
                batch: list[tuple[int, Preprocessed[Any]]] = []
                batch_cost = 0
                while pending:
                    index, item = pending[0]
                    exceeds_cost = (
                        backend.max_batch_cost is not None
                        and batch_cost + item.cost > backend.max_batch_cost
                    )
                    exceeds_items = (
                        backend.max_batch_items is not None
                        and len(batch) >= backend.max_batch_items
                    )
                    if batch and (exceeds_cost or exceeds_items):
                        break
                    if exceeds_cost:
                        raise ValueError(
                            f"encoder item cost {item.cost} exceeds "
                            f"max_batch_cost {backend.max_batch_cost}"
                        )
                    batch.append((index, item))
                    batch_cost += item.cost
                    pending.pop(0)
                encoded = backend.forward_batch([item.item for _, item in batch])
                if len(encoded) != len(batch):
                    raise RuntimeError(
                        f"encoder returned {len(encoded)} artifacts for "
                        f"{len(batch)} items"
                    )
                for (index, _), artifact in zip(batch, encoded):
                    output[index] = artifact
        if any(artifact is _MISSING for artifact in output):
            raise RuntimeError("encoder did not produce every requested artifact")
        return output

    def _require_pipeline(
        self,
    ) -> tuple[VisionEncoderBackend[Any, Any, Any], CustomEncoderAdapter[Any], LLM,]:
        if self._backend is None or self._adapter is None or self._llm is None:
            raise RuntimeError("control pipeline is not loaded")
        return self._backend, self._adapter, self._llm

    @staticmethod
    def _request_output_to_chunks(
        request: GenerateRequest,
        request_output: Any,
    ) -> list[GenerateChunk]:
        prompt_token_ids = request_output.prompt_token_ids
        prompt_tokens = (
            len(prompt_token_ids)
            if prompt_token_ids is not None
            else len(request.get("token_ids") or [])
        )
        chunks: list[GenerateChunk] = []
        for completion in request_output.outputs:
            finish_reason = completion.finish_reason
            if finish_reason is None:
                raise RuntimeError("offline LLM returned an unfinished completion")
            token_ids = [int(token_id) for token_id in completion.token_ids]
            chunks.append(
                {
                    "token_ids": token_ids,
                    "index": int(completion.index),
                    "finish_reason": normalize_finish_reason(finish_reason),
                    "completion_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(token_ids),
                        "total_tokens": prompt_tokens + len(token_ids),
                    },
                }
            )
        if not chunks:
            raise RuntimeError("offline LLM returned no completions")
        return chunks

    @staticmethod
    def _single_image_url(request: GenerateRequest) -> str:
        multimodal = request.get("multi_modal_data") or {}
        unsupported = sorted(
            key for key, value in multimodal.items() if key != IMAGE_URL_KEY and value
        )
        if unsupported:
            raise InvalidArgument(
                "BatchedControlEngine supports image inputs only; got "
                f"unsupported multimodal data: {unsupported}"
            )
        image_items = multimodal.get(IMAGE_URL_KEY) or []
        if len(image_items) != 1:
            raise InvalidArgument(
                "BatchedControlEngine requires exactly one image per request; "
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

    @staticmethod
    def _cancelled_chunk(prompt_tokens: int) -> GenerateChunk:
        return {
            "token_ids": [],
            "index": 0,
            "finish_reason": "cancelled",
            "completion_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        }


def main() -> None:
    run(BatchedControlEngine)


if __name__ == "__main__":
    main()
