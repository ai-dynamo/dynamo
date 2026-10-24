# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated token backend for the local DeepInfer low-level engine API."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib
import math
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.token_ids import normalize_request_token_ids
from dynamo.llm.exceptions import Cancelled, EngineShutdown, InvalidArgument

_QUEUE_SIZE = 64


def _load_engine_args():
    return importlib.import_module("deepinfer.engine.arg_utils").EngineArgs


def _load_llm_engine():
    return importlib.import_module("deepinfer.engine.llm_engine").LLMEngine


def _load_request_output():
    return importlib.import_module("deepinfer._C.engine").RequestOutput


def _unsupported(request: GenerateRequest) -> None:
    sampling = request.get("sampling_options") or {}
    stops = request.get("stop_conditions") or {}
    output = request.get("output_options") or {}
    defaults = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1,
        "min_p": 0,
        "use_beam_search": False,
        "length_penalty": 1,
    }
    supported = {"temperature", "top_p", "top_k", "seed", "include_stop_str_in_output"}
    for key, value in sampling.items():
        if value is None or key in supported:
            continue
        if key in defaults and value == defaults[key]:
            continue
        raise InvalidArgument(f"DeepInfer does not support sampling_options.{key}")
    # String/token stops and visibility are enforced by Dynamo's token decoder.
    for key, value in stops.items():
        if key in {
            "max_tokens",
            "ignore_eos",
            "stop",
            "stop_token_ids",
            "stop_token_ids_visible",
            "stop_token_ids_hidden",
        }:
            continue
        if value is not None and not (key == "min_tokens" and value == 0):
            raise InvalidArgument(f"DeepInfer does not support stop_conditions.{key}")
    for key, value in output.items():
        if (
            key
            in {
                "no_stop_trim",
                "skip_special_tokens",
                "formatted_prompt",
                "return_tokens_as_token_ids",
            }
            or value is None
        ):
            continue
        raise InvalidArgument(f"DeepInfer does not support output_options.{key}")
    for key in (
        "multi_modal_data",
        "mm_processor_kwargs",
        "mm_routing_info",
        "encoder_result",
        "prefill_result",
        "bootstrap_info",
        "extra_args",
    ):
        if request.get(key):
            raise InvalidArgument(f"DeepInfer does not support {key}")


def _sampling(request: GenerateRequest) -> dict[str, Any]:
    source = request.get("sampling_options") or {}
    stops = request.get("stop_conditions") or {}
    params = {
        key: source[key]
        for key in ("temperature", "top_p", "top_k")
        if source.get(key) is not None
    }
    for key in ("temperature", "top_p"):
        value = params.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise InvalidArgument(f"{key} must be a finite number")
    if params.get("temperature", 1) < 0:
        raise InvalidArgument("temperature must be nonnegative")
    if not 0 < params.get("top_p", 1) <= 1:
        raise InvalidArgument("top_p must be in (0, 1]")
    top_k = params.get("top_k", -1)
    if type(top_k) is not int or not (top_k == -1 or 1 <= top_k < 2**31):
        raise InvalidArgument("top_k must be -1 or a positive int32")
    if source.get("seed") is not None:
        if type(source["seed"]) is not int:
            raise InvalidArgument("seed must be an integer")
        params["seq_seed"] = source["seed"]
    if stops.get("ignore_eos") is not None:
        if type(stops["ignore_eos"]) is not bool:
            raise InvalidArgument("ignore_eos must be a boolean")
        params["ignore_eos"] = stops["ignore_eos"]
    if stops.get("max_tokens") is not None:
        value = stops["max_tokens"]
        if type(value) is not int or not 1 <= value < 2**31:
            raise InvalidArgument("max_tokens must be a positive int32")
        params["max_tokens"] = value
    return params


@dataclass
class _Request:
    native_id: str
    tokens: list[int]
    params: dict[str, Any]
    queue: asyncio.Queue
    submitted: bool = False
    cancelled: bool = False


class DeepInferLLMEngine(LLMEngine):
    """Async state stays on the event loop; native calls share one thread."""

    def __init__(self, args: Any, worker_config: WorkerConfig):
        self.args = args
        self.worker_config = worker_config
        self.engine: Any = None
        self._executor = None
        self._pump_task = None
        self._wake = asyncio.Event()
        self._requests: dict[str, _Request] = {}
        self._next_id = 0
        self._closing = False
        self._closed = False
        self._failure: BaseException | None = None
        self._output_type = None
        self._model_max_len = args.max_model_len
        self._vocab_size = None
        self._bos_token_id = None

    @classmethod
    async def from_args(cls, argv=None):
        parser = argparse.ArgumentParser(
            description="Dynamo DeepInfer backend", allow_abbrev=False
        )
        parser.add_argument("--model", required=True)
        parser.add_argument("--served-model-name")
        parser.add_argument("--trust-remote-code", action="store_true")
        parser.add_argument("--max-model-len", type=int)
        parser.add_argument("--max-num-seqs", type=int, default=256)
        parser.add_argument("--max-num-batched-tokens", type=int)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        parser.add_argument("--expert-parallel-size", type=int, choices=[1], default=1)
        parser.add_argument("--context-parallel-size", type=int, choices=[1], default=1)
        parser.add_argument("--data-parallel-size", type=int, choices=[1], default=1)
        parser.add_argument("--namespace", default="dynamo")
        parser.add_argument("--component", default="deepinfer")
        parser.add_argument("--endpoint", default="generate")
        parser.add_argument("--discovery-backend", default="etcd")
        parser.add_argument("--request-plane", default="tcp")
        parser.add_argument("--response-plane", choices=["tcp", "quic"], default="tcp")
        parser.add_argument("--event-plane")
        parser.add_argument("--disaggregation-mode", choices=["agg"], default="agg")
        parsed = parser.parse_args(argv)
        for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
            value = getattr(parsed, key)
            if value is not None and value <= 0:
                parser.error(f"{key} must be positive")
        if not 0 < parsed.gpu_memory_utilization <= 1:
            parser.error("gpu_memory_utilization must be in (0, 1]")
        args = SimpleNamespace(
            model=parsed.model,
            tokenizer=parsed.model,
            served_model_name=parsed.served_model_name or parsed.model,
            trust_remote_code=parsed.trust_remote_code,
            max_model_len=parsed.max_model_len,
            max_num_seqs=parsed.max_num_seqs,
            max_num_batched_tokens=parsed.max_num_batched_tokens,
            gpu_memory_utilization=parsed.gpu_memory_utilization,
            expert_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=1,
            stop_preempted_request=False,
        )
        config = WorkerConfig(
            namespace=parsed.namespace,
            component=parsed.component,
            endpoint=parsed.endpoint,
            model_name=parsed.model,
            served_model_name=args.served_model_name,
            discovery_backend=parsed.discovery_backend,
            request_plane=parsed.request_plane,
            response_plane=parsed.response_plane,
            event_plane=parsed.event_plane,
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            enable_kv_routing=False,
            enable_local_indexer=False,
        )
        return cls(args, config), config

    async def _native(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, function, *args
        )

    def _initialize(self):
        self._output_type = _load_request_output()
        self.engine = _load_llm_engine().from_engine_args(
            _load_engine_args()(**vars(self.args))
        )
        self._model_max_len = self.engine.model_config.max_model_len
        self._vocab_size = self.engine.model_config.get_vocab_size()
        self._bos_token_id = getattr(self.engine.tokenizer, "bos_token_id", None)
        if self._bos_token_id is None:
            self._bos_token_id = self.engine.eos_id
        return EngineConfig(
            model=self.args.model,
            served_model_name=self.args.served_model_name,
            llm=LlmRegistration(
                context_length=self._model_max_len,
                max_num_seqs=self.engine.scheduler_config.max_num_seqs,
                max_num_batched_tokens=self.engine.scheduler_config.max_num_batched_tokens,
                data_parallel_size=1,
            ),
        )

    async def start(self, worker_id: int) -> EngineConfig:
        if self._closing or self._closed or self._executor is not None:
            raise EngineShutdown("DeepInfer cannot be started twice")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="deepinfer-driver"
        )
        try:
            config = await self._native(self._initialize)
            self._pump_task = asyncio.create_task(self._pump())
            return config
        except BaseException:
            await self.cleanup()
            raise

    async def health_check_payload(self):
        if self._bos_token_id is None:
            return None
        return {
            "token_ids": [self._bos_token_id],
            "stop_conditions": {"max_tokens": 1, "ignore_eos": True},
        }

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self._closing or self._failure or self._pump_task is None:
            raise EngineShutdown("DeepInfer is not available") from self._failure
        normalize_request_token_ids(request)
        _unsupported(request)
        tokens = request.get("token_ids", [])
        if not tokens or any(
            type(t) is not int or not 0 <= t < 2**31 for t in tokens
        ):
            raise InvalidArgument("token_ids must be nonempty nonnegative int32 values")
        if self._vocab_size is not None and max(tokens) >= self._vocab_size:
            raise InvalidArgument("token ID is outside the model vocabulary")
        params = _sampling(request)
        if self._model_max_len is not None:
            remaining = self._model_max_len - len(tokens)
            if remaining <= 0 or params.get("max_tokens", 1) > remaining:
                raise InvalidArgument(
                    "prompt plus generation exceeds model context length"
                )
            params.setdefault("max_tokens", remaining)
        rid = str(context.id())
        if context.is_stopped():
            raise Cancelled("Request cancelled before submission")
        if rid in self._requests:
            raise InvalidArgument("Duplicate active request ID")
        if len(self._requests) >= self.args.max_num_seqs:
            raise EngineShutdown("DeepInfer admission capacity exceeded")
        self._next_id += 1
        state = _Request(
            str(self._next_id), list(tokens), params, asyncio.Queue(_QUEUE_SIZE)
        )
        self._requests[rid] = state
        self._wake.set()
        try:
            while True:
                item = await state.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if "finish_reason" in item:
                    return
        finally:
            if self._requests.get(rid) is state:
                state.cancelled = True
                self._wake.set()

    @staticmethod
    def _fail(state, error):
        while not state.queue.empty():
            state.queue.get_nowait()
        state.queue.put_nowait(error)
        state.cancelled = True

    async def abort(self, context: Context) -> None:
        state = self._requests.get(str(context.id()))
        if state is not None:
            self._fail(state, Cancelled("DeepInfer request cancelled"))
            self._wake.set()

    def _add_native(self, state):
        result = self.engine.add_request(
            user_id="dynamo",
            request_id=state.native_id,
            sampling_params=state.params,
            prompt_token_ids=np.asarray(state.tokens, dtype=np.int32),
            multi_modal_data=None,
        )
        if result is None or result is False:
            raise InvalidArgument("DeepInfer rejected request")

    def _advance(self):
        unfinished = self.engine.has_unfinished_requests()
        raw = self.engine.step() if unfinished else self.engine.flush_inflight_reqs()
        # No native pointer or pooled object crosses the executor boundary.
        outputs = []
        for pointer in raw:
            out = self._output_type.clone(int(pointer))
            outputs.append(
                (
                    str(out.request_id),
                    int(out.token_id),
                    bool(out.finished),
                    str(out.finish_reason),
                    int(out.prompt_len),
                    int(out.num_output_tokens),
                )
            )
        return outputs, unfinished

    async def _pump(self):
        try:
            while not self._closing:
                if not self._requests:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                for rid, state in list(self._requests.items()):
                    if state.cancelled:
                        if state.submitted:
                            await self._native(
                                self.engine.abort_request, state.native_id
                            )
                        self._requests.pop(rid, None)
                    elif not state.submitted:
                        try:
                            await self._native(self._add_native, state)
                            state.submitted = True
                        except InvalidArgument as exc:
                            self._fail(state, exc)
                            self._requests.pop(rid, None)
                if not self._requests or self._closing:
                    continue
                outputs, unfinished = await self._native(self._advance)
                by_native = {s.native_id: (r, s) for r, s in self._requests.items()}
                for native_id, token, finished, reason, prompt, completion in outputs:
                    match = by_native.get(native_id)
                    if match is None:
                        continue
                    rid, state = match
                    if state.cancelled or self._requests.get(rid) is not state:
                        continue
                    if finished and reason not in ("stop", "length", "abort"):
                        self._fail(
                            state,
                            EngineShutdown(f"DeepInfer terminated request: {reason}"),
                        )
                        continue
                    chunk = {"token_ids": [token] if token >= 0 else [], "index": 0}
                    if finished:
                        chunk["finish_reason"] = (
                            "cancelled" if reason == "abort" else reason
                        )
                        chunk["completion_usage"] = {
                            "prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "total_tokens": prompt + completion,
                        }
                    try:
                        state.queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        self._fail(
                            state,
                            EngineShutdown("DeepInfer stream consumer is too slow"),
                        )
                        continue
                    if finished:
                        self._requests.pop(rid, None)
                if not unfinished:
                    for state in self._requests.values():
                        if state.submitted and not state.cancelled:
                            self._fail(
                                state,
                                EngineShutdown(
                                    "DeepInfer stream ended without a terminal output"
                                ),
                            )
        except Exception as exc:
            self._failure = exc
            for state in self._requests.values():
                self._fail(state, EngineShutdown(f"DeepInfer driver failed: {exc}"))

    async def cleanup(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._wake.set()
        for state in self._requests.values():
            self._fail(state, EngineShutdown("DeepInfer is shutting down"))
        try:
            if self._pump_task is not None:
                await self._pump_task
            if self.engine is not None:
                await self._native(self.engine.close)
        finally:
            self.engine = None
            self._requests.clear()
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._closed = True
