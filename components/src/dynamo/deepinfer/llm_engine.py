# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed-style LLMEngine adapter over DeepInfer's step-based interface."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path
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
from dynamo.deepinfer.args import Config, parse_args
from dynamo.deepinfer.mock_engine import MockDeepInfer
from dynamo.llm import ModelInput
from dynamo.llm.exceptions import InvalidArgument

logger = logging.getLogger(__name__)


class DeepInferLLMEngine(LLMEngine):
    def __init__(self, config: Config):
        self.config = config
        self.engine: Any = None
        self.executor: ThreadPoolExecutor | None = None
        self.driver: asyncio.Task | None = None
        self.wake = asyncio.Event()
        self.streams: dict[str, asyncio.Queue] = {}
        self.failure: Exception | None = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[DeepInferLLMEngine, WorkerConfig]:
        config = parse_args(argv)
        return cls(config), WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            enable_kv_routing=False,
            enable_local_indexer=False,
        )

    async def _on_owner(self, fn, *args, **kwargs):
        assert self.executor is not None
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, partial(fn, *args, **kwargs)
        )

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="deepinfer"
        )
        try:
            self.engine = await self._on_owner(create_engine, self.config)
        except Exception:
            logger.exception("DeepInfer engine initialization failed")
            raise
        ready_dir = os.environ.get("DYN_DEEPINFER_READY_DIR")
        if ready_dir and self.config.rank_worker:
            rank = self.config.engine_args.get("engine_rank", 0)
            Path(ready_dir, str(rank)).touch()
            ranks = range(self.config.engine_args.get("data_parallel_size", 1))
            while not all(Path(ready_dir, str(peer)).exists() for peer in ranks):
                await asyncio.sleep(0.01)
        self.driver = asyncio.create_task(self._drive())
        return EngineConfig(
            model=self.config.model,
            served_model_name=self.config.served_model_name,
            llm=LlmRegistration(context_length=self.config.max_model_len),
        )

    async def _drive(self) -> None:
        assert self.engine is not None
        distributed = self.config.engine_args.get("data_parallel_size", 1) > 1
        try:
            while True:
                # Empty DP ranks must still enter the native work-vote/step collectives.
                if not distributed:
                    await self.wake.wait()
                outputs = await self._on_owner(self.engine.step)
                for output in outputs:
                    stream = self.streams.get(output.request_id)
                    if stream is None:
                        continue
                    if output.finished and output.finish_reason not in (
                        "stop",
                        "length",
                    ):
                        stream.put_nowait(
                            RuntimeError(
                                f"DeepInfer request failed: {output.finish_reason}"
                            )
                        )
                        continue
                    chunk: GenerateChunk = {
                        "index": 0,
                        "token_ids": [output.token_id] if output.token_id >= 0 else [],
                    }
                    if output.finished:
                        chunk["finish_reason"] = output.finish_reason
                        chunk["completion_usage"] = {
                            "prompt_tokens": output.prompt_len,
                            "completion_tokens": output.num_output_tokens,
                            "total_tokens": output.prompt_len
                            + output.num_output_tokens,
                        }
                    stream.put_nowait(chunk)
                if not self.streams:
                    self.wake.clear()
                delay = self.config.mock_token_delay if self.config.mock else 0
                if distributed and not self.engine.has_work:
                    delay = 0.005
                await asyncio.sleep(delay)
        except Exception as error:
            self.failure = error
            for stream in self.streams.values():
                stream.put_nowait(error)
            if distributed:
                logger.exception("DeepInfer rank failed; stopping its collective group")
                os._exit(1)
            raise

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine is not None
        if self.failure is not None:
            raise self.failure
        params = build_sampling_params(request, self.config.max_model_len)
        request_id = context.id()
        if request_id is None or request_id in self.streams:
            raise InvalidArgument("DeepInfer requires a unique request context ID")
        stream: asyncio.Queue = asyncio.Queue()
        self.streams[request_id] = stream
        try:
            await self._on_owner(
                self.engine.add_request,
                user_id="dynamo",
                request_id=request_id,
                sampling_params=params,
                prompt_token_ids=np.asarray(request["token_ids"], dtype=np.int32),
            )
            self.wake.set()
            while True:
                chunk = await stream.get()
                if chunk is None:
                    return
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
                if "finish_reason" in chunk:
                    return
        finally:
            await self._on_owner(self.engine.abort_request, request_id)
            self.streams.pop(request_id, None)

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if self.engine is not None and request_id is not None:
            await self._on_owner(self.engine.abort_request, request_id)
            stream = self.streams.get(request_id)
            if stream is not None:
                stream.put_nowait(None)

    async def cleanup(self) -> None:
        try:
            if self.driver is not None:
                self.driver.cancel()
                with suppress(asyncio.CancelledError):
                    await self.driver
        finally:
            self.driver = None
            try:
                if self.engine is not None:
                    await self._on_owner(self.engine.close)
            finally:
                if self.executor is not None:
                    self.executor.shutdown(wait=True, cancel_futures=True)
                    self.executor = None
                self.engine = None


def create_engine(config: Config):
    if config.mock:
        return MockDeepInfer(config.model)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES before starting native DeepInfer")
    # Text-only serving does not need DeepInfer's 8 GiB shared image pool per rank.
    os.environ.setdefault("DEEPINFER_USE_MULTIMODAL_POOL", "0")
    # Keep the optional CUDA engine and its dependencies out of mock mode.
    module = importlib.import_module("dynamo.deepinfer.native_engine")
    return module.NativeDeepInfer(config)


def build_sampling_params(request: GenerateRequest, context_length: int) -> dict:
    sampling = request.get("sampling_options") or {}
    stops = request.get("stop_conditions") or {}
    if sampling.get("n") is not None and sampling["n"] != 1:
        raise InvalidArgument("DeepInfer backend supports only n=1")
    if sampling.get("guided_decoding"):
        raise InvalidArgument(
            "DeepInfer backend does not implement constrained decoding"
        )
    if request.get("multi_modal_data"):
        raise InvalidArgument("DeepInfer backend supports text input only")
    remaining = context_length - len(request["token_ids"])
    if remaining <= 0:
        raise InvalidArgument("prompt exceeds DeepInfer context length")
    maximum = stops.get("max_tokens")
    maximum = min(remaining, maximum if maximum is not None else 512)
    if maximum <= 0:
        raise InvalidArgument("max_tokens must be positive")
    params = {
        key: sampling[key]
        for key in ("temperature", "top_p", "top_k")
        if sampling.get(key) is not None
    }
    if sampling.get("seed") is not None:
        params["seq_seed"] = sampling["seed"]
    params["max_tokens"] = maximum
    params["ignore_eos"] = stops.get("ignore_eos", False)
    return params
