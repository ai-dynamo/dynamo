# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepInfer LLMEngine implementation (experimental, aggregated serving only).

DeepInfer ships offline inference: a blocking ``LLM.generate()`` and a
non-thread-safe ``step()`` loop. This engine builds the DeepInfer ``LLM`` for
its engine object only and drives that object through
:class:`~dynamo.deepinfer.runner.EngineRunner`, which owns the native engine on a
single thread. ``generate()`` submits a request and streams the snapshots the
runner hands back through an ``asyncio.Queue``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
)
from dynamo.common.backend.health_check import build_health_check_payload
from dynamo.common.backend.logprobs import parse_logprob_options
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.token_ids import normalize_request_token_ids
from dynamo.deepinfer.args import Config, parse_args
from dynamo.deepinfer.runner import EngineRunner, RunnerClosed, TokenOutput
from dynamo.deepinfer.sampling import build_sampling_kwargs, validate_prompt
from dynamo.llm import ModelInput
from dynamo.llm.exceptions import EngineShutdown, InvalidArgument, Unknown

logger = logging.getLogger(__name__)

# Environment DeepInfer's own offline example sets before importing the engine.
# Operators can still override any of them from the process environment.
_ENGINE_ENV_DEFAULTS = {
    "DEEPINFER_USE_NEXTN_DECODE": "5",
    "DEEPINFER_SAMPLER_FORCE_TOPK": "4096",
    "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "32",
    "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE": "3",
    "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "30",
    "DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE": "1",
    "DEEPINFER_LOGGING_INTERVAL_STEP": "10",
}

# Native finish reasons that are not a normal completion.
_PROMPT_TOO_LONG = "prompt_length_exceeded"

_CANCELLED = object()
_CHUNK_INDEX = 0


def resolve_model_path(model: str) -> str:
    """Return a local checkpoint directory for a path or Hugging Face id."""
    if os.path.isdir(os.path.expanduser(model)):
        path = os.path.abspath(os.path.expanduser(model))
    else:
        from huggingface_hub import snapshot_download

        logger.info("Resolving %s from the Hugging Face Hub", model)
        path = snapshot_download(model)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"HF checkpoint requires config.json: {path}")
    return path


def read_bos_token_id(model_path: str, default: int = 1) -> int:
    """Return the checkpoint's ``bos_token_id`` (for the health-check prompt)."""
    with open(os.path.join(model_path, "config.json")) as f:
        bos = json.load(f).get("bos_token_id")
    return int(bos) if bos is not None else default


class DeepInferLLMEngine(LLMEngine):
    def __init__(self, config: Config):
        self.config = config
        self._runner: Optional[EngineRunner] = None
        self._sampling_params_cls: Any = None
        self._vocab_size: Optional[int] = None
        self._bos_token_id = 1
        # request id -> that request's queue; only touched on the event loop.
        self._streams: dict[str, asyncio.Queue] = {}
        self._request_counter = itertools.count()

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[DeepInferLLMEngine, WorkerConfig]:
        config = parse_args(argv)
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            # DeepInfer publishes no KV events or load metrics.
            enable_kv_routing=False,
        )
        return cls(config), worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        model_path = await asyncio.to_thread(resolve_model_path, self.config.model)
        self._bos_token_id = read_bos_token_id(model_path)
        runner = EngineRunner()
        # The native engine is built on the runner's own thread so that
        # construction and every later step share one thread.
        await runner.start(lambda: self._create_native_engine(model_path))
        self._runner = runner
        logger.info(
            "DeepInfer engine ready: model=%s ep=%d cp=%d",
            self.config.model,
            self.config.expert_parallel_size,
            self.config.context_parallel_size,
        )
        return EngineConfig(
            model=self.config.model,
            served_model_name=self.config.served_model_name,
            llm=LlmRegistration(
                context_length=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                data_parallel_size=1,
            ),
        )

    def _create_native_engine(self, model_path: str) -> tuple[Any, Any]:
        """Construct the native engine and return ``(engine, clone)``.

        Runs on the runner's worker thread: it blocks for the model load, and
        the environment must be set before DeepInfer (and torch) are imported.
        """
        config = self.config
        for key, value in _ENGINE_ENV_DEFAULTS.items():
            os.environ.setdefault(key, value)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(config.device_offset + i) for i in range(config.context_parallel_size)
        )

        from deepinfer._C.engine import RequestOutput  # type: ignore
        from deepinfer.entrypoints.llm import LLM  # type: ignore
        from deepinfer.sampling_params import SamplingParams  # type: ignore

        llm = LLM(
            model=model_path,
            expert_parallel_size=config.expert_parallel_size,
            context_parallel_size=config.context_parallel_size,
            engine_rank=0,
            data_parallel_size=1,
            trust_remote_code=True,
            chunked_prefill=True,
            decode_sequence=False,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            lookahead_tokens=min(1, int(os.environ["DEEPINFER_USE_NEXTN_DECODE"])),
        )
        self._sampling_params_cls = SamplingParams
        self._vocab_size = llm.llm_engine.model_config.get_vocab_size()
        return llm.llm_engine, RequestOutput.clone

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        runner = self._runner
        if runner is None:
            raise EngineShutdown("DeepInfer engine is not running")
        normalize_request_token_ids(request)
        token_ids = request.get("token_ids") or []
        validate_prompt(
            token_ids,
            vocab_size=self._vocab_size,
            max_model_len=self.config.max_model_len,
        )

        try:
            params = self._sampling_params_cls(
                **build_sampling_kwargs(
                    request,
                    model_max_len=self.config.max_model_len,
                )
            )
        except ValueError as exc:
            raise InvalidArgument(str(exc)) from exc

        request_id = self._request_id(context)

        import numpy as np

        finished = False
        try:
            try:
                results = runner.submit(
                    request_id, np.asarray(token_ids, dtype=np.int32), params
                )
            except RunnerClosed as exc:
                raise EngineShutdown(str(exc)) from exc
            except ValueError as exc:  # duplicate request id
                raise InvalidArgument(str(exc)) from exc
            self._streams[request_id] = results
            logprobs, _ = parse_logprob_options(request.get("output_options") or {})
            completion_tokens = 0
            while True:
                item = await results.get()
                if item is _CANCELLED:
                    finished = True  # abort() already told the engine
                    yield self._terminal_chunk(
                        "cancelled", len(token_ids), completion_tokens
                    )
                    return
                if isinstance(item, BaseException):
                    finished = True
                    raise self._to_dynamo_error(item)
                completion_tokens += len(item)
                chunk, finished = self._to_chunk(
                    item, len(token_ids), completion_tokens, logprobs
                )
                yield chunk
                if finished:
                    return
        finally:
            self._streams.pop(request_id, None)
            if not finished:
                # Stream closed early (client went away): free the native slot.
                runner.abort(request_id)

    def _to_chunk(
        self,
        batch: list[TokenOutput],
        prompt_tokens: int,
        completion_tokens: int,
        logprobs: Optional[int],
    ) -> tuple[GenerateChunk, bool]:
        last = batch[-1]
        token_ids = [out.token_id for out in batch]
        chunk: GenerateChunk = {"token_ids": token_ids, "index": _CHUNK_INDEX}
        if logprobs is not None:
            self._add_logprobs(chunk, batch, logprobs)
        if not last.finished:
            return chunk, False
        if last.finish_reason == _PROMPT_TOO_LONG:
            raise InvalidArgument(
                f"prompt of {prompt_tokens} tokens exceeds the model context "
                f"length ({self.config.max_model_len})"
            )
        if last.finish_reason not in ("stop", "length") and not (
            last.finish_reason.startswith("abort")
        ):
            raise Unknown(f"DeepInfer finished the request: {last.finish_reason}")
        chunk["finish_reason"] = normalize_finish_reason(last.finish_reason)
        chunk["completion_usage"] = self._usage(prompt_tokens, completion_tokens)
        return chunk, True

    @staticmethod
    def _add_logprobs(
        chunk: GenerateChunk, batch: list[TokenOutput], logprobs: int
    ) -> None:
        # The response builder zips these against token_ids by position, so a
        # partial array would misalign every later token: emit all or nothing.
        if not batch or any(out.logprob is None for out in batch):
            return
        chunk["log_probs"] = [out.logprob for out in batch]  # type: ignore[misc]
        if logprobs > 0:
            chunk["top_logprobs"] = [
                [
                    {
                        "rank": rank,
                        "token_id": token_id,
                        "token": f"token_id:{token_id}",
                        "logprob": logprob,
                    }
                    for rank, (token_id, logprob) in enumerate(
                        zip(out.top_token_ids, out.top_logprobs), start=1
                    )
                ]
                for out in batch
            ]

    @staticmethod
    def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _terminal_chunk(
        self, finish_reason: str, prompt_tokens: int, completion_tokens: int
    ) -> GenerateChunk:
        return {
            "token_ids": [],
            "index": _CHUNK_INDEX,
            "finish_reason": finish_reason,
            "completion_usage": self._usage(prompt_tokens, completion_tokens),
        }

    @staticmethod
    def _to_dynamo_error(exc: BaseException) -> Exception:
        if isinstance(exc, (InvalidArgument, EngineShutdown)):
            return exc
        if isinstance(exc, RunnerClosed):
            return EngineShutdown(str(exc))
        if isinstance(exc, ValueError):
            return InvalidArgument(str(exc))
        return Unknown(f"DeepInfer engine error: {exc!r}")

    def _request_id(self, context: Context) -> str:
        context_id = context.id()
        suffix = context_id if context_id is not None else next(self._request_counter)
        return f"dyn-{suffix}"

    async def abort(self, context: Context) -> None:
        if context.id() is None:
            return
        request_id = self._request_id(context)
        stream = self._streams.get(request_id)
        if stream is None or self._runner is None:
            return
        self._runner.abort(request_id)
        stream.put_nowait(_CANCELLED)
        logger.debug("Aborted DeepInfer request %s", request_id)

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        return build_health_check_payload(bos_token_id=self._bos_token_id)

    async def cleanup(self) -> None:
        runner, self._runner = self._runner, None
        if runner is None:
            return
        # Lets the in-flight native step finish before the engine is closed.
        if not await runner.stop(60.0):
            logger.warning("DeepInfer engine did not stop within 60s")
        logger.info("DeepInfer engine shutdown")
