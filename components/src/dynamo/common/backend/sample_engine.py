# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import queue
import threading
import uuid
from collections.abc import AsyncGenerator
from typing import Optional

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import KvEventPublisher

from .disagg import enforce_prefill_max_tokens, require_prefill_result
from .engine import EngineConfig, GenerateChunk, GenerateRequest, LLMEngine
from .publisher import KvEventSource, Metrics, PushSource, SnapshotSource
from .worker import WorkerConfig

logger = logging.getLogger(__name__)

_SAMPLE_BLOCK_SIZE = 16


class SampleLLMEngine(LLMEngine):
    """Reference LLMEngine implementation.

    Generates rotating token IDs with configurable per-token latency.
    Useful for testing the Worker lifecycle end-to-end and as a template
    for engine leads implementing real backends.

    Disaggregation:
        ``--disaggregation-mode {agg,prefill,decode}`` selects the role.
        AGGREGATED is the default and produces ``max_tokens`` rotating
        tokens. PREFILL caps generation at one token and stamps a
        synthetic ``disaggregated_params`` payload on the terminal so
        the frontend's PrefillRouter has something to forward. DECODE
        requires the request to carry ``prefill_result`` (otherwise the
        frontend forgot to route through the prefill peer); on success
        it generates normally.
    """

    def __init__(
        self,
        model_name: str = "sample-model",
        max_tokens: int = 16,
        delay: float = 0.01,
        disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.delay = delay
        self.disaggregation_mode = disaggregation_mode
        self._metrics = Metrics(kv_used_blocks=0)
        self._publish_queue: queue.SimpleQueue[tuple[str, dict]] = queue.SimpleQueue()
        self._publish_stop = threading.Event()
        self._publish_thread: Optional[threading.Thread] = None
        # itertools.count is thread-safe — concurrent generate() calls
        # won't race on hash issuance.
        self._block_hash_counter = itertools.count(1)

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SampleLLMEngine, WorkerConfig]:
        parser = argparse.ArgumentParser(description="Sample Dynamo backend")
        parser.add_argument("--model-name", default="sample-model")
        parser.add_argument("--namespace", default="dynamo")
        parser.add_argument("--component", default="sample")
        parser.add_argument("--endpoint", default="generate")
        parser.add_argument("--max-tokens", type=int, default=16)
        parser.add_argument("--delay", type=float, default=0.01)
        parser.add_argument("--endpoint-types", default="chat,completions")
        parser.add_argument("--discovery-backend", default="etcd")
        parser.add_argument("--request-plane", default="tcp")
        parser.add_argument("--event-plane", default=None)
        parser.add_argument(
            "--disaggregation-mode",
            choices=[
                m.value for m in DisaggregationMode if m != DisaggregationMode.ENCODE
            ],
            default=DisaggregationMode.AGGREGATED.value,
            help="Disaggregation role: 'agg' (default), 'prefill', or 'decode'.",
        )
        args = parser.parse_args(argv)

        mode = DisaggregationMode(args.disaggregation_mode)
        engine = cls(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            delay=args.delay,
            disaggregation_mode=mode,
        )
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.model_name,
            served_model_name=args.model_name,
            endpoint_types=args.endpoint_types,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
            disaggregation_mode=mode,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.model_name,
            context_length=2048,
            kv_cache_block_size=_SAMPLE_BLOCK_SIZE,
            total_kv_blocks=1000,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
        )

    async def kv_event_sources(self) -> list[KvEventSource]:
        return [PushSource(on_ready=self._start_publisher_thread, dp_rank=0)]

    async def metrics_sources(self) -> list[SnapshotSource]:
        return [SnapshotSource(snapshot=lambda: self._metrics, dp_rank=0)]

    def _start_publisher_thread(self, publisher: KvEventPublisher) -> None:
        self._publish_thread = threading.Thread(
            target=self._publish_loop,
            args=(publisher,),
            daemon=True,
            name="sample-kv-publisher",
        )
        self._publish_thread.start()

    def _publish_loop(self, publisher: KvEventPublisher) -> None:
        while not self._publish_stop.is_set():
            try:
                kind, payload = self._publish_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if kind == "stored":
                    publisher.publish_stored(**payload)
                elif kind == "removed":
                    publisher.publish_removed(**payload)
            except Exception:
                logger.exception("Sample publisher dropped event of kind=%s", kind)

    def _emit_synthetic_events(self, prompt_len: int) -> list[int]:
        block_count = max(1, prompt_len // _SAMPLE_BLOCK_SIZE)
        hashes: list[int] = []
        for _ in range(block_count):
            h = next(self._block_hash_counter)
            hashes.append(h)
        self._publish_queue.put(
            (
                "stored",
                {
                    "token_ids": list(range(block_count * _SAMPLE_BLOCK_SIZE)),
                    "num_block_tokens": [_SAMPLE_BLOCK_SIZE] * block_count,
                    "block_hashes": hashes,
                },
            )
        )
        self._metrics = Metrics(
            kv_used_blocks=(self._metrics.kv_used_blocks or 0) + block_count
        )
        return hashes

    def _release_synthetic_blocks(self, hashes: list[int]) -> None:
        self._publish_queue.put(("removed", {"block_hashes": hashes}))
        self._metrics = Metrics(
            kv_used_blocks=max(0, (self._metrics.kv_used_blocks or 0) - len(hashes))
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            require_prefill_result(request, self.disaggregation_mode)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            enforce_prefill_max_tokens(request)

        token_ids = request.get("token_ids", [])
        prompt_len = len(token_ids)
        stop_conditions = request.get("stop_conditions", {})
        max_new = stop_conditions.get("max_tokens") or self.max_tokens

        block_hashes = self._emit_synthetic_events(prompt_len)
        try:
            async for chunk in self._generate_tokens(prompt_len, max_new, context):
                yield chunk
        finally:
            self._release_synthetic_blocks(block_hashes)

    async def _generate_tokens(
        self, prompt_len: int, max_new: int, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        for i in range(max_new):
            if context.is_stopped():
                yield {
                    "token_ids": [],
                    "index": 0,
                    "finish_reason": "cancelled",
                    "completion_usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": i,
                        "total_tokens": prompt_len + i,
                    },
                }
                break
            await asyncio.sleep(self.delay)
            token_id = (i + 1) % 32000
            out: GenerateChunk = {"token_ids": [token_id], "index": 0}
            if i == max_new - 1:
                out["finish_reason"] = "length"
                out["completion_usage"] = {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": max_new,
                    "total_tokens": prompt_len + max_new,
                }
                if self.disaggregation_mode == DisaggregationMode.PREFILL:
                    out["disaggregated_params"] = {
                        "sample_handle": uuid.uuid4().hex,
                        "completed_tokens": [token_id],
                    }
            yield out

    async def cleanup(self) -> None:
        self._publish_stop.set()
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=1.0)
