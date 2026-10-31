# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``DeepInferEngine``: DeepInfer DP ranks behind one Dynamo unified-backend worker.

``from_args`` spawns this node's ranks 1..R-1 as child processes and builds
rank 0 on the main thread before the Rust Worker (and its Tokio threads)
exists. On an engine that spans nodes it also accepts the other nodes' ranks
on ``--control-port``. ``start`` launches the controller thread; ``generate``
feeds it and streams the outputs it dispatches back onto the event loop.

With ``--no-rank0-in-process`` the controller and every rank of this node run
in a process of their own instead (``controller_process``), started at the
same point, so this process's GIL only serves requests and outputs.
``DYN_DEEPINFER_CONTROLLER_PROCESS=0`` keeps the controller on a thread here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
import weakref
from collections.abc import AsyncGenerator
from typing import Any, Optional, Union

from dynamo._core import Context
from dynamo.common.backend.disagg import (
    enforce_prefill_max_tokens,
    require_prefill_result,
)
from dynamo.common.backend.dp_rank import forced_dp_rank, validate_global_dp_rank
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
)
from dynamo.common.backend.health_check import build_health_check_payload, is_probe
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, ZmqSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.token_ids import token_ids_len
from dynamo.deepinfer.args import DeepInferArgs, apply_env_defaults, parse_args
from dynamo.deepinfer.controller import Controller
from dynamo.deepinfer.controller_process import ControllerProcess
from dynamo.deepinfer.disagg import HANDOFF_KEY, handoff_from_prefill_result
from dynamo.deepinfer.launch import (  # noqa: F401 (ChildWatch: re-exported)
    ChildWatch,
    Launch,
    launch_ranks,
)
from dynamo.deepinfer.protocol import AddReq, EngineMeta, Output, RankStats
from dynamo.deepinfer.request import build_sampling_params, prompt_bytes
from dynamo.llm import ModelInput
from dynamo.llm.exceptions import InvalidArgument

logger = logging.getLogger(__name__)


def _controller_process() -> bool:
    return os.environ.get("DYN_DEEPINFER_CONTROLLER_PROCESS", "1") not in (
        "0",
        "",
        "false",
        "False",
    )


def _shutdown_ranks(
    controller: Union[Controller, ControllerProcess], launch: Optional[Launch]
) -> None:
    """Interpreter-exit teardown for an engine whose ``start()`` never ran
    (``cleanup()`` is not called then): one CLOSE iteration on every rank, so
    no rank is left in DeepInfer's ``close()`` barrier."""
    try:
        controller.close()
    finally:
        if launch is not None:
            launch.close()


def _usage(
    prompt_tokens: int, completion_tokens: int, cached_tokens: Optional[int] = None
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


class DeepInferEngine(LLMEngine):
    def __init__(
        self,
        args: DeepInferArgs,
        controller: Union[Controller, ControllerProcess],
        meta: EngineMeta,
        launch: Optional[Launch] = None,
    ) -> None:
        self.args = args
        self.controller = controller
        self.meta = meta
        self.mode = args.disaggregation_mode
        self.dp_size = controller.dp_size
        self._launch = launch
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._streams: dict[str, asyncio.Queue[Output]] = {}
        # Aborts that arrived before their generate() registered a stream.
        self._early_aborts: set[str] = set()
        self._publisher: Any = None
        self._closed = False
        self._finalizer: Optional[weakref.finalize] = None
        controller.on_stats = self._publish_snapshot

    @classmethod
    async def from_args(
        cls, argv: Optional[list[str]] = None
    ) -> tuple[DeepInferEngine, WorkerConfig]:
        args = parse_args(argv)
        apply_env_defaults(args)
        controller: Union[Controller, ControllerProcess]
        if args.rank0_in_process or not _controller_process():
            launch = launch_ranks(args)
            controller = Controller(
                launch.handles, launch.meta, engine_type=args.engine_type
            )
        else:
            controller = ControllerProcess.spawn(
                sys.argv[1:] if argv is None else argv, args.startup_timeout_s
            )
            launch = Launch(
                [], controller.meta, controller.ipc_dir, controller.kv_endpoints
            )
        engine = cls(args, controller, launch.meta, launch)
        # Runs at interpreter exit unless cleanup() ran first.
        engine._finalizer = weakref.finalize(
            engine, _shutdown_ranks, controller, launch
        )
        worker_config = WorkerConfig.from_runtime_config(
            args,
            model_name=args.model,
            served_model_name=args.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        self._loop = asyncio.get_running_loop()
        self.controller.start(self._emit)
        meta = self.meta
        logger.info(
            "DeepInfer engine started (%s, dp_size=%d, nnodes=%d)",
            self.mode.value,
            self.dp_size,
            self.args.nnodes,
        )
        return EngineConfig(
            model=self.args.model,
            served_model_name=self.args.served_model_name,
            llm=LlmRegistration(
                context_length=meta.max_model_len,
                kv_cache_block_size=meta.block_size,
                total_kv_blocks=meta.num_gpu_blocks,
                max_num_seqs=meta.max_num_seqs,
                max_num_batched_tokens=meta.max_num_batched_tokens,
                data_parallel_size=self.dp_size,
                data_parallel_start_rank=0,
            ),
        )

    # -- controller thread -> event loop --------------------------------------

    def _emit(self, outputs: dict[str, Output]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._dispatch, outputs)
        except RuntimeError:
            pass  # the loop closed between the check and the call

    def _dispatch(self, outputs: dict[str, Output]) -> None:
        for rid, out in outputs.items():
            stream = self._streams.get(rid)
            if stream is not None:
                stream.put_nowait(out)
            else:
                self._release_undelivered(out)

    def _release_undelivered(self, out: Output) -> None:
        """A prefill handoff that never reached the caller pins KV until its
        lease expires; no decode peer can read it, so free it now."""
        handoff = (out.disaggregated_params or {}).get(HANDOFF_KEY)
        if isinstance(handoff, dict):
            logger.debug("Releasing undelivered KV handoff of %s", out.request_id)
            self.controller.release(handoff, int(handoff.get("dp_rank", 0)))

    def _publish_snapshot(self, rank: int, stats: RankStats) -> None:
        if self._publisher is None:
            return
        total = stats.kv_total_blocks or self.meta.num_gpu_blocks
        self._publisher.publish(
            rank,
            ComponentSnapshot(
                kv_used_blocks=stats.kv_used_blocks,
                kv_total_blocks=total,
                gpu_cache_usage=stats.kv_used_blocks / total if total else 0.0,
                dp_rank=rank,
                kv_cache_hit_rate=stats.kv_hit_rate,
            ),
        )

    # -- requests ---------------------------------------------------------------

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self._closed or self._loop is None:
            raise RuntimeError("DeepInfer engine is not running")
        # A list, or with DYN_TOKEN_IDS_AS_BYTES on the frontend one packed int32
        # buffer that goes to the ranks as is (no Python int per token).
        token_ids = request.get("token_ids")
        if token_ids is None:
            token_ids = []
        probe = is_probe(request)
        if self.mode == DisaggregationMode.DECODE and probe:
            # A decode-only engine cannot prefill; answer the canary directly.
            yield {
                "token_ids": [],
                "index": 0,
                "finish_reason": "stop",
                "completion_usage": _usage(token_ids_len(token_ids), 0),
            }
            return
        if self.mode == DisaggregationMode.PREFILL:
            enforce_prefill_max_tokens(request)
        rid = context.id() or uuid.uuid4().hex
        prompt_len = token_ids_len(token_ids)
        handoff = None
        dp_rank = None
        # Decode reports the prefill side's cache hit (vLLM convention).
        prefill_cached = (
            ((request.get("prefill_result") or {}).get("prompt_tokens_details") or {})
        ).get("cached_tokens")
        try:
            if self.mode == DisaggregationMode.DECODE:
                handoff = handoff_from_prefill_result(
                    require_prefill_result(request, self.mode)
                )
            sampling, logprobs = build_sampling_params(request, self.meta.max_model_len)
            dp_rank = validate_global_dp_rank(
                forced_dp_rank(request), 0, self.dp_size, "DeepInfer"
            )
            add = AddReq(
                request_id=rid,
                token_ids=prompt_bytes(token_ids),
                sampling=sampling,
                handoff=handoff,
                hold_kv=self.mode == DisaggregationMode.PREFILL and not probe,
            )
        except (ValueError, TypeError, OverflowError) as e:
            self._release_handoff(handoff, dp_rank)
            raise InvalidArgument(str(e)) from e

        if rid in self._early_aborts:
            self._early_aborts.discard(rid)
            self._release_handoff(handoff, dp_rank)
            yield {
                "token_ids": [],
                "index": 0,
                "finish_reason": "cancelled",
                "completion_usage": _usage(prompt_len, 0),
            }
            return

        # A prefill worker answers with exactly one (terminal) chunk.
        buffer_until_finish = self.mode == DisaggregationMode.PREFILL
        pending = _Pending(logprobs)
        completion_tokens = 0
        finished = False
        stream: asyncio.Queue[Output] = asyncio.Queue()
        self._streams[rid] = stream
        try:
            self.controller.submit(add, dp_rank)
            while not finished:
                pending.add(await stream.get())
                while not stream.empty() and pending.finish is None:
                    pending.add(stream.get_nowait())
                finished = pending.finish is not None
                if not finished and (buffer_until_finish or not pending.token_ids):
                    continue
                chunk = pending.take()
                completion_tokens += len(chunk["token_ids"])
                if finished:
                    finish = pending.finish
                    if (
                        buffer_until_finish
                        and not probe
                        and finish == "length"
                        and pending.params is None
                    ):
                        logger.error("Prefill of %s finished without a KV handoff", rid)
                        finish = "error"
                    chunk["finish_reason"] = finish
                    chunk["completion_usage"] = _usage(
                        prompt_len,
                        completion_tokens,
                        pending.cached_tokens
                        if pending.cached_tokens is not None
                        else prefill_cached,
                    )
                    if pending.params is not None:
                        chunk["disaggregated_params"] = pending.params
                yield chunk
        finally:
            self._streams.pop(rid, None)
            # e.g. a handoff queued behind a "cancelled" from abort().
            while not stream.empty():
                self._release_undelivered(stream.get_nowait())
            if not finished:
                self.controller.abort(rid)

    def _release_handoff(
        self, handoff: Optional[dict[str, Any]], dp_rank: Optional[int]
    ) -> None:
        """Decode role: a handoff this worker will not read; tell its prefill
        peer through one of our ranks."""
        if handoff is not None:
            self.controller.release(handoff, dp_rank or 0)

    async def abort(self, context: Context) -> None:
        rid = context.id()
        if rid is None:
            return
        stream = self._streams.get(rid)
        if stream is None:
            if len(self._early_aborts) > 10_000:
                self._early_aborts.clear()
            self._early_aborts.add(rid)
            return
        # A running DeepInfer request emits nothing when aborted: end the stream here.
        stream.put_nowait(Output(request_id=rid, finish_reason="cancelled"))
        self.controller.abort(rid)
        logger.debug("Aborted DeepInfer request %s", rid)

    # -- lifecycle --------------------------------------------------------------

    async def is_quiescent(self) -> Optional[bool]:
        if self.mode != DisaggregationMode.PREFILL:
            return None
        held = sum(s.num_held for s in self.controller.stats())
        return held == 0 and not self._streams

    async def kv_event_sources(self) -> list[KvEventSource]:
        if self.args.kv_cache_manager == "none":
            return []
        remote = self._launch.kv_endpoints if self._launch is not None else {}
        return [
            ZmqSource(
                endpoint=remote.get(rank) or self.args.kv_event_endpoint(rank),
                topic="",
                dp_rank=rank,
            )
            for rank in range(self.dp_size)
        ]

    def component_metrics_dp_ranks(self) -> list[int]:
        return list(range(self.dp_size))

    def attach_snapshot_publisher(self, publisher: Any) -> None:
        self._publisher = publisher

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        if self.mode == DisaggregationMode.DECODE:
            return None
        return build_health_check_payload(bos_token_id=self.meta.bos_token_id)

    async def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._finalizer is not None:
            self._finalizer.detach()
        try:
            await asyncio.to_thread(self.controller.close)
        finally:
            for rid, stream in self._streams.items():
                stream.put_nowait(Output(request_id=rid, finish_reason="cancelled"))
            if self._launch is not None:
                self._launch.close()
            logger.info("DeepInfer engine shutdown")


class _Pending:
    """Outputs of one request accumulated between yields."""

    def __init__(self, logprobs: Optional[int]) -> None:
        self.logprobs = logprobs
        self.finish: Optional[str] = None
        self.params: Optional[dict[str, Any]] = None
        self.cached_tokens: Optional[int] = None
        self._reset()

    def _reset(self) -> None:
        self.token_ids: list[int] = []
        self._lps: list[float] = []
        self._top: list[list[dict[str, Any]]] = []
        # The decode side's first token comes from the prefill handoff, without
        # logprobs; a chunk whose logprobs don't line up carries none.
        self._aligned = True

    def add(self, out: Output) -> None:
        if self.finish is not None:
            return
        self.token_ids.extend(out.token_ids)
        if len(out.logprobs) != len(out.token_ids):
            self._aligned = False
        self._lps.extend(out.logprobs)
        for ids, lps in zip(out.top_token_ids, out.top_logprobs):
            self._top.append(
                [
                    {"rank": i + 1, "token_id": t, "token": None, "logprob": lp}
                    for i, (t, lp) in enumerate(zip(ids, lps))
                ]
            )
        if out.disaggregated_params is not None:
            self.params = out.disaggregated_params
        if out.cached_tokens is not None:
            self.cached_tokens = out.cached_tokens
        self.finish = out.finish_reason

    def take(self) -> GenerateChunk:
        chunk: GenerateChunk = {"token_ids": self.token_ids, "index": 0}
        if self.logprobs is not None and self.token_ids and self._aligned:
            chunk["log_probs"] = self._lps
            if self.logprobs > 0 and len(self._top) == len(self.token_ids):
                chunk["top_logprobs"] = self._top
        self._reset()
        return chunk
