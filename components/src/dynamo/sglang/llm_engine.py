# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang LLMEngine implementation for the unified backend.

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
from collections.abc import AsyncGenerator, Callable
from typing import Any, Optional

import sglang as sgl
import zmq
import zmq.asyncio
from sglang.srt.disaggregation.kv_events import ZmqEventPublisher
from sglang.srt.utils.network import get_local_ip_auto, get_zmq_socket

from dynamo._core import Context
from dynamo.common.backend.dp_rank import forced_dp_rank, validate_global_dp_rank
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.publisher import (
    KvEventSource,
    Metrics,
    SnapshotSource,
    ZmqSource,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import ModelInput
from dynamo.sglang._compat import get_scheduler_info
from dynamo.sglang._disagg import (
    SGLANG_WORKER_GROUP_ID_KEY,
    compute_bootstrap_address,
    get_sglang_worker_group_id,
    warmup_prefill_engine,
)
from dynamo.sglang.args import parse_args
from dynamo.sglang.publisher import format_zmq_endpoint

logger = logging.getLogger(__name__)

# Bound on prefill drain during graceful shutdown. After this, force-cancel
# any still-running consume tasks. Matches TRT-LLM's drain timeout.
_PREFILL_DRAIN_TIMEOUT_S = 30.0

# Operators can opt out of the prefill warmup for fast-iteration / smoke
# environments where the warmup adds avoidable startup latency. The default
# (`0`/unset) keeps warmup on; set to `1`/`true` to skip.
_DYN_SGLANG_SKIP_WARMUP_ENV = "DYN_SGLANG_SKIP_PREFILL_WARMUP"


def _warmup_enabled() -> bool:
    raw = os.environ.get(_DYN_SGLANG_SKIP_WARMUP_ENV, "")
    return raw.strip().lower() not in ("1", "true", "yes", "on")


def _get_runtime_data(server_args) -> dict[str, Any] | None:
    worker_group_id = get_sglang_worker_group_id(server_args)
    if worker_group_id is None:
        return None
    return {SGLANG_WORKER_GROUP_ID_KEY: worker_group_id}


def _local_dp_rank_range(server_args) -> tuple[int, int]:
    """Per-node local-rank slice for this worker. Under DP attention each
    node owns ``dp_size // nnodes`` ranks starting at
    ``node_rank * local_dp_size``; otherwise rank 0 only."""
    dp_size = getattr(server_args, "dp_size", 1) or 1
    enable_dp_attention = getattr(server_args, "enable_dp_attention", False)
    nnodes = getattr(server_args, "nnodes", 1) or 1
    node_rank = getattr(server_args, "node_rank", 0) or 0
    if enable_dp_attention and dp_size > 1:
        local_dp_size = dp_size // nnodes if nnodes > 0 else dp_size
        start = node_rank * local_dp_size
        return start, start + local_dp_size
    return 0, 1


class SglangLLMEngine(LLMEngine):
    def __init__(self, server_args, dynamo_args, serving_mode: DisaggregationMode):
        self.server_args = server_args
        self.dynamo_args = dynamo_args
        # SGLang's local name for disaggregation_mode. Same enum.
        self.serving_mode = serving_mode
        self.engine: Any = None
        self._bootstrap_host: str | None = None
        self._bootstrap_port: int | None = None
        self._input_param_manager: InputParamManager | None = None
        self._skip_tokenizer_init = server_args.skip_tokenizer_init
        self._use_sglang_tokenizer = dynamo_args.use_sglang_tokenizer
        # Background drain tasks for prefill stream after the bootstrap
        # chunk yields (Completed path only). Cancelled in cleanup().
        self._prefill_consume_tasks: set[asyncio.Task[Any]] = set()
        # Populated by `_metrics_pull_loop`; read by `metrics_sources()`.
        self._latest_metrics: dict[int, Metrics] = {}
        self._metrics_task: Optional[asyncio.Task[None]] = None
        self._metrics_zmq_ctx: Optional[zmq.asyncio.Context] = None
        self._metrics_zmq_sock = None
        # Local DP-rank slice this worker owns; resolved in `start()`
        # via `_local_dp_rank_range`. Used to validate router-supplied
        # `dp_rank` against this node's range before forwarding to SGLang.
        self._dp_start: int = 0
        self._dp_size: int = 1

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SglangLLMEngine, WorkerConfig]:
        config = await parse_args(argv if argv is not None else sys.argv[1:])
        server_args = config.server_args
        dynamo_args = config.dynamo_args

        model_input = (
            ModelInput.Text if dynamo_args.use_sglang_tokenizer else ModelInput.Tokens
        )

        engine = cls(server_args, dynamo_args, config.serving_mode)
        worker_config = WorkerConfig.from_runtime_config(
            dynamo_args,
            model_name=server_args.model_path,
            served_model_name=server_args.served_model_name,
            model_input=model_input,
            disaggregation_mode=config.serving_mode,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id  # SGLang bootstrap uses host/port/room triples
        self.engine = sgl.Engine(server_args=self.server_args)

        tokenizer = (
            self.engine.tokenizer_manager.tokenizer
            if not self._skip_tokenizer_init
            else None
        )
        self._input_param_manager = InputParamManager(tokenizer)

        if self.serving_mode == DisaggregationMode.PREFILL:
            self._bootstrap_host, self._bootstrap_port = compute_bootstrap_address(
                self.engine
            )
            if self._bootstrap_host is None or self._bootstrap_port is None:
                raise RuntimeError(
                    "prefill worker could not resolve bootstrap host/port; "
                    "SGLang server_args.disaggregation_bootstrap_port is unset"
                )
            if _warmup_enabled():
                await warmup_prefill_engine(self.engine, self._bootstrap_port)
            else:
                logger.info(
                    "Skipping SGLang prefill warmup (%s set)",
                    _DYN_SGLANG_SKIP_WARMUP_ENV,
                )

        # Capacity fields — match register.py in the legacy path.
        total_kv_blocks = None
        scheduler_info = get_scheduler_info(self.engine)
        max_total_tokens = scheduler_info.get("max_total_num_tokens")
        page_size = self.server_args.page_size
        if max_total_tokens and page_size:
            total_kv_blocks = (max_total_tokens + page_size - 1) // page_size

        # Prefer max_prefill_tokens; fall back so planner has a signal.
        max_num_batched_tokens = (
            getattr(self.server_args, "max_prefill_tokens", None) or max_total_tokens
        )

        self._start_metrics_task()

        dp_start, dp_end = _local_dp_rank_range(self.server_args)
        self._dp_start = dp_start
        self._dp_size = dp_end - dp_start
        return EngineConfig(
            model=self.server_args.model_path,
            served_model_name=self.server_args.served_model_name,
            context_length=self.server_args.context_length,
            kv_cache_block_size=page_size,
            total_kv_blocks=total_kv_blocks,
            max_num_seqs=getattr(self.server_args, "max_running_requests", None),
            max_num_batched_tokens=max_num_batched_tokens,
            # Router needs the rank range to enumerate per-rank load.
            data_parallel_size=self._dp_size,
            data_parallel_start_rank=self._dp_start,
            # Prefill-only — drives PrefillRouter's Bootstrap path.
            bootstrap_host=self._bootstrap_host,
            bootstrap_port=self._bootstrap_port,
            runtime_data=_get_runtime_data(self.server_args),
        )

    def _kv_routing_enabled(self) -> bool:
        # Matches legacy `DynamoSglangPublisher.init_kv_event_publish` —
        # every node publishes its local ranks; no decode gate.
        return bool(getattr(self.server_args, "kv_events_config", None))

    def _is_metrics_leader(self) -> bool:
        # Scheduler-metrics PULL socket binds on the leader only.
        return (getattr(self.server_args, "node_rank", 0) or 0) == 0

    def _start_metrics_task(self) -> None:
        assert self.engine is not None, "Engine not initialized"
        if not (self._kv_routing_enabled() and self._is_metrics_leader()):
            return
        self._metrics_zmq_ctx = zmq.asyncio.Context()
        self._metrics_zmq_sock = get_zmq_socket(
            self._metrics_zmq_ctx,
            zmq.PULL,
            self.engine.port_args.metrics_ipc_name,
            True,
        )
        self._metrics_task = asyncio.create_task(
            self._metrics_pull_loop(), name="sglang-metrics-pull"
        )

    async def _metrics_pull_loop(self) -> None:
        sock = self._metrics_zmq_sock
        assert sock is not None
        while True:
            try:
                kv_metrics = await sock.recv_pyobj()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Log and exit; matches the legacy publisher's stance.
                logger.exception("SGLang metrics pull loop terminated")
                return
            dp_rank = (
                kv_metrics.data_parallel_rank
                if kv_metrics.data_parallel_rank is not None
                else 0
            )
            self._latest_metrics[dp_rank] = Metrics(
                kv_used_blocks=kv_metrics.kv_active_blocks,
            )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine is not None, "Engine not initialized"

        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)

        # SGLang disagg keys NIXL transport on a (host, port, room) triple
        # exchanged between prefill and decode peers.
        bootstrap_kwargs: dict[str, Any] = {}
        if self.serving_mode == DisaggregationMode.PREFILL:
            bootstrap_kwargs = self._resolve_prefill_bootstrap(request)
        elif self.serving_mode == DisaggregationMode.DECODE:
            bootstrap_kwargs = self._resolve_decode_bootstrap(request)

        # Honour the router's DP rank decision; without it SGLang picks
        # its own rank and KV events land on the wrong publisher.
        sgl_dp_rank = validate_global_dp_rank(
            forced_dp_rank(request),
            self._dp_start,
            self._dp_size,
            "SGLang",
        )

        stream = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            rid=context.trace_id,
            data_parallel_rank=sgl_dp_rank,
            **bootstrap_kwargs,
        )

        # ORDER MATTERS: async_generate must register the room (the await
        # above) before we yield the bootstrap chunk — otherwise the
        # decode peer can connect to a room that doesn't exist yet.
        if self.serving_mode == DisaggregationMode.PREFILL:
            yield {
                "token_ids": [],
                "index": 0,
                "disaggregated_params": dict(bootstrap_kwargs),
            }
            # Bootstrap path (router-populated bootstrap_info): drain
            # inline so cancellation propagates to engine.abort().
            # Completed path: router awaits our stream end before
            # forwarding to decode — a sync drain deadlocks, so spawn.
            if request.get("bootstrap_info"):
                await self._consume_prefill_stream(stream, context, context.trace_id)
                return
            task = asyncio.create_task(
                self._consume_prefill_stream(stream, context, context.trace_id)
            )
            self._prefill_consume_tasks.add(task)
            task.add_done_callback(self._prefill_consume_tasks.discard)
            return

        async for res in stream:
            # SGLang sets index when n>1; default to 0 otherwise.
            output_idx = res.get("index") or 0
            out: GenerateChunk = {"token_ids": [], "index": output_idx}
            meta_info = res["meta_info"]
            finish_reason = meta_info["finish_reason"]

            output_ids = res.get("output_ids", [])
            if not output_ids and not finish_reason:
                if context.is_stopped():
                    prompt_tokens = meta_info.get("prompt_tokens", 0)
                    completion_tokens = meta_info.get("completion_tokens", 0)
                    yield {
                        "token_ids": [],
                        "index": output_idx,
                        "finish_reason": "cancelled",
                        "completion_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                    break
                continue

            out["token_ids"] = output_ids

            if finish_reason:
                prompt_tokens = meta_info["prompt_tokens"]
                completion_tokens = meta_info["completion_tokens"]
                out["finish_reason"] = finish_reason["type"]
                out["completion_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

            if context.is_stopped():
                prompt_tokens = meta_info.get("prompt_tokens", 0)
                completion_tokens = meta_info.get("completion_tokens", 0)
                yield {
                    "token_ids": output_ids,
                    "index": output_idx,
                    "finish_reason": "cancelled",
                    "completion_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                break

            yield out

    async def abort(self, context: Context) -> None:
        rid = context.trace_id
        if self.engine is None or rid is None:
            return
        tokenizer_manager = getattr(self.engine, "tokenizer_manager", None)
        if tokenizer_manager is not None:
            tokenizer_manager.abort_request(rid=rid, abort_all=False)
            logger.debug("Aborted request %s", rid)

    async def drain(self) -> None:
        """Await background prefill consume tasks before cleanup (#7319)."""
        pending = [t for t in self._prefill_consume_tasks if not t.done()]
        if not pending:
            return
        logger.info(
            "Draining %d background prefill consume task(s) (timeout=%.1fs)",
            len(pending),
            _PREFILL_DRAIN_TIMEOUT_S,
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=_PREFILL_DRAIN_TIMEOUT_S,
            )
            logger.info("All prefill consume tasks drained")
        except asyncio.TimeoutError:
            logger.warning(
                "Drain timeout (%.1fs) reached; cleanup() will cancel "
                "remaining tasks — some NIXL transfers may not complete",
                _PREFILL_DRAIN_TIMEOUT_S,
            )

    async def cleanup(self) -> None:
        # Anything still running here either timed out in drain() or was
        # never drained (e.g. start failed). Force-cancel.
        for task in self._prefill_consume_tasks:
            if not task.done():
                task.cancel()
        self._prefill_consume_tasks.clear()

        if self._metrics_task is not None:
            self._metrics_task.cancel()
            # CancelledError is a BaseException (not Exception) in 3.8+; the
            # tuple catches both the expected cancel and any teardown error.
            try:
                await self._metrics_task
            except (asyncio.CancelledError, Exception):
                pass
            self._metrics_task = None
        if self._metrics_zmq_sock is not None:
            try:
                self._metrics_zmq_sock.close(linger=0)
            except Exception:
                pass
            self._metrics_zmq_sock = None
        if self._metrics_zmq_ctx is not None:
            try:
                self._metrics_zmq_ctx.term()
            except Exception:
                pass
            self._metrics_zmq_ctx = None
        if self.engine is not None:
            self.engine.shutdown()
            logger.info("SGLang engine shutdown")

    def _resolve_prefill_bootstrap(self, request: GenerateRequest) -> dict[str, Any]:
        """Pick the (host, port, room) triple this prefill request will use.

        Bootstrap path: router pre-populates ``request.bootstrap_info``
        and the same triple is on the decode side. Completed path: fall
        back to engine defaults + a locally generated room; the router
        forwards this triple via ``prefill_result.disaggregated_params``.

        Partial ``bootstrap_info`` is a router contract violation; we
        warn and fill the gaps so the request doesn't fail outright.
        """
        assert (
            self._bootstrap_host is not None and self._bootstrap_port is not None
        ), "prefill workers must resolve bootstrap host/port in start()"

        bootstrap_info_from_req = request.get("bootstrap_info") or {}
        if isinstance(bootstrap_info_from_req, dict) and bootstrap_info_from_req:
            missing = [
                k
                for k in ("bootstrap_host", "bootstrap_port", "bootstrap_room")
                if k not in bootstrap_info_from_req
            ]
            if missing:
                logger.warning(
                    "incomplete prefill bootstrap_info (missing %s); "
                    "filling from engine defaults — decode peer may not "
                    "find this room. PrefillRouter contract violation?",
                    missing,
                )
            host = bootstrap_info_from_req.get("bootstrap_host", self._bootstrap_host)
            port = bootstrap_info_from_req.get("bootstrap_port", self._bootstrap_port)
            room = bootstrap_info_from_req.get("bootstrap_room")
        else:
            host, port, room = self._bootstrap_host, self._bootstrap_port, None

        if room is None:
            room = random.randint(0, 2**63 - 1)
        return {
            "bootstrap_host": host,
            "bootstrap_port": port,
            "bootstrap_room": room,
        }

    @staticmethod
    def _resolve_decode_bootstrap(request: GenerateRequest) -> dict[str, Any]:
        """Read the triple from ``bootstrap_info`` (Bootstrap path) or
        ``prefill_result.disaggregated_params`` (Completed path)."""
        bootstrap_info: Any = request.get("bootstrap_info")
        if not bootstrap_info:
            prefill_result: Any = request.get("prefill_result")
            if prefill_result is not None:
                bootstrap_info = prefill_result.get("disaggregated_params")

        if not bootstrap_info:
            raise ValueError(
                "decode worker received request without bootstrap info "
                "from PrefillRouter (bootstrap_info or prefill_result)"
            )

        try:
            return {
                "bootstrap_host": bootstrap_info["bootstrap_host"],
                "bootstrap_port": bootstrap_info["bootstrap_port"],
                "bootstrap_room": bootstrap_info["bootstrap_room"],
            }
        except KeyError as e:
            raise ValueError(
                "decode worker received bootstrap info missing required "
                f"field: {e.args[0]} (need host/port/room)"
            ) from e

    async def _consume_prefill_stream(
        self,
        stream: AsyncGenerator[Any, None],
        context: Context,
        rid: str | None,
    ) -> None:
        """Drain a prefill engine stream after the bootstrap chunk has
        been yielded. Awaited inline on the Bootstrap path, run as a
        background task on the Completed path (see ``generate``).

        On stream failure (NIXL transport error, engine crash) abort the
        SGLang request so the decode peer's NIXL connect fails fast
        instead of hanging on a KV transfer that will not arrive.
        """
        try:
            async for _ in stream:
                if context.is_stopped():
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            # Abort releases the bootstrap room so the decode peer fails
            # fast instead of waiting on KV that won't arrive.
            logger.warning(
                "prefill consume task ended with exception (rid=%s); "
                "aborting to release the bootstrap room",
                rid,
                exc_info=True,
            )
            if rid is not None:
                self._abort_sglang_request(rid)

    def _abort_sglang_request(self, rid: str) -> None:
        """Best-effort abort. Failures here are swallowed — SGLang is
        already in a bad state and we want to surface the original
        failure, not a follow-up abort error."""
        if self.engine is None:
            return
        tokenizer_manager = getattr(self.engine, "tokenizer_manager", None)
        if tokenizer_manager is None:
            return
        try:
            tokenizer_manager.abort_request(rid=rid, abort_all=False)
        except Exception:
            logger.debug(
                "abort_request failed while releasing bootstrap room (rid=%s)",
                rid,
                exc_info=True,
            )

    async def kv_event_sources(self) -> list[KvEventSource]:
        if not self._kv_routing_enabled():
            return []
        kv_events = json.loads(self.server_args.kv_events_config)
        base_ep = kv_events.get("endpoint")
        if not base_ep:
            return []
        local_ip = get_local_ip_auto()
        start, end = _local_dp_rank_range(self.server_args)
        sources: list[KvEventSource] = []
        for dp_rank in range(start, end):
            zmq_ep = ZmqEventPublisher.offset_endpoint_port(base_ep, dp_rank)
            if not zmq_ep:
                logger.warning(
                    "Skipping ZMQ subscriber for dp_rank=%d: offset_endpoint_port "
                    "returned None for base_ep=%s",
                    dp_rank,
                    base_ep,
                )
                continue
            sources.append(
                ZmqSource(
                    endpoint=format_zmq_endpoint(zmq_ep, local_ip),
                    dp_rank=dp_rank,
                )
            )
        return sources

    async def metrics_sources(self) -> list[SnapshotSource]:
        if not (self._kv_routing_enabled() and self._is_metrics_leader()):
            return []

        def snapshot_for(r: int) -> Callable[[], Optional[Metrics]]:
            return lambda: self._latest_metrics.get(r)

        start, end = _local_dp_rank_range(self.server_args)
        return [
            SnapshotSource(snapshot=snapshot_for(rank), dp_rank=rank)
            for rank in range(start, end)
        ]

    def _build_sampling_params(self, request: GenerateRequest) -> dict:
        if not self._use_sglang_tokenizer:
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "n": sampling_opts.get("n"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
                **self._get_guided_decoding_params(
                    sampling_opts.get("guided_decoding")
                ),
            }
        else:
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "n": request.get("n"),
                "max_new_tokens": request.get("max_tokens"),
                **self._get_guided_decoding_params(request.get("guided_decoding")),
            }
        return {k: v for k, v in param_mapping.items() if v is not None}

    @staticmethod
    def _get_guided_decoding_params(guided_decoding: object) -> dict:
        if isinstance(guided_decoding, dict):
            json_schema = guided_decoding.get("json")
            if json_schema is not None:
                return {"json_schema": json.dumps(json_schema)}
            structural_tag = guided_decoding.get("structural_tag")
            if structural_tag is not None:
                if hasattr(structural_tag, "model_dump"):
                    structural_tag = structural_tag.model_dump()
                return {"structural_tag": json.dumps(structural_tag)}
        return {}

    def _get_input_param(self, request: GenerateRequest) -> dict:
        assert self._input_param_manager is not None, "Engine not initialized"
        request_input = self._input_param_manager.get_input_param(
            dict(request), use_tokenizer=self._use_sglang_tokenizer
        )
        return {
            "prompt" if isinstance(request_input, str) else "input_ids": request_input
        }
