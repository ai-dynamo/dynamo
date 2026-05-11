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
from collections.abc import AsyncGenerator
from typing import Any

import sglang as sgl

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import ModelInput
from dynamo.sglang._compat import NetworkAddress, get_local_ip_auto, get_scheduler_info
from dynamo.sglang.args import parse_args

logger = logging.getLogger(__name__)

# Mirrors `_warmup_prefill_engine` in init_llm.py — must finish before the
# prefill worker starts serving so the first real request doesn't pay the
# JIT/CUDA-graph compile cost. SGLang's FAKE_BOOTSTRAP_HOST drives the
# warmup through the disagg path without needing a real decode peer.
_PREFILL_WARMUP_TIMEOUT_S = 1800.0

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
            # Cache bootstrap host/port; warm the engine to avoid first-
            # request JIT/cuda-graph compile latency.
            (
                self._bootstrap_host,
                self._bootstrap_port,
            ) = self._resolve_bootstrap_info()
            if _warmup_enabled():
                await self._warmup_prefill()
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

        return EngineConfig(
            model=self.server_args.model_path,
            served_model_name=self.server_args.served_model_name,
            context_length=self.server_args.context_length,
            kv_cache_block_size=page_size,
            total_kv_blocks=total_kv_blocks,
            max_num_seqs=getattr(self.server_args, "max_running_requests", None),
            max_num_batched_tokens=max_num_batched_tokens,
            # Prefill-only — drives PrefillRouter's Bootstrap path.
            bootstrap_host=self._bootstrap_host,
            bootstrap_port=self._bootstrap_port,
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

        stream = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            rid=context.trace_id,
            **bootstrap_kwargs,
        )

        # ORDER MATTERS: async_generate must register the room (the await
        # above) before we yield the bootstrap chunk — otherwise the
        # decode peer can connect to a room that doesn't exist yet.
        if self.serving_mode == DisaggregationMode.PREFILL:
            yield {
                "token_ids": [],
                "index": 0,
                "disaggregated_params": dict(bootstrap_kwargs),  # type: ignore[typeddict-unknown-key]
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
        if self.engine is not None and rid is not None:
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                self.engine.tokenizer_manager.abort_request(rid=rid, abort_all=False)
                logger.debug("Aborted request %s", rid)

    async def drain(self) -> None:
        """Await background prefill consume tasks before cleanup.

        On the Completed path these tasks are still draining in-flight
        NIXL KV transfers for already-routed decode requests; cancelling
        them in cleanup() abandons the transfer mid-flight (#7319).
        """
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

    def _resolve_bootstrap_info(self) -> tuple[str, int]:
        """Mirrors `BaseWorkerHandler._get_bootstrap_info` — returns the
        `(host, port)` tuple this prefill worker advertises to decode
        peers. Source of truth is the SGLang engine's ServerArgs.

        TODO: dedup with `dynamo.sglang.register._get_bootstrap_info_for_config`.
        """
        assert self.engine is not None
        inner_tm = self.engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            dist_init = NetworkAddress.parse(inner_tm.server_args.dist_init_addr)
            bootstrap_host = (
                NetworkAddress(dist_init.resolved().host, bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )
        else:
            bootstrap_host = (
                NetworkAddress(get_local_ip_auto(), bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )
        return bootstrap_host, bootstrap_port

    async def _warmup_prefill(self) -> None:
        """Warm the engine through the disagg path so the first real
        request doesn't pay JIT/CUDA-graph compile cost. Uses
        FAKE_BOOTSTRAP_HOST so no decode peer is required. Startup
        fails on warmup error — an unwarmed prefill silently drops
        requests in production."""
        assert self.engine is not None
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        }

        async def _do_warmup() -> None:
            results = await self.engine.async_generate(
                input_ids=[0, 1, 2, 3],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=FAKE_BOOTSTRAP_HOST,
                bootstrap_port=self._bootstrap_port,
                bootstrap_room=999999,
            )
            async for _ in results:
                pass

        logger.info("SGLang prefill warmup starting...")
        await asyncio.wait_for(_do_warmup(), timeout=_PREFILL_WARMUP_TIMEOUT_S)
        logger.info("SGLang prefill warmup complete")

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
