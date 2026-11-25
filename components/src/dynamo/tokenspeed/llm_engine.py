# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed LLMEngine implementation for the unified backend.

TokenSpeed's engine ranks (Rust ``ts-host`` processes) serve an external
frontend over a msgpack ZMQ wire. This worker is that frontend: it binds the
frontend sockets through ``tokenspeed_engine.FrontendListener``, starts the
headless launcher (``python -m tokenspeed.runtime.entrypoints.engine``), whose
ranks dial in, then submits token-id requests and routes each request's output
chunks back to its ``generate()`` call.
"""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from dynamo._core import Context
from dynamo.common.backend.dp_rank import forced_dp_rank, validate_global_dp_rank
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
)
from dynamo.common.backend.publisher import KvEventSource, ZmqSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.structural_tag import serialize_structural_tag
from dynamo.common.utils.token_ids import normalize_request_token_ids
from dynamo.llm import ModelInput
from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed.args import parse_args
from dynamo.tokenspeed.disagg import (
    attention_dp_size,
    bootstrap_kwargs,
    resolve_disaggregation_mode,
    runtime_disaggregated_endpoint,
)
from dynamo.tokenspeed.kv_events import (
    kv_event_source,
    kv_events_config_dict,
    kv_events_enabled,
)

logger = logging.getLogger(__name__)

LAUNCHER_MODULE = "tokenspeed.runtime.entrypoints.engine"
# How long one message may wait for an engine's socket.
_SEND_TIMEOUT_S = 60.0
# The launcher gives its ranks 30 s to exit before it kills them.
_SHUTDOWN_GRACE_S = 60.0
# The status code TokenSpeed puts on an abort that refuses a client error.
_CLIENT_ERROR_STATUS = 400
# Temporary, until the TokenSpeed pin includes lightseekorg/tokenspeed#3227
# (ai-dynamo/dynamo#18571): the pinned client sends no final chunk for a
# request aborted while still queued, so abort() ends the stream itself if the
# engine has not within this grace. The Rust EngineAdapter gives abort() 5 s.
_ABORT_GRACE_S = 3.0
# Per-token NVTX ranges are guarded on this so they cost one branch when off.
_NVTX = _nvtx.ENABLED


class TokenspeedLLMEngine(LLMEngine):
    def __init__(
        self,
        server_args: Any,
        engine_argv: list[str] | None = None,
        kv_block_size: int | None = None,
        engine_startup_timeout: float = 3600.0,
        bootstrap_host: str | None = None,
    ):
        self.server_args = server_args
        self.engine_argv = list(engine_argv or [])
        self.kv_block_size = kv_block_size
        self.engine_startup_timeout = engine_startup_timeout
        self.bootstrap_host = bootstrap_host
        self.disaggregation_mode = resolve_disaggregation_mode(server_args)
        # Attention-DP ranks, one engine each.
        self._dp_size = attention_dp_size(server_args)
        self._bootstrap_endpoint: tuple[str, int] | None = None
        self._kv_source: ZmqSource | None = None
        self._kv_event_dir: tempfile.TemporaryDirectory | None = None
        self._model_max_len: int | None = None
        # The connected tokenspeed_engine.Frontend.
        self.frontend: Any = None
        self._process: subprocess.Popen | None = None
        self._encoder: Any = None
        self._request_cls: Any = None
        self._sampling_params_cls: Any = None
        # Each in-flight request's state, by request id: the one place a
        # request lives in the worker, removed when its generate() ends.
        self._requests: dict[str, _Request] = {}
        self._background: set[asyncio.Task] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dispatcher: asyncio.Task | None = None
        self._monitor: asyncio.Task | None = None
        self._stopping = False
        self._failure: str | None = None
        self._abort_grace_s = _ABORT_GRACE_S

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[TokenspeedLLMEngine, WorkerConfig]:
        return cls.from_config(parse_args(argv))

    @classmethod
    def from_config(cls, config: Any) -> tuple[TokenspeedLLMEngine, WorkerConfig]:
        engine = cls(
            config.server_args,
            engine_argv=config.engine_argv,
            kv_block_size=config.kv_block_size,
            engine_startup_timeout=config.engine_startup_timeout,
            bootstrap_host=config.bootstrap_host,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        bootstrap_host, bootstrap_port = None, None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._bootstrap_endpoint = runtime_disaggregated_endpoint(
                self.server_args, self.bootstrap_host
            )
            bootstrap_host, bootstrap_port = self._bootstrap_endpoint
        engine_overrides = self._configure_kv_events()

        native = _tokenspeed_engine_module()
        self._loop = asyncio.get_running_loop()
        io_struct = _io_struct_module()
        self._encoder = io_struct.MsgpackEncoder()
        self._request_cls = io_struct.TokenizedGenerateReqInput
        self._sampling_params_cls = _sampling_params_cls()
        host, port = handshake_endpoint(self.server_args)
        listener = native.FrontendListener.bind(
            handshake_address=_tcp_address(host, port),
            input_address=_tcp_address(host, 0),
            output_address=_tcp_address(host, 0),
            first_engine_index=int(getattr(self.server_args, "zmq_engine_index", 0)),
            num_engines=self._dp_size,
            routing=(
                "least_in_flight"
                if self.disaggregation_mode == DisaggregationMode.AGGREGATED
                else "bootstrap_room"
            ),
            handshake_timeout_s=self.engine_startup_timeout,
            send_timeout_s=_SEND_TIMEOUT_S,
        )
        handshake_port = listener.handshake_address.rsplit(":", 1)[1]
        pending = listener.accept()
        self._process = _start_launcher(
            [
                *self.engine_argv,
                *engine_overrides,
                "--data-parallel-address",
                host,
                "--data-parallel-rpc-port",
                handshake_port,
            ]
        )
        self.frontend = await self._wait_for_engines(pending)

        engines = [dict(engine) for engine in self.frontend.engines()]
        if not engines:
            raise RuntimeError("TokenSpeed registered no engines")
        self._model_max_len = _optional_int(engines[0].get("max_model_len"))
        # TokenSpeed reports its prefix granularity only in a handshake field
        # tokenspeed_engine does not pass on, so it comes from --kv-block-size.
        block_size = self.kv_block_size

        cache_tokens = [
            int(engine["kv_cache_size_tokens"])
            for engine in engines
            if engine.get("kv_cache_size_tokens") is not None
        ]
        total_kv_blocks = (
            min(cache_tokens) // block_size if cache_tokens and block_size else None
        )
        chunked_prefill_size = _optional_int(
            getattr(self.server_args, "chunked_prefill_size", None)
        )

        self._dispatcher = asyncio.create_task(self._dispatch_outputs())
        self._monitor = asyncio.create_task(self._watch_launcher())
        logger.info(
            "TokenSpeed engine initialized: %d engine(s), max_model_len=%s",
            len(engines),
            self._model_max_len,
        )

        return EngineConfig(
            model=self.server_args.model,
            served_model_name=self.server_args.served_model_name,
            llm=LlmRegistration(
                context_length=self._model_max_len,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                kv_cache_block_size=block_size,
                total_kv_blocks=total_kv_blocks,
                max_num_seqs=_optional_int(
                    getattr(self.server_args, "max_num_seqs", None)
                ),
                max_num_batched_tokens=(
                    chunked_prefill_size
                    if chunked_prefill_size and chunked_prefill_size > 0
                    else None
                ),
                # The router picks among these ranks and, for prefill, mints
                # bootstrap rooms with room % data_parallel_size == its rank.
                data_parallel_size=self._dp_size if self._dp_size > 1 else None,
                data_parallel_start_rank=0 if self._dp_size > 1 else None,
            ),
        )

    def _configure_kv_events(self) -> list[str]:
        """Set up the KV event source (args.py validated the config) and
        return the launcher's override that points the engine at it."""
        config = kv_events_config_dict(
            getattr(self.server_args, "kv_events_config", None)
        )
        if not kv_events_enabled(config):
            return []
        if not config.get("endpoint"):
            self._kv_event_dir = tempfile.TemporaryDirectory(
                prefix="dynamo-tokenspeed-"
            )
            config["endpoint"] = f"ipc://{self._kv_event_dir.name}/kv-events"
        self._kv_source = kv_event_source(config)
        self.server_args.kv_events_config = json.dumps(config)
        return ["--kv-events-config", self.server_args.kv_events_config]

    async def kv_event_sources(self) -> list[KvEventSource]:
        return [self._kv_source] if self._kv_source is not None else []

    async def is_quiescent(self) -> bool | None:
        """A prefill engine finishes a request only once its KV reached the
        decode node or the transfer failed (``ts-control``'s
        ``finish_prefill`` / ``fail_transfer``), so no unfinished requests
        means no transfer still reads this engine's cache."""
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return None
        frontend = self.frontend
        if frontend is None:
            return True
        return not any(frontend.in_flight())

    async def _wait_for_engines(self, pending: Any) -> Any:
        """Wait for every rank to register, failing early if the launcher exits."""
        loop = asyncio.get_running_loop()
        accepted: asyncio.Future = loop.create_future()

        def settle(result: Any, error: BaseException | None) -> None:
            if accepted.done():
                return
            if error is not None:
                accepted.set_exception(error)
            else:
                accepted.set_result(result)

        def wait() -> None:
            result: tuple[Any, BaseException | None]
            try:
                result = (pending.wait(), None)
            except BaseException as error:  # noqa: BLE001 - re-raised on the loop
                result = (None, error)
            try:
                loop.call_soon_threadsafe(settle, *result)
            except RuntimeError:
                # start() already failed (the launcher exited) and the loop
                # is gone; nobody is waiting for this result.
                pass

        # A daemon thread: a launcher that dies leaves the native wait blocked
        # until the handshake timeout, which must not hold up shutdown.
        threading.Thread(target=wait, name="tokenspeed-handshake", daemon=True).start()
        while True:
            done, _ = await asyncio.wait({accepted}, timeout=0.5)
            if done:
                try:
                    return accepted.result()
                except Exception as error:
                    raise RuntimeError(
                        f"TokenSpeed engines did not register: {error}"
                    ) from error
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    "TokenSpeed engine launcher exited with code "
                    f"{self._process.returncode} before its engines registered"
                )

    async def _dispatch_outputs(self) -> None:
        """Route every output chunk to its request."""
        requests = self._requests
        frontend = self.frontend
        try:
            while True:
                batch = await frontend.next_outputs()
                rng = _nvtx.start_range("ts:dispatch", "blue") if _NVTX else None
                for rid, chunk in batch:
                    if isinstance(chunk, BaseException):
                        # The native client fails a stream only once it has
                        # closed for good (an engine shut down, or its output
                        # socket failed), so every request is lost with it.
                        self._engine_lost(f"TokenSpeed engines unavailable: {chunk}")
                        return
                    request = requests.get(rid)
                    if request is not None:
                        # _Request.push, inlined: this runs once per token.
                        request.chunks.append(chunk)
                        waiter = request.waiter
                        if waiter is not None and not waiter.done():
                            waiter.set_result(None)
                if rng is not None:
                    _nvtx.end_range(rng)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._engine_lost(f"TokenSpeed output stream failed: {error}")

    async def _watch_launcher(self) -> None:
        """Shut the worker down if the engine launcher exits on its own."""
        process = self._process
        if process is None:
            return
        returncode = await asyncio.to_thread(process.wait)
        self._engine_lost(f"TokenSpeed engine launcher exited with code {returncode}")

    def _engine_lost(self, message: str) -> None:
        """Fail every request, and unless the worker is already stopping, stop
        it: every later request would fail too, so it must leave the routing
        pool rather than stay registered."""
        if self._failure is not None:
            return
        self._failure = message
        for request in self._requests.values():
            request.push(RuntimeError(message))
        if not self._stopping:
            logger.error("%s; shutting down the worker", message)
            os.kill(os.getpid(), signal.SIGTERM)

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.frontend is None:
            raise RuntimeError("Engine not initialized")
        if self._failure is not None:
            raise RuntimeError(self._failure)
        normalize_request_token_ids(request)

        dp_rank = self._routed_dp_rank(request)
        bootstrap = bootstrap_kwargs(
            request,
            self.disaggregation_mode,
            self._bootstrap_endpoint,
            dp_rank=dp_rank,
            dp_size=self._dp_size,
        )
        _validate_single_choice_sampling(request)
        sampling_params = build_sampling_params(request, self._model_max_len)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            sampling_params["max_new_tokens"] = 1
            sampling_params.pop("min_new_tokens", None)
            # TokenSpeed runs a prefill request on rank room % dp_size, where
            # decode looks for its KV cache; the room already carries the rank.
            dp_rank = None
        rid = context.id() or uuid.uuid4().hex
        frames = self._encode_request(
            rid, request.get("token_ids", []), sampling_params, bootstrap, dp_rank
        )

        state = _Request()
        self._requests[rid] = state
        finished = False
        try:
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # The router needs a handoff even when it supplied the bootstrap
                # address. Preserve its trace identity; fallback rooms use the
                # same validated parameters passed to the native prefill engine.
                yield {
                    "index": 0,
                    "token_ids": [],
                    "disaggregated_params": request.get("bootstrap_info") or bootstrap,
                }
            # An abort while the handoff was yielded must prevent submission:
            # the engine cannot abort a request it does not hold yet.
            if state.cancelled:
                finished = True
                yield {"index": 0, "token_ids": [], "finish_reason": "cancelled"}
                return
            rng = _nvtx.start_range("ts:submit", "green") if _NVTX else None
            try:
                await self.frontend.generate(frames)
            except Exception as error:
                # ts-client's Error::Unavailable: the client closed for good.
                if str(error).startswith("engines unavailable"):
                    self._engine_lost(f"TokenSpeed {error}")
                raise
            finally:
                if rng is not None:
                    _nvtx.end_range(rng)
            state.submitted = True
            if state.cancelled:
                # abort() ran while the submission was in flight.
                await self._abort_native(rid)
            loop = asyncio.get_running_loop()
            while not finished:
                if not state.chunks:
                    state.waiter = waiter = loop.create_future()
                    try:
                        await waiter
                    finally:
                        state.waiter = None
                    continue
                items, state.chunks = state.chunks, []
                chunk = _coalesce(items, state.cancelled)
                if chunk is not None:
                    finished = "finish_reason" in chunk
                    yield chunk
        finally:
            if self._requests.get(rid) is state:
                del self._requests[rid]
            if state.ended is not None and not state.ended.done():
                state.ended.set_result(None)
            if state.submitted and not finished and self._failure is None:
                # The caller stopped reading before the request finished; the
                # engine must not keep generating for nobody.
                self._schedule_abort(rid)

    def _routed_dp_rank(self, request: GenerateRequest) -> int | None:
        """The attention-DP rank the router chose for this request, if any."""
        if self._dp_size <= 1:
            return None
        rank = forced_dp_rank(request)
        if rank is None and self.disaggregation_mode == DisaggregationMode.PREFILL:
            routing = request.get("routing") or {}
            prefill_rank = routing.get("prefill_dp_rank")
            rank = None if prefill_rank is None else int(prefill_rank)
        return validate_global_dp_rank(rank, 0, self._dp_size, "TokenSpeed")

    def _encode_request(
        self,
        rid: str,
        token_ids: list[int],
        sampling_params: dict[str, Any],
        bootstrap: dict[str, Any],
        dp_rank: int | None = None,
    ) -> list[bytes]:
        pin = {} if dp_rank is None else {"data_parallel_rank": dp_rank}
        # The encoder serializes token_ids as they are, so they are not copied:
        # normalize_request_token_ids already made them a list, and the
        # encoder returns bytes frames the native client takes as they are.
        native_request = self._request_cls(
            rid=rid,
            input_ids=token_ids,
            sampling_params=self._sampling_params_cls(**sampling_params),
            stream=True,
            **bootstrap,
            **pin,
        )
        return self._encoder.encode(native_request)

    async def abort(self, context: Context) -> None:
        """Forward the cancellation to the engine, whose final chunk then ends
        the request's stream, as vLLM's and TRT-LLM's handlers do. A request
        that already finished is left alone."""
        rid = context.id()
        state = self._requests.get(rid) if rid is not None else None
        if state is None or state.cancelled:
            return
        state.cancelled = True
        # Before submission the engine does not hold the request yet;
        # generate() then aborts it, or skips submitting it, itself.
        if state.submitted:
            await self._abort_native(rid)
        logger.debug("Aborted TokenSpeed request %s", rid)
        # Temporary (tokenspeed#3227): stand in for the final chunk the pinned
        # engine never sends for a request aborted while queued.
        state.ended = asyncio.get_running_loop().create_future()
        try:
            async with asyncio.timeout(self._abort_grace_s):
                await state.ended
        except TimeoutError:
            logger.debug("TokenSpeed sent no final chunk for aborted %s", rid)
            # The final chunk the engine sends for an aborted request.
            state.push({"output_ids": [], "finish_reason": "abort"})

    async def _abort_native(self, rid: str) -> None:
        frontend = self.frontend
        if frontend is None:
            return
        try:
            await frontend.abort(rid)
        except Exception as error:  # noqa: BLE001 - the engine may be gone
            logger.debug("TokenSpeed abort of %s failed: %s", rid, error)

    def _schedule_abort(self, rid: str) -> None:
        """Abort ``rid`` from any context. A generator the caller drops
        without closing is finalized by the garbage collector, outside the
        event loop, so this cannot assume a running loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def spawn() -> None:
            task = loop.create_task(self._abort_native(rid))
            self._background.add(task)
            task.add_done_callback(self._background.discard)

        try:
            loop.call_soon_threadsafe(spawn)
        except RuntimeError:
            pass  # The loop closed meanwhile; the engine went with it.

    async def cleanup(self) -> None:
        self._stopping = True
        try:
            for task in (self._dispatcher, self._monitor, *self._background):
                if task is not None:
                    task.cancel()
            self._dispatcher = self._monitor = None
            self._engine_lost("TokenSpeed engine shut down")
            process, self._process = self._process, None
            if process is not None:
                await asyncio.to_thread(_stop_launcher, process)
                logger.info("TokenSpeed engine shutdown")
        finally:
            self.frontend = None
            self._bootstrap_endpoint = None
            self._kv_source = None
            if self._kv_event_dir is not None:
                self._kv_event_dir.cleanup()
                self._kv_event_dir = None


class _Request:
    """An in-flight request: the chunks the dispatcher delivered and the
    caller has not taken yet, and the future its generate() sleeps on."""

    __slots__ = ("chunks", "waiter", "cancelled", "submitted", "ended")

    def __init__(self) -> None:
        self.chunks: list[Any] = []
        self.waiter: asyncio.Future | None = None
        self.cancelled = False
        self.submitted = False
        # Set by abort(), resolved when generate() ends (tokenspeed#3227).
        self.ended: asyncio.Future | None = None

    def push(self, item: Any) -> None:
        self.chunks.append(item)
        waiter = self.waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)


def _coalesce(items: list[Any], cancelled: bool) -> GenerateChunk | None:
    """The wire chunks a request received since it last ran, as one Dynamo
    chunk: their new tokens, and the final chunk's finish reason and usage.
    None when they carry no tokens and no finish."""
    rng = _nvtx.start_range("ts:emit", "cyan") if _NVTX else None
    try:
        token_ids: list[int] | None = None
        for item in items:
            if isinstance(item, BaseException):
                raise RuntimeError(str(item)) from item
            if item.get("finish_reason") is not None:
                chunk = convert_output_to_chunk(item, cancelled=cancelled)
                if token_ids:
                    token_ids.extend(chunk["token_ids"])
                    chunk["token_ids"] = token_ids
                return chunk
            ids = item.get("output_ids")
            if ids:
                if token_ids is None:
                    token_ids = ids if isinstance(ids, list) else list(ids)
                else:
                    token_ids.extend(ids)
        return {"index": 0, "token_ids": token_ids} if token_ids else None
    finally:
        if rng is not None:
            _nvtx.end_range(rng)


def build_sampling_params(
    request: GenerateRequest,
    model_max_len: int | None = None,
) -> dict[str, Any]:
    sampling_options = request.get("sampling_options", {}) or {}
    stop_conditions = request.get("stop_conditions", {}) or {}

    params: dict[str, Any] = {}

    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "n",
    ):
        value = sampling_options.get(key)
        if value is not None:
            params[key] = value

    logit_bias = sampling_options.get("logit_bias")
    if logit_bias:
        # The wire types logit_bias as dict[str, float].
        params["logit_bias"] = {
            str(token_id): float(bias) for token_id, bias in logit_bias.items()
        }

    guided_decoding = sampling_options.get("guided_decoding")
    if isinstance(guided_decoding, dict):
        params.update(_guided_decoding_params(guided_decoding))

    max_tokens = stop_conditions.get("max_tokens")
    if max_tokens is not None:
        params["max_new_tokens"] = max_tokens
    elif model_max_len is not None:
        params["max_new_tokens"] = max(
            1, model_max_len - len(request.get("token_ids", []))
        )

    min_tokens = stop_conditions.get("min_tokens")
    if min_tokens is not None:
        params["min_new_tokens"] = min_tokens

    ignore_eos = stop_conditions.get("ignore_eos")
    if ignore_eos is not None:
        params["ignore_eos"] = ignore_eos

    stop_token_ids = _merge_stop_token_ids(
        stop_conditions.get("stop_token_ids_hidden"),
        stop_conditions.get("stop_token_ids"),
    )
    if stop_token_ids:
        params["stop_token_ids"] = stop_token_ids

    return params


def convert_output_to_chunk(
    out: dict[str, Any], cancelled: bool = False
) -> GenerateChunk:
    """One wire output chunk (``tokenspeed_engine.Frontend.next_outputs``) as
    a Dynamo chunk. Its ``output_ids`` are the tokens new in this chunk."""
    ids = out.get("output_ids") or []
    chunk: GenerateChunk = {
        "index": 0,
        "token_ids": ids if isinstance(ids, list) else list(ids),
    }

    finish_reason = out.get("finish_reason")
    if finish_reason is None:
        return chunk
    if finish_reason == "abort":
        finish_reason = _abort_finish_reason(out.get("finish_details"), cancelled)
    chunk["finish_reason"] = normalize_finish_reason(str(finish_reason))
    prompt_tokens = int(out.get("prompt_tokens") or 0)
    completion_tokens = int(out.get("completion_tokens") or 0)
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    cached_tokens = out.get("cached_tokens")
    if cached_tokens:
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached_tokens)}
    chunk["completion_usage"] = usage
    return chunk


def _abort_finish_reason(details: Any, cancelled: bool) -> str:
    """Classify an abort: a refusal, a cancellation, or an engine failure."""
    details = details if isinstance(details, dict) else {}
    message = details.get("message") or "Unknown backend error"
    if details.get("status_code") == _CLIENT_ERROR_STATUS:
        raise InvalidArgument(f"TokenSpeed rejected the request: {message}")
    if cancelled:
        return "cancelled"
    raise RuntimeError(f"TokenSpeed generation aborted: {message}")


def _guided_decoding_params(guided_decoding: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_schema = guided_decoding.get("json")
    regex = guided_decoding.get("regex")
    choice = guided_decoding.get("choice")
    grammar = guided_decoding.get("grammar")
    structural_tag = guided_decoding.get("structural_tag")

    if regex is None and choice:
        valid_choices = [str(c) for c in choice if c is not None]
        if valid_choices:
            regex = "(" + "|".join(re.escape(c) for c in valid_choices) + ")"

    constraints = {
        "json": json_schema,
        "regex": regex,
        "grammar": grammar,
        "structural_tag": structural_tag,
    }
    active_constraints = [
        name for name, value in constraints.items() if value is not None
    ]
    if len(active_constraints) > 1:
        raise InvalidArgument(
            "TokenSpeed guided decoding supports one constraint at a time; "
            f"got {', '.join(active_constraints)}"
        )

    if json_schema is not None:
        params["json_schema"] = (
            json_schema if isinstance(json_schema, str) else json.dumps(json_schema)
        )

    if regex is not None:
        params["regex"] = regex

    if grammar is not None:
        params["ebnf"] = grammar

    if structural_tag is not None:
        params["structural_tag"] = serialize_structural_tag(structural_tag)

    return params


def _merge_stop_token_ids(*token_id_lists: Any) -> list[int]:
    return list(dict.fromkeys(t for ids in token_id_lists for t in ids or ()))


def _validate_single_choice_sampling(request: GenerateRequest) -> None:
    sampling_options = request.get("sampling_options", {}) or {}
    n = sampling_options.get("n", 1)
    if isinstance(n, int) and not isinstance(n, bool) and n > 1:
        raise InvalidArgument(
            f"TokenSpeed Dynamo backend does not support n={n}; only n=1 is supported"
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _num_nodes(server_args: Any) -> int:
    return int(server_args.mapping.nnodes)


def is_follower_node(server_args: Any) -> bool:
    """Whether this process serves a node other than the engine's first."""
    return int(getattr(server_args, "node_rank", None) or 0) != 0


def handshake_endpoint(server_args: Any) -> tuple[str, int]:
    """The handshake address the worker binds and every node's ranks dial.

    A single node binds a free port on ``--data-parallel-address``, so that
    several workers can share a host. Across nodes, every node must agree on
    the address without talking to the worker first: the configured
    ``--data-parallel-rpc-port``, on ``--data-parallel-address`` or, if that
    is loopback (TokenSpeed's default), on ``--dist-init-addr``'s host, the
    first node, where the worker runs.
    """
    host = str(getattr(server_args, "data_parallel_address", None) or "127.0.0.1")
    if _num_nodes(server_args) <= 1:
        return host, 0
    if _is_loopback(host):
        dist_init_addr = getattr(server_args, "dist_init_addr", None)
        if not dist_init_addr:
            raise ValueError(
                "A multi-node TokenSpeed worker needs --data-parallel-address "
                "reachable from every node, or --dist-init-addr"
            )
        host = _resolve_host(str(dist_init_addr).rsplit(":", 1)[0])
    return host, int(server_args.data_parallel_rpc_port)


def run_follower_node(config: Any) -> None:
    """Run a follower node's engine ranks in place of this process.

    The worker on the first node registers every rank with Dynamo; another
    node's ranks only dial its handshake. ``exec`` hands this process to
    TokenSpeed's launcher, so a shutdown signal reaches it directly.
    """
    server_args = config.server_args
    host, port = handshake_endpoint(server_args)
    argv = [
        sys.executable,
        "-m",
        LAUNCHER_MODULE,
        *config.engine_argv,
        "--data-parallel-address",
        host,
        "--data-parallel-rpc-port",
        str(port),
    ]
    logger.info(
        "TokenSpeed node %s runs its engine ranks only: %s",
        getattr(server_args, "node_rank", None),
        " ".join(argv[1:]),
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, argv)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False  # A hostname.


def _resolve_host(host: str) -> str:
    host = host.strip("[]")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return socket.gethostbyname(host)


def _tcp_address(host: str, port: int) -> str:
    try:
        if ipaddress.ip_address(host.strip("[]")).version == 6:
            return f"tcp://[{host.strip('[]')}]:{port}"
    except ValueError:
        pass  # A hostname.
    return f"tcp://{host}:{port}"


def _start_launcher(argv: list[str]) -> subprocess.Popen:
    logger.info("Starting TokenSpeed engine launcher: %s", " ".join(argv))
    return subprocess.Popen(
        [sys.executable, "-m", LAUNCHER_MODULE, *argv],
        # Its own process group, so that a shutdown reaches every rank.
        start_new_session=True,
    )


def _stop_launcher(process: subprocess.Popen) -> None:
    """Stop the launcher, which stops its ranks; kill what is left."""
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + _SHUTDOWN_GRACE_S
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _tokenspeed_engine_module() -> Any:
    return importlib.import_module("tokenspeed_engine")


def _io_struct_module() -> Any:
    return importlib.import_module("tokenspeed.runtime.engine.io_struct")


def _sampling_params_cls() -> Any:
    module = importlib.import_module("tokenspeed.runtime.sampling.sampling_params")
    return module.SamplingParams
