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
    validate_disagg_compatibility,
)
from dynamo.tokenspeed.kv_events import (
    kv_event_source,
    kv_events_config_dict,
    kv_events_enabled,
)

logger = logging.getLogger(__name__)

LAUNCHER_MODULE = "tokenspeed.runtime.entrypoints.engine"
STARTUP_TIMEOUT_ENV = "DYN_TOKENSPEED_STARTUP_TIMEOUT_S"
# Large checkpoints take tens of minutes to load before the ranks register.
_DEFAULT_STARTUP_TIMEOUT_S = 3600.0
# How long one message may wait for an engine's socket.
_SEND_TIMEOUT_S = 60.0
# The launcher gives its ranks 30 s to exit before it kills them.
_SHUTDOWN_GRACE_S = 60.0
# The status code TokenSpeed puts on an abort that refuses a client error.
_CLIENT_ERROR_STATUS = 400


class TokenspeedLLMEngine(LLMEngine):
    def __init__(
        self,
        server_args: Any,
        engine_argv: list[str] | None = None,
        kv_block_size: int | None = None,
    ):
        self.server_args = server_args
        self.engine_argv = list(engine_argv or [])
        self.kv_block_size = kv_block_size
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
        # Each in-flight request's outputs, by request id.
        self._streams: dict[str, asyncio.Queue] = {}
        self._cancelled: set[str] = set()
        self._background: set[asyncio.Task] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dispatcher: asyncio.Task | None = None
        self._monitor: asyncio.Task | None = None
        self._stopping = False
        self._failure: str | None = None

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
        validate_disagg_compatibility(self.disaggregation_mode, self.server_args)
        if is_follower_node(self.server_args):
            raise ValueError(
                "A TokenSpeed node other than the first runs its engine ranks "
                "only; start it through `python -m dynamo.tokenspeed`, which "
                "launches them without registering a worker"
            )
        bootstrap_host, bootstrap_port = None, None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._bootstrap_endpoint = runtime_disaggregated_endpoint(self.server_args)
            bootstrap_host, bootstrap_port = self._bootstrap_endpoint
        engine_overrides = self._configure_kv_events()

        native = _tokenspeed_engine_module()
        self._loop = asyncio.get_running_loop()
        self._encoder = _io_struct_module().MsgpackEncoder()
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
            handshake_timeout_s=_startup_timeout_s(),
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
        block_size = self._registered_block_size(engines[0])
        if self._bootstrap_endpoint is not None:
            reported_port = engines[0].get("bootstrap_port")
            if reported_port is not None and int(reported_port) != bootstrap_port:
                logger.warning(
                    "TokenSpeed prefill bootstrap server listens on port %s, but "
                    "Dynamo advertises port %s",
                    reported_port,
                    bootstrap_port,
                )

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

    def _registered_block_size(self, engine: dict[str, Any]) -> int | None:
        # tokenspeed_engine does not report the prefix granularity at the
        # pinned versions; prefer it, should a later one start to.
        reported = _optional_int(engine.get("prefix_granularity"))
        if reported:
            if self.kv_block_size is not None and self.kv_block_size != reported:
                raise ValueError(
                    f"--kv-block-size {self.kv_block_size} does not match "
                    f"TokenSpeed's prefix granularity {reported}"
                )
            return reported
        return self.kv_block_size

    def _configure_kv_events(self) -> list[str]:
        """Validate the KV event config; return the launcher's overrides."""
        config = kv_events_config_dict(
            getattr(self.server_args, "kv_events_config", None)
        )
        if not kv_events_enabled(config):
            return []
        if not getattr(self.server_args, "enable_prefix_caching", True):
            # Native publisher construction is independent of prefix caching.
            config["enable_kv_cache_events"] = False
            config["publisher"] = "null"
            self.server_args.kv_events_config = json.dumps(config)
            logger.warning(
                "TokenSpeed KV events were requested but enable_prefix_caching=False "
                "(--disable-prefix-caching); no KV events will be published"
            )
            return ["--kv-events-config", self.server_args.kv_events_config]
        validate_kv_events_topology(self.server_args)
        if self.kv_block_size is None:
            raise ValueError(
                "TokenSpeed KV events require --kv-block-size, the engine's prefix "
                "granularity (TokenSpeed logs prefix_granularity=N at startup)"
            )
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
        """Route every output chunk to its request's stream."""
        try:
            while True:
                for rid, chunk in await self.frontend.next_outputs():
                    stream = self._streams.get(rid)
                    if stream is not None:
                        stream.put_nowait(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            message = f"TokenSpeed output stream failed: {error}"
            self._fail_streams(message)
            if self._stopping:
                return
            # Every later request would fail too; stop so the worker leaves
            # the routing pool instead of staying registered.
            logger.error("%s; shutting down the worker", message)
            os.kill(os.getpid(), signal.SIGTERM)

    async def _watch_launcher(self) -> None:
        """Shut the worker down if the engine launcher exits on its own."""
        process = self._process
        if process is None:
            return
        returncode = await asyncio.to_thread(process.wait)
        if self._stopping:
            return
        message = f"TokenSpeed engine launcher exited with code {returncode}"
        logger.error("%s; shutting down the worker", message)
        self._fail_streams(message)
        os.kill(os.getpid(), signal.SIGTERM)

    def _fail_streams(self, message: str) -> None:
        self._failure = self._failure or message
        for stream in self._streams.values():
            stream.put_nowait(RuntimeError(message))

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
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            dp_rank = self._prefill_pin(dp_rank, bootstrap["bootstrap_room"])
        _validate_single_choice_sampling(request)
        sampling_params = build_sampling_params(request, self._model_max_len)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            sampling_params["max_new_tokens"] = 1
            sampling_params.pop("min_new_tokens", None)
        frames = self._encode_request(
            request.get("token_ids", []), sampling_params, bootstrap, context, dp_rank
        )
        rid = frames.rid

        stream: asyncio.Queue = asyncio.Queue()
        self._streams[rid] = stream
        submitted = finished = False
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
            if rid in self._cancelled:
                finished = True
                yield {"index": 0, "token_ids": [], "finish_reason": "cancelled"}
                return
            await self.frontend.generate(frames.frames)
            submitted = True
            if rid in self._cancelled:
                # abort() ran while the submission was in flight.
                await self._abort_native(rid)
            while not finished:
                item = await stream.get()
                if isinstance(item, BaseException):
                    raise RuntimeError(str(item)) from item
                chunk = convert_output_to_chunk(item, cancelled=rid in self._cancelled)
                finished = "finish_reason" in chunk
                yield chunk
        finally:
            self._streams.pop(rid, None)
            self._cancelled.discard(rid)
            if submitted and not finished and self._failure is None:
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

    def _prefill_pin(self, dp_rank: int | None, room: int) -> int | None:
        """A prefill request runs on rank ``room % dp_size``: decode locates
        its KV cache by that residue, and TokenSpeed rejects a prefill pin
        that disagrees. Keep the router's pin only when it agrees."""
        if dp_rank is None or room % self._dp_size == dp_rank:
            return dp_rank
        logger.warning(
            "Ignoring prefill DP rank %d: bootstrap room %d places the request "
            "on rank %d of %d",
            dp_rank,
            room,
            room % self._dp_size,
            self._dp_size,
        )
        return None

    def _encode_request(
        self,
        token_ids: list[int],
        sampling_params: dict[str, Any],
        bootstrap: dict[str, Any],
        context: Context,
        dp_rank: int | None = None,
    ) -> _EncodedRequest:
        io_struct = _io_struct_module()
        rid = context.id() or uuid.uuid4().hex
        pin = {} if dp_rank is None else {"data_parallel_rank": dp_rank}
        native_request = io_struct.TokenizedGenerateReqInput(
            rid=rid,
            input_ids=list(token_ids),
            sampling_params=_sampling_params_cls()(**sampling_params),
            stream=True,
            **bootstrap,
            **pin,
        )
        frames = [bytes(frame) for frame in self._encoder.encode(native_request)]
        return _EncodedRequest(rid, frames)

    async def abort(self, context: Context) -> None:
        rid = context.id()
        if self.frontend is None or rid is None or rid not in self._streams:
            return
        self._cancelled.add(rid)
        await self._abort_native(rid)
        logger.debug("Aborted TokenSpeed request %s", rid)

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
            self._fail_streams("TokenSpeed engine shut down")
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


class _EncodedRequest:
    __slots__ = ("rid", "frames")

    def __init__(self, rid: str, frames: list[bytes]):
        self.rid = rid
        self.frames = frames


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
    chunk: GenerateChunk = {
        "index": 0,
        "token_ids": list(out.get("output_ids") or []),
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
    merged: list[int] = []
    seen: set[int] = set()
    for token_ids in token_id_lists:
        for token_id in token_ids or []:
            if token_id not in seen:
                seen.add(token_id)
                merged.append(token_id)
    return merged


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
    mapping = getattr(server_args, "mapping", None)
    return int(
        getattr(mapping, "nnodes", None) or getattr(server_args, "nnodes", None) or 1
    )


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


def validate_kv_events_topology(server_args: Any) -> None:
    if attention_dp_size(server_args) != 1:
        # Each rank publishes its own KV events (on its own node); the worker
        # subscribes to one publisher and tags no DP rank.
        raise ValueError(
            "TokenSpeed KV events currently require attention DP=1; with "
            "attention DP > 1, route without --router-kv-events"
        )


def run_follower_node(config: Any) -> None:
    """Run a follower node's engine ranks in place of this process.

    The worker on the first node registers every rank with Dynamo; another
    node's ranks only dial its handshake. ``exec`` hands this process to
    TokenSpeed's launcher, so a shutdown signal reaches it directly.
    """
    server_args = config.server_args
    validate_disagg_compatibility(resolve_disaggregation_mode(server_args), server_args)
    if kv_events_enabled(
        kv_events_config_dict(getattr(server_args, "kv_events_config", None))
    ):
        validate_kv_events_topology(server_args)
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


def _startup_timeout_s() -> float:
    value = os.environ.get(STARTUP_TIMEOUT_ENV)
    return float(value) if value else _DEFAULT_STARTUP_TIMEOUT_S


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
