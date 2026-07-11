# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental async facade for vLLM's synchronous in-process engine."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import warnings
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, AsyncGenerator, Callable, Literal, Mapping, Optional

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EngineInput, PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.executor import Executor
from vllm.v1.metrics.prometheus import shutdown_prometheus

logger = logging.getLogger(__name__)

_COMMAND_QUEUE_CAPACITY = 1024
_MAX_COMMANDS_PER_STEP = 64


class _DriverState(Enum):
    STARTING = auto()
    RUNNING = auto()
    CLOSED = auto()
    FAILED = auto()


class _RequestState(Enum):
    PENDING_ADD = auto()
    ACTIVE = auto()
    FINISHED = auto()
    ABORTED = auto()


class _CommandKind(Enum):
    ADD = auto()
    ABORT = auto()
    CALL = auto()
    CORE_CALL = auto()
    RENDERER_CALL = auto()
    HEALTH = auto()
    SHUTDOWN = auto()


@dataclass
class _RequestContext:
    request_id: str
    collector: RequestOutputCollector
    loop: asyncio.AbstractEventLoop
    state: _RequestState = _RequestState.PENDING_ADD


@dataclass
class _Command:
    kind: _CommandKind
    result: Future[Any]
    args: tuple[Any, ...] = ()
    kwargs: Optional[dict[str, Any]] = None
    request: Optional[_RequestContext] = None


class _EngineCoreAsyncProxy:
    """Compatibility surface used by Dynamo's cache metadata setup."""

    def __init__(self, owner: "SyncInprocEngineClient") -> None:
        self._owner = owner

    async def call_utility_async(self, method: str, *args: Any) -> Any:
        return await self._owner._submit_async(
            _CommandKind.CORE_CALL,
            args=(method, *args),
        )


class SyncInprocEngineClient:
    """Drive ``LLMEngine`` on one thread while exposing AsyncLLM-like methods.

    This class is intentionally narrow and experimental. It preserves online
    continuous batching by admitting queued commands between every engine step,
    while never blocking for commands as long as vLLM reports unfinished work.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        usage_context: UsageContext,
        disable_log_stats: bool,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.log_stats = not disable_log_stats
        self.engine_core = _EngineCoreAsyncProxy(self)

        self._usage_context = usage_context
        self._disable_log_stats = disable_log_stats
        self._commands: queue.Queue[_Command] = queue.Queue(
            maxsize=_COMMAND_QUEUE_CAPACITY
        )
        self._state = _DriverState.STARTING
        self._state_lock = threading.Lock()
        self._failure: Optional[BaseException] = None
        self._startup: Future[None] = Future()
        self._terminated = threading.Event()
        self._tokenizer: Any = None
        self._thread = threading.Thread(
            target=self._run,
            name="vllm-sync-inproc",
            daemon=False,
        )
        self._thread.start()
        self._startup.result()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        *,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        disable_log_stats: bool = False,
    ) -> "SyncInprocEngineClient":
        return cls(
            vllm_config,
            usage_context=usage_context,
            disable_log_stats=disable_log_stats,
        )

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def _run(self) -> None:
        engine: Optional[LLMEngine] = None
        requests: dict[str, _RequestContext] = {}
        try:
            engine = LLMEngine(
                vllm_config=self.vllm_config,
                executor_class=Executor.get_class(self.vllm_config),
                log_stats=not self._disable_log_stats,
                usage_context=self._usage_context,
                stat_loggers=None,
                multiprocess_mode=False,
            )
            self._tokenizer = engine.tokenizer
            with self._state_lock:
                self._state = _DriverState.RUNNING
            self._complete(self._startup, None)

            stop = False
            while not stop:
                commands = self._collect_commands(engine)
                for command in commands:
                    stop = self._process_command(
                        command,
                        engine,
                        requests,
                    )
                    if stop:
                        break

                if not stop and self._engine_has_work(engine):
                    outputs = engine.step()
                    self._route_outputs(outputs, requests)
        except BaseException as exc:
            self._fail(exc, requests)
        finally:
            try:
                self._fail_live_requests(EngineDeadError(), requests)
                self._fail_queued_commands(EngineDeadError())
                if engine is not None:
                    self._shutdown_engine(engine)
            finally:
                with self._state_lock:
                    if self._state is not _DriverState.FAILED:
                        self._state = _DriverState.CLOSED
                self._terminated.set()

    def _collect_commands(self, engine: LLMEngine) -> list[_Command]:
        commands: list[_Command] = []
        if not self._engine_has_work(engine):
            commands.append(self._commands.get())
        while len(commands) < _MAX_COMMANDS_PER_STEP:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                break
        return commands

    @staticmethod
    def _engine_has_work(engine: LLMEngine) -> bool:
        if engine.has_unfinished_requests():
            return True
        # With vLLM async scheduling, aborting the last request can empty the
        # OutputProcessor while a GPU future is still queued in EngineCore. Keep
        # stepping until that internal queue drains or shutdown can strand it.
        batch_queue = engine.engine_core.engine_core.batch_queue
        return batch_queue is not None and bool(batch_queue)

    def _process_command(
        self,
        command: _Command,
        engine: LLMEngine,
        requests: dict[str, _RequestContext],
    ) -> bool:
        try:
            if command.kind is _CommandKind.ADD:
                if command.request is None:
                    raise RuntimeError("ADD command is missing request context")
                actual_request_id = engine.add_request(
                    *command.args, **(command.kwargs or {})
                )
                command.request.state = _RequestState.ACTIVE
                # LLMEngine returns its internal wave-suffixed ID, but its
                # RequestOutputs and abort_request() use the public ID.
                requests[command.request.request_id] = command.request
                self._complete(command.result, actual_request_id)
                return False

            if command.kind is _CommandKind.ABORT:
                requested_ids = command.args[0]
                active_ids: list[str] = []
                for request_id in requested_ids:
                    context = requests.pop(request_id, None)
                    if context is not None:
                        context.state = _RequestState.ABORTED
                        active_ids.append(request_id)
                if active_ids:
                    engine.abort_request(active_ids)
                self._complete(command.result, None)
                return False

            if command.kind is _CommandKind.CALL:
                method = command.args[0]
                call_args = command.args[1:]
                result = getattr(engine, method)(*call_args, **(command.kwargs or {}))
                self._complete(command.result, result)
                return False

            if command.kind is _CommandKind.CORE_CALL:
                method = command.args[0]
                call_args = command.args[1:]
                core = engine.engine_core.engine_core
                result = getattr(core, method)(*call_args, **(command.kwargs or {}))
                self._complete_or_chain(command.result, result)
                return False

            if command.kind is _CommandKind.RENDERER_CALL:
                method = command.args[0]
                call_args = command.args[1:]
                result = getattr(engine.renderer, method)(
                    *call_args, **(command.kwargs or {})
                )
                self._complete_or_chain(command.result, result)
                return False

            if command.kind is _CommandKind.HEALTH:
                self._complete(command.result, None)
                return False

            if command.kind is _CommandKind.SHUTDOWN:
                self._complete(command.result, None)
                return True

            raise RuntimeError(f"unknown sync engine command: {command.kind}")
        except Exception as exc:
            self._complete_exception(command.result, exc)
            if command.request is not None:
                self._deliver(command.request, exc)
            return False

    def _route_outputs(
        self,
        outputs: list[RequestOutput],
        requests: dict[str, _RequestContext],
    ) -> None:
        for output in outputs:
            context = requests.get(output.request_id)
            if context is None:
                logger.debug(
                    "sync-inproc received output for unknown request %s; active=%s",
                    output.request_id,
                    list(requests),
                )
                continue
            if output.finished:
                context.state = _RequestState.FINISHED
                requests.pop(output.request_id, None)
            self._deliver(context, output)

    def _deliver(self, context: _RequestContext, value: Any) -> None:
        try:
            context.loop.call_soon_threadsafe(context.collector.put, value)
        except RuntimeError:
            logger.debug(
                "Dropping sync-engine output for request %s because its event loop closed",
                context.request_id,
            )

    def _fail(self, exc: BaseException, requests: dict[str, _RequestContext]) -> None:
        with self._state_lock:
            self._state = _DriverState.FAILED
            self._failure = exc
        self._complete_exception(self._startup, exc)
        self._fail_live_requests(exc, requests)
        logger.error(
            "Synchronous in-process vLLM driver failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def _fail_live_requests(
        self,
        exc: BaseException,
        requests: dict[str, _RequestContext],
    ) -> None:
        for context in list(requests.values()):
            self._deliver(context, exc)
        requests.clear()

    def _fail_queued_commands(self, exc: BaseException) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            self._complete_exception(command.result, exc)

    @staticmethod
    def _shutdown_engine(engine: LLMEngine) -> None:
        try:
            engine.renderer.shutdown()
        finally:
            try:
                engine.engine_core.shutdown()
            finally:
                shutdown_prometheus()

    @staticmethod
    def _complete(result: Future[Any], value: Any) -> None:
        if not result.done():
            result.set_result(value)

    @staticmethod
    def _complete_exception(result: Future[Any], exc: BaseException) -> None:
        if not result.done():
            result.set_exception(exc)

    @classmethod
    def _complete_or_chain(cls, result: Future[Any], value: Any) -> None:
        if isinstance(value, Future):
            value.add_done_callback(
                lambda completed: cls._copy_future_result(completed, result)
            )
        else:
            cls._complete(result, value)

    @classmethod
    def _copy_future_result(
        cls,
        source: Future[Any],
        destination: Future[Any],
    ) -> None:
        try:
            cls._complete(destination, source.result())
        except BaseException as exc:
            cls._complete_exception(destination, exc)

    def _enqueue(self, command: _Command) -> None:
        with self._state_lock:
            if self._state is _DriverState.FAILED:
                raise EngineDeadError() from self._failure
            if self._state is not _DriverState.RUNNING:
                raise EngineDeadError()
            try:
                self._commands.put_nowait(command)
            except queue.Full as exc:
                raise RuntimeError("sync-inproc engine command queue is full") from exc

    async def _submit_async(
        self,
        kind: _CommandKind,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        request: Optional[_RequestContext] = None,
    ) -> Any:
        result: Future[Any] = Future()
        self._enqueue(
            _Command(
                kind=kind,
                result=result,
                args=args,
                kwargs=kwargs,
                request=request,
            )
        )
        return await asyncio.wrap_future(result)

    async def generate(
        self,
        prompt: PromptType | EngineInput,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
        reasoning_ended: Optional[bool] = None,
        reasoning_parser_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        if sampling_params.n != 1:
            raise ValueError("sync-inproc mode supports sampling_params.n=1 only")
        if data_parallel_rank is not None:
            raise ValueError("sync-inproc mode does not support data_parallel_rank")
        if reasoning_ended is not None or reasoning_parser_kwargs is not None:
            raise ValueError("sync-inproc mode does not support reasoning state inputs")

        collector = RequestOutputCollector(sampling_params.output_kind, request_id)
        context = _RequestContext(
            request_id=request_id,
            collector=collector,
            loop=asyncio.get_running_loop(),
        )
        try:
            await self._submit_async(
                _CommandKind.ADD,
                args=(request_id, prompt, sampling_params),
                kwargs={
                    "lora_request": lora_request,
                    "tokenization_kwargs": tokenization_kwargs,
                    "trace_headers": trace_headers,
                    "priority": priority,
                    "prompt_text": prompt_text,
                },
                request=context,
            )
            finished = False
            while not finished:
                output = collector.get_nowait() or await collector.get()
                if not isinstance(output, RequestOutput):
                    raise RuntimeError(
                        f"sync-inproc generated unexpected output type {type(output)!r}"
                    )
                finished = output.finished
                yield output
        except (asyncio.CancelledError, GeneratorExit):
            await asyncio.shield(self.abort(request_id, internal=True))
            raise
        finally:
            collector.close()

    async def abort(
        self,
        request_id: str | Iterable[str],
        internal: bool = False,
    ) -> None:
        del internal
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        await self._submit_async(_CommandKind.ABORT, args=(request_ids,))

    async def check_health(self) -> None:
        await self._submit_async(_CommandKind.HEALTH)

    async def do_log_stats(self) -> None:
        if self.log_stats:
            await self._submit_async(_CommandKind.CALL, args=("do_log_stats",))

    async def start_profile(self, profile_prefix: Optional[str] = None) -> None:
        await self._submit_async(
            _CommandKind.CALL,
            args=("start_profile", profile_prefix),
        )

    async def stop_profile(self) -> None:
        await self._submit_async(_CommandKind.CALL, args=("stop_profile",))

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        return await self._submit_async(
            _CommandKind.CALL,
            args=("reset_prefix_cache", reset_running_requests, reset_connector),
        )

    async def sleep(
        self,
        level: int = 1,
        mode: Literal["abort", "wait", "keep"] = "abort",
    ) -> None:
        await self._submit_async(_CommandKind.CALL, args=("sleep", level, mode))

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        await self._submit_async(_CommandKind.CALL, args=("wake_up", tags))

    async def pause_generation(
        self,
        *,
        mode: Literal["abort", "wait", "keep"] = "abort",
        wait_for_inflight_requests: Optional[bool] = None,
        clear_cache: bool = True,
    ) -> None:
        if wait_for_inflight_requests:
            warnings.warn(
                "wait_for_inflight_requests is deprecated; use mode='wait'",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "wait"
        if clear_cache:
            await self._submit_async(
                _CommandKind.RENDERER_CALL,
                args=("clear_mm_cache",),
            )
        await self._submit_async(
            _CommandKind.CORE_CALL,
            args=("pause_scheduler",),
            kwargs={"mode": mode, "clear_cache": clear_cache},
        )
        await asyncio.sleep(0.02)

    async def resume_generation(self) -> None:
        await self._submit_async(
            _CommandKind.CORE_CALL,
            args=("resume_scheduler",),
        )

    async def collective_rpc(
        self,
        method: str | Callable[..., Any],
        timeout: Optional[float] = None,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        return await self._submit_async(
            _CommandKind.CORE_CALL,
            args=("collective_rpc", method, timeout, args, kwargs),
        )

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        del new_data_parallel_size
        raise RuntimeError(
            "sync-inproc mode does not support elastic expert parallelism"
        )

    def shutdown(self, timeout: Optional[float] = None) -> None:
        enqueue_shutdown = False
        with self._state_lock:
            if self._state is _DriverState.RUNNING:
                self._state = _DriverState.CLOSED
                enqueue_shutdown = True
        if enqueue_shutdown:
            self._commands.put(
                _Command(kind=_CommandKind.SHUTDOWN, result=Future()),
            )
        wait_timeout = timeout if timeout is not None else 30.0
        self._terminated.wait(wait_timeout)
        if self._thread.is_alive():
            raise TimeoutError("sync-inproc vLLM driver did not stop before timeout")


def validate_sync_inproc_config(
    engine_args: EngineArgs,
    vllm_config: VllmConfig,
) -> None:
    """Fail fast on vLLM features outside the experimental adapter's scope."""

    parallel = vllm_config.parallel_config
    unsupported: list[str] = []
    if parallel.tensor_parallel_size != 1:
        unsupported.append("tensor parallelism")
    if parallel.pipeline_parallel_size != 1:
        unsupported.append("pipeline parallelism")
    if parallel.data_parallel_size != 1:
        unsupported.append("data parallelism")
    if engine_args.enable_lora:
        unsupported.append("LoRA")
    if unsupported:
        raise ValueError(
            "--engine-client-mode=sync-inproc does not support "
            + ", ".join(unsupported)
        )
