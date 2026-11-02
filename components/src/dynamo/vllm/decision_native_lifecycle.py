# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own native initialization, bounded child execution, and cancellation cleanup."""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from dynamo.llm.exceptions import DynamoException, EngineShutdown, HttpError

from .decision_native_errors import NativeExecutionError, NativeRuntimeError
from .decision_native_loader import NativeApi, load_native
from .decision_native_registry import PROFILES, NativeProfile
from .decision_native_wire import MAX_NATIVE_QUESTIONS

MAX_NATIVE_CHILD_ID_LENGTH = 256
NATIVE_HEALTH_CACHE_SALT = "dynamo-native-head-health-v1"
# Heads append -head-N; reads append -N or -prefix within the same child-ID budget.
MAX_NATIVE_REQUEST_ID_LENGTH = MAX_NATIVE_CHILD_ID_LENGTH - max(
    len(f"-head-{MAX_NATIVE_QUESTIONS - 1}"), len("-prefix")
)


def _create_strategy(engine: Any, native: NativeApi, name: str, max_concurrency: int):
    """Initialize a strategy after the adapter validates the effective runtime."""
    capacity = engine.vllm_config.scheduler_config.max_num_seqs
    if (
        type(max_concurrency) is not int
        or not 1 <= max_concurrency <= 64
        or type(capacity) is not int
        or capacity < 1
    ):
        raise NativeRuntimeError("native concurrency is invalid")
    try:
        renderer = native.online_renderer(
            model_config=engine.model_config,
            renderer=engine.renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        context = native.strategies.ReadContext(
            engine_client=engine,
            online_renderer=renderer,
            chat_template=None,
            chat_template_content_format="auto",
            default_chat_template_kwargs={},
        )
        strategy = getattr(native.strategies, name)(context)
    except (
        ValueError,
        KeyError,
        AttributeError,
        TypeError,
        native.template_error,
    ) as cause:
        raise NativeRuntimeError(
            "native strategy cannot initialize on the effective runtime"
        ) from cause
    return strategy, min(max_concurrency, capacity)


_T = TypeVar("_T")


async def run_with_context_cancellation(
    operation: Awaitable[_T], context: Any, cancelled: Callable[[], bool]
) -> _T:
    """Cancel and drain owned Python work when the runtime context terminates.

    This does not establish physical GPU abort settlement. Actual admitted
    children still use the handler's abort monitor and explicit engine cleanup.
    """
    task = asyncio.ensure_future(operation)
    signal = None
    try:
        if cancelled():
            raise asyncio.CancelledError
        # The PyO3 runtime returns an awaitable, not necessarily a coroutine.
        signal = asyncio.ensure_future(context.async_killed_or_stopped())
        done, _ = await asyncio.wait(
            (task, signal), return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return await task
        if signal in done:
            await signal
            raise asyncio.CancelledError
        return await task
    finally:
        owned = [task] if signal is None else [task, signal]
        for pending in owned:
            if not pending.done():
                pending.cancel()
        await asyncio.gather(*owned, return_exceptions=True)


async def translate_native_errors(
    operation: Awaitable[_T], native: Any, on_engine_dead
) -> _T:
    """Preserve source-qualified engine failure and admission semantics."""
    try:
        return await operation
    except native.overload_errors as cause:
        raise native.overload_error(
            "Native decision engine capacity is exhausted"
        ) from cause
    except native.engine_generate_error as cause:
        raise HttpError(503, "Native decision engine generation failed") from cause
    except native.retryable_request_error as cause:
        raise HttpError(
            503, "Native decision request temporarily unavailable"
        ) from cause
    except native.engine_dead_error as cause:
        if on_engine_dead is not None:
            on_engine_dead()
        raise EngineShutdown("Native decision engine is shut down") from cause


async def run_children(
    count: int, concurrency: int, operation: Callable[[int], Awaitable[_T]]
) -> list[_T]:
    """Bound child tasks; cancel and drain every owned child on terminal failure."""
    results: list[Any] = [None] * count
    indices = iter(range(count))

    async def worker():
        for index in indices:
            results[index] = await operation(index)

    tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, count))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


@dataclass(frozen=True)
class NativeChildBatch:
    count: int
    operation: Callable[[int], Awaitable[Any]]


class NativeAdapter(ABC):
    """One owner for native initialization, cancellation, capacity and child cleanup."""

    kind: str

    @staticmethod
    @abstractmethod
    def validate_runtime(engine: Any, selected: NativeProfile, tasks: tuple[str, ...]):
        """Check profile-specific runtime requirements before initialization."""

    @abstractmethod
    async def _prepare(self, wire: Any):
        """Validate every input before admission, guarding private-tokenizer work."""

    @abstractmethod
    def _children(
        self, prepared, context, *, request_id, cancelled, trace_headers
    ) -> Iterable[NativeChildBatch]:
        """Plan ordered batches; any warmup must finish before the next batch."""

    @abstractmethod
    def _respond(self, prepared, results) -> dict[str, Any]:
        """Project completed, validated child evidence into the private response."""

    def __init__(
        self,
        engine,
        profile: NativeProfile,
        native,
        strategy,
        abort_monitor,
        concurrency,
        *,
        on_engine_dead=None,
    ):
        self.engine, self.spec, self.native = engine, profile, native
        self.strategy = strategy
        self.abort_monitor, self.concurrency = abort_monitor, concurrency
        self.on_engine_dead = on_engine_dead
        self.semaphore = asyncio.Semaphore(concurrency)
        tokenizer = engine.renderer.get_tokenizer()
        try:
            self.preflight_tokenizer = copy.deepcopy(tokenizer)
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if self.preflight_tokenizer is tokenizer or (
                backend is not None
                and getattr(self.preflight_tokenizer, "backend_tokenizer", None)
                is backend
            ):
                raise NativeRuntimeError(
                    "native preflight requires a private tokenizer"
                )
        except (NativeRuntimeError, EngineShutdown):
            raise
        except Exception as cause:
            raise NativeRuntimeError(
                "native preflight requires a private tokenizer"
            ) from cause
        self.preflight_lock = asyncio.Lock()
        self._initialize_profile()

    def _initialize_profile(self):
        """Validate and initialize profile state after all shared fields exist."""

    def _check_canary(self, prepared):
        """Check profile-specific preparation constraints before advertisement."""

    async def run_preflight(self, operation: Callable[[], _T]) -> _T:
        """Serialize private-tokenizer work off-loop and drain started CPU work.

        The shared renderer tokenizer never enters this worker thread. Cancellation
        cannot stop a running tokenizer; retain its guard until that work settles.
        """
        async with self.preflight_lock:
            task = asyncio.create_task(asyncio.to_thread(operation))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task.done():
                        raise
                    cancelled = True
                except BaseException:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
                else:
                    if cancelled:
                        raise asyncio.CancelledError
                    return result

    @classmethod
    async def create(
        cls,
        engine,
        profile: NativeProfile,
        *,
        abort_monitor,
        max_concurrency=4,
        on_engine_dead=None,
    ):
        selected = profile
        if (
            not isinstance(selected, NativeProfile)
            or PROFILES.get(selected.name) != selected
            or selected.kind != cls.kind
        ):
            raise NativeRuntimeError("unsupported native execution profile")
        native = load_native()
        tasks = await translate_native_errors(
            engine.get_supported_tasks(), native, on_engine_dead
        )
        cls.validate_runtime(engine, selected, tuple(tasks))
        strategy, capacity = _create_strategy(
            engine, native, selected.strategy, max_concurrency
        )
        actual_limit = strategy.limits().max_options
        if type(actual_limit) is not int or actual_limit < 2:
            raise NativeRuntimeError("native strategy has an invalid candidate limit")
        qualified = replace(
            selected, max_candidates=min(actual_limit, selected.max_candidates)
        )
        return cls(
            engine,
            qualified,
            native,
            strategy,
            abort_monitor,
            capacity,
            on_engine_dead=on_engine_dead,
        )

    async def prepare_canary(self):
        """Prepare the bounded startup request and verify its profile contract."""
        payload = {
            f"decision_{self.kind}": {
                "protocol_version": 1,
                "profile": self.spec.name,
                "model": "dynamo-native-canary",
                "input": "probe",
                "questions": [
                    {
                        "ordinal": 0,
                        "kind": "Predicate",
                        "instructions": "Is this a health probe?",
                        "candidates": [{"value": True}, {"value": False}],
                    }
                ],
                "temperature": 1,
                "cache_salt": NATIVE_HEALTH_CACHE_SALT,
                "max_input_tokens": min(self.engine.model_config.max_model_len, 4096),
                "max_tokenization_work": 8192,
                "max_encoding_bytes": 16384,
            }
        }
        prepared = await self._prepare(payload)
        self._check_canary(prepared)
        return payload, prepared

    async def execute(
        self, wire, context, *, request_id, cancelled, trace_headers=None
    ):
        if (
            not isinstance(request_id, str)
            or not 1 <= len(request_id) <= MAX_NATIVE_REQUEST_ID_LENGTH
        ):
            raise NativeExecutionError("invalid native engine request ID")

        async def run():
            if cancelled():
                raise asyncio.CancelledError
            prepared = await self._prepare(wire)
            if cancelled():
                raise asyncio.CancelledError
            # Advisory fast-fail: each child can still hit a live queue limit.
            # translate_native_errors preserves late overloads after sibling cleanup.
            self.engine.check_admission(prepared.admission_count)
            results = []
            for batch in self._children(
                prepared,
                context,
                request_id=request_id,
                cancelled=cancelled,
                trace_headers=trace_headers,
            ):
                results.extend(
                    await run_children(batch.count, self.concurrency, batch.operation)
                )
            if cancelled():
                raise asyncio.CancelledError
            return self._respond(prepared, results)

        try:
            return await translate_native_errors(
                run_with_context_cancellation(run(), context, cancelled),
                self.native,
                self.on_engine_dead,
            )
        except (DynamoException, HttpError):
            raise
        except Exception as cause:
            raise HttpError(500, "native decision executor failed") from cause

    async def read_child(
        self, factory, request_id, context, cancelled, validate, *, admission=None
    ):
        async with self.semaphore if admission is None else admission:
            if cancelled():
                raise asyncio.CancelledError
            generator = factory()
            succeeded = False
            try:
                terminal = None
                async with self.abort_monitor(context, request_id):
                    async for output in generator:
                        if cancelled():
                            raise asyncio.CancelledError
                        if terminal is not None:
                            raise NativeExecutionError(
                                "native engine returned data after its terminal"
                            )
                        if (
                            output.request_id != request_id
                            or getattr(output, "error", None) is not None
                        ):
                            raise NativeExecutionError(
                                "native engine failed or returned a different request"
                            )
                        if output.finished is True:
                            terminal = output
                        elif output.finished is not False:
                            raise NativeExecutionError(
                                "native engine returned an invalid terminal marker"
                            )
                    if terminal is None:
                        raise NativeExecutionError(
                            "native engine did not return a finished result"
                        )
                    result = validate(terminal)
                succeeded = True
                return result
            except self.native.validation_error as cause:
                raise NativeExecutionError(
                    "native engine rejected a prepared request"
                ) from cause
            finally:
                try:
                    if not succeeded:
                        await self.engine.abort(request_id)
                finally:
                    await generator.aclose()
