# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang request cancellation and shutdown coordination."""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator

from dynamo._core import Context
from dynamo.llm.exceptions import EngineShutdown
from dynamo.sglang._compat import resolved_server_args

_CANCELLATION_POLL_MAX_DELAY_S = 0.05
_CANCELLATION_ABORT_RETRY_LIMIT = 8
_CANCELLATION_DRAIN_TIMEOUT_S = 1.0


def _consume_detached_task(task: asyncio.Task[Any]) -> None:
    """Consume a detached cancellation task result without surfacing closure."""
    try:
        task.result()
    except (asyncio.CancelledError, StopAsyncIteration):
        pass
    except Exception:
        logging.exception("Detached SGLang task failed during cancellation")


def _cancel_and_detach(task: asyncio.Task[Any]) -> None:
    """Request cancellation without allowing resistant iterators to block cleanup."""
    task.cancel()
    task.add_done_callback(_consume_detached_task)


class CancellationMixin:
    """Coordinate Dynamo cancellation with SGLang request aborts."""

    engine: Any
    config: Any
    shutdown_event: asyncio.Event | None
    _abort_retry_tasks: set[asyncio.Task[Any]]

    def _start_abort_retry(
        self,
        tokenizer_manager: Any,
        request_id: str,
        registry: Mapping[str, Any],
        state: Any,
    ) -> None:
        retry_task = asyncio.create_task(
            self._retry_abort(tokenizer_manager, request_id, registry, state)
        )
        self._abort_retry_tasks.add(retry_task)
        retry_task.add_done_callback(self._abort_retry_tasks.discard)
        retry_task.add_done_callback(_consume_detached_task)

    def _cancel_abort_retries(self) -> None:
        for retry_task in tuple(self._abort_retry_tasks):
            retry_task.cancel()

    async def _stream_until_cancelled(
        self,
        stream_source: AsyncIterator[Any],
        cancellation_task: asyncio.Task[Any],
    ) -> AsyncGenerator[Any, None]:
        """Drain briefly after abort, then stop a stream that did not terminate."""
        iterator = aiter(stream_source)
        drain_deadline: float | None = None
        while True:
            next_item = asyncio.create_task(anext(iterator))
            try:
                if drain_deadline is None:
                    done, _ = await asyncio.wait(
                        (next_item, cancellation_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if next_item in done:
                        try:
                            yield next_item.result()
                        except StopAsyncIteration:
                            return
                        continue
                    cancellation_task.result()
                    drain_deadline = (
                        asyncio.get_running_loop().time()
                        + _CANCELLATION_DRAIN_TIMEOUT_S
                    )

                remaining = drain_deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    logging.warning(
                        "Timed out draining SGLang stream after cancellation"
                    )
                    return
                done, _ = await asyncio.wait((next_item,), timeout=remaining)
                if next_item not in done:
                    logging.warning(
                        "Timed out draining SGLang stream after cancellation"
                    )
                    return
                try:
                    yield next_item.result()
                except StopAsyncIteration:
                    return
            finally:
                if not next_item.done():
                    _cancel_and_detach(next_item)

    @asynccontextmanager
    async def _wait_for_signal(self, context: Context) -> AsyncGenerator[bool, None]:
        cancellation_future = context.async_killed_or_stopped()
        shutdown_task = (
            asyncio.create_task(self.shutdown_event.wait())
            if self.shutdown_event
            else None
        )
        wait_for = [cancellation_future]
        if shutdown_task is not None:
            wait_for.append(shutdown_task)
        try:
            done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)
            yield shutdown_task is not None and shutdown_task in done
        finally:
            pending = [awaitable for awaitable in wait_for if not awaitable.done()]
            for awaitable in pending:
                awaitable.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _request_registry(
        tokenizer_manager: Any, submitted_request_id: str | None
    ) -> Mapping[str, Any] | None:
        if tokenizer_manager is None or submitted_request_id is None:
            return None
        candidate = getattr(tokenizer_manager, "rid_to_state", None)
        return candidate if isinstance(candidate, Mapping) else None

    async def _wait_for_registration(
        self,
        submitted_request_id: str,
        request_id_future: asyncio.Future,
        registry: Mapping[str, Any],
    ) -> str:
        delay = 0.001
        while submitted_request_id not in registry:
            if request_id_future.done():
                return request_id_future.result()
            await asyncio.sleep(delay)
            delay = min(delay * 2, _CANCELLATION_POLL_MAX_DELAY_S)
        return submitted_request_id

    @staticmethod
    def _resolved_request_id(request_id_future: asyncio.Future) -> str:
        if request_id_future.done() and not request_id_future.cancelled():
            try:
                return request_id_future.result()
            except Exception:
                pass
        return "unknown"

    async def _handle_cancellation(
        self,
        request_id_future: asyncio.Future,
        context: Context,
        submitted_request_id: str | None = None,
    ) -> None:
        """Wait for cancellation, then order an exact SGLang abort."""
        logging.debug("Cancellation monitor started for Context: %s", context.id())
        request_id = submitted_request_id
        try:
            if request_id is None:
                request_id = await request_id_future
            async with self._wait_for_signal(context) as shutdown_requested:
                tokenizer_manager = getattr(self.engine, "tokenizer_manager", None)
                registry = self._request_registry(
                    tokenizer_manager, submitted_request_id
                )
                if submitted_request_id is not None and registry is not None:
                    request_id = await self._wait_for_registration(
                        submitted_request_id, request_id_future, registry
                    )
                logging.info(
                    "Cancellation or shutdown signal received for SGLang Request ID %s, Context: %s",
                    request_id,
                    context.id(),
                )
                if tokenizer_manager is None:
                    logging.error(
                        "SGLang tokenizer_manager not found for abort request: %s",
                        context.id(),
                    )
                else:
                    await self._abort_sglang_request(
                        tokenizer_manager, request_id, registry, context.id()
                    )
                if shutdown_requested:
                    raise EngineShutdown("Engine was shut down during token generation")
        except asyncio.CancelledError:
            logging.debug(
                "Cancellation monitor task cancelled for SGLang Request ID %s, Context: %s",
                self._resolved_request_id(request_id_future),
                context.id(),
            )
            raise

    async def _wait_for_dispatch(
        self,
        registry: Mapping[str, Any],
        request_id: str,
        state: Any,
    ) -> bool:
        time_stats = getattr(state, "time_stats", None)
        if not hasattr(time_stats, "api_server_dispatch_finish_time"):
            return False
        delay = 0.001
        while registry.get(request_id) is state:
            if time_stats.api_server_dispatch_finish_time:
                return True
            await asyncio.sleep(delay)
            delay = min(delay * 2, _CANCELLATION_POLL_MAX_DELAY_S)
        return False

    @staticmethod
    def _submit_abort(tokenizer_manager: Any, request_id: str, context_id: str) -> None:
        tokenizer_manager.abort_request(rid=request_id, abort_all=False)
        logging.info("Aborted Request ID: %s", context_id)

    @staticmethod
    def _requires_abort_retry(server_args: Any, dispatch_observed: bool) -> bool:
        return (
            not dispatch_observed
            or getattr(server_args, "pp_size", 1) > 1
            or (
                getattr(server_args, "enable_dp_attention", False)
                and not getattr(
                    server_args, "enable_dp_attention_local_control_broadcast", False
                )
            )
        )

    async def _retry_abort(
        self,
        tokenizer_manager: Any,
        request_id: str,
        registry: Mapping[str, Any],
        state: Any,
    ) -> None:
        delay = 0.05
        for retry in range(_CANCELLATION_ABORT_RETRY_LIMIT):
            await asyncio.sleep(delay)
            if registry.get(request_id) is not state:
                return
            logging.debug("Retrying SGLang abort_request for Request ID %s", request_id)
            tokenizer_manager.abort_request(rid=request_id, abort_all=False)
            delay = min(delay * 2, 1.0)
        if registry.get(request_id) is state:
            logging.warning(
                "SGLang request %s remained registered after %d abort retries",
                request_id,
                retry + 1,
            )

    async def _abort_sglang_request(
        self,
        tokenizer_manager: Any,
        request_id: str,
        registry: Mapping[str, Any] | None,
        context_id: str,
    ) -> None:
        state = registry.get(request_id) if registry is not None else None
        if registry is None or state is None:
            self._submit_abort(tokenizer_manager, request_id, context_id)
            return
        dispatch_observed = await self._wait_for_dispatch(registry, request_id, state)
        if registry.get(request_id) is not state:
            return
        self._submit_abort(tokenizer_manager, request_id, context_id)
        server_args = resolved_server_args(self.config.server_args)
        if self._requires_abort_retry(server_args, dispatch_observed):
            self._start_abort_retry(tokenizer_manager, request_id, registry, state)

    @asynccontextmanager
    async def _cancellation_monitor(
        self,
        request_id_future: asyncio.Future,
        context: Context,
        submitted_request_id: str | None = None,
    ) -> AsyncGenerator[asyncio.Task, None]:
        """Own the cancellation monitor task for one response stream."""
        logging.debug(
            "Creating cancellation monitor task for Context: %s", context.id()
        )
        cancellation_task = asyncio.create_task(
            self._handle_cancellation(request_id_future, context, submitted_request_id)
        )

        try:
            yield cancellation_task
        finally:
            request_id = self._resolved_request_id(request_id_future)
            if not cancellation_task.done():
                logging.debug(
                    "Cancelling cancellation monitor task for SGLang Request ID %s, Context: %s",
                    request_id,
                    context.id(),
                )
                cancellation_task.cancel()
                try:
                    await cancellation_task
                except asyncio.CancelledError:
                    pass
            else:
                cancellation_task.result()

            if self.shutdown_event and self.shutdown_event.is_set():
                raise EngineShutdown("Engine was shut down during token generation")
