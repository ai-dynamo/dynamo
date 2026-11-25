# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duru request handling and startup using the ThunderAgent router service."""

import asyncio
import json
import logging
import os
import time
import uuid
import weakref
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import uvloop

from dynamo.common.utils.token_ids import token_ids_len
from dynamo.duru_router.args import parse_args
from dynamo.duru_router.router import DuruScheduler
from dynamo.duru_router.scoring import Ingress, RouterAccounting, score_channel
from dynamo.llm import HttpError
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.thunderagent_router.__main__ import (
    ThunderAgentRouterHandler,
    _extract_program_id,
    _is_session_final,
    serve_router,
)

MAX_TRACKED_SESSIONS = 100000
logger = logging.getLogger(__name__)


class AccountingUnavailable(HttpError):
    """Reject generation while the router cannot calculate valid scores."""

    def __init__(self):
        super().__init__(
            503, "Duru accounting is unavailable; new admissions are stopped"
        )


@dataclass(eq=False)
class Request:
    stream: weakref.ReferenceType | None = None
    task: asyncio.Task | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    dispatched: bool = False


@dataclass
class Session:
    root: str
    agent: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent: str | None = None
    requests: set[Request] = field(default_factory=set)
    closing: bool = False
    ended: bool = False


class DuruRouterHandler(ThunderAgentRouterHandler):
    def __init__(self, runtime, config, *, accounting, ingress):
        super().__init__(runtime, config)
        self.scheduler_cls = partial(DuruScheduler, reader=accounting.reader)
        self.accounting = accounting
        self.ingress = ingress
        self.sessions = {}
        self.roots = {}
        self.requests = {}
        self.failure = None
        self.stopping = False

    def fail(self, error):
        if self.failure is not None:
            return
        self.failure = error
        logger.error("Duru accounting failed; rejecting new admissions", exc_info=error)
        if self._scheduler is not None:
            task = self._scheduler._scheduler_task
            if task is not None:
                task.cancel()
        current = asyncio.current_task()
        for request in self.requests.values():
            if (
                not request.dispatched
                and request.task is not None
                and request.task is not current
            ):
                request.task.cancel()

    def check(self):
        if self.stopping:
            raise RuntimeError("Duru router is stopping")
        if self.failure is not None:
            raise AccountingUnavailable() from self.failure
        try:
            self.accounting.check()
        except Exception as error:
            self.fail(error)
            raise AccountingUnavailable() from error

    def emit(self, event, **fields):
        try:
            self.ingress.emit(event, time.monotonic(), **fields)
        except Exception as error:
            self.fail(error)
            raise AccountingUnavailable() from error

    async def monitor(self):
        try:
            while not self.stopping:
                self.check()
                if self._scheduler is not None:
                    task = self._scheduler._scheduler_task
                    if task is not None and task.done():
                        raise RuntimeError("Duru admission scheduler stopped")
                self.emit("u_clock")
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.fail(error)
            raise AccountingUnavailable() from error

    def session(self, program_id, parent):
        current = self.sessions.get(program_id)
        if current is not None:
            if current.closing or (parent is not None and current.parent != parent):
                raise ValueError("Session has ended or changed parent")
            return current
        if len(self.sessions) >= MAX_TRACKED_SESSIONS:
            raise RuntimeError(
                "Duru session limit reached; send session-final notifications"
            )
        if parent is not None:
            ancestor = self.sessions.get(parent)
            if ancestor is None or ancestor.closing:
                raise ValueError(
                    "Parent session must be registered and open on this router"
                )
            root = ancestor.root
        else:
            # Reusing an externally supplied session ID after termination must not
            # join the previous root's late worker acknowledgements.
            root = uuid.uuid4().hex
        current = Session(root=root, parent=parent)
        self.sessions[program_id] = current
        self.roots.setdefault(root, set()).add(program_id)
        return current

    async def end_session(self, program_id):
        session = self.sessions.get(program_id)
        if session is None or session.ended:
            return
        if session.closing:
            raise ValueError("Session termination is already in progress")
        session.closing = True
        await self.cancel_requests(list(session.requests))
        await self._scheduler.end_program(program_id)
        session.ended = True
        members = self.roots[session.root]
        group_ended = all(self.sessions[pid].ended for pid in members)
        try:
            if self.failure is None:
                self.emit("agent_end", program_id=program_id, root=session.root)
                if group_ended:
                    self.emit("agent_group_end", root=session.root)
        finally:
            if group_ended:
                for pid in members:
                    del self.sessions[pid]
                del self.roots[session.root]

    async def _before_dispatch(self, request):
        self.check()
        identity = request["extra_args"]["sampling_options"]["vllm_xargs"]
        request_id = identity["duru_request_id"]
        self.emit("request_dispatched", request_id=request_id)
        self.requests[request_id].dispatched = True

    async def cancel_requests(self, requests):
        tasks = []
        for request in requests:
            if request.task is not None:
                request.task.cancel()
                tasks.append(request.task)
            else:
                stream = request.stream()
                if stream is not None:
                    tasks.append(asyncio.create_task(stream.aclose()))
        await asyncio.gather(*tasks, return_exceptions=True)
        # A dropped iterator may already be queued for async-generator finalization.
        await asyncio.gather(*(request.done.wait() for request in requests))

    def generate(self, request, context=None):
        state = Request()
        stream = self._generate(request, context, state)
        # Do not keep abandoned iterators alive and prevent their finalizers.
        state.stream = weakref.ref(stream)
        return stream

    async def _generate(self, request, context, state):
        if self._scheduler is None or self._kv_router is None:
            raise RuntimeError("Duru router used before initialization")
        program_id = _extract_program_id(request)
        if _is_session_final(request):
            if program_id is None:
                raise ValueError("Session-final requires a session ID")
            self._stat_requests_total += 1
            self._stat_session_final_requests += 1
            await self.end_session(program_id)
            return
        self.check()
        tokens = token_ids_len(request.get("token_ids"))
        if tokens <= 0:
            raise ValueError("Duru requires a nonempty token prompt")
        anonymous = program_id is None
        program_id = program_id or uuid.uuid4().hex
        agent_context = dict(request.get("agent_context") or {})
        parent = agent_context.get("parent_session_id")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ValueError("Parent session ID must be a nonempty string")
        request_id = uuid.uuid4().hex
        extra = dict(request.get("extra_args") or {})
        options = dict(extra.get("sampling_options") or {})
        xargs = dict((request.get("sampling_options") or {}).get("vllm_xargs") or {})
        xargs.update(options.get("vllm_xargs") or {})
        session = self.session(program_id, parent)
        xargs.update(
            duru_request_id=request_id,
            duru_root_id=session.root,
            duru_agent_id=session.agent,
        )
        options["vllm_xargs"] = xargs
        extra["sampling_options"] = options
        agent_context["session_id"] = program_id
        prepared = dict(request, extra_args=extra, agent_context=agent_context)
        state.task = asyncio.current_task()
        session.requests.add(state)
        self.requests[request_id] = state
        outcome = "request_completed"
        try:
            self.emit(
                "request_arrived",
                request_id=request_id,
                program_id=program_id,
                root=session.root,
                agent=session.agent,
                prompt_tokens=tokens,
            )
            async with aclosing(super().generate(prepared, context=context)) as stream:
                while True:
                    try:
                        chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                    state.task = None
                    try:
                        yield chunk
                    finally:
                        # The runtime may use a new task for each __anext__.
                        state.task = asyncio.current_task()
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "request_cancelled"
            if self.failure is not None and not state.dispatched:
                raise AccountingUnavailable() from self.failure
            raise
        except Exception:
            outcome = "request_failed"
            if not state.dispatched:
                self.check()
            raise
        finally:
            session.requests.discard(state)
            self.requests.pop(request_id, None)
            try:
                try:
                    if self.failure is None:
                        self.emit(outcome, request_id=request_id)
                finally:
                    if anonymous:
                        await self.end_session(program_id)
            except AccountingUnavailable:
                # A reporting failure must not replace a dispatched request's
                # successful result, engine error, or client cancellation.
                if not state.dispatched:
                    raise
            finally:
                state.task = None
                state.done.set()

    async def status(self, request=None):
        try:
            self.check()
            snapshot = self.accounting.reader.refresh()
        except Exception as error:
            yield {
                "status": "unhealthy",
                "component": "duru_router",
                "error": str(error),
            }
            return
        if not snapshot:
            yield {"status": "starting", "component": "duru_router"}
            return
        async for value in super().status(request):
            value["component"] = "duru_router"
            yield value

    async def shutdown(self):
        self.stopping = True
        await self.cancel_requests(
            [
                request
                for session in self.sessions.values()
                for request in session.requests
            ]
        )
        await super().shutdown()
        self.sessions.clear()
        self.roots.clear()
        self.requests.clear()


async def serve(runtime, config, handler):
    serving = asyncio.create_task(serve_router(runtime, config, handler))
    monitoring = asyncio.create_task(handler.monitor())
    try:
        done, _ = await asyncio.wait(
            (serving, monitoring), return_when=asyncio.FIRST_COMPLETED
        )
        if monitoring in done:
            try:
                monitoring.result()
            except AccountingUnavailable:
                # Keep endpoints available to return explicit request errors
                # and status, while already-dispatched requests finish.
                await serving
                return
        for task in done:
            task.result()
    finally:
        serving.cancel()
        monitoring.cancel()
        await asyncio.gather(serving, monitoring, return_exceptions=True)
        await handler.shutdown()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    accounting_config = os.environ["DURU_ACCOUNTING_CONFIG"]
    settings = json.loads(Path(accounting_config).read_text())
    ingress_path = Path(settings["ingress"])
    if ingress_path.name != "duru-ingress.jsonl":
        raise ValueError("Ingress channel must be named duru-ingress.jsonl")
    os.environ["DURU_SCORES"] = settings["scores"]
    score_channel()
    ingress = Ingress(ingress_path.parent, in_process=True)
    accounting = RouterAccounting(accounting_config, ingress=ingress)
    try:
        accounting.start()
        handler = DuruRouterHandler(
            runtime, config, accounting=accounting, ingress=ingress
        )
        await serve(runtime, config, handler)
    finally:
        try:
            await asyncio.to_thread(ingress.close)
        finally:
            await asyncio.to_thread(accounting.close)


def main():
    for name in ("DURU_ACCOUNTING_CONFIG",):
        if not os.environ.get(name):
            raise ValueError(f"Set {name} before starting the router")
    uvloop.run(worker())


if __name__ == "__main__":
    main()
