# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous session API, hosted by a frontend-owned helper process."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import msgspec
from aiohttp import web

from .collectors import CollectionError, PerfCollector, TorchCollector
from .models import Config, Plan

logger = logging.getLogger(__name__)
COLLECTION_ERRORS = (
    CollectionError,
    OSError,
    asyncio.TimeoutError,
    aiohttp.ClientError,
    ValueError,
)


@dataclass
class Session:
    session_id: str
    plan: Plan
    directory: Path
    state: str = "accepted"
    accepted_at_ns: int = field(default_factory=time.time_ns)
    collectors: list[PerfCollector | TorchCollector] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    deadline_monotonic: float = 0.0


class ProfilingService:
    """Own one profiling lease; client retries never extend its deadline."""

    def __init__(self, config: Config, frontend_pid: int, frontend_epoch: str):
        if config.max_duration_ms <= 0 or config.max_sessions <= 0:
            raise ValueError("Profiling limits must be positive")
        if config.start_timeout_s <= 0 or config.finalize_timeout_s <= 0:
            raise ValueError("Profiling timeouts must be positive")
        self.config = config
        self.frontend_pid = frontend_pid
        self.frontend_epoch = frontend_epoch
        self.workers = {worker.target_id: worker for worker in config.workers}
        if len(self.workers) != len(config.workers) or "frontend" in self.workers:
            raise ValueError("Worker target IDs must be unique and cannot be frontend")
        self.sessions: dict[str, Session] = {}
        self.requests: dict[str, Session] = {}
        self.active: Session | None = None

    def targets(self) -> list[dict]:
        return [
            {
                "target_id": "frontend",
                "process_epoch": self.frontend_epoch,
                "collectors": ["linux_perf"]
                if self.config.frontend_perf_enabled
                else [],
                "scope": "process_and_threads",
            },
            *[
                {
                    "target_id": worker.target_id,
                    "process_epoch": worker.process_epoch,
                    "collectors": ["torch_profiler"],
                    "scope": "engine_group",
                    "identity_verification": "operator_config_only",
                }
                for worker in self.workers.values()
            ],
        ]

    def validate(self, plan: Plan) -> None:
        if not plan.client_request_id or len(plan.client_request_id) > 128:
            raise web.HTTPBadRequest(
                text="client_request_id must contain 1..128 characters"
            )
        if plan.mode != "torch_profile":
            raise web.HTTPUnprocessableEntity(
                text="nsys adapter is deferred; requires a prepared launch/session"
            )
        if not 0 < plan.duration_ms <= self.config.max_duration_ms:
            raise web.HTTPBadRequest(
                text="duration_ms exceeds configured capture budget"
            )
        target_ids = [target.target_id for target in plan.targets]
        if not target_ids or len(target_ids) != len(set(target_ids)):
            raise web.HTTPBadRequest(text="targets must be nonempty and unique")
        epochs = {item["target_id"]: item["process_epoch"] for item in self.targets()}
        for target in plan.targets:
            if target.target_id not in epochs:
                raise web.HTTPBadRequest(text="Unknown target; use GET targets")
            if epochs[target.target_id] != target.process_epoch:
                raise web.HTTPConflict(
                    text="Target process epoch does not match inventory"
                )
        if not self.config.frontend_perf_enabled and "frontend" in target_ids:
            raise web.HTTPUnprocessableEntity(
                text="Frontend profiling is disabled by operator configuration"
            )
        # Frontend perf is implicit unless explicitly disabled by the operator.

    def submit(self, plan: Plan) -> Session:
        previous = self.requests.get(plan.client_request_id)
        if previous is not None:
            if previous.plan != plan:
                raise web.HTTPConflict(
                    text="client_request_id already has a different plan"
                )
            return previous
        self.validate(plan)
        if self.active is not None:
            raise web.HTTPConflict(
                text="A profiling session is already active or finalizing"
            )
        if len(self.sessions) >= self.config.max_sessions:
            raise web.HTTPServiceUnavailable(
                text="PoC session retention limit reached; restart helper"
            )
        session_id = uuid.uuid4().hex
        directory = Path(self.config.output_dir) / session_id
        directory.mkdir(parents=True, mode=0o700)
        session = Session(session_id, plan, directory)
        session.deadline_monotonic = (
            asyncio.get_running_loop().time() + plan.duration_ms / 1000
        )
        self.active = session
        self.sessions[session_id] = session
        self.requests[plan.client_request_id] = session
        self.persist(session)
        session.task = asyncio.create_task(self.capture(session))
        return session

    def manifest(self, session: Session) -> dict:
        stop_due = session.accepted_at_ns + session.plan.duration_ms * 1000000
        return {
            "schema_version": 1,
            "session_id": session.session_id,
            "state": session.state,
            "plan": msgspec.to_builtins(session.plan),
            "accepted_at_ns": session.accepted_at_ns,
            "stop_due_at_ns": stop_due,
            "frontend": {
                "pid": self.frontend_pid,
                "process_epoch": self.frontend_epoch,
                "profiling_enabled": self.config.frontend_perf_enabled,
                "omission_reason": None
                if self.config.frontend_perf_enabled
                else "disabled_by_operator_config",
            },
            "profiles": [collector.evidence for collector in session.collectors],
            "lifecycle_otel": {
                "trace_ids": session.plan.trace_ids,
                "request_ids": session.plan.request_ids,
                "evidence_location": self.config.lifecycle_evidence_location,
                "coverage": "unknown",
                "instrumentation_mode": "unchanged",
                "export_status": "external_reference"
                if self.config.lifecycle_evidence_location
                else "not_configured",
            },
            "clock": {"timestamps": "unix_ns", "cross_host_alignment": "unknown"},
            "errors": session.errors,
        }

    def persist(self, session: Session) -> None:
        pending = session.directory / "manifest.json.tmp"
        pending.write_text(json.dumps(self.manifest(session), indent=2) + "\n")
        pending.replace(session.directory / "manifest.json")

    async def capture(self, session: Session) -> None:
        deadline = session.deadline_monotonic
        session.state = "starting"
        self.persist(session)
        candidates = []
        if self.config.frontend_perf_enabled:
            candidates.append(
                PerfCollector(self.frontend_pid, self.frontend_epoch, session.directory)
            )
        candidates.extend(
            TorchCollector(self.workers[target.target_id], self.config)
            for target in session.plan.targets
            if target.target_id != "frontend"
        )
        capture_ok = False
        try:
            for collector in candidates:
                if session.stop_requested.is_set():
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise CollectionError("Capture deadline expired during startup")
                # Register before starting: a timeout may have taken effect
                # remotely, so cleanup must also attempt stop on uncertain starts.
                session.collectors.append(collector)
                await asyncio.wait_for(
                    collector.start(), min(remaining, self.config.start_timeout_s)
                )
                self.persist(session)
            session.state = "recording"
            self.persist(session)
            remaining = max(0, deadline - asyncio.get_running_loop().time())
            try:
                await asyncio.wait_for(session.stop_requested.wait(), remaining)
            except asyncio.TimeoutError:
                pass  # The lease deadline is a normal stop trigger.
            capture_ok = True
        except COLLECTION_ERRORS as exc:
            logger.warning(
                "Profiling startup failed for %s: %s", session.session_id, exc
            )
            session.errors.append(str(exc) or type(exc).__name__)
        finally:
            if not capture_ok and not session.errors:
                session.errors.append("Capture interrupted by an unexpected failure")
            session.state = "finalizing"
            self.persist(session)
            # Stop all collectors concurrently: slow worker trace serialization
            # must not keep frontend perf recording beyond its lease.
            results = await asyncio.gather(
                *(self.stop_collector(collector) for collector in session.collectors)
            )
            session.errors.extend(error for error in results if error is not None)
            session.state = "failed" if session.errors else "completed"
            self.persist(session)
            self.active = None

    async def stop_collector(
        self, collector: PerfCollector | TorchCollector
    ) -> str | None:
        try:
            await collector.stop()
        except COLLECTION_ERRORS as exc:
            collector.evidence["coverage"] = "partial_or_unknown"
            logger.warning("Profiling finalization failed: %s", exc)
            return str(exc) or type(exc).__name__
        return None

    async def close(self) -> None:
        if self.active is not None:
            self.active.stop_requested.set()
            if self.active.task is not None:
                await self.active.task

    def application(self) -> web.Application:
        app = web.Application(client_max_size=65536)

        async def targets(request: web.Request) -> web.Response:
            return web.json_response({"targets": self.targets()})

        async def read_plan(request: web.Request) -> Plan:
            try:
                return msgspec.json.decode(await request.read(), type=Plan)
            except msgspec.DecodeError as exc:
                raise web.HTTPBadRequest(text=str(exc)) from exc

        async def validate(request: web.Request) -> web.Response:
            plan = await read_plan(request)
            self.validate(plan)
            return web.json_response(
                {
                    "valid": True,
                    "frontend_collector": "linux_perf"
                    if self.config.frontend_perf_enabled
                    else None,
                    "worker_collector": "torch_profiler",
                }
            )

        async def submit(request: web.Request) -> web.Response:
            session = self.submit(await read_plan(request))
            return web.json_response(self.manifest(session), status=202)

        def lookup(request: web.Request) -> Session:
            session = self.sessions.get(request.match_info["session_id"])
            if session is None:
                raise web.HTTPNotFound(text="Unknown profiling session")
            return session

        async def status(request: web.Request) -> web.Response:
            return web.json_response(self.manifest(lookup(request)))

        async def stop(request: web.Request) -> web.Response:
            session = lookup(request)
            session.stop_requested.set()
            return web.json_response(self.manifest(session), status=202)

        prefix = "/admin/profiling/v1"
        app.add_routes(
            [
                web.get(prefix + "/targets", targets),
                web.post(prefix + "/plans/validate", validate),
                web.post(prefix + "/sessions", submit),
                web.get(prefix + "/sessions/{session_id}", status),
                web.post(prefix + "/sessions/{session_id}/stop", stop),
                web.get(prefix + "/sessions/{session_id}/artifacts", status),
            ]
        )
        return app
