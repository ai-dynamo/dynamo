# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Callable coordinator; HTTP and process hosting are separate adapters."""

import asyncio
import json
import logging
import os
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import msgspec

from .adapters import Collector, CollectorRegistry, default_registry
from .artifacts import ArtifactValidator, SharedFilesystemValidator
from .clock import ClockSampler, WorkerClockSampler
from .collectors import CollectionError
from .errors import BadRequest, Conflict, NotFound, Unavailable, Unsupported
from .models import Config, Plan, RuntimeSnapshot

logger = logging.getLogger(__name__)
COLLECTION_ERRORS = (
    CollectionError,
    OSError,
    asyncio.TimeoutError,
    aiohttp.ClientError,
    ValueError,
)


async def errors_of(operations) -> list[str]:
    """Run operations concurrently and keep only the errors they report."""
    return [error for error in await asyncio.gather(*operations) if error]


@dataclass
class Session:
    session_id: str
    plan: Plan
    directory: Path
    state: str = "accepted"
    accepted_at_ns: int = field(default_factory=time.time_ns)
    collectors: list[Collector] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    deadline_monotonic: float = 0.0
    runtime: RuntimeSnapshot | None = None
    clocks: dict = field(default_factory=dict)
    artifact_validation: list[dict] = field(default_factory=list)


class Coordinator:
    """Own one profiling lease; client retries never extend its deadline."""

    def __init__(
        self,
        config: Config,
        runtime: RuntimeSnapshot,
        *,
        registry: CollectorRegistry | None = None,
        clock_sampler: ClockSampler | None = None,
        artifact_validator: ArtifactValidator | None = None,
    ):
        # Frozen structs can still contain mutable inventory/options/context.
        # The callable API owns snapshots rather than caller-owned containers.
        config = deepcopy(config)
        runtime = deepcopy(runtime)
        if config.max_duration_ms <= 0 or config.max_sessions <= 0:
            raise ValueError("Profiling limits must be positive")
        if config.start_timeout_s <= 0 or config.finalize_timeout_s <= 0:
            raise ValueError("Profiling timeouts must be positive")
        if (
            config.clock_probe_timeout_s <= 0
            or not 1 <= config.clock_probe_samples <= 16
        ):
            raise ValueError(
                "Clock sampling limits must be positive and at most 16 samples"
            )
        if config.artifact_validation_timeout_s <= 0 or config.max_trace_bytes <= 0:
            raise ValueError("Artifact validation limits must be positive")
        for worker in runtime.workers:
            if len(set(worker.expected_rank_ids)) != len(worker.expected_rank_ids):
                raise ValueError("Expected rank IDs must be unique")
        self.config = config
        self.runtime = runtime
        self.registry = registry if registry is not None else default_registry()
        self.builtin_collectors = registry is None
        self.clock_sampler = (
            clock_sampler if clock_sampler is not None else WorkerClockSampler()
        )
        self.artifact_validator = (
            artifact_validator
            if artifact_validator is not None
            else SharedFilesystemValidator()
        )
        self.frontend_pid = runtime.frontend_pid
        self.frontend_epoch = runtime.frontend_epoch
        self.workers = {worker.target_id: worker for worker in runtime.workers}
        if len(self.workers) != len(runtime.workers) or "frontend" in self.workers:
            raise ValueError("Worker target IDs must be unique and cannot be frontend")
        self.sessions: dict[str, Session] = {}
        self.requests: dict[str, Session] = {}
        self.active: Session | None = None
        self.closed = False
        self.cleanup_uncertain = False

    def targets(self) -> list[dict]:
        return [
            {
                "target_id": "frontend",
                "process_epoch": self.frontend_epoch,
                "collectors": (
                    ["linux_perf"] if self.config.frontend_perf_enabled else []
                )
                + (["nsys"] if self.config.nsys_frontend_output_prefix else []),
                "scope": "process_and_threads",
            },
            *[
                {
                    "target_id": worker.target_id,
                    "process_epoch": worker.process_epoch,
                    "collectors": (
                        []
                        if worker.backend == "vllm" and worker.nsys_output_prefix
                        else ["torch_profiler"]
                    )
                    + (["nsys"] if worker.nsys_output_prefix else []),
                    "scope": "engine_group",
                    "identity_verification": "control_process_epoch_required_at_start"
                    if worker.verify_runtime
                    else "operator_config_only",
                    "expected_rank_ids": list(worker.expected_rank_ids),
                }
                for worker in self.workers.values()
            ],
        ]

    def nsys_prefix(self, target_id: str) -> str | None:
        if target_id == "frontend":
            return self.config.nsys_frontend_output_prefix
        return self.workers[target_id].nsys_output_prefix

    def validate(self, plan: Plan) -> None:
        if not plan.client_request_id or len(plan.client_request_id) > 128:
            raise BadRequest("client_request_id must contain 1..128 characters")
        if not self.registry.supports(plan.mode):
            raise Unsupported("Collector mode is not installed")
        if not 0 < plan.duration_ms <= self.config.max_duration_ms:
            raise BadRequest("duration_ms exceeds configured capture budget")
        target_ids = [target.target_id for target in plan.targets]
        if not target_ids or len(target_ids) != len(set(target_ids)):
            raise BadRequest("targets must be nonempty and unique")
        epochs = {item["target_id"]: item["process_epoch"] for item in self.targets()}
        prefixes = []
        for target in plan.targets:
            if target.target_id not in epochs:
                raise BadRequest("Unknown target; use GET targets")
            if epochs[target.target_id] != target.process_epoch:
                raise Conflict("Target process epoch does not match inventory")
            if not self.builtin_collectors:
                continue
            prefix = self.nsys_prefix(target.target_id)
            if plan.mode == "torch_profile":
                if (
                    target.target_id != "frontend"
                    and prefix
                    and self.workers[target.target_id].backend == "vllm"
                ):
                    raise Unsupported(
                        "This vLLM launch uses its CUDA profiler; Torch profiling requires a separate launch"
                    )
                continue
            if not prefix:
                raise Unsupported("Nsight requires a prepared launch/output prefix")
            if not Path(prefix).is_absolute():
                raise Unsupported("Nsight output prefixes must be absolute")
            if target.target_id == "frontend" and self.frontend_pid != os.getpid():
                raise Unsupported("NVTX frontend capture requires an embedded host")
            prefixes.append(prefix)
        if len(set(prefixes)) != len(prefixes):
            raise BadRequest(
                "Selected Nsight launches must have distinct output prefixes"
            )
        if (
            plan.mode == "torch_profile"
            and not self.config.frontend_perf_enabled
            and "frontend" in target_ids
        ):
            raise Unsupported(
                "Frontend profiling is disabled by operator configuration"
            )
        # Frontend perf is implicit unless explicitly disabled by the operator.

    def submit(self, plan: Plan) -> Session:
        # A caller of the Python API may retain mutable lists inside its plan.
        # Copy and type-check the accepted value before using it as a retry key.
        plan = msgspec.json.decode(msgspec.json.encode(plan), type=Plan)
        previous = self.requests.get(plan.client_request_id)
        if previous is not None:
            if previous.plan != plan:
                raise Conflict("client_request_id already has a different plan")
            return previous
        if self.closed:
            raise Unavailable("Profiling coordinator is shutting down")
        self.validate(plan)
        if self.cleanup_uncertain:
            raise Conflict(
                "Collector cleanup is uncertain; reconcile worker state before restarting the coordinator"
            )
        if self.active is not None:
            raise Conflict("A profiling session is already active or finalizing")
        if len(self.sessions) >= self.config.max_sessions:
            raise Unavailable(
                "Profiling session retention limit reached; restart coordinator"
            )
        session_id = uuid.uuid4().hex
        directory = Path(self.config.output_dir) / session_id
        directory.mkdir(parents=True, mode=0o700)
        session = Session(session_id, plan, directory)
        session.runtime = deepcopy(
            msgspec.structs.replace(self.runtime, workers=list(self.workers.values()))
        )
        session.deadline_monotonic = (
            asyncio.get_running_loop().time() + plan.duration_ms / 1000
        )
        self.persist(session)
        self.active = session
        self.sessions[session_id] = session
        self.requests[plan.client_request_id] = session
        session.task = asyncio.create_task(self.capture(session))
        return session

    def manifest(self, session: Session) -> dict:
        # Callers receive a detached copy; persistence serializes in place.
        return deepcopy(self.build_manifest(session))

    def build_manifest(self, session: Session) -> dict:
        statuses = {item["status"] for item in session.artifact_validation}
        frontend_enabled = (
            any(target.target_id == "frontend" for target in session.plan.targets)
            if session.plan.mode == "nsys"
            else self.config.frontend_perf_enabled
        )
        stop_due = session.accepted_at_ns + session.plan.duration_ms * 1000000
        return {
            "schema_version": 1,
            "session_id": session.session_id,
            "state": session.state,
            "plan": msgspec.to_builtins(session.plan),
            "accepted_at_ns": session.accepted_at_ns,
            "stop_due_at_ns": stop_due,
            "timing_contract": {
                "duration_semantics": "best_effort_lease_from_acceptance",
                "startup_counts_toward_duration": True,
                "synchronized_start": False,
                "native_recording_bounds": (
                    "artifact_validation.recording_bounds"
                    if session.plan.mode == "nsys"
                    else "artifact_validation.artifacts.recording_bounds"
                ),
                "start_timeout_s": self.config.start_timeout_s,
                "finalize_timeout_s": self.config.finalize_timeout_s,
                "artifact_validation_timeout_s": self.config.artifact_validation_timeout_s,
            },
            "frontend": {
                "pid": self.frontend_pid,
                "process_epoch": self.frontend_epoch,
                "profiling_enabled": frontend_enabled,
                "omission_reason": None
                if frontend_enabled
                else "not_selected"
                if session.plan.mode == "nsys"
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
            "runtime": {
                "hostname": session.runtime.hostname,
                "context": session.runtime.context,
            },
            "clock": {
                "timestamps": "unix_ns",
                "method": "four_timestamp_offset_estimate",
                "system_clocks_modified": False,
                "workers": session.clocks,
            },
            "artifact_validation": session.artifact_validation,
            # "present": every target's files exist and are readable; full
            # inspection runs offline in the bundle job, off the capture path.
            "artifact_status": "verified"
            if statuses == {"verified"}
            else "partial"
            if statuses & {"partial", "invalid"}
            else "present"
            if statuses and statuses <= {"verified", "present"}
            else "unverified",
            "errors": session.errors,
        }

    def persist(self, session: Session) -> None:
        pending = session.directory / "manifest.json.tmp"
        pending.write_text(json.dumps(self.build_manifest(session), indent=2) + "\n")
        pending.replace(session.directory / "manifest.json")

    def persist_progress(self, session: Session) -> None:
        # Once a collector is owned, a full/unavailable artifact filesystem
        # must not prevent stop and cleanup. Initial acceptance still requires
        # a successful manifest write through persist().
        try:
            self.persist(session)
        except Exception as exc:
            error = f"Manifest persistence failed: {exc}"
            if error not in session.errors:
                session.errors.append(error)
            if session.state == "completed":
                session.state = "failed"
            logger.warning("%s: %s", session.session_id, error)

    def start_budget(self, session: Session, phase: str) -> float:
        remaining = session.deadline_monotonic - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise CollectionError(f"Capture deadline expired during {phase}")
        return min(remaining, self.config.start_timeout_s)

    async def capture(self, session: Session) -> None:
        session.state = "starting"
        self.persist_progress(session)
        assert session.runtime is not None
        workers = {worker.target_id: worker for worker in session.runtime.workers}
        selected = [
            workers[target.target_id]
            for target in session.plan.targets
            if target.target_id != "frontend"
        ]
        capture_ok = False
        try:
            if session.errors:
                raise CollectionError("Manifest write failed before collector startup")
            # Probe and start targets concurrently: startup skew is bounded by
            # the slowest target rather than accumulating across targets.
            budget = self.start_budget(session, "runtime validation")
            measurements = await asyncio.wait_for(
                asyncio.gather(
                    *(self.clock_sampler.measure(w, self.config) for w in selected)
                ),
                budget,
            )
            for worker, measurement in zip(selected, measurements):
                session.clocks[worker.target_id] = {
                    "start": measurement,
                    "end": {"alignment": "unknown"},
                }
            if session.stop_requested.is_set():
                raise CollectionError("Capture stopped before recording started")
            # Register before starting: a timeout may have taken effect
            # remotely, so cleanup must also attempt stop on uncertain starts.
            session.collectors = self.registry.create(
                self.config, session.runtime, session.plan, session.directory
            )
            for collector in session.collectors:
                clock = session.clocks.get(collector.evidence.get("target_id"), {})
                if (
                    clock.get("start", {}).get("identity_verification")
                    == "control_process_epoch"
                ):
                    collector.evidence[
                        "identity_verification"
                    ] = "control_process_epoch"
            budget = self.start_budget(session, "startup")
            failures = await asyncio.wait_for(
                errors_of(self.start_collector(c) for c in session.collectors), budget
            )
            if failures:
                raise CollectionError("; ".join(failures))
            self.persist_progress(session)
            if session.errors:
                raise CollectionError("Manifest write failed after collector startup")
            session.state = "recording"
            self.persist_progress(session)
            remaining = session.deadline_monotonic - asyncio.get_running_loop().time()
            try:
                await asyncio.wait_for(session.stop_requested.wait(), max(0, remaining))
            except asyncio.TimeoutError:
                pass  # The lease deadline is a normal stop trigger.
            capture_ok = True
        except COLLECTION_ERRORS as exc:
            logger.warning(
                "Profiling startup failed for %s: %s", session.session_id, exc
            )
            session.errors.append(str(exc) or type(exc).__name__)
        finally:
            try:
                if not capture_ok and not session.errors:
                    session.errors.append(
                        "Capture interrupted by an unexpected failure"
                    )
                session.state = "finalizing"
                self.persist_progress(session)
                # Stop all collectors concurrently: slow worker trace serialization
                # must not keep other targets recording beyond their lease.
                stop_errors = await errors_of(
                    self.stop_collector(c) for c in session.collectors
                )
                session.errors.extend(stop_errors)
                self.cleanup_uncertain |= bool(stop_errors)
                session.errors.extend(
                    await errors_of(
                        self.measure_end(session, workers[target_id])
                        for target_id in session.clocks
                        if workers[target_id].verify_runtime
                    )
                )
                validations = await asyncio.gather(
                    *(
                        self.validate_artifacts(collector)
                        for collector in session.collectors
                    )
                )
                for collector, validation in zip(session.collectors, validations):
                    session.artifact_validation.append(validation)
                    if validation["status"] in ("partial", "invalid"):
                        if collector.evidence.get("collector") == "nsys":
                            # An unfinished export could be misassociated with the
                            # next window; require reconciliation before admitting it.
                            self.cleanup_uncertain = True
                        session.errors.append(
                            f"{validation['target_id']}: artifacts {validation['status']}"
                        )
                session.state = "failed" if session.errors else "completed"
                self.persist_progress(session)
            finally:
                # Release the lease even if finalization fails unexpectedly.
                self.active = None

    async def measure_end(self, session: Session, worker) -> str | None:
        measurements = session.clocks[worker.target_id]
        try:
            measurements["end"] = await asyncio.wait_for(
                self.clock_sampler.measure(worker, self.config),
                self.config.start_timeout_s,
            )
        except Exception as exc:
            measurements["end"] = {"alignment": "unknown", "error": str(exc)}
            return f"{worker.target_id}: final runtime verification failed: {exc}"
        start, end = measurements["start"], measurements["end"]
        if start.get("alignment") == end.get("alignment") == "estimated":
            measurements["offset_delta_ns"] = end["offset_ns"] - start["offset_ns"]
        return None

    async def validate_artifacts(self, collector: Collector) -> dict:
        try:
            validation = await self.artifact_validator.validate(
                collector.evidence, self.config
            )
        except Exception as exc:
            validation = {"status": "invalid", "coverage": "partial", "error": str(exc)}
        return {"target_id": collector.evidence.get("target_id"), **validation}

    async def start_collector(self, collector: Collector) -> str | None:
        try:
            await collector.start()
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return None

    async def stop_collector(self, collector: Collector) -> str | None:
        try:
            await asyncio.wait_for(collector.stop(), self.config.finalize_timeout_s)
        except Exception as exc:
            collector.evidence["coverage"] = "partial_or_unknown"
            logger.warning("Profiling finalization failed: %s", exc)
            return str(exc) or type(exc).__name__
        return None

    def lookup(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise NotFound("Unknown profiling session")
        return session

    def stop(self, session_id: str) -> Session:
        session = self.lookup(session_id)
        session.stop_requested.set()
        return session

    async def close(self) -> None:
        self.closed = True
        if self.active is not None:
            self.active.stop_requested.set()
            if self.active.task is not None:
                await self.active.task
