# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculate live U from local router events and worker IPC.

A common observation cutoff keeps entitlement and accepted work at the same
observation time. Recorded events are for diagnostics and offline validation only.
"""

import heapq
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path

from dynamo.duru_router.admission import priority_key as priority_key
from dynamo.duru_router.calibration.measure import load as load_calibration
from dynamo.duru_router.calibration.predict import predict
from dynamo.duru_router.diagnostics import PerformanceLogger
from dynamo.duru_router.events import open_feed, open_journal
from dynamo.duru_router.transport import (
    InProcessChannel,
    Receiver,
    ScorePublisher,
    ScoreSubscriber,
    channel_path,
    transport,
)

logger = logging.getLogger(__name__)
MAX_PENDING_EVENTS = 262144


class AccountingError(RuntimeError):
    """The router-owned accounting service has failed or stopped unexpectedly."""


class DispatchAckTimeout(AccountingError):
    """New admissions stopped while uncertain worker delivery remains accounted."""

    def __init__(self, request_id, timeout_seconds, failed_at):
        self.request_id = request_id
        self.timeout_seconds = timeout_seconds
        self.failed_at = failed_at
        super().__init__(
            f"Worker acknowledgement for request {request_id} timed out after "
            f"{timeout_seconds:g}s; delivery remains uncertain and new admissions are stopped"
        )


class LiveU:
    def __init__(self, capacity, calibration):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.calibration = calibration
        self.prefill_budget = calibration.get("max_prefill", 8192)
        if type(self.prefill_budget) is not int or self.prefill_budget <= 0:
            raise ValueError("Calibration prefill budget must be a positive integer")
        self.active = {}
        self.agent_active_requests = Counter()
        # Distinct active agents per root, not the number of requests.
        self.root_active = Counter()
        self.entitlement = Counter()
        self.work = Counter()
        self.invalid = Counter()
        self.last = None
        self.agent_root = {}
        self.root_agents = defaultdict(set)
        self.accepted_steps = 0
        self.output_tokens = 0
        # Counts predictor evaluations, including lookahead and cache re-evaluations;
        # it is not a count of accepted steps or affected roots.
        self.extrapolated_cost_evaluations = 0
        # The cache belongs to this accounting session, not a class-level wrapper
        # that keeps completed sessions and their calibration tables alive.
        self.cost = lru_cache(maxsize=32768)(self._cost)

    def advance(self, now):
        if self.last is not None:
            if not now >= self.last:
                raise ValueError(f"Accounting time regressed: {now} < {self.last}")
            roots = [r for r, count in self.root_active.items() if count]
            share = self.capacity / len(roots) if roots else 0.0
            for root in roots:
                self.entitlement[root] += (now - self.last) * min(
                    self.root_active[root], share
                )
        self.last = now

    def submit(self, request_id, root, agent, prompt_tokens, now):
        if request_id in self.active:
            raise ValueError(f"Request ID already active: {request_id}")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("Prompt tokens must be a positive integer")
        if self.agent_root.get(agent, root) != root:
            raise ValueError(f"Agent root changed: {agent}")
        self.advance(now)
        self.agent_root[agent] = root
        self.root_agents[root].add(agent)
        self.active[request_id] = dict(
            root=root, agent=agent, prompt_tokens=prompt_tokens, arrival=now
        )
        if not self.agent_active_requests[agent]:
            self.root_active[root] += 1
        self.agent_active_requests[agent] += 1

    def finish(self, request_id, now):
        self.advance(now)
        req = self.active.pop(request_id, None)
        if req is not None:
            agent = req["agent"]
            self.agent_active_requests[agent] -= 1
            if not self.agent_active_requests[agent]:
                del self.agent_active_requests[agent]
                self.root_active[req["root"]] -= 1
                if not self.root_active[req["root"]]:
                    del self.root_active[req["root"]]

    def finish_root(self, root, now):
        """Called only after the root and all descendants have stopped producing work."""
        self.advance(now)
        if self.root_active[root]:
            raise ValueError("Root completed with outstanding requests: " + root)
        for agent in self.root_agents.pop(root, ()):
            del self.agent_root[agent]
            self.agent_active_requests.pop(agent, None)
        for table in (self.entitlement, self.work, self.invalid, self.root_active):
            table.pop(root, None)

    def _cost(self, phase, p, d, k):
        ms, outside = predict(self.calibration[phase], dict(phase=phase, p=p, d=d, k=k))
        if not math.isfinite(ms) or ms <= 0:
            return None
        # predict continues the grid's edge slope, exactly as offline analysis.
        # Out-of-grid coordinates alone must not permanently invalidate a root.
        if outside:
            self.extrapolated_cost_evaluations += 1
        return ms / 1000.0

    def accepted(self, step):
        if not step["accepted"]:
            return
        root = step["root"]
        cost = self.cost(step["phase"], step["p"], step["d"], step["k"])
        self.accepted_steps += 1
        self.output_tokens += step["generated_tokens"]
        if cost is None:
            self.invalid[root] += 1
        else:
            self.work[root] += cost

    def measured(self, root):
        if self.invalid[root] or self.work[root] <= 0:
            return None
        return self.entitlement[root] / self.work[root]

    def next_five(self, prompt_tokens, known_cached_tokens=0, budget=None):
        if budget is None:
            budget = self.prefill_budget
        # Predict five engine steps from the prompt and observed cache state.
        # Unknown cache reuse is conservatively zero.
        d = min(max(0, known_cached_tokens), prompt_tokens - 1)
        total = 0.0
        for _ in range(5):
            phase = "prefill" if d < prompt_tokens else "decode"
            p = min(budget, prompt_tokens - d) if phase == "prefill" else 1
            cost = self.cost(phase, p, d, d + p)
            if cost is None:
                return None
            total += cost
            d += p
        return total

    def score(self, request_id, known_cached_tokens=0):
        req = self.active[request_id]
        root = req["root"]
        measured = self.measured(root)
        if self.work[root] > 0 or self.invalid[root]:
            return dict(U=measured, score=measured, kind="measured", lookahead_s=None)
        predicted = self.next_five(req["prompt_tokens"], known_cached_tokens)
        value = None
        if predicted is not None and not self.invalid[root]:
            value = self.entitlement[root] / (self.work[root] + predicted)
        return dict(
            U=measured,
            score=value,
            kind="zero_work_root_next_five",
            lookahead_s=predicted,
        )


class LifecycleIndex:
    """Retain identities until their root has no outstanding worker acknowledgements.

    Root membership indexes bound cleanup to that root's history. Active request
    counts separately permit program termination before the root is released.
    """

    def __init__(self):
        self.request_identities = {}
        self.request_programs = {}
        self.program_identities = {}
        self.worker_requests = set()
        self.root_requests = defaultdict(set)
        self.root_programs = defaultdict(set)
        self.root_workers = Counter()
        self.program_active = Counter()

    def arrive(self, request_id, program_id, root, agent):
        """Register one arrival without allowing retained identities to change."""
        if request_id in self.request_identities:
            raise ValueError("Request ID reused: " + request_id)
        identity = (root, agent)
        if self.program_identities.get(program_id, identity) != identity:
            raise ValueError("Router program identity changed: " + program_id)
        self.request_identities[request_id] = identity
        self.request_programs[request_id] = program_id
        self.program_identities[program_id] = identity
        self.root_requests[root].add(request_id)
        self.root_programs[root].add(program_id)
        self.program_active[program_id] += 1

    def dispatch(self, request_id):
        """Track a worker acknowledgement once even if dispatch is repeated."""
        if request_id not in self.request_identities:
            raise ValueError("Request dispatched without an arrival event")
        if request_id not in self.worker_requests:
            self.worker_requests.add(request_id)
            self.root_workers[self.request_identities[request_id][0]] += 1

    def undispatch(self, request_id):
        """Release a confirmed rejection or a validated worker acknowledgement."""
        self.worker_requests.remove(request_id)
        root = self.request_identities[request_id][0]
        self.root_workers[root] -= 1
        if not self.root_workers[root]:
            del self.root_workers[root]

    def finish(self, request_id):
        """End active membership while retaining identity for late worker output."""
        program_id = self.request_programs.pop(request_id, None)
        if program_id is not None:
            self.program_active[program_id] -= 1
            if not self.program_active[program_id]:
                del self.program_active[program_id]

    def end_program(self, program_id):
        """Release an idle program without scanning other programs' requests."""
        if self.program_active[program_id]:
            raise ValueError(
                "Program completed with outstanding requests: " + program_id
            )
        identity = self.program_identities.pop(program_id, None)
        if identity is not None:
            programs = self.root_programs[identity[0]]
            programs.remove(program_id)
            if not programs:
                del self.root_programs[identity[0]]

    def release_root(self, root):
        """Forget only this root after its active work and worker barrier clear."""
        if self.root_workers[root]:
            raise ValueError("Root completed with outstanding worker requests: " + root)
        for request_id in self.root_requests.pop(root, ()):
            self.request_identities.pop(request_id)
            self.request_programs.pop(request_id, None)
        for program_id in self.root_programs.pop(root, ()):
            self.program_identities.pop(program_id)
            self.program_active.pop(program_id, None)


class DispatchAckMonitor:
    """Latch admission failure without dropping uncertain work or stopping accounting.

    Each pass receives available acknowledgements before checking deadlines;
    expiry is bounded by accounting polling, not a real-time scheduling guarantee.
    """

    def __init__(self, identities, timeout_seconds):
        if timeout_seconds is not None and (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "Dispatch acknowledgement timeout must be finite and positive"
            )
        self.identities = identities
        self.timeout_seconds = timeout_seconds
        self.failures = {}
        self.acknowledgements = {}
        self.observed_arrivals = {}
        self.observed_dispatches = set()
        self.observed_rejections = set()
        self.deadlines = {}
        self.heap = []
        self.latched = None

    def arrived(self, request_id, identity):
        """Retain identity for watchdog checks while ordered arrival is pending."""
        if self.timeout_seconds is None or self.latched is not None:
            return
        if request_id not in self.identities.request_identities:
            self.observed_arrivals.setdefault(request_id, identity)

    def dispatched(self, request_id):
        """Observe possible delivery without advancing accounting's worker barrier."""
        if self.timeout_seconds is None or self.latched is not None:
            return
        self.observed_dispatches.add(request_id)
        self.arm(request_id)

    def rejected(self, request_id):
        """Confirmed synchronous rejection suppresses a timer before cutoff catches up."""
        if self.timeout_seconds is None or self.latched is not None:
            return
        self.forget(request_id)
        self.observed_rejections.add(request_id)

    def ordered_arrival(self, request_id):
        self.observed_arrivals.pop(request_id, None)

    def ordered_dispatch(self, request_id):
        self.observed_dispatches.discard(request_id)
        self.arm(request_id)

    def _identity(self, request_id):
        return self.identities.request_identities.get(
            request_id, self.observed_arrivals.get(request_id)
        )

    def failed(self, request_id, at):
        """Observe caller failure even when accounting's common cutoff is frozen."""
        if (
            self.timeout_seconds is None
            or self.latched is not None
            or request_id in self.observed_rejections
        ):
            return
        self.failures.setdefault(request_id, at)
        self.arm(request_id)

    def acknowledged(self, request_id, identity):
        """Cancel a matching received acknowledgement without advancing worker state."""
        if self.timeout_seconds is None or self.latched is not None:
            return
        known = self._identity(request_id)
        if known is not None:
            if identity != known:
                return
            self.acknowledgements[request_id] = [identity]
            self.deadlines.pop(request_id, None)
        else:
            self.acknowledgements.setdefault(request_id, []).append(identity)
        self._compact()

    def arm(self, request_id):
        """Start a timer only for a failed request still awaiting worker acknowledgement."""
        if (
            request_id not in self.failures
            or (
                request_id not in self.identities.worker_requests
                and request_id not in self.observed_dispatches
            )
            or request_id in self.observed_rejections
            or self.latched is not None
        ):
            return
        identity = self._identity(request_id)
        if identity is None:
            return
        if identity in self.acknowledgements.get(request_id, ()):
            self.deadlines.pop(request_id, None)
            return
        if request_id not in self.deadlines:
            failed_at = self.failures[request_id]
            self.deadlines[request_id] = failed_at
            heapq.heappush(
                self.heap, (failed_at + self.timeout_seconds, request_id, failed_at)
            )

    def forget(self, request_id):
        """Drop timer bookkeeping after confirmed completion or non-delivery."""
        self.failures.pop(request_id, None)
        self.acknowledgements.pop(request_id, None)
        self.observed_arrivals.pop(request_id, None)
        self.observed_dispatches.discard(request_id)
        self.observed_rejections.discard(request_id)
        self.deadlines.pop(request_id, None)
        self._compact()

    def _compact(self):
        if len(self.heap) > 2 * len(self.deadlines) + 128:
            self.heap = [
                (at + self.timeout_seconds, request_id, at)
                for request_id, at in self.deadlines.items()
            ]
            heapq.heapify(self.heap)

    def expire(self, now):
        """Latch the first expired deadline; later acknowledgement never reopens admission."""
        while self.latched is None and self.heap and self.heap[0][0] <= now:
            _, request_id, failed_at = heapq.heappop(self.heap)
            if self.deadlines.get(request_id) != failed_at:
                continue
            self.latched = dict(
                request_id=request_id,
                timeout_seconds=self.timeout_seconds,
                failed_at=failed_at,
            )
            # Admission remains latched; further timers cannot change that decision.
            self.failures.clear()
            self.acknowledgements.clear()
            self.observed_arrivals.clear()
            self.observed_dispatches.clear()
            self.observed_rejections.clear()
            self.deadlines.clear()
            self.heap.clear()
        return self.latched


def score_channel():
    value = os.environ.get("DURU_SCORES", "")
    if not value.strip():
        raise ValueError("DURU_SCORES must name the configured score IPC channel")
    return Path(value)


class Reader:
    """Read scores whose observation_time uses this host's monotonic clock."""

    def __init__(self, snapshot_source=None):
        self.snapshot_source = snapshot_source
        self.path = None if snapshot_source is not None else score_channel()
        self._subscriber = None
        self.snapshot = {}
        self.reason = "missing_snapshot"
        self.age = None
        self.pinned = None

    def pin(self, snapshot):
        @contextmanager
        def decision():
            previous = self.pinned
            self.pinned = snapshot
            try:
                yield
            finally:
                self.pinned = previous

        return decision()

    def refresh(self):
        if self.pinned is not None:
            return self.pinned
        if self.snapshot_source is not None:
            self.snapshot = self.snapshot_source()
        else:
            if self._subscriber is None:
                self._subscriber = ScoreSubscriber(channel_path(self.path))
            self.snapshot = self._subscriber.snapshot()
        self.age = (
            time.monotonic() - self.snapshot.get("observation_time", 0)
            if self.snapshot
            else None
        )
        if not self.snapshot:
            self.reason = "missing_snapshot"
        elif self.age < 0:
            self.reason = "future_snapshot"
        elif self.age > 2:
            self.reason = "stale_snapshot"
        else:
            self.reason = "fresh"
            return self.snapshot
        return {}

    def close(self):
        if self._subscriber is not None:
            self._subscriber.close()
        self.snapshot = {}


class EventStreamReader:
    """Read ordered local or IPC events and track producer progress.

    received_through is the latest received event/clock timestamp. Ordered
    delivery lets accounting use it as this source's safe observation cutoff.
    """

    def __init__(self, path, *, receiver=None):
        self.path = Path(path)
        self.sequence = 0
        self.received_through = 0.0
        self.closed = False
        self.receiver = receiver

    def read(self):
        if self.receiver is None:
            try:
                self.receiver = Receiver(channel_path(self.path))
            except FileNotFoundError:
                return []
        rows = self.receiver.read()
        for row in rows:
            if self.closed or row["sequence"] != self.sequence + 1:
                raise ValueError("Duru event stream sequence gap or event after footer")
            self.sequence += 1
            self.received_through = max(self.received_through, row["time"])
            self.closed = row["event"] == "footer"
        return rows

    def close(self):
        if self.receiver is not None:
            self.receiver.close()


def _program_id(fields):
    program_id = fields.get("program_id")
    if not isinstance(program_id, str) or not program_id:
        raise ValueError("Request arrival requires a nonempty program_id")
    return program_id


class Ingress:
    """Publish lifecycle events with program identity and monotonic timestamps."""

    def __init__(self, path, *, in_process=False):
        if in_process:
            recording = (
                open_journal(path, name="duru-ingress")
                if os.environ.get("DURU_RECORD_EVENTS") == "1"
                else None
            )
            self.journal = InProcessChannel(recording=recording)
        else:
            # Retain the external-producer path for standalone/replay harnesses.
            self.journal = open_feed(path, name="duru-ingress")
        self.sequence = 0

    def emit(self, event, timestamp, **fields):
        if event not in (
            "request_arrived",
            "request_dispatched",
            "request_dispatch_rejected",
            "request_completed",
            "request_cancelled",
            "request_failed",
            "agent_end",
            "agent_group_end",
            "u_clock",
            "footer",
        ):
            return
        if event == "request_arrived":
            _program_id(fields)
            keys = ("request_id", "program_id", "root", "agent", "prompt_tokens")
        elif event == "agent_end":
            keys = ("request_id", "program_id", "root", "agent")
        else:
            keys = ("request_id", "program_id", "root")
        self.sequence += 1
        self.journal.submit(
            "record",
            dict(
                sequence=self.sequence,
                event=event,
                time=timestamp,
                **{k: fields[k] for k in keys if k in fields},
            ),
        )

    def close(self):
        try:
            self.emit("footer", time.monotonic())
        finally:
            self.journal.close()


def run(
    config,
    stop=None,
    *,
    router_publish=None,
    ingress_source=None,
    dispatch_ack_timeout_seconds=None,
):
    if transport() != "memory":
        raise ValueError(
            "Live U accounting requires IPC (DURU_ACCOUNTING_TRANSPORT=memory)"
        )
    cfg = json.loads(Path(config).read_text())
    stop = stop if stop is not None else threading.Event()
    drain_wait = threading.Event()
    score_path = Path(cfg["scores"])
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg["ledgers"]:
        raise ValueError("At least one worker ledger is required")
    run_id = uuid.uuid4().hex
    book = LiveU(
        cfg["capacity"], load_calibration(cfg["calibration"], cfg["runtime_contract"])
    )
    sources = []
    audit = None
    sequence = 0
    heap = []
    serial = 0
    last_save = 0.0
    last_snapshot = None
    performance = PerformanceLogger()
    # Completed requests can still receive accepted-work callbacks.
    identities = LifecycleIndex()
    dispatch_acks = DispatchAckMonitor(identities, dispatch_ack_timeout_seconds)
    request_identities = identities.request_identities
    request_programs = identities.request_programs
    program_identities = identities.program_identities
    worker_requests = identities.worker_requests
    closing_roots = set()
    stopped_cleanly = False
    max_identities = 100000
    shutdown_deadline = None
    worker_publisher = None
    cleanup = ExitStack()

    def release_root(root, at):
        if identities.root_workers[root]:
            return
        book.finish_root(root, at)
        for request_id in identities.root_requests.get(root, ()):
            dispatch_acks.forget(request_id)
        identities.release_root(root)
        closing_roots.discard(root)

    try:
        if router_publish is not None:
            cleanup.callback(router_publish, {})
        ingress_reader = (
            EventStreamReader(cfg["ingress"])
            if ingress_source is None
            else EventStreamReader(cfg["ingress"], receiver=ingress_source)
        )
        cleanup.callback(ingress_reader.close)
        sources.append(ingress_reader)
        for path in cfg["ledgers"]:
            source = EventStreamReader(path)
            cleanup.callback(source.close)
            sources.append(source)
        if os.environ.get("DURU_RECORD_EVENTS") == "1":
            audit = open_journal(
                cfg.get("audit_dir", score_path.parent),
                name="duru-control",
                fresh=False,
            )
            cleanup.callback(audit.close)
        worker_publisher = ScorePublisher(channel_path(score_path))
        cleanup.callback(worker_publisher.close)
        worker_publish = worker_publisher.publish
        while True:
            stopping = stop.is_set()
            if stopping and shutdown_deadline is None:
                shutdown_deadline = time.monotonic() + 5
            made_progress = False
            for index, source in enumerate(sources):
                for row in source.read():
                    made_progress = True
                    if index == 0:
                        if row["event"] == "request_arrived":
                            dispatch_acks.arrived(
                                row["request_id"], (row["root"], row["agent"])
                            )
                        elif row["event"] == "request_dispatched":
                            dispatch_acks.dispatched(row["request_id"])
                        elif row["event"] == "request_dispatch_rejected":
                            dispatch_acks.rejected(row["request_id"])
                        if row["event"] in ("request_failed", "request_cancelled"):
                            dispatch_acks.failed(row["request_id"], row["time"])
                        if row["event"] in (
                            "request_arrived",
                            "request_dispatched",
                            "request_dispatch_rejected",
                            "request_completed",
                            "request_cancelled",
                            "request_failed",
                            "agent_end",
                            "agent_group_end",
                        ):
                            serial += 1
                            heapq.heappush(heap, (row["time"], serial, "ingress", row))
                            if len(heap) > MAX_PENDING_EVENTS:
                                raise RuntimeError("Duru pending-event limit exceeded")
                    elif row["event"] == "worker_request_finished":
                        for identity in row["requests"]:
                            dispatch_acks.acknowledged(
                                identity["client_request_id"],
                                (identity.get("root"), identity.get("agent")),
                            )
                            serial += 1
                            heapq.heappush(
                                heap,
                                (
                                    row["time"],
                                    serial,
                                    "worker_request_finished",
                                    identity,
                                ),
                            )
                            if len(heap) > MAX_PENDING_EVENTS:
                                raise RuntimeError("Duru pending-event limit exceeded")
                    elif row["event"] == "accepted_output":
                        for st in row["steps"]:
                            at = st["end"]
                            serial += 1
                            heapq.heappush(heap, (at, serial, row["event"], st))
                            if len(heap) > MAX_PENDING_EVENTS:
                                raise RuntimeError("Duru pending-event limit exceeded")
            # Advance only through the time covered by every open event stream.
            # Closed streams cannot produce more events or hold back the cutoff.
            accounting_cutoff = min(
                (s.received_through for s in sources if not s.closed),
                default=max(s.received_through for s in sources),
            )
            if accounting_cutoff > 0:
                while heap and heap[0][0] <= accounting_cutoff:
                    at, _, kind, event_data = heapq.heappop(heap)
                    if book.last is not None and at < book.last:
                        raise ValueError(
                            f"Late compact event: {kind} at {at} precedes {book.last}"
                        )
                    if (
                        len(heap) > MAX_PENDING_EVENTS
                        or len(request_identities) > max_identities
                    ):
                        raise RuntimeError(
                            "Duru accounting state limit exceeded; check completion events"
                        )
                    # Accepted steps do not change active roots or agents.
                    # Integrate E only on ingress transitions and at publication.
                    # This is the same integral without O(steps * active roots).
                    if kind == "ingress":
                        if event_data["event"] == "request_arrived":
                            request_id = event_data["request_id"]
                            program_id = _program_id(event_data)
                            identities.arrive(
                                request_id,
                                program_id,
                                event_data["root"],
                                event_data["agent"],
                            )
                            dispatch_acks.ordered_arrival(request_id)
                            book.submit(
                                event_data["request_id"],
                                event_data["root"],
                                event_data["agent"],
                                event_data["prompt_tokens"],
                                at,
                            )
                        elif event_data["event"] == "request_dispatched":
                            identities.dispatch(event_data["request_id"])
                            dispatch_acks.ordered_dispatch(event_data["request_id"])
                        elif event_data["event"] == "request_dispatch_rejected":
                            # The router confirmed rejection before async dispatch.
                            # Request failure still closes the active accounting entry.
                            identities.undispatch(event_data["request_id"])
                            dispatch_acks.forget(event_data["request_id"])
                        elif event_data["event"] == "agent_group_end":
                            root = event_data["root"]
                            if book.root_active[root]:
                                raise ValueError(
                                    "Root completed with outstanding requests: " + root
                                )
                            closing_roots.add(root)
                            release_root(root, at)
                        elif event_data["event"] == "agent_end":
                            identities.end_program(_program_id(event_data))
                        else:
                            book.finish(event_data["request_id"], at)
                            identities.finish(event_data["request_id"])
                            if event_data["request_id"] not in worker_requests:
                                dispatch_acks.forget(event_data["request_id"])
                    else:
                        request_id = event_data["client_request_id"]
                        # Terminal acknowledgements are idempotent. Old harnesses
                        # without dispatch tracking do not require this barrier.
                        if (
                            kind == "worker_request_finished"
                            and request_id not in worker_requests
                        ):
                            dispatch_acks.forget(request_id)
                            continue
                        if request_id not in request_identities:
                            raise ValueError("Unjoined work: " + request_id)
                        if (
                            event_data.get("root"),
                            event_data.get("agent"),
                        ) != request_identities[request_id]:
                            raise ValueError(
                                "Worker request identity mismatch: " + request_id
                            )
                        if kind == "worker_request_finished":
                            identities.undispatch(request_id)
                            dispatch_acks.forget(request_id)
                            if event_data["root"] in closing_roots:
                                release_root(event_data["root"], at)
                        else:
                            book.accepted(event_data)
                dispatch_timeout = dispatch_acks.expire(time.monotonic())
                book.advance(accounting_cutoff)
                requests = {
                    rid: dict(book.score(rid), program_id=request_programs[rid], **r)
                    for rid, r in book.active.items()
                }
                programs = {}
                for rid, request in requests.items():
                    program_id = request["program_id"]
                    previous = programs.get(program_id)
                    # One reservation serves every active request in a program.
                    # Use its highest request score, with stable arrival/ID ties.
                    if previous is None or priority_key(
                        request["score"], request["arrival"], rid
                    ) < priority_key(
                        previous["score"], previous["arrival"], previous["request_id"]
                    ):
                        programs[program_id] = dict(request, request_id=rid)
                for program_id, (root, agent) in program_identities.items():
                    programs.setdefault(
                        program_id,
                        dict(
                            program_id=program_id,
                            root=root,
                            agent=agent,
                            U=book.measured(root),
                            score=None,
                            kind="idle_measured_only",
                        ),
                    )
                snapshot = dict(
                    schema=1,
                    run_id=run_id,
                    extrapolated_cost_evaluations=book.extrapolated_cost_evaluations,
                    time=time.monotonic(),
                    observation_time=accounting_cutoff,
                    capacity=book.capacity,
                    accounting_state=dict(
                        request_identities=len(request_identities),
                        program_identities=len(program_identities),
                        agents=len(book.agent_root),
                        roots=len(book.entitlement),
                        pending_events=len(heap),
                        worker_requests=len(worker_requests),
                        closing_roots=len(closing_roots),
                    ),
                    requests=requests,
                    programs=programs,
                    accepted_steps=book.accepted_steps,
                    output_tokens=book.output_tokens,
                )
                if dispatch_timeout is not None:
                    snapshot["dispatch_timeout"] = dispatch_timeout
                if audit is not None:
                    snapshot.update(
                        source_received_through=[x.received_through for x in sources],
                        source_sequences=[x.sequence for x in sources],
                    )
                if audit is not None or performance.enabled:
                    snapshot["roots"] = {
                        r: dict(
                            E=book.entitlement[r],
                            W=book.work[r],
                            U=book.measured(r),
                            invalid_steps=book.invalid[r],
                            active_agents=book.root_active[r],
                        )
                        for r in set(book.agent_root.values())
                    }
                worker_publish(snapshot)
                if router_publish is not None:
                    router_publish(snapshot)
                last_snapshot = snapshot
                performance.observe(snapshot, len(heap))
                if audit is not None and time.monotonic() - last_save >= 1:
                    sequence += 1
                    audit.submit(
                        "record", dict(sequence=sequence, event="snapshot", **snapshot)
                    )
                    last_save = time.monotonic()
            if stopping:
                if all(source.closed for source in sources) and not heap:
                    if made_progress:
                        # Check for already-buffered records after the last footer.
                        continue
                    stopped_cleanly = True
                    break
                remaining = shutdown_deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not made_progress:
                    # stop is already set, so waiting on it would busy-spin.
                    drain_wait.wait(min(0.05, remaining))
                continue
            stop.wait(0.05)
    finally:
        try:
            if heap:
                logger.warning(
                    "Stopping with %d events beyond the common observation cutoff",
                    len(heap),
                )
            sequence += 1
            if audit is not None:
                audit.submit(
                    "record",
                    dict(
                        sequence=sequence,
                        event="footer",
                        run_id=run_id,
                        time=time.monotonic(),
                        last_snapshot=last_snapshot,
                        pending_events=len(heap),
                        inputs_closed=all(source.closed for source in sources),
                        complete=stopped_cleanly
                        and all(source.closed for source in sources)
                        and not heap,
                    ),
                )
        finally:
            cleanup.close()


class RouterAccounting:
    """Router-owned accounting, with one background thread for bounded event batches.

    Published snapshots are replaced, never mutated. Router decisions read the
    latest object directly; engine processes receive compact scores over IPC.
    Fatal accounting errors invalidate the local snapshot and surface through
    check()/close(). Live serving uses DURU_DISPATCH_ACK_TIMEOUT_SECONDS (default
    300) to stop new admissions when failed dispatches remain unacknowledged,
    independently of the common cutoff. This latched admission-only condition
    keeps accounting and worker score publication running for late completion.
    """

    def __init__(self, config, *, ingress=None):
        self.config = config
        try:
            self.dispatch_ack_timeout_seconds = float(
                os.environ.get("DURU_DISPATCH_ACK_TIMEOUT_SECONDS", "300")
            )
        except ValueError as exc:
            raise ValueError(
                "DURU_DISPATCH_ACK_TIMEOUT_SECONDS must be finite and positive"
            ) from exc
        if (
            not math.isfinite(self.dispatch_ack_timeout_seconds)
            or self.dispatch_ack_timeout_seconds <= 0
        ):
            raise ValueError(
                "DURU_DISPATCH_ACK_TIMEOUT_SECONDS must be finite and positive"
            )
        self._ingress_source = None if ingress is None else ingress.journal
        if self._ingress_source is not None and not isinstance(
            self._ingress_source, InProcessChannel
        ):
            raise ValueError("Router-owned ingress must use in_process=True")
        self._snapshot = {}
        self._stop = threading.Event()
        self._result = Future()
        self._thread = threading.Thread(
            target=self._run, name="duru-accounting", daemon=True
        )
        self.reader = Reader(snapshot_source=self.snapshot)

    def snapshot(self):
        self.check()
        return self._snapshot

    def _router_publish(self, snapshot):
        self._snapshot = snapshot

    def _run(self):
        try:
            run(
                self.config,
                self._stop,
                router_publish=self._router_publish,
                ingress_source=self._ingress_source,
                dispatch_ack_timeout_seconds=self.dispatch_ack_timeout_seconds,
            )
        except Exception as exc:
            # Hand ordinary task failures to the owner without losing their
            # original traceback. Process-control exceptions still escape.
            self._result.set_exception(exc)
        else:
            self._result.set_result(None)
        finally:
            self._snapshot = {}

    def start(self):
        self._thread.start()
        return self

    def check(self, *, admissions=True):
        if self._result.done():
            try:
                self._result.result()
            except Exception as exc:
                raise AccountingError("Router U accounting failed") from exc
        if (
            self._thread.ident is not None
            and not self._thread.is_alive()
            and not self._stop.is_set()
        ):
            raise AccountingError("Router U accounting stopped unexpectedly")
        timeout = self._snapshot.get("dispatch_timeout")
        if admissions and timeout is not None:
            raise DispatchAckTimeout(**timeout)

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(15)
            if self._thread.is_alive():
                raise TimeoutError("Router U accounting did not stop")
        self._snapshot = {}
        self.check(admissions=False)
