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
from collections import Counter
from contextlib import contextmanager
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


class LiveU:
    def __init__(self, capacity, calibration):
        assert isinstance(capacity, int) and capacity > 0
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
        self.accepted_steps = 0
        self.output_tokens = 0
        # Counts predictor evaluations, including lookahead and cache re-evaluations;
        # it is not a count of accepted steps or affected roots.
        self.extrapolated_cost_evaluations = 0

    def advance(self, now):
        if self.last is not None:
            assert now >= self.last, (now, self.last)
            roots = [r for r, count in self.root_active.items() if count]
            share = self.capacity / len(roots) if roots else 0.0
            for root in roots:
                self.entitlement[root] += (now - self.last) * min(
                    self.root_active[root], share
                )
        self.last = now

    def submit(self, request_id, root, agent, prompt_tokens, now):
        self.advance(now)
        assert request_id not in self.active and prompt_tokens > 0
        assert self.agent_root.get(agent, root) == root
        self.agent_root[agent] = root
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
        agents = [agent for agent, owner in self.agent_root.items() if owner == root]
        for agent in agents:
            del self.agent_root[agent]
            self.agent_active_requests.pop(agent, None)
        for table in (self.entitlement, self.work, self.invalid, self.root_active):
            table.pop(root, None)

    @lru_cache(maxsize=32768)
    def cost(self, phase, p, d, k):
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


def score_channel():
    value = os.environ.get("DURU_SCORES", "")
    if not value.strip():
        raise ValueError("DURU_SCORES must name the configured score IPC channel")
    return Path(value)


class Reader:
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
            time.time() - self.snapshot.get("observation_time", 0)
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
        if self.closed:
            return []
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
    """Publish request lifecycle events with explicit router program identity."""

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
        keys = (
            ("request_id", "program_id", "root", "agent", "prompt_tokens")
            if event == "request_arrived"
            else ("request_id", "program_id", "root")
        )
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
            self.emit("footer", time.time())
        finally:
            self.journal.close()


def run(config, stop=None, *, router_publish=None, ingress_source=None):
    if transport() != "memory":
        raise ValueError(
            "Live U accounting requires IPC (DURU_ACCOUNTING_TRANSPORT=memory)"
        )
    cfg = json.loads(Path(config).read_text())
    stop = stop if stop is not None else threading.Event()
    score_path = Path(cfg["scores"])
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg["ledgers"]:
        raise ValueError("At least one worker ledger is required")
    run_id = uuid.uuid4().hex
    book = LiveU(
        cfg["capacity"], load_calibration(cfg["calibration"], cfg["runtime_contract"])
    )
    ingress_reader = (
        EventStreamReader(cfg["ingress"])
        if ingress_source is None
        else EventStreamReader(cfg["ingress"], receiver=ingress_source)
    )
    sources = [ingress_reader] + [EventStreamReader(p) for p in cfg["ledgers"]]
    audit = (
        open_journal(
            cfg.get("audit_dir", score_path.parent), name="duru-control", fresh=False
        )
        if os.environ.get("DURU_RECORD_EVENTS") == "1"
        else None
    )
    sequence = 0
    heap = []
    serial = 0
    last_save = 0.0
    last_snapshot = None
    performance = PerformanceLogger()
    # Completed requests can still receive accepted-work callbacks.
    request_identities = {}
    request_programs = {}
    program_identities = {}
    worker_requests = set()
    closing_roots = set()
    stopped_cleanly = False
    max_identities = 100000
    shutdown_deadline = None
    worker_publisher = None

    def release_root(root, at):
        if any(request_identities[rid][0] == root for rid in worker_requests):
            return
        book.finish_root(root, at)
        for rid, identity in list(request_identities.items()):
            if identity[0] == root:
                request_identities.pop(rid)
                request_programs.pop(rid, None)
        for pid, identity in list(program_identities.items()):
            if identity[0] == root:
                program_identities.pop(pid)
        closing_roots.discard(root)

    try:
        worker_publisher = ScorePublisher(channel_path(score_path))
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
                        if row["event"] in (
                            "request_arrived",
                            "request_dispatched",
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
                    assert book.last is None or at >= book.last, (
                        "Late compact event",
                        at,
                        book.last,
                        kind,
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
                            if request_id in request_identities:
                                raise ValueError("Request ID reused: " + request_id)
                            program_id = _program_id(event_data)
                            identity = (event_data["root"], event_data["agent"])
                            if program_identities.get(program_id, identity) != identity:
                                raise ValueError(
                                    "Router program identity changed: " + program_id
                                )
                            program_identities[program_id] = identity
                            request_programs[event_data["request_id"]] = program_id
                            request_identities[request_id] = identity
                            book.submit(
                                event_data["request_id"],
                                event_data["root"],
                                event_data["agent"],
                                event_data["prompt_tokens"],
                                at,
                            )
                        elif event_data["event"] == "request_dispatched":
                            if event_data["request_id"] not in request_identities:
                                raise ValueError(
                                    "Request dispatched without an arrival event"
                                )
                            worker_requests.add(event_data["request_id"])
                        elif event_data["event"] == "agent_group_end":
                            root = event_data["root"]
                            if book.root_active[root]:
                                raise ValueError(
                                    "Root completed with outstanding requests: " + root
                                )
                            closing_roots.add(root)
                            release_root(root, at)
                        elif event_data["event"] == "agent_end":
                            pid = _program_id(event_data)
                            if pid in request_programs.values():
                                raise ValueError(
                                    "Program completed with outstanding requests: "
                                    + pid
                                )
                            program_identities.pop(pid, None)
                        else:
                            book.finish(event_data["request_id"], at)
                            request_programs.pop(event_data["request_id"], None)
                    else:
                        request_id = event_data["client_request_id"]
                        # Terminal acknowledgements are idempotent. Old harnesses
                        # without dispatch tracking do not require this barrier.
                        if (
                            kind == "worker_request_finished"
                            and request_id not in worker_requests
                        ):
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
                            worker_requests.discard(request_id)
                            if event_data["root"] in closing_roots:
                                release_root(event_data["root"], at)
                        else:
                            book.accepted(event_data)
                book.advance(accounting_cutoff)
                requests = {
                    rid: dict(book.score(rid), program_id=request_programs[rid], **r)
                    for rid, r in book.active.items()
                }
                programs = {
                    r["program_id"]: dict(r, request_id=rid)
                    for rid, r in requests.items()
                }
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
                    time=time.time(),
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
                if made_progress and time.monotonic() < shutdown_deadline:
                    continue
                stopped_cleanly = True
                break
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
                        time=time.time(),
                        last_snapshot=last_snapshot,
                        pending_events=len(heap),
                        inputs_closed=all(source.closed for source in sources),
                        complete=stopped_cleanly
                        and all(source.closed for source in sources)
                        and not heap,
                    ),
                )
        finally:
            try:
                if audit is not None:
                    audit.close()
                for source in sources:
                    source.close()
            finally:
                if router_publish is not None:
                    router_publish({})
                if worker_publisher is not None:
                    worker_publisher.close()


class RouterAccounting:
    """Router-owned accounting, with one background thread for bounded event batches.

    Published snapshots are replaced, never mutated. Router decisions read the
    latest object directly; engine processes receive compact scores over IPC.
    Errors invalidate the local snapshot and are surfaced by check()/close().
    """

    def __init__(self, config, *, ingress=None):
        self.config = config
        self._ingress_source = None if ingress is None else ingress.journal
        if self._ingress_source is not None and not isinstance(
            self._ingress_source, InProcessChannel
        ):
            raise ValueError("Router-owned ingress must use in_process=True")
        self._snapshot = {}
        self._stop = threading.Event()
        self._error = None
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
            )
        except BaseException as exc:
            self._snapshot = {}
            self._error = exc

    def start(self):
        self._thread.start()
        return self

    def check(self):
        if self._error is not None:
            raise RuntimeError("Router U accounting failed") from self._error
        if (
            self._thread.ident is not None
            and not self._thread.is_alive()
            and not self._stop.is_set()
        ):
            raise RuntimeError("Router U accounting stopped unexpectedly")

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(15)
            if self._thread.is_alive():
                raise TimeoutError("Router U accounting did not stop")
        self._snapshot = {}
        self.check()
