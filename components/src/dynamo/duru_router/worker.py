# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Highest-U waiting queues and accepted-work accounting for native vLLM."""

import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue

from dynamo.duru_router.admission import priority_key
from dynamo.duru_router.events import CompactFeed
from dynamo.duru_router.scoring import Reader


class Queue(FCFSRequestQueue):
    def __init__(self, old, owner):
        super().__init__(old)
        self.owner = owner
        self._peeked = None

    def peek_request(self):
        if not self:
            raise IndexError("empty Highest-U queue")
        self._peeked = min(
            enumerate(self), key=lambda pair: self.owner.key(pair[1], pair[0])
        )[1]
        return self._peeked

    def pop_request(self):
        # Allocation uses the peeked identity, even if scores subsequently change.
        request = self._peeked
        if request is None or not any(item is request for item in self):
            request = self.peek_request()
        self._peeked = None
        self.remove_request(request)
        return request


class DuruScheduler(AsyncScheduler):
    """Record accepted engine work, with score-based scheduling enabled by default.

    Set DURU_ENABLE_SCORING=0 for isolated calibration: native scheduling runs
    without a score reader while accepted-work recording remains enabled.
    Use DURU_ACCOUNTING_TRANSPORT=journal to collect calibration work without
    a live accounting consumer.
    """

    def __init__(self, *args, **kwargs):
        scoring = os.environ.get("DURU_ENABLE_SCORING", "1")
        if scoring not in ("0", "1"):
            raise ValueError("DURU_ENABLE_SCORING must be 0 or 1")
        self.u_reader = Reader() if scoring == "1" else None
        super().__init__(*args, **kwargs)
        if self.num_spec_tokens or self.num_sampled_tokens_per_step != 1:
            raise ValueError("Duru requires non-speculative autoregressive generation")
        if self.u_reader is not None:
            self.waiting = Queue(self.waiting, self)
            self.kv_holding_waiting = Queue(self.kv_holding_waiting, self)
        self._pending_work = {}
        self._kv_rejected_work = set()
        self.compact_feed = CompactFeed(os.environ["DURU_WORK_DIR"])

    def identity(self, request):
        extra = request.sampling_params.extra_args or {}
        client_id = extra.get("duru_request_id") or request.request_id
        root = extra.get("duru_root_id") or client_id
        return dict(
            client_request_id=client_id,
            root=root,
            agent=extra.get("duru_agent_id") or root,
        )

    def add_request(self, request, *args, **kwargs):
        if request.sampling_params is None or request.resumable:
            raise ValueError("Duru requires non-resumable generation requests")
        return super().add_request(request, *args, **kwargs)

    def key(self, request, index):
        snapshot = self.u_reader.refresh()
        row = snapshot.get("requests", {}).get(
            self.identity(request)["client_request_id"], {}
        )
        score = (
            row.get("score")
            if request.status.name in ("WAITING", "PREEMPTED")
            else None
        )
        return priority_key(score, request.arrival_time, index)

    def schedule(self, *args, **kwargs):
        if self.u_reader is None:
            return super().schedule(*args, **kwargs)
        snapshot = self.u_reader.refresh()
        # Requests here have already passed router admission. An unavailable
        # snapshot must let native scheduling drain them, including preemptions.
        # Pin the empty snapshot so both queues keep native order for this pass.
        held = []
        try:
            for queue in (self.waiting, self.kv_holding_waiting):
                queue._peeked = None
                deferred = [
                    request
                    for request in queue
                    if snapshot
                    and snapshot.get("requests", {}).get(
                        self.identity(request)["client_request_id"]
                    )
                    is None
                ]
                held.append((queue, deferred))
                if deferred:
                    queue.remove_requests(deferred)
            with self.u_reader.pin(snapshot):
                return super().schedule(*args, **kwargs)
        finally:
            for queue, deferred in held:
                if deferred:
                    queue.prepend_requests(deferred)

    def _update_after_schedule(self, scheduler_output):
        if scheduler_output.scheduled_spec_decode_tokens:
            raise ValueError("Speculative work is outside Duru's accounting model")
        with self.compact_feed.lock:
            started = time.monotonic()
            pending = {}
            for request_id, count in scheduler_output.num_scheduled_tokens.items():
                request = self.requests[request_id]
                computed = request.num_computed_tokens
                pending[request_id] = (
                    request,
                    dict(
                        **self.identity(request),
                        scheduled_at=started,
                        p=count,
                        d=computed,
                        k=computed + count,
                        phase=(
                            "prefill"
                            if computed < request.num_prompt_tokens or count > 1
                            else "decode"
                        ),
                    ),
                )
            self._pending_work[id(scheduler_output)] = pending
            return super()._update_after_schedule(scheduler_output)

    def update_from_output(self, scheduler_output, runner_output):
        if scheduler_output.scheduled_spec_decode_tokens:
            raise ValueError("Speculative work is outside Duru's accounting model")
        with self.compact_feed.lock:
            self._kv_rejected_work.clear()
            pending = self._pending_work.pop(id(scheduler_output))
            acceptance = {}
            for request_id, (request, _) in pending.items():
                accepted = (
                    self.requests.get(request_id) is request
                    and not request.is_finished()
                    and not (
                        request.num_stale_output_tokens > 0
                        and request.drop_stale_output
                    )
                )
                acceptance[request_id] = (accepted, request.num_output_tokens)
            result = super().update_from_output(scheduler_output, runner_output)
            ended = time.monotonic()
            steps = []
            for request_id, (request, step) in pending.items():
                accepted, previous_tokens = acceptance[request_id]
                accepted = accepted and request_id not in self._kv_rejected_work
                steps.append(
                    dict(
                        step,
                        accepted=accepted,
                        end=ended,
                        generated_tokens=(
                            max(0, request.num_output_tokens - previous_tokens)
                            if accepted
                            else 0
                        ),
                    )
                )
            self.compact_feed.capture("accepted_output", dict(steps=steps))
            finished = [
                self.identity(request)
                for rid, (request, _) in pending.items()
                if acceptance[rid][0] and request.is_finished()
            ]
            if finished:
                self.compact_feed.emit("worker_request_finished", requests=finished)
            return result

    def _handle_failed_recving(self, failed_req_ids):
        rejected = super()._handle_failed_recving(failed_req_ids)
        self._kv_rejected_work.update(rejected)
        return rejected

    def _handle_invalid_blocks(self, invalid_block_ids, num_scheduled_tokens):
        rejected = super()._handle_invalid_blocks(
            invalid_block_ids, num_scheduled_tokens
        )
        # Use the native skip set: a transfer failure can trigger recovery
        # without rejecting any scheduled output in this batch.
        self._kv_rejected_work.update(rejected)
        return rejected

    def finish_requests(self, *args, **kwargs):
        # Abort acknowledgement follows native removal. Later stale output cannot
        # be accepted, so accounting may release a terminal root after this event.
        with self.compact_feed.lock:
            finished = super().finish_requests(*args, **kwargs)
            if finished:
                self.compact_feed.emit(
                    "worker_request_finished",
                    requests=[self.identity(r) for r in finished],
                )
            return finished

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            try:
                self.compact_feed.close()
            finally:
                if self.u_reader is not None:
                    self.u_reader.close()
