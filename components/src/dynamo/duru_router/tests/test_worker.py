# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Duru accounting against CPU-only native scheduler callbacks."""

import importlib.util
import sys
import threading
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class NativeQueue(deque):
    def remove_request(self, request):
        self.remove(request)

    def remove_requests(self, requests):
        removed = set(requests)
        retained = [request for request in self if request not in removed]
        self.clear()
        self.extend(retained)

    def prepend_requests(self, requests):
        self.extendleft(requests)


class NativeScheduler:
    def __init__(self, num_spec_tokens=0, num_sampled_tokens_per_step=1):
        self.num_spec_tokens = num_spec_tokens
        self.num_sampled_tokens_per_step = num_sampled_tokens_per_step
        self.waiting = NativeQueue()
        self.kv_holding_waiting = NativeQueue()
        self.requests = {}
        self.selected_queues = None
        self.failure = None
        self.output_calls = 0
        self.stopped = False

    def add_request(self, request):
        self.requests[request.request_id] = request

    def schedule(self):
        self.selected_queues = (list(self.waiting), list(self.kv_holding_waiting))
        if self.failure is not None:
            raise self.failure
        return self.selected_queues

    def _update_after_schedule(self, scheduler_output):
        for request_id, count in scheduler_output.num_scheduled_tokens.items():
            self.requests[request_id].num_computed_tokens += count

    def update_from_output(self, scheduler_output, runner_output):
        self.output_calls += 1
        for request_id, count in runner_output.generated.items():
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                continue
            if request.num_stale_output_tokens and request.drop_stale_output:
                request.num_stale_output_tokens = 0
                continue
            request.num_output_tokens += count
            if request_id in runner_output.finished:
                request.finished = True
                del self.requests[request_id]
        if self.failure is not None:
            raise self.failure
        return "native-output"

    def finish_requests(self, request_ids, status=None):
        finished = [
            self.requests.pop(rid) for rid in request_ids if rid in self.requests
        ]
        for request in finished:
            request.finished = True
        return finished

    def shutdown(self):
        self.stopped = True
        if self.failure is not None:
            raise self.failure


class Feed:
    def __init__(self, directory):
        self.lock = threading.RLock()
        self.events = []
        self.closed = False

    def capture(self, event, fields):
        if not self.lock._is_owned():
            raise RuntimeError("Capture must hold the clock lock")
        self.events.append((event, fields))

    def emit(self, event, **fields):
        self.capture(event, fields)

    def close(self):
        self.closed = True


class Request:
    def __init__(self, request_id="r", *, computed=0, prompt=8, extra=None):
        self.request_id = request_id
        self.sampling_params = SimpleNamespace(extra_args=extra)
        self.resumable = False
        self.status = SimpleNamespace(name="WAITING")
        self.arrival_time = 1
        self.num_computed_tokens = computed
        self.num_prompt_tokens = prompt
        self.num_output_tokens = 0
        self.num_stale_output_tokens = 0
        self.drop_stale_output = False
        self.finished = False

    def is_finished(self):
        return self.finished


@pytest.fixture
def worker(monkeypatch):
    """Replace GPU imports only; execute the actual Duru adapter."""
    native = ModuleType("vllm.v1.core.sched.async_scheduler")
    native.AsyncScheduler = NativeScheduler
    queues = ModuleType("vllm.v1.core.sched.request_queue")
    queues.FCFSRequestQueue = NativeQueue
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, queues.__name__, queues)
    path = Path(__file__).parents[1] / "worker.py"
    spec = importlib.util.spec_from_file_location("duru_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DURU_SCORES", "unused")
    monkeypatch.setenv("DURU_WORK_DIR", "unused")
    return module


@pytest.fixture
def scheduler(worker):
    return worker.DuruScheduler(event_feed=Feed(None))


def scheduled(count=1):
    return SimpleNamespace(
        num_scheduled_tokens={"r": count}, scheduled_spec_decode_tokens={}
    )


def output(count=1, *, finished=False):
    return SimpleNamespace(
        kv_connector_output=None,
        generated={"r": count},
        finished={"r"} if finished else set(),
    )


@pytest.fixture
def batch(scheduler):
    scheduler.add_request(Request(computed=8))
    result = scheduled()
    scheduler._update_after_schedule(result)
    return result


@pytest.mark.parametrize(
    "computed,count,phase,tokens,finished",
    [
        (3, 5, "prefill", 1, True),
        (0, 4, "prefill", 0, False),
        (8, 1, "decode", 0, False),
    ],
)
def test_coordinates_and_accepted_work(
    scheduler, computed, count, phase, tokens, finished
):
    request = Request(computed=computed)
    scheduler.add_request(request)
    batch = scheduled(count)
    scheduler._update_after_schedule(batch)
    assert scheduler.compact_feed.events == []
    assert request.num_computed_tokens == computed + count
    assert (
        scheduler.update_from_output(batch, output(tokens, finished=finished))
        == "native-output"
    )
    accepted = next(
        fields["steps"][0]
        for event, fields in scheduler.compact_feed.events
        if event == "accepted_output"
    )
    assert (accepted["p"], accepted["d"], accepted["k"], accepted["phase"]) == (
        count,
        computed,
        computed + count,
        phase,
    )
    if finished:
        assert scheduler.compact_feed.events[-1] == (
            "worker_request_finished",
            {"requests": [scheduler.identity(request)]},
        )
    assert accepted["accepted"] and accepted["generated_tokens"] == tokens
    assert accepted["end"] >= accepted["scheduled_at"]
    assert ("r" not in scheduler.requests) == finished
    assert not scheduler._pending_work


@pytest.mark.parametrize(
    "state", ["cancelled", "replaced", "finished", "drop_stale", "deliver_stale"]
)
def test_output_acceptance(scheduler, batch, state):
    request = scheduler.requests["r"]
    if state == "cancelled":
        scheduler.requests.pop("r")
    elif state == "replaced":
        scheduler.requests["r"] = Request()
    elif state == "finished":
        request.finished = True
    else:
        request.num_stale_output_tokens = 1
        request.drop_stale_output = state == "drop_stale"
    scheduler.update_from_output(batch, output())
    step = scheduler.compact_feed.events[-1][1]["steps"][0]
    assert step["accepted"] == (state == "deliver_stale")
    assert step["generated_tokens"] == (1 if state == "deliver_stale" else 0)


def test_inflight_batches_retain_coordinates(scheduler, batch):
    second = scheduled()
    scheduler._update_after_schedule(second)
    for pending in (batch, second):
        scheduler.update_from_output(pending, output())
    steps = [
        fields["steps"][0]
        for event, fields in scheduler.compact_feed.events
        if event == "accepted_output"
    ]
    assert [(step["d"], step["generated_tokens"]) for step in steps] == [(8, 1), (9, 1)]
    assert not scheduler._pending_work


@pytest.mark.parametrize(
    "kwargs", [dict(num_spec_tokens=1), dict(num_sampled_tokens_per_step=0)]
)
def test_unsupported_modes_fail_before_feed_opens(worker, kwargs):
    feed = Feed(None)
    with pytest.raises(ValueError, match="non-speculative"):
        worker.DuruScheduler(event_feed=feed, **kwargs)
    assert not feed.events


@pytest.mark.parametrize("pooling", [False, True])
def test_unsupported_requests_fail_before_admission(scheduler, pooling):
    request = Request()
    request.resumable = not pooling
    if pooling:
        request.sampling_params = None
    with pytest.raises(ValueError, match="non-resumable"):
        scheduler.add_request(request)
    assert not scheduler.requests


@pytest.mark.parametrize("failure", ["invalid_kv", "failed_transfer", "speculative"])
def test_invalid_outputs_fail_before_native_acceptance(scheduler, batch, failure):
    result = output()
    result.kv_connector_output = SimpleNamespace(
        invalid_block_ids={1} if failure == "invalid_kv" else set(),
        failed_recving={"r"} if failure == "failed_transfer" else set(),
    )
    if failure == "speculative":
        batch.scheduled_spec_decode_tokens = {"r": [2]}
    with pytest.raises(ValueError):
        scheduler.update_from_output(batch, result)
    assert id(batch) in scheduler._pending_work
    assert scheduler.output_calls == 0


def test_native_failure_emits_no_work_and_closes_feed(scheduler, batch):
    scheduler.failure = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        scheduler.update_from_output(batch, output())
    assert scheduler.compact_feed.events == []
    with pytest.raises(RuntimeError, match="native failure"):
        scheduler.shutdown()
    assert scheduler.stopped and scheduler.compact_feed.closed


def test_abort_acknowledges_native_removal_before_stale_output(scheduler, batch):
    request = scheduler.requests["r"]
    assert scheduler.finish_requests(["r"]) == [request]
    assert scheduler.compact_feed.events[-1] == (
        "worker_request_finished",
        {"requests": [scheduler.identity(request)]},
    )
    scheduler.update_from_output(batch, output())
    step = scheduler.compact_feed.events[-1][1]["steps"][0]
    assert not step["accepted"]
    assert step["generated_tokens"] == 0


def test_collection_keeps_native_queue_order(scheduler):
    requests = [Request("first"), Request("second")]
    scheduler.waiting.extend(requests)
    assert scheduler.schedule() == (requests, [])
    assert list(scheduler.waiting) == requests
