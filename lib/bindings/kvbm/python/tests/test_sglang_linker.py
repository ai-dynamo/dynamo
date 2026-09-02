# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fake-core lifecycle tests for :class:`DynamoKvbmLinker`."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.unit,
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _PoolName(Enum):
    KV = "kv"


class _CancelOutcome(Enum):
    LOOKUP_RELEASED = "lookup_released"
    QUEUED_LOAD_CANCELLED = "queued_load_cancelled"
    SUBMITTED_LOAD_RETAINED = "submitted_load_retained"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class _Ticket:
    ticket_id: int
    generation: int
    hit_pages: int


@dataclass(frozen=True)
class _Operation:
    operation_id: int
    generation: int
    kind: str


@dataclass(frozen=True)
class _Completion:
    operation_id: int
    generation: int
    kind: str
    success: bool
    error: str | None = None


class _FakeCore:
    def __init__(self, *, hit_pages: int = 2, events: list[str] | None = None):
        self.hit_pages = hit_pages
        self.events = [] if events is None else events
        self.lookups = []
        self.cancelled = []
        self.load_calls = []
        self.store_calls = []
        self._tickets = set()
        self._operations = {}
        self._completions = []
        self._next_ticket = 1
        self._next_operation = 100
        self._lock = threading.Lock()
        self.completion_polled = threading.Event()
        self.poll_error = None
        self.reset_calls = 0
        self.close_calls = 0

    def lookup_prefix(self, keys):
        with self._lock:
            ticket = _Ticket(
                ticket_id=self._next_ticket,
                generation=0,
                hit_pages=min(self.hit_pages, len(keys)),
            )
            self._next_ticket += 1
            self._tickets.add(ticket.ticket_id)
            self.lookups.append(list(keys))
            return ticket

    def cancel_lookup(self, ticket):
        with self._lock:
            if ticket.ticket_id not in self._tickets:
                raise KeyError("lookup ticket already consumed")
            self._tickets.remove(ticket.ticket_id)
            self.cancelled.append(ticket.ticket_id)

    def enqueue_load(self, ticket, keys, blocks):
        with self._lock:
            if ticket.ticket_id not in self._tickets:
                raise KeyError("lookup ticket already consumed")
            self._tickets.remove(ticket.ticket_id)
            operation = self._new_operation("load")
            self.load_calls.append((ticket, list(keys), list(blocks), operation))
            return operation

    def enqueue_store(self, keys, blocks):
        with self._lock:
            operation = self._new_operation("store")
            self.store_calls.append((list(keys), list(blocks), operation))
            return operation

    def _new_operation(self, kind):
        operation = _Operation(self._next_operation, 0, kind)
        self._next_operation += 1
        self._operations[operation.operation_id] = operation
        return operation

    def complete(self, operation, *, success=True, error=None):
        with self._lock:
            self._operations.pop(operation.operation_id)
            self._completions.append(
                _Completion(
                    operation_id=operation.operation_id,
                    generation=operation.generation,
                    kind=operation.kind,
                    success=success,
                    error=error,
                )
            )

    def poll_completions(self):
        with self._lock:
            if self.poll_error is not None:
                raise self.poll_error
            completions = self._completions
            self._completions = []
            if completions:
                self.completion_polled.set()
            return completions

    def pending_counts(self):
        with self._lock:
            return len(self._tickets), len(self._operations), len(self._completions)

    def reset(self):
        with self._lock:
            self.reset_calls += 1
            self._tickets.clear()
            self._operations.clear()
            self._completions.clear()

    def close(self):
        self.events.append("core.close")
        self.close_calls += 1


class _HostRegion:
    def __init__(self, events: list[str] | None = None):
        self.events = [] if events is None else events
        self.close_calls = 0

    def close(self):
        self.events.append("region.close")
        self.close_calls += 1


def _wait_for_completion_pump(linker, core):
    assert core.completion_polled.wait(timeout=1.0)
    # The pump holds this lock while polling and publishing the completion.
    with linker._lock:
        pass
    core.completion_polled.clear()


@pytest.fixture
def linker_contract(install_module, load_source):
    class UnifiedCacheLinker:
        pass

    class DynamoPlhKeyCodec:
        def __init__(self, manager_namespace):
            self.manager_namespace = manager_namespace

    install_module(
        "sglang.srt.mem_cache.hicache_storage",
        PoolName=_PoolName,
        PoolTransfer=object,
    )
    install_module(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker",
        LinkerCancelOutcome=_CancelOutcome,
        UnifiedCacheLinker=UnifiedCacheLinker,
    )
    install_module(
        "kvbm._core",
        SglangLocalKvStore=object,
        SglangLookupTicket=_Ticket,
    )
    install_module(
        "kvbm.sglang_integration.key_codec",
        DynamoPlhKeyCodec=DynamoPlhKeyCodec,
    )
    module = load_source("test_kvbm_sglang_linker", "linker.py")
    return SimpleNamespace(module=module, pool_name=_PoolName, outcome=_CancelOutcome)


def _transfer(pool_name, *, page_size=2, pages=(0, 1)):
    slots = [
        slot
        for page in pages
        for slot in range(page * page_size, (page + 1) * page_size)
    ]
    return SimpleNamespace(
        name=pool_name.KV,
        keys=[f"sha-{page}" for page in pages],
        linker_keys=[f"key-{page}".encode() for page in pages],
        device_indices=torch.tensor(slots, dtype=torch.int64),
    )


def _linker(contract, core, region=None):
    return contract.module.DynamoKvbmLinker(
        core=core,
        manager_namespace=b"n" * 32,
        page_size=2,
        num_device_blocks=8,
        num_layers=3,
        host_region=_HostRegion() if region is None else region,
    )


def test_lookup_and_queued_cancel_release_ticket(linker_contract):
    core = _FakeCore()
    linker = _linker(linker_contract, core)
    transfer = _transfer(linker_contract.pool_name)
    try:
        assert linker.lookup("request", [transfer]) == [1, 2]
        assert linker.load("request", [transfer]) is True

        assert (
            linker.cancel_request("request")
            is linker_contract.outcome.QUEUED_LOAD_CANCELLED
        )
        assert core.cancelled == [1]
        assert linker.has_pending_operations() is False
    finally:
        linker.close()


def test_load_cardinality_failure_leaves_lookup_ticket_cancellable(linker_contract):
    core = _FakeCore(hit_pages=1)
    linker = _linker(linker_contract, core)
    lookup_transfer = _transfer(linker_contract.pool_name, pages=(0,))
    load_transfer = _transfer(linker_contract.pool_name, pages=(0, 1))
    load_transfer.keys = lookup_transfer.keys
    load_transfer.linker_keys = lookup_transfer.linker_keys
    try:
        assert linker.lookup("request", [lookup_transfer]) == [1]
        with pytest.raises(ValueError, match="load key/page cardinality"):
            linker.load("request", [load_transfer])
        assert (
            linker.cancel_request("request") is linker_contract.outcome.LOOKUP_RELEASED
        )
    finally:
        linker.close()


def test_submitted_cancel_retains_guards_until_completion(linker_contract):
    core = _FakeCore()
    linker = _linker(linker_contract, core)
    transfer = _transfer(linker_contract.pool_name)
    try:
        linker.lookup("request", [transfer])
        linker.load("request", [transfer])
        consumer_index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(consumer_index)

        assert (
            linker.cancel_request("request")
            is linker_contract.outcome.SUBMITTED_LOAD_RETAINED
        )
        operation = core.load_calls[0][3]
        core.complete(operation)
        _wait_for_completion_pump(linker, core)
        assert linker.num_completed_loads() == 1

        linker.layer_done_counter.wait_until(2)
        assert linker.pop_completed_load() == ["request"]
        assert linker.cancel_request("request") is linker_contract.outcome.NOT_FOUND
        assert linker.has_pending_operations() is False
    finally:
        linker.close()


def test_offload_completions_are_published_in_submission_order(linker_contract):
    core = _FakeCore()
    linker = _linker(linker_contract, core)
    transfer = _transfer(linker_contract.pool_name)
    try:
        assert linker.offload([transfer]) is True
        assert linker.offload([transfer]) is True
        first = core.store_calls[0][2]
        second = core.store_calls[1][2]

        core.complete(second, success=False, error="D2H failed")
        _wait_for_completion_pump(linker, core)
        assert linker.num_completed_offloads() == 0

        core.complete(first)
        _wait_for_completion_pump(linker, core)
        assert linker.num_completed_offloads() == 2
        assert linker.pop_completed_offload() is True
        assert linker.pop_completed_offload() is False
        assert core.store_calls[0][:2] == ([b"key-0", b"key-1"], [0, 1])
    finally:
        linker.close()


def test_reset_cancels_unsubmitted_tickets_before_resetting_core(linker_contract):
    core = _FakeCore()
    linker = _linker(linker_contract, core)
    transfer = _transfer(linker_contract.pool_name)
    try:
        linker.lookup("lookup-only", [transfer])
        linker.lookup("queued", [transfer])
        linker.load("queued", [transfer])

        linker.reset()

        assert core.cancelled == [1, 2]
        assert core.reset_calls == 1
        assert linker.has_pending_operations() is False
    finally:
        linker.close()


def test_close_releases_core_before_owner_region(linker_contract):
    events = []
    core = _FakeCore(events=events)
    region = _HostRegion(events=events)
    linker = _linker(linker_contract, core, region)

    linker.close()
    linker.close()

    assert events == ["core.close", "region.close"]
    assert core.close_calls == 1
    assert region.close_calls == 1


@pytest.mark.parametrize(
    ("slots", "message"),
    [
        ([1, 2], "aligned contiguous pages"),
        ([0, 2], "aligned contiguous pages"),
        ([16, 17], "outside the registered G1 layout"),
        ([0, 1, 0, 1], "cannot reuse a device page"),
    ],
)
def test_device_page_validation_fails_closed(linker_contract, slots, message):
    core = _FakeCore()
    linker = _linker(linker_contract, core)
    transfer = _transfer(linker_contract.pool_name, pages=(0,))
    transfer.device_indices = torch.tensor(slots, dtype=torch.int64)
    try:
        with pytest.raises(ValueError, match=message):
            linker.offload([transfer])
        assert core.store_calls == []
    finally:
        linker.close()
