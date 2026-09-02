# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct SGLang GPU-pool linker backed by the typed Rust KVBM core."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LinkerCancelOutcome,
    UnifiedCacheLinker,
)

# isort: split

from kvbm._core import SglangLocalKvStore, SglangLookupTicket
from kvbm.sglang_integration.key_codec import DynamoPlhKeyCodec


@dataclass
class _CounterSlot:
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class FullTransferLayerCounter:
    """Conservative layer counter which gates every layer on full H2D."""

    def __init__(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError("A layer counter requires at least one model layer.")
        self.num_layers = num_layers
        self._slots: dict[int, _CounterSlot] = {}
        self._producer = -1
        self._consumer = -1
        self._lock = threading.Lock()

    def begin(self) -> int:
        with self._lock:
            self._producer += 1
            self._slots[self._producer] = _CounterSlot()
            return self._producer

    def set_consumer(self, index: int) -> None:
        with self._lock:
            self._consumer = index

    def complete(self, index: int) -> None:
        with self._lock:
            slot = self._slots[index]
        slot.done.set()

    def fail(self, index: int, error: BaseException) -> None:
        with self._lock:
            slot = self._slots[index]
            slot.error = error
        slot.done.set()

    def wait_until(self, threshold: int) -> None:
        if not 0 <= threshold < self.num_layers:
            raise ValueError(f"Layer threshold {threshold} is out of range.")
        with self._lock:
            index = self._consumer
            slot = self._slots.get(index)
        if index < 0:
            return
        if slot is None:
            raise RuntimeError(f"Unknown Dynamo KVBM counter slot {index}.")
        slot.done.wait()
        if slot.error is not None:
            raise RuntimeError("Dynamo KVBM H2D transfer failed.") from slot.error
        if threshold == self.num_layers - 1:
            with self._lock:
                self._slots.pop(index, None)

    def reset(self) -> None:
        with self._lock:
            self._slots.clear()
            self._producer = -1
            self._consumer = -1


@dataclass
class _QueuedLoad:
    ticket: SglangLookupTicket
    keys: list[bytes]
    blocks: list[int]


@dataclass
class _LoadBatch:
    rids: list[str]
    pending_operations: set[int]
    consumer_index: int
    error: BaseException | None = None


class DynamoKvbmLinker(UnifiedCacheLinker):
    """Asynchronous request adapter around :class:`SglangLocalKvStore`."""

    def __init__(
        self,
        core: SglangLocalKvStore,
        manager_namespace: bytes,
        page_size: int,
        num_device_blocks: int,
        num_layers: int,
        host_region,
    ):
        self.core = core
        self.key_codec = DynamoPlhKeyCodec(manager_namespace)
        self.page_size = page_size
        self.num_device_blocks = num_device_blocks
        self.host_region = host_region
        self.layer_done_counter = FullTransferLayerCounter(num_layers)
        self._lookup_tickets: dict[str, SglangLookupTicket] = {}
        self._queued_loads: dict[str, _QueuedLoad] = {}
        self._submitted_loads: set[str] = set()
        self._load_operations: dict[int, tuple[int, str, int]] = {}
        self._load_batches: dict[int, _LoadBatch] = {}
        self._completed_loads: queue.Queue[list[str]] = queue.Queue()
        self._offload_operations: dict[int, tuple[int, int]] = {}
        self._offload_ready: dict[int, bool] = {}
        self._next_offload_sequence = 0
        self._next_offload_completion = 0
        self._completed_offloads: queue.Queue[bool] = queue.Queue()
        self._fatal_error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._pump = threading.Thread(
            target=self._completion_pump,
            name="dynamo-kvbm-completions",
            daemon=True,
        )
        self._pump.start()

    def _check_healthy(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                "Dynamo KVBM linker is unhealthy."
            ) from self._fatal_error

    def _keys(self, transfer: PoolTransfer) -> list[bytes]:
        if transfer.name is not PoolName.KV:
            raise ValueError("Dynamo KVBM V1 only supports the KV pool.")
        if transfer.linker_keys is None or not transfer.linker_keys:
            raise ValueError("Dynamo KVBM transfer is missing linker-owned page keys.")
        if transfer.keys is None or len(transfer.linker_keys) != len(transfer.keys):
            raise ValueError("Dynamo KVBM linker/SGLang key cardinality differs.")
        return list(transfer.linker_keys)

    def _blocks(self, transfer: PoolTransfer) -> list[int]:
        indices = transfer.device_indices
        if indices is None or indices.numel() == 0:
            raise ValueError("Dynamo KVBM transfer has no device slots.")
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError("Device slots do not contain complete KVBM pages.")
        pages = slots.reshape(-1, self.page_size)
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + offsets
        ):
            raise ValueError("Device slots must be aligned contiguous pages.")
        blocks = starts.div(self.page_size, rounding_mode="floor")
        if int(blocks.min()) < 0 or int(blocks.max()) >= self.num_device_blocks:
            raise ValueError("Device page falls outside the registered G1 layout.")
        if blocks.unique().numel() != blocks.numel():
            raise ValueError("A KVBM transfer cannot reuse a device page.")
        return blocks.tolist()

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        with self._lock:
            self._check_healthy()
            if len(transfers) != 1:
                raise ValueError("Dynamo KVBM V1 requires exactly one KV transfer.")
            if (
                rid in self._lookup_tickets
                or rid in self._queued_loads
                or rid in self._submitted_loads
            ):
                raise RuntimeError(f"Duplicate KVBM lookup for rid={rid!r}.")
            ticket = self.core.lookup_prefix(self._keys(transfers[0]))
            if ticket.hit_pages == 0:
                self.core.cancel_lookup(ticket)
                return []
            self._lookup_tickets[rid] = ticket
            return list(range(1, ticket.hit_pages + 1))

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        with self._lock:
            self._check_healthy()
            if len(transfers) != 1:
                raise ValueError(
                    "Dynamo KVBM V1 requires exactly one KV load transfer."
                )
            transfer = transfers[0]
            keys = self._keys(transfer)
            blocks = self._blocks(transfer)
            if len(keys) != len(blocks):
                raise ValueError("KVBM load key/page cardinality differs.")
            ticket = self._lookup_tickets.get(rid)
            if ticket is None:
                raise RuntimeError(f"KVBM load has no lookup ticket for rid={rid!r}.")
            if len(keys) > ticket.hit_pages:
                raise ValueError("KVBM load exceeds its lookup-ticket prefix.")
            del self._lookup_tickets[rid]
            self._queued_loads[rid] = _QueuedLoad(
                ticket=ticket,
                keys=keys,
                blocks=blocks,
            )
            return True

    def start_layer_wise_loading(self) -> int:
        with self._lock:
            self._check_healthy()
            if not self._queued_loads:
                return -1
            queued = self._queued_loads
            self._queued_loads = {}
            consumer_index = self.layer_done_counter.begin()
            batch = _LoadBatch(
                rids=[], pending_operations=set(), consumer_index=consumer_index
            )
            self._load_batches[consumer_index] = batch
            queued_items = list(queued.items())
            try:
                for _item_index, (rid, load) in enumerate(queued_items):
                    operation = self.core.enqueue_load(
                        load.ticket, load.keys, load.blocks
                    )
                    if operation.kind != "load":
                        raise RuntimeError("KVBM returned a non-load operation handle.")
                    batch.rids.append(rid)
                    batch.pending_operations.add(operation.operation_id)
                    self._load_operations[operation.operation_id] = (
                        consumer_index,
                        rid,
                        operation.generation,
                    )
                    self._submitted_loads.add(rid)
            except Exception as error:
                try:
                    self.core.cancel_lookup(load.ticket)
                except KeyError:
                    # enqueue_load consumed the ticket before its transfer failed.
                    pass
                for _, unsubmitted in queued_items[_item_index + 1 :]:
                    self.core.cancel_lookup(unsubmitted.ticket)
                self._fatal_error = error
                batch.error = error
                self.layer_done_counter.fail(consumer_index, error)
                raise
            return consumer_index

    def cancel_queued_load(self, rid: str) -> bool:
        with self._lock:
            return self._cancel_queued_load(rid)

    def _cancel_queued_load(self, rid: str) -> bool:
        queued = self._queued_loads.get(rid)
        if queued is None:
            return False
        self.core.cancel_lookup(queued.ticket)
        del self._queued_loads[rid]
        return True

    def cancel_request(self, rid: str) -> LinkerCancelOutcome:
        with self._lock:
            ticket = self._lookup_tickets.get(rid)
            if ticket is not None:
                self.core.cancel_lookup(ticket)
                del self._lookup_tickets[rid]
                return LinkerCancelOutcome.LOOKUP_RELEASED
            if self._cancel_queued_load(rid):
                return LinkerCancelOutcome.QUEUED_LOAD_CANCELLED
            if rid in self._submitted_loads:
                return LinkerCancelOutcome.SUBMITTED_LOAD_RETAINED
            return LinkerCancelOutcome.NOT_FOUND

    def num_completed_loads(self) -> int:
        self._check_healthy()
        return self._completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        with self._lock:
            rids = self._completed_loads.get_nowait()
            self._submitted_loads.difference_update(rids)
            return rids

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        with self._lock:
            self._check_healthy()
            if len(transfers) != 1:
                raise ValueError(
                    "Dynamo KVBM V1 requires exactly one KV store transfer."
                )
            transfer = transfers[0]
            keys = self._keys(transfer)
            blocks = self._blocks(transfer)
            if len(keys) != len(blocks):
                raise ValueError("KVBM store key/page cardinality differs.")
            operation = self.core.enqueue_store(keys, blocks)
            if operation.kind != "store":
                raise RuntimeError("KVBM returned a non-store operation handle.")
            sequence = self._next_offload_sequence
            self._next_offload_sequence += 1
            self._offload_operations[operation.operation_id] = (
                sequence,
                operation.generation,
            )
            return True

    def num_completed_offloads(self) -> int:
        self._check_healthy()
        return self._completed_offloads.qsize()

    def pop_completed_offload(self) -> bool:
        return self._completed_offloads.get_nowait()

    def has_pending_operations(self) -> bool:
        with self._lock:
            tickets, operations, completions = self.core.pending_counts()
            return bool(
                tickets
                or operations
                or completions
                or self._lookup_tickets
                or self._queued_loads
                or self._submitted_loads
                or self._offload_operations
                or not self._completed_loads.empty()
                or not self._completed_offloads.empty()
            )

    def _completion_pump(self) -> None:
        while not self._stop.wait(0.001):
            with self._lock:
                try:
                    completions = self.core.poll_completions()
                except RuntimeError as error:
                    if not self._stop.is_set():
                        self._fatal_error = error
                    return
                for completion in completions:
                    if completion.kind == "load":
                        self._complete_load(completion)
                    elif completion.kind == "store":
                        self._complete_offload(completion)
                    else:
                        self._fatal_error = RuntimeError(
                            f"Unknown KVBM completion kind {completion.kind!r}."
                        )

    def _complete_load(self, completion) -> None:
        operation = self._load_operations.pop(completion.operation_id, None)
        if operation is None:
            self._fatal_error = RuntimeError("Unknown KVBM load completion.")
            return
        consumer_index, _, generation = operation
        if completion.generation != generation:
            self._fatal_error = RuntimeError("Stale KVBM load completion generation.")
            return
        batch = self._load_batches[consumer_index]
        batch.pending_operations.remove(completion.operation_id)
        if not completion.success:
            error = RuntimeError(completion.error or "KVBM H2D failed")
            self._fatal_error = error
            if batch.error is None:
                batch.error = error
                self.layer_done_counter.fail(consumer_index, error)
        if not batch.pending_operations:
            if batch.error is None:
                self.layer_done_counter.complete(consumer_index)
                self._completed_loads.put(batch.rids)
            del self._load_batches[consumer_index]

    def _complete_offload(self, completion) -> None:
        operation = self._offload_operations.pop(completion.operation_id, None)
        if operation is None:
            self._fatal_error = RuntimeError("Unknown KVBM store completion.")
            return
        sequence, generation = operation
        if completion.generation != generation:
            self._fatal_error = RuntimeError("Stale KVBM store completion generation.")
            return
        self._offload_ready[sequence] = completion.success
        while self._next_offload_completion in self._offload_ready:
            self._completed_offloads.put(
                self._offload_ready.pop(self._next_offload_completion)
            )
            self._next_offload_completion += 1

    def reset(self) -> None:
        with self._lock:
            for rid, ticket in list(self._lookup_tickets.items()):
                self.core.cancel_lookup(ticket)
                del self._lookup_tickets[rid]
            for rid, load in list(self._queued_loads.items()):
                self.core.cancel_lookup(load.ticket)
                del self._queued_loads[rid]
            self.core.reset()
            self._submitted_loads.clear()
            self._load_operations.clear()
            self._load_batches.clear()
            self._offload_operations.clear()
            self._offload_ready.clear()
            self._next_offload_sequence = 0
            self._next_offload_completion = 0
            self._drain_queue(self._completed_loads)
            self._drain_queue(self._completed_offloads)
            self._fatal_error = None
            self.layer_done_counter.reset()

    @staticmethod
    def _drain_queue(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._pump.join()
        self.core.close()
        self.host_region.close()
        self._closed = True
