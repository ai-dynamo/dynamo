# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side KV block lease table.

The lease table is intentionally separate from CUDA allocation ownership.
Persistent allocations give multiple cooperating engines access to the same
physical KV pool. Leases decide which engine is allowed to hand out a KV
block for writes and when immutable blocks may be pinned for reads.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

STATE_FREE = "free"
STATE_WRITING = "writing"
STATE_SEALED = "sealed"
STATE_RETIRING = "retiring"


class KVLeaseError(Exception):
    """Base class for lease-table errors."""


class KVLeaseConflictError(KVLeaseError):
    """A requested lease transition conflicts with current block state."""


@dataclass
class KVBlockLeaseRecord:
    block_id: int
    generation: int
    lease_epoch: int
    owner_id: str
    state: str
    read_pins: int = 0


@dataclass
class _BlockState:
    block_id: int
    state: str = STATE_FREE
    owner_id: str = ""
    generation: int = 0
    lease_epoch: int = 0
    read_pins: dict[str, int] = field(default_factory=dict)

    @property
    def total_pins(self) -> int:
        return sum(self.read_pins.values())

    def as_record(self) -> KVBlockLeaseRecord:
        return KVBlockLeaseRecord(
            block_id=self.block_id,
            generation=self.generation,
            lease_epoch=self.lease_epoch,
            owner_id=self.owner_id,
            state=self.state,
            read_pins=self.total_pins,
        )


@dataclass
class _Namespace:
    name: str
    total_blocks: int
    blocks: list[_BlockState]
    free_blocks: deque[int]


class KVLeaseManager:
    """Thread-safe, in-memory lease table for one GMS daemon."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._namespaces: dict[str, _Namespace] = {}

    def init_namespace(
        self,
        namespace: str,
        total_blocks: int,
        *,
        reserved_blocks: Iterable[int] = (),
    ) -> int:
        if not namespace:
            raise ValueError("namespace must be non-empty")
        if total_blocks <= 0:
            raise ValueError(f"total_blocks must be > 0, got {total_blocks}")
        reserved = {int(b) for b in reserved_blocks}
        for block_id in reserved:
            if block_id < 0 or block_id >= total_blocks:
                raise ValueError(
                    f"reserved block {block_id} out of range for {total_blocks} blocks"
                )

        with self._lock:
            existing = self._namespaces.get(namespace)
            if existing is not None:
                if total_blocks > existing.total_blocks:
                    raise KVLeaseConflictError(
                        f"namespace {namespace!r} already exists with "
                        f"{existing.total_blocks} blocks, not {total_blocks}"
                    )
                self._reserve_free_blocks(existing, reserved)
                return existing.total_blocks

            blocks = [_BlockState(i) for i in range(total_blocks)]
            free = deque(i for i in range(total_blocks) if i not in reserved)
            ns = _Namespace(namespace, total_blocks, blocks, free)
            self._namespaces[namespace] = ns
            for block_id in reserved:
                block = ns.blocks[block_id]
                block.state = STATE_SEALED
                block.owner_id = "__gms_reserved__"
                block.generation += 1
                block.lease_epoch += 1
            return total_blocks

    def acquire(
        self,
        namespace: str,
        owner_id: str,
        count: int,
        *,
        preferred_blocks: Iterable[int] = (),
        allow_partial: bool = False,
        strict_preferred: bool = False,
    ) -> list[KVBlockLeaseRecord]:
        if not owner_id:
            raise ValueError("owner_id must be non-empty")
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return []

        with self._lock:
            ns = self._require_namespace(namespace)
            selected = self._select_free_blocks(
                ns,
                count,
                [int(b) for b in preferred_blocks],
                strict_preferred=strict_preferred,
            )
            if len(selected) < count and not allow_partial:
                raise KVLeaseConflictError(
                    f"namespace {namespace!r} has only {len(selected)} leaseable "
                    f"blocks for owner {owner_id!r}, need {count}"
                )
            if selected:
                selected_set = set(selected)
                ns.free_blocks = deque(
                    block_id
                    for block_id in ns.free_blocks
                    if block_id not in selected_set
                )
            records: list[KVBlockLeaseRecord] = []
            for block_id in selected:
                block = ns.blocks[block_id]
                block.state = STATE_WRITING
                block.owner_id = owner_id
                block.generation += 1
                block.lease_epoch += 1
                block.read_pins.clear()
                records.append(block.as_record())
            return records

    def seal(
        self,
        namespace: str,
        owner_id: str,
        block_ids: Iterable[int],
        generations: Iterable[int] = (),
    ) -> list[KVBlockLeaseRecord]:
        ids = self._unique_ids(block_ids)
        gens = list(generations)
        with self._lock:
            ns = self._require_namespace(namespace)
            self._validate_owned(ns, owner_id, ids, gens, {STATE_WRITING, STATE_SEALED})
            records: list[KVBlockLeaseRecord] = []
            for block_id in ids:
                block = ns.blocks[block_id]
                block.state = STATE_SEALED
                records.append(block.as_record())
            return records

    def release(
        self,
        namespace: str,
        owner_id: str,
        block_ids: Iterable[int],
        generations: Iterable[int] = (),
    ) -> list[KVBlockLeaseRecord]:
        ids = self._unique_ids(block_ids)
        gens = list(generations)
        with self._lock:
            ns = self._require_namespace(namespace)
            self._validate_owned(
                ns, owner_id, ids, gens, {STATE_WRITING, STATE_SEALED, STATE_RETIRING}
            )
            records: list[KVBlockLeaseRecord] = []
            for block_id in ids:
                block = ns.blocks[block_id]
                if block.total_pins:
                    block.state = STATE_RETIRING
                    records.append(block.as_record())
                    continue
                self._make_free(ns, block)
                records.append(block.as_record())
            return records

    def pin(
        self,
        namespace: str,
        reader_id: str,
        block_ids: Iterable[int],
        generations: Iterable[int] = (),
    ) -> list[KVBlockLeaseRecord]:
        if not reader_id:
            raise ValueError("reader_id must be non-empty")
        ids = self._unique_ids(block_ids)
        gens = list(generations)
        with self._lock:
            ns = self._require_namespace(namespace)
            for i, block_id in enumerate(ids):
                block = self._require_block(ns, block_id)
                if block.state != STATE_SEALED:
                    raise KVLeaseConflictError(
                        f"block {block_id} is {block.state}, not sealed/readable"
                    )
                self._validate_generation(block, gens, i)
            records: list[KVBlockLeaseRecord] = []
            for block_id in ids:
                block = ns.blocks[block_id]
                block.read_pins[reader_id] = block.read_pins.get(reader_id, 0) + 1
                records.append(block.as_record())
            return records

    def unpin(
        self,
        namespace: str,
        reader_id: str,
        block_ids: Iterable[int],
        generations: Iterable[int] = (),
    ) -> list[KVBlockLeaseRecord]:
        if not reader_id:
            raise ValueError("reader_id must be non-empty")
        ids = self._unique_ids(block_ids)
        gens = list(generations)
        with self._lock:
            ns = self._require_namespace(namespace)
            for i, block_id in enumerate(ids):
                block = self._require_block(ns, block_id)
                self._validate_generation(block, gens, i)
                if block.read_pins.get(reader_id, 0) <= 0:
                    raise KVLeaseConflictError(
                        f"reader {reader_id!r} does not hold a pin on block {block_id}"
                    )
            records: list[KVBlockLeaseRecord] = []
            for block_id in ids:
                block = ns.blocks[block_id]
                remaining = block.read_pins[reader_id] - 1
                if remaining:
                    block.read_pins[reader_id] = remaining
                else:
                    block.read_pins.pop(reader_id, None)
                if block.state == STATE_RETIRING and block.total_pins == 0:
                    self._make_free(ns, block)
                records.append(block.as_record())
            return records

    def list(self, namespace: str) -> list[KVBlockLeaseRecord]:
        with self._lock:
            ns = self._require_namespace(namespace)
            return [block.as_record() for block in ns.blocks]

    def count_free(self, namespace: str) -> int:
        with self._lock:
            ns = self._require_namespace(namespace)
            return len(ns.free_blocks)

    def _require_namespace(self, namespace: str) -> _Namespace:
        ns = self._namespaces.get(namespace)
        if ns is None:
            raise KVLeaseConflictError(f"unknown KV lease namespace {namespace!r}")
        return ns

    def _require_block(self, ns: _Namespace, block_id: int) -> _BlockState:
        if block_id < 0 or block_id >= ns.total_blocks:
            raise KVLeaseConflictError(
                f"block {block_id} out of range for namespace {ns.name!r}"
            )
        return ns.blocks[block_id]

    def _reserve_free_blocks(self, ns: _Namespace, reserved: set[int]) -> None:
        if not reserved:
            return
        free = deque()
        for block_id in ns.free_blocks:
            if block_id in reserved:
                block = ns.blocks[block_id]
                block.state = STATE_SEALED
                block.owner_id = "__gms_reserved__"
                block.generation += 1
                block.lease_epoch += 1
            else:
                free.append(block_id)
        ns.free_blocks = free

    def _select_free_blocks(
        self,
        ns: _Namespace,
        count: int,
        preferred: list[int],
        *,
        strict_preferred: bool,
    ) -> list[int]:
        selected: list[int] = []
        selected_set: set[int] = set()
        if preferred:
            for block_id in preferred:
                if len(selected) == count:
                    break
                if block_id in selected_set:
                    continue
                if block_id < 0 or block_id >= ns.total_blocks:
                    continue
                if ns.blocks[block_id].state == STATE_FREE:
                    selected.append(block_id)
                    selected_set.add(block_id)
        if not strict_preferred and len(selected) < count:
            for block_id in list(ns.free_blocks):
                if len(selected) == count:
                    break
                if block_id in selected_set:
                    continue
                selected.append(block_id)
                selected_set.add(block_id)
        return selected

    def _validate_owned(
        self,
        ns: _Namespace,
        owner_id: str,
        ids: list[int],
        generations: list[int],
        allowed_states: set[str],
    ) -> None:
        if not owner_id:
            raise ValueError("owner_id must be non-empty")
        for i, block_id in enumerate(ids):
            block = self._require_block(ns, block_id)
            if block.owner_id != owner_id:
                raise KVLeaseConflictError(
                    f"block {block_id} is owned by {block.owner_id!r}, not {owner_id!r}"
                )
            if block.state not in allowed_states:
                raise KVLeaseConflictError(
                    f"block {block_id} is {block.state}, expected one of "
                    f"{sorted(allowed_states)}"
                )
            self._validate_generation(block, generations, i)

    def _validate_generation(
        self,
        block: _BlockState,
        generations: list[int],
        index: int,
    ) -> None:
        if not generations:
            return
        if index >= len(generations):
            raise KVLeaseConflictError("generation list shorter than block list")
        expected = int(generations[index])
        if expected != block.generation:
            raise KVLeaseConflictError(
                f"stale generation for block {block.block_id}: "
                f"expected {expected}, current {block.generation}"
            )

    def _make_free(self, ns: _Namespace, block: _BlockState) -> None:
        block.state = STATE_FREE
        block.owner_id = ""
        block.read_pins.clear()
        ns.free_blocks.append(block.block_id)

    @staticmethod
    def _unique_ids(block_ids: Iterable[int]) -> list[int]:
        ids = [int(b) for b in block_ids]
        if len(ids) != len(set(ids)):
            raise KVLeaseConflictError("duplicate block ids in lease operation")
        return ids
