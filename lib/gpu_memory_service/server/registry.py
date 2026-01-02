"""Embedded Artifact Registry for the GPU Memory Service.

This is an in-process key/value store served over the same Unix-domain-socket
connection as the Allocation Server RPCs (connection = lock session).

The registry is intentionally generic:
- key: str
- value: opaque bytes (client-defined)

Additionally, each key is associated with an allocation reference:
- allocation_id: str
- offset_bytes: int

This extra linkage enables pruning and integrity tooling.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class RegistryEntry:
    allocation_id: str
    offset_bytes: int
    value: bytes


class ArtifactRegistry:
    """Thread-safe in-memory artifact registry with pruning helpers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kv: Dict[str, RegistryEntry] = {}
        self._alloc_to_keys: Dict[str, Set[str]] = {}

    def put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> None:
        with self._lock:
            # Remove old reverse-index entry if overwriting.
            old = self._kv.get(key)
            if old is not None:
                keys = self._alloc_to_keys.get(old.allocation_id)
                if keys is not None:
                    keys.discard(key)
                    if not keys:
                        self._alloc_to_keys.pop(old.allocation_id, None)

            entry = RegistryEntry(
                allocation_id=allocation_id, offset_bytes=offset_bytes, value=value
            )
            self._kv[key] = entry
            self._alloc_to_keys.setdefault(allocation_id, set()).add(key)

    def get(self, key: str) -> Optional[RegistryEntry]:
        with self._lock:
            return self._kv.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            entry = self._kv.pop(key, None)
            if entry is None:
                return False
            keys = self._alloc_to_keys.get(entry.allocation_id)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._alloc_to_keys.pop(entry.allocation_id, None)
            return True

    def list_keys(self, prefix: str = "") -> List[str]:
        with self._lock:
            if not prefix:
                return sorted(self._kv.keys())
            return sorted([k for k in self._kv.keys() if k.startswith(prefix)])

    def delete_prefix(self, prefix: str) -> int:
        with self._lock:
            to_delete = [k for k in self._kv.keys() if k.startswith(prefix)]
        # Delete outside the initial scan lock to reuse delete() logic safely.
        count = 0
        for k in to_delete:
            if self.delete(k):
                count += 1
        return count

    def prune_allocation(self, allocation_id: str) -> int:
        with self._lock:
            keys = list(self._alloc_to_keys.get(allocation_id, set()))
        count = 0
        for k in keys:
            if self.delete(k):
                count += 1
        return count

    def prune_missing_allocations(self, valid_allocation_ids: Set[str]) -> int:
        with self._lock:
            missing = [
                alloc_id
                for alloc_id in self._alloc_to_keys.keys()
                if alloc_id not in valid_allocation_ids
            ]
        count = 0
        for alloc_id in missing:
            count += self.prune_allocation(alloc_id)
        return count

    def clear(self) -> int:
        with self._lock:
            count = len(self._kv)
            self._kv.clear()
            self._alloc_to_keys.clear()
            return count
