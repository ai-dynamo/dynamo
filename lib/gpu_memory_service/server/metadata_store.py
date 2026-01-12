"""Metadata Store for the GPU Memory Service.

This is an in-process key/value store served over the same Unix-domain-socket
connection as the server RPCs (connection = lock session).

The store is intentionally generic:
- key: str
- value: opaque bytes (client-defined, typically JSON tensor metadata)

Additionally, each key is associated with an allocation reference:
- allocation_id: str
- offset_bytes: int
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MetadataEntry:
    allocation_id: str
    offset_bytes: int
    value: bytes


class GMSMetadataStore:
    """In-memory metadata store for GMS.

    NOT thread-safe. Callers must provide external synchronization via
    GlobalLockFSM's RW/RO semantics:
    - Only RW sessions can mutate (put, delete, clear)
    - RO sessions can only read (get, list_keys)
    - When RW is active, no RO sessions exist
    """

    def __init__(self) -> None:
        self._kv: Dict[str, MetadataEntry] = {}

    def put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> None:
        entry = MetadataEntry(
            allocation_id=allocation_id, offset_bytes=offset_bytes, value=value
        )
        self._kv[key] = entry

    def get(self, key: str) -> Optional[MetadataEntry]:
        return self._kv.get(key)

    def delete(self, key: str) -> bool:
        entry = self._kv.pop(key, None)
        return entry is not None

    def list_keys(self, prefix: str = "") -> List[str]:
        if not prefix:
            return sorted(self._kv.keys())
        return sorted([k for k in self._kv.keys() if k.startswith(prefix)])

    def clear(self) -> int:
        count = len(self._kv)
        self._kv.clear()
        return count
