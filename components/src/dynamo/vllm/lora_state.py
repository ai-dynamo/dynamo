# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass

from vllm.lora.request import LoRARequest

from dynamo.common.lora.manager import LoRAInfo


@dataclass(frozen=True)
class RuntimeLoRAInfo:
    """Private worker-local residency record for a request-time adapter."""

    adapter_key: str
    full_identity_digest: bytes
    base_model_name: str
    source_revision: str
    id: int
    path: str


class LoRAState:
    """Shared LoRA tracking and lock management for vLLM handlers."""

    def __init__(self):
        # name -> LoRAInfo(id, path)
        self.loaded_loras: dict[str, LoRAInfo] = {}
        # Per-LoRA lock to serialize concurrent load/unload operations.
        # Weak values ensure lock entries are reclaimed once no coroutine keeps
        # a strong reference to that lock (held, queued, or local variable).
        self.lora_load_locks: weakref.WeakValueDictionary[
            str, asyncio.Lock
        ] = weakref.WeakValueDictionary()
        self.lora_load_locks_guard = threading.Lock()
        # Runtime adapters are intentionally private and are never published as
        # model cards. Raw source URIs are not retained after resolution.
        self.runtime_loras: OrderedDict[str, RuntimeLoRAInfo] = OrderedDict()
        self.runtime_load_tasks: dict[str, asyncio.Task[RuntimeLoRAInfo]] = {}
        self.runtime_load_digests: dict[str, bytes] = {}
        self.runtime_reserved_ids: dict[str, int] = {}
        self.runtime_eviction_events: dict[str, asyncio.Event] = {}
        self.admin_reserved_ids: dict[str, int] = {}
        self.rollback_reserved_ids: set[int] = set()
        self.uncertain_engine_lora_ids: set[int] = set()
        self.discovery_uncertain_loras: set[str] = set()
        self.runtime_pending_admissions: dict[str, tuple[str, asyncio.TimerHandle]] = {}
        self.runtime_pending_leases: dict[str, int] = {}
        self.runtime_active_leases: dict[str, int] = {}
        self.runtime_expiry_tasks: set[asyncio.Task[None]] = set()
        self.runtime_resolution_semaphore: asyncio.Semaphore | None = None
        self.runtime_cache_guard: asyncio.Lock | None = None

        self.runtime_cache_reserved_bytes = 0

    def resolve_request(
        self,
        model_name: str | None,
        *,
        base_model_names: tuple[str | None, ...] = (),
        lora_enabled: bool = False,
    ) -> LoRARequest | None:
        """Resolve a model name to a loaded LoRA request.

        Returns None for missing model name, base-model aliases, and uncommitted
        placeholder reservations (id=-1). Raises ValueError for unknown non-base
        names when LoRA is enabled.
        """
        if not model_name or model_name in base_model_names:
            return None

        if lora := self.loaded_loras.get(model_name):
            # Skip placeholder reservations (id=-1) that are not yet committed.
            # These are transient entries inserted during load_lora but not yet
            # the real LoRA; returning them would create invalid LoRARequests.
            if lora.id == -1:
                if lora_enabled:
                    # Treat as loading/unknown during the reservation window
                    raise ValueError(f"unknown model or LoRA adapter: '{model_name}'")
                return None

            return LoRARequest(
                lora_name=model_name,
                lora_int_id=lora.id,
                lora_path=lora.path,
            )

        if lora_enabled:
            raise ValueError(f"unknown model or LoRA adapter: '{model_name}'")

        return None

    def get_lock(self, lora_name: str) -> asyncio.Lock:
        """Get/create per-LoRA lock without eagerly allocating locks.

        Lock objects are shared per adapter name while in active use, then
        automatically reclaimed when no task still references them. This bounds
        memory for long-lived workers that churn through many distinct adapter
        names, while preserving per-name serialization for concurrent operations.
        """
        with self.lora_load_locks_guard:
            lock = self.lora_load_locks.get(lora_name)
            if lock is None:
                lock = asyncio.Lock()
                self.lora_load_locks[lora_name] = lock
            return lock

    def is_runtime_managed(self, lora_name: str) -> bool:
        """Return whether runtime loading currently owns an adapter identity."""
        return (
            lora_name in self.runtime_loras
            or lora_name in self.runtime_load_tasks
            or lora_name in self.runtime_eviction_events
        )

    def allocate_lora_id(self, lora_name: str, preferred_id: int) -> int:
        """Return an unused positive ID across committed and in-flight loads."""
        used = {
            info.id: name for name, info in self.loaded_loras.items() if info.id > 0
        }
        used.update(
            {lora_id: name for name, lora_id in self.runtime_reserved_ids.items()}
        )
        used.update(
            {lora_id: name for name, lora_id in self.admin_reserved_ids.items()}
        )
        used.update(
            {lora_id: "<rollback-reserved>" for lora_id in self.rollback_reserved_ids}
        )
        candidate = max(1, int(preferred_id))
        for _ in range(len(used) + 1):
            owner = used.get(candidate)
            if owner is None or owner == lora_name:
                return candidate
            candidate = candidate % 0x7FFFFFFF + 1
        raise ValueError("no collision-free LoRA ID is available")

    def list_lora_ids(self) -> dict[str, int]:
        """Return map of loaded LoRA names to integer IDs.

        Excludes placeholder entries (id=-1) that are reserved during load but not yet committed.
        """
        return {
            name: lora.id
            for name, lora in self.loaded_loras.items()
            if lora.id != -1  # Skip uncommitted placeholder reservations
        }
