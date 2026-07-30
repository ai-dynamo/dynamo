# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import weakref

from vllm.lora.request import LoRARequest

from dynamo.common.lora.manager import LoRAInfo


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

    def list_lora_ids(self) -> dict[str, int]:
        """Return map of loaded LoRA names to integer IDs.

        Excludes placeholder entries (id=-1) that are reserved during load but not yet committed.
        """
        return {
            name: lora.id
            for name, lora in self.loaded_loras.items()
            if lora.id != -1  # Skip uncommitted placeholder reservations
        }
