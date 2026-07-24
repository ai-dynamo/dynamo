# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading

from vllm.lora.request import LoRARequest

from dynamo.common.lora.manager import LoRAInfo


class LoRAState:
    """Shared LoRA tracking and lock management for vLLM handlers."""

    def __init__(self):
        # name -> LoRAInfo(id, path)
        self.loaded_loras: dict[str, LoRAInfo] = {}
        # Per-LoRA lock to serialize concurrent load/unload operations.
        self.lora_load_locks: dict[str, asyncio.Lock] = {}
        self.lora_load_locks_guard = threading.Lock()

    def resolve_request(
        self,
        model_name: str | None,
        *,
        base_model_names: tuple[str | None, ...] = (),
        lora_enabled: bool = False,
    ) -> LoRARequest | None:
        """Resolve a model name to a loaded LoRA request.

        Returns None for missing model name and base-model aliases.
        Raises ValueError for unknown non-base names when LoRA is enabled.
        """
        if not model_name or model_name in base_model_names:
            return None

        if lora := self.loaded_loras.get(model_name):
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

        Locks are retained indefinitely (never evicted) to ensure the invariant
        that a given adapter name always maps to the same asyncio.Lock. This is
        critical for serializing load, unload, and inference-admission operations
        on the same adapter, preventing races where an unload deletes bookkeeping
        during vLLM's lazy adapter activation.

        This non-evicting design is analogous to the striped locks in llm_engine.py.
        """
        with self.lora_load_locks_guard:
            lock = self.lora_load_locks.get(lora_name)
            if lock is None:
                lock = asyncio.Lock()
                self.lora_load_locks[lora_name] = lock
            return lock

    def list_lora_ids(self) -> dict[str, int]:
        """Return map of loaded LoRA names to integer IDs."""
        return {name: lora.id for name, lora in self.loaded_loras.items()}
