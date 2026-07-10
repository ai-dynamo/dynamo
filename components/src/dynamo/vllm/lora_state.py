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

    def resolve_request(self, model_name: str | None) -> LoRARequest | None:
        """Return a LoRARequest when model_name targets a loaded adapter."""
        if model_name and (lora := self.loaded_loras.get(model_name)):
            return LoRARequest(
                lora_name=model_name,
                lora_int_id=lora.id,
                lora_path=lora.path,
            )
        return None

    def get_lock(self, lora_name: str) -> asyncio.Lock:
        """Get/create per-LoRA lock without eagerly allocating locks."""
        with self.lora_load_locks_guard:
            lock = self.lora_load_locks.get(lora_name)
            if lock is None:
                lock = asyncio.Lock()
                self.lora_load_locks[lora_name] = lock
            return lock

    def cleanup_lock_if_not_loaded(self, lora_name: str, lock: asyncio.Lock) -> None:
        """Drop lock map entry when adapter is not loaded anymore."""
        with self.lora_load_locks_guard:
            if (
                lora_name not in self.loaded_loras
                and self.lora_load_locks.get(lora_name) is lock
            ):
                self.lora_load_locks.pop(lora_name, None)

    def list_lora_ids(self) -> dict[str, int]:
        """Return map of loaded LoRA names to integer IDs."""
        return {name: lora.id for name, lora in self.loaded_loras.items()}
