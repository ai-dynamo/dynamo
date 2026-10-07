# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-local cache accounting for request-time LoRA snapshots."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from dynamo.llm import HttpError

from .lora_lifecycle import run_lora_mutation
from .lora_state import LoRAState


class RuntimeLoRACacheSettings(Protocol):
    @property
    def max_download_bytes(self) -> int:
        ...

    @property
    def max_cache_bytes(self) -> int:
        ...


def _runtime_error(status: int, code: str) -> HttpError:
    return HttpError(status, code)


async def _release_cache_reservation(
    state: LoRAState,
    guard: asyncio.Lock,
    reservation: int,
) -> None:
    async with guard:
        state.runtime_cache_reserved_bytes -= reservation


def _cache_size(cache_root: Path, stop_after: int) -> int:
    try:
        canonical_root = Path(cache_root).resolve(strict=True)
    except OSError:
        raise _runtime_error(500, "lora_plugin_error") from None

    total_bytes = 0
    try:
        for root, directories, files in os.walk(canonical_root, followlinks=False):
            for name in [*directories, *files]:
                metadata = (Path(root) / name).lstat()
                if stat.S_ISREG(metadata.st_mode):
                    total_bytes += metadata.st_size
                    if total_bytes > stop_after:
                        return total_bytes
    except OSError:
        raise _runtime_error(500, "lora_plugin_error") from None
    return total_bytes


@asynccontextmanager
async def cache_reservation(
    state: LoRAState,
    cache_root: Path,
    settings: RuntimeLoRACacheSettings,
) -> AsyncIterator[int]:
    guard = state.runtime_cache_guard
    if guard is None:
        guard = asyncio.Lock()
        state.runtime_cache_guard = guard
    async with guard:
        cache_bytes = await asyncio.to_thread(
            _cache_size,
            cache_root,
            settings.max_cache_bytes,
        )
        available = (
            settings.max_cache_bytes - cache_bytes - state.runtime_cache_reserved_bytes
        )
        if available < 0:
            raise _runtime_error(429, "lora_capacity_exceeded")
        reservation = min(settings.max_download_bytes, available)
        state.runtime_cache_reserved_bytes += reservation
    try:
        yield reservation
    finally:
        cancelled, _released, error = await run_lora_mutation(
            _release_cache_reservation(state, guard, reservation)
        )
        if error is not None:
            raise error
        if cancelled:
            raise asyncio.CancelledError


async def check_cache_limit(
    state: LoRAState,
    cache_root: Path,
    max_cache_bytes: int,
) -> None:
    guard = state.runtime_cache_guard
    if guard is None:
        raise _runtime_error(500, "lora_plugin_error")
    async with guard:
        cache_bytes = await asyncio.to_thread(
            _cache_size,
            cache_root,
            max_cache_bytes,
        )
        if cache_bytes > max_cache_bytes:
            raise _runtime_error(429, "lora_capacity_exceeded")
