# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache accounting and artifact validation for request-time vLLM LoRAs."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from itertools import pairwise
from math import prod
from pathlib import Path
from typing import Protocol

from dynamo.llm import HttpError

from .lora_lifecycle import run_lora_mutation
from .lora_state import LoRAState

_MAX_FILES = 4096
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_ALLOWED_SNAPSHOT_FILES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "new_embeddings.safetensors",
    }
)
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class RuntimeLoRACacheSettings(Protocol):
    max_download_bytes: int
    max_cache_bytes: int


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


def _validate_safetensors(weights_path: Path) -> frozenset[str]:
    try:
        file_size = weights_path.stat().st_size
        with weights_path.open("rb") as weights:
            raw_header_length = weights.read(8)
            if len(raw_header_length) != 8:
                raise _runtime_error(422, "invalid_lora_adapter")
            header_length = int.from_bytes(raw_header_length, "little")
            if (
                header_length <= 2
                or header_length > _MAX_SAFETENSORS_HEADER_BYTES
                or header_length > file_size - 8
            ):
                raise _runtime_error(422, "invalid_lora_adapter")
            header = json.loads(weights.read(header_length))
    except HttpError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _runtime_error(422, "invalid_lora_adapter") from None

    if not isinstance(header, dict):
        raise _runtime_error(422, "invalid_lora_adapter")
    payload_size = file_size - 8 - header_length
    ranges: list[tuple[int, int]] = []
    tensor_names: set[str] = set()
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise _runtime_error(422, "invalid_lora_adapter")
            continue
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, dict)
            or tensor.get("dtype") not in _SAFETENSORS_DTYPE_BYTES
        ):
            raise _runtime_error(422, "invalid_lora_adapter")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or len(shape) > 8
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            )
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in offsets
            )
        ):
            raise _runtime_error(422, "invalid_lora_adapter")
        start, end = offsets
        expected_bytes = prod(shape) * _SAFETENSORS_DTYPE_BYTES[tensor["dtype"]]
        if (
            start < 0
            or end < start
            or end > payload_size
            or end - start != expected_bytes
        ):
            raise _runtime_error(422, "invalid_lora_adapter")
        ranges.append((start, end))
        tensor_names.add(name)

    if not tensor_names:
        raise _runtime_error(422, "invalid_lora_adapter")
    ranges.sort()
    if any(previous[1] > current[0] for previous, current in pairwise(ranges)):
        raise _runtime_error(422, "invalid_lora_adapter")
    return frozenset(tensor_names)


def validate_snapshot(
    snapshot: Path,
    cache_root: Path,
    max_snapshot_bytes: int,
    max_lora_rank: int | None,
    compatible_base_names: Iterable[str | None],
) -> Path:
    raw_snapshot = Path(snapshot)
    if raw_snapshot.is_symlink():
        raise _runtime_error(422, "invalid_lora_adapter")
    try:
        canonical_root = Path(cache_root).resolve(strict=True)
        canonical_snapshot = raw_snapshot.resolve(strict=True)
    except OSError:
        raise _runtime_error(422, "invalid_lora_adapter") from None
    if not canonical_snapshot.is_dir() or not canonical_snapshot.is_relative_to(
        canonical_root
    ):
        raise _runtime_error(422, "invalid_lora_adapter")

    total_bytes = 0
    file_count = 0
    for root, directories, files in os.walk(canonical_snapshot, followlinks=False):
        for name in [*directories, *files]:
            path = Path(root) / name
            metadata = path.lstat()
            mode = metadata.st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise _runtime_error(422, "invalid_lora_adapter")
            relative_path = path.relative_to(canonical_snapshot)
            if stat.S_ISDIR(mode) or (
                relative_path.parent != Path(".")
                or relative_path.name not in _ALLOWED_SNAPSHOT_FILES
            ):
                raise _runtime_error(422, "invalid_lora_adapter")
            if stat.S_ISREG(mode):
                if metadata.st_nlink != 1:
                    raise _runtime_error(422, "invalid_lora_adapter")
                file_count += 1
                total_bytes += metadata.st_size
                if file_count > _MAX_FILES or total_bytes > max_snapshot_bytes:
                    raise _runtime_error(422, "invalid_lora_adapter")

    config_path = canonical_snapshot / "adapter_config.json"
    weights_path = canonical_snapshot / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise _runtime_error(422, "invalid_lora_adapter")
    try:
        adapter_config = json.loads(config_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _runtime_error(422, "invalid_lora_adapter") from None
    if not isinstance(adapter_config, dict):
        raise _runtime_error(422, "invalid_lora_adapter")
    rank = adapter_config.get("r")
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank <= 0
        or (max_lora_rank is not None and rank > max_lora_rank)
    ):
        raise _runtime_error(422, "invalid_lora_adapter")

    compatible_bases = {
        name for name in compatible_base_names if isinstance(name, str) and name
    }
    if adapter_config.get("base_model_name_or_path") not in compatible_bases:
        raise _runtime_error(422, "invalid_lora_adapter")

    raw_targets = adapter_config.get("target_modules")
    if isinstance(raw_targets, str):
        if (
            not raw_targets
            or len(raw_targets.encode("utf-8")) > 256
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in raw_targets)
        ):
            raise _runtime_error(422, "invalid_lora_adapter")
        targets = None
    elif isinstance(raw_targets, list):
        targets = raw_targets
        if not targets or any(
            not isinstance(target, str)
            or not target
            or len(target.encode("utf-8")) > 256
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in target)
            for target in targets
        ):
            raise _runtime_error(422, "invalid_lora_adapter")
    else:
        raise _runtime_error(422, "invalid_lora_adapter")
    tensor_names = _validate_safetensors(weights_path)
    embeddings_path = canonical_snapshot / "new_embeddings.safetensors"
    if embeddings_path.exists():
        _validate_safetensors(embeddings_path)
    if targets is not None and any(
        not any(target in name for name in tensor_names) for target in targets
    ):
        raise _runtime_error(422, "invalid_lora_adapter")
    return canonical_snapshot
