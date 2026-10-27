# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact validation for request-time vLLM LoRAs."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NoReturn

from dynamo.llm import HttpError

_MAX_FILES = 4096
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_MAX_USIZE = 2 * sys.maxsize + 1
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


def _runtime_error(status: int, code: str) -> HttpError:
    return HttpError(status, code)


def _reject_json_constant(_value: str) -> NoReturn:
    raise _runtime_error(422, "invalid_lora_adapter")


def _parse_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise _runtime_error(422, "invalid_lora_adapter")
    return result


def _parse_json_int(value: str) -> int:
    _parse_json_float(value)
    return int(value)


def _parse_json(encoded: bytes) -> object:
    text = encoded.decode("utf-8")
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                raise _runtime_error(422, "invalid_lora_adapter")
        elif char in "]}":
            depth -= 1
    return json.loads(
        text,
        parse_constant=_reject_json_constant,
        parse_float=_parse_json_float,
        parse_int=_parse_json_int,
    )


def _scan_error(error: OSError) -> None:
    raise error


def _safetensors_byte_size(shape: list[int], dtype_bytes: int) -> int:
    elements = 1
    for dimension in shape:
        if dimension > _MAX_USIZE:
            raise _runtime_error(422, "invalid_lora_adapter")
        elements *= dimension
        if elements > _MAX_USIZE:
            raise _runtime_error(422, "invalid_lora_adapter")
    size = elements * dtype_bytes
    if size > _MAX_USIZE:
        raise _runtime_error(422, "invalid_lora_adapter")
    return size


def _validate_safetensors(
    weights_path: Path,
    max_lora_rank: int | None = None,
    num_experts: int | None = None,
) -> frozenset[str]:
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
            header = _parse_json(weights.read(header_length))
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
            if not isinstance(tensor, dict) or any(
                not isinstance(value, str) for value in tensor.values()
            ):
                raise _runtime_error(422, "invalid_lora_adapter")
            continue
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, dict)
            or not isinstance(tensor.get("dtype"), str)
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
        if max_lora_rank is not None:
            rank_axis = None
            if name.endswith((".lora_A.weight", ".lora_embedding_A")):
                rank_axis = -2
            elif name.endswith((".lora_B.weight", ".lora_embedding_B")):
                rank_axis = -1
            if rank_axis is not None:
                if len(shape) < 2:
                    raise _runtime_error(422, "invalid_lora_adapter")
                tensor_rank = shape[rank_axis]
                if (
                    num_experts is not None
                    and len(shape) == 2
                    and name.endswith(
                        (
                            ".experts.lora_A.weight",
                            ".experts.lora_B.weight",
                            ".experts.base_layer.lora_A.weight",
                            ".experts.base_layer.lora_B.weight",
                        )
                    )
                ):
                    # PEFT packs expert ranks along this axis. Only the base
                    # model's trusted expert count can disambiguate the rank.
                    tensor_rank, remainder = divmod(tensor_rank, num_experts)
                    if remainder:
                        raise _runtime_error(422, "invalid_lora_adapter")
                if tensor_rank > max_lora_rank:
                    raise _runtime_error(422, "invalid_lora_adapter")
        start, end = offsets
        expected_bytes = _safetensors_byte_size(
            shape, _SAFETENSORS_DTYPE_BYTES[tensor["dtype"]]
        )
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
    expected_start = 0
    for start, end in sorted(ranges):
        if start != expected_start:
            raise _runtime_error(422, "invalid_lora_adapter")
        expected_start = end
    if expected_start != payload_size:
        raise _runtime_error(422, "invalid_lora_adapter")
    return frozenset(tensor_names)


def validate_snapshot(
    snapshot: Path,
    cache_root: Path,
    max_snapshot_bytes: int,
    max_lora_rank: int | None,
    compatible_base_names: Iterable[str | None],
    *,
    num_experts: int | None = None,
) -> Path:
    try:
        if num_experts is not None and (
            isinstance(num_experts, bool)
            or not isinstance(num_experts, int)
            or num_experts <= 0
        ):
            raise _runtime_error(422, "invalid_lora_adapter")
        return _validate_snapshot(
            snapshot,
            cache_root,
            max_snapshot_bytes,
            max_lora_rank,
            compatible_base_names,
            num_experts,
        )
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise _runtime_error(422, "invalid_lora_adapter") from None


def _validate_snapshot(
    snapshot: Path,
    cache_root: Path,
    max_snapshot_bytes: int,
    max_lora_rank: int | None,
    compatible_base_names: Iterable[str | None],
    num_experts: int | None,
) -> Path:
    raw_snapshot = Path(snapshot)
    if raw_snapshot.is_symlink():
        raise _runtime_error(422, "invalid_lora_adapter")
    try:
        canonical_root = Path(cache_root).resolve(strict=True)
        canonical_snapshot = raw_snapshot.resolve(strict=True)
    except (OSError, RuntimeError):
        raise _runtime_error(422, "invalid_lora_adapter") from None
    if (
        not canonical_snapshot.is_dir()
        or canonical_snapshot == canonical_root
        or not canonical_snapshot.is_relative_to(canonical_root)
    ):
        raise _runtime_error(422, "invalid_lora_adapter")

    total_bytes = 0
    file_count = 0
    for root, directories, files in os.walk(
        canonical_snapshot, followlinks=False, onerror=_scan_error
    ):
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
        with config_path.open("rb") as config_file:
            encoded_config = config_file.read(_MAX_CONFIG_BYTES + 1)
        if len(encoded_config) > _MAX_CONFIG_BYTES:
            raise _runtime_error(422, "invalid_lora_adapter")
        adapter_config = _parse_json(encoded_config)
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

    rank_pattern = adapter_config.get("rank_pattern")
    if rank_pattern is not None and (
        not isinstance(rank_pattern, dict)
        or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or (max_lora_rank is not None and value > max_lora_rank)
            for key, value in rank_pattern.items()
        )
    ):
        raise _runtime_error(422, "invalid_lora_adapter")

    compatible_bases = {
        name for name in compatible_base_names if isinstance(name, str) and name
    }
    base_name = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_name, str) or base_name not in compatible_bases:
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
    tensor_names = _validate_safetensors(weights_path, max_lora_rank, num_experts)
    embeddings_path = canonical_snapshot / "new_embeddings.safetensors"
    if embeddings_path.exists():
        _validate_safetensors(embeddings_path)
    if targets is not None and any(
        not any(target in name for name in tensor_names) for target in targets
    ):
        raise _runtime_error(422, "invalid_lora_adapter")
    return canonical_snapshot
