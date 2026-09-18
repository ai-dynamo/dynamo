# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated-vLLM request-time LoRA admission and load-on-miss."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import prod
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from dynamo.common.lora.manager import LoRAInfo, get_lora_manager
from dynamo.common.lora.runtime import (
    ResolveContext,
    RuntimeLoRAConfigurationError,
    RuntimeLoRANotFoundError,
    RuntimeLoRAPluginError,
    RuntimeLoRAResolverUnavailableError,
)
from dynamo.llm import HttpError, ModelRuntimeConfig, WorkerType, lora_name_to_id

from vllm.lora.request import LoRARequest

from .constants import DisaggregationMode
from .lora_state import RuntimeLoRAInfo

_IDENTITY_DOMAIN = b"dynamo-runtime-lora-v1\0"
_RUNTIME_KEY_PREFIX = "dyn-lora-"
_RUNTIME_PROTOCOL_VERSION = 2
_MAX_BASE_BYTES = 512
_MAX_SOURCE_BYTES = 3072
_MAX_FILES = 4096
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
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
_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*$")
_TRUE_VALUES = {"1", "true", "yes", "on"}
_SECRET_QUERY_TERMS = (
    "access_key",
    "api_key",
    "credential",
    "password",
    "secret",
    "signature",
    "token",
)

logger = logging.getLogger(__name__)


def _runtime_error(status: int, code: str) -> HttpError:
    return HttpError(status, code)


@dataclass(frozen=True)
class _RuntimeRequest:
    adapter_key: str
    base_model_name: str
    source_uri: str
    scheme: str
    identity_digest: bytes
    protocol_version: int


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"runtime_lora_unsupported: invalid {name}") from exc
    if value <= 0:
        raise ValueError(f"runtime_lora_unsupported: invalid {name}")
    return value


def _identity_digest(base_model_name: str, source_uri: str) -> bytes:
    return hashlib.sha256(
        _IDENTITY_DOMAIN
        + base_model_name.encode("utf-8")
        + b"\0"
        + source_uri.encode("utf-8")
    ).digest()


def _adapter_key(digest: bytes) -> str:
    return f"{_RUNTIME_KEY_PREFIX}{digest[:16].hex()}"


def _validate_source_uri(source_uri: str) -> str:
    try:
        encoded = source_uri.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _runtime_error(400, "invalid_lora_model_id") from exc
    if (
        not source_uri
        or len(encoded) > _MAX_SOURCE_BYTES
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in source_uri)
        or "://" not in source_uri
    ):
        raise _runtime_error(400, "invalid_lora_model_id")

    parsed = urlsplit(source_uri)
    scheme = parsed.scheme.lower()
    if (
        not _SCHEME_PATTERN.fullmatch(scheme)
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise _runtime_error(400, "invalid_lora_model_id")
    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized = key.lower()
        if any(term in normalized for term in _SECRET_QUERY_TERMS):
            raise _runtime_error(400, "invalid_lora_model_id")
    return scheme


def _parse_request(request: Mapping[str, Any]) -> _RuntimeRequest | None:
    routing = request.get("routing")
    if not isinstance(routing, Mapping):
        return None

    adapter_key = routing.get("lora_name")
    base_model_name = routing.get("base_model_name")
    source_uri = routing.get("lora_source_uri")
    protocol_version = routing.get("lora_resolution_version")
    runtime_values = (base_model_name, source_uri, protocol_version)
    if all(value is None for value in runtime_values):
        return None
    has_runtime_key = isinstance(adapter_key, str) and adapter_key.startswith(
        _RUNTIME_KEY_PREFIX
    )
    if (
        not has_runtime_key
        or not isinstance(base_model_name, str)
        or not isinstance(source_uri, str)
        or isinstance(protocol_version, bool)
        or protocol_version != _RUNTIME_PROTOCOL_VERSION
    ):
        raise _runtime_error(400, "invalid_lora_model_id")
    if (
        not base_model_name
        or len(base_model_name.encode("utf-8")) > _MAX_BASE_BYTES
        or request.get("model") != base_model_name
    ):
        raise _runtime_error(400, "invalid_lora_model_id")

    scheme = _validate_source_uri(source_uri)
    digest = _identity_digest(base_model_name, source_uri)
    if _adapter_key(digest) != adapter_key:
        raise _runtime_error(400, "runtime_lora_identity_mismatch")
    return _RuntimeRequest(
        adapter_key=adapter_key,
        base_model_name=base_model_name,
        source_uri=source_uri,
        scheme=scheme,
        identity_digest=digest,
        protocol_version=protocol_version,
    )


def _lora_request(record: RuntimeLoRAInfo) -> LoRARequest:
    return LoRARequest(
        lora_name=record.adapter_key,
        lora_int_id=record.id,
        lora_path=record.path,
    )


def publish_runtime_lora_capability(
    runtime_config: ModelRuntimeConfig,
    handler_config: Any,
    worker_type: WorkerType,
) -> None:
    """Publish capability only after the configured resolver validates."""
    if not _env_bool("DYN_LORA_RUNTIME_LOAD_ENABLED"):
        return
    if not _env_bool("DYN_LORA_ENABLED") or not bool(
        getattr(handler_config.engine_args, "enable_lora", False)
    ):
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA loading requires vLLM LoRA support"
        )
    if worker_type != WorkerType.Aggregated:
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA protocol version 2 supports aggregated workers only"
        )

    manager = get_lora_manager()
    if manager is None or not manager.runtime_lora_schemes:
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA resolver did not advertise any allowed schemes"
        )
    try:
        max_loras = int(getattr(handler_config.engine_args, "max_loras", 4))
        if max_loras <= 0:
            raise ValueError("vLLM max_loras must be positive")
        max_resident = _positive_int(
            "DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS",
            max_loras,
        )
        max_download_bytes = _positive_int(
            "DYN_LORA_MAX_DOWNLOAD_BYTES",
            10 * 1024 * 1024 * 1024,
        )
        max_cache_bytes = _positive_int(
            "DYN_LORA_MAX_CACHE_BYTES",
            100 * 1024 * 1024 * 1024,
        )
        _positive_int("DYN_LORA_RESOLVE_TIMEOUT_SECONDS", 300)
        _positive_int("DYN_LORA_MAX_CONCURRENT_RESOLUTIONS", 2)
        _positive_int("DYN_LORA_MAX_PENDING_RUNTIME_KEYS", 64)
    except (TypeError, ValueError) as exc:
        raise RuntimeLoRAConfigurationError(str(exc)) from exc
    if max_resident > max_loras:
        raise RuntimeLoRAConfigurationError(
            "DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS cannot exceed vLLM max_loras"
        )
    if max_download_bytes > max_cache_bytes:
        raise RuntimeLoRAConfigurationError(
            "DYN_LORA_MAX_DOWNLOAD_BYTES cannot exceed DYN_LORA_MAX_CACHE_BYTES"
        )
    try:
        cache_root = manager.cache_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA cache root is unavailable"
        ) from exc
    if not cache_root.is_dir():
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA cache root must be a directory"
        )
    runtime_config.set_engine_specific("supports_runtime_lora_resolution", "true")
    runtime_config.set_engine_specific(
        "runtime_lora_protocol_versions",
        json.dumps([_RUNTIME_PROTOCOL_VERSION]),
    )
    runtime_config.set_engine_specific(
        "runtime_lora_schemes", json.dumps(sorted(manager.runtime_lora_schemes))
    )
    runtime_config.taints = set(runtime_config.taints) | {
        f"dynamo.runtime-lora/v{_RUNTIME_PROTOCOL_VERSION}"
    }


class RuntimeLoRACoordinator:
    """Coordinates private runtime resolver work for one vLLM handler."""

    def __init__(self, handler: Any):
        self._handler = handler

    async def ensure_from_request(
        self,
        request: Mapping[str, Any],
        request_id: str,
    ) -> LoRARequest | None:
        metadata = _parse_request(request)
        if metadata is None:
            routing = request.get("routing")
            routed_name = (
                routing.get("lora_name") if isinstance(routing, Mapping) else None
            )
            if (
                isinstance(routed_name, str)
                and routed_name.startswith(_RUNTIME_KEY_PREFIX)
                and routed_name not in self._handler._lora_state.loaded_loras
            ):
                raise _runtime_error(400, "invalid_lora_model_id")
            return None
        self._validate_worker_capability(metadata)

        resident = self._handler._lora_state.runtime_loras.get(metadata.adapter_key)
        if resident is not None:
            return self._validate_resident(metadata, resident)

        state = self._handler._lora_state
        task = state.runtime_load_tasks.get(metadata.adapter_key)
        if task is None:
            max_pending = _positive_int("DYN_LORA_MAX_PENDING_RUNTIME_KEYS", 64)
            if len(state.runtime_load_tasks) >= max_pending:
                raise _runtime_error(429, "lora_capacity_exceeded")
            task = asyncio.create_task(
                self._resolve_validate_and_load(metadata, request_id)
            )
            state.runtime_load_tasks[metadata.adapter_key] = task
            state.runtime_load_digests[metadata.adapter_key] = metadata.identity_digest
            task.add_done_callback(
                lambda done, key=metadata.adapter_key: self._cleanup_task(key, done)
            )
        elif (
            state.runtime_load_digests.get(metadata.adapter_key)
            != metadata.identity_digest
        ):
            raise _runtime_error(500, "runtime_lora_identity_collision")

        record = await asyncio.shield(task)
        return self._validate_resident(metadata, record)

    def _cleanup_task(
        self,
        adapter_key: str,
        task: asyncio.Task[RuntimeLoRAInfo],
    ) -> None:
        state = self._handler._lora_state
        if state.runtime_load_tasks.get(adapter_key) is task:
            state.runtime_load_tasks.pop(adapter_key, None)
            state.runtime_load_digests.pop(adapter_key, None)
        if not task.cancelled():
            task.exception()

    def _validate_worker_capability(self, metadata: _RuntimeRequest) -> None:
        if not _env_bool("DYN_LORA_ENABLED") or not _env_bool(
            "DYN_LORA_RUNTIME_LOAD_ENABLED"
        ):
            raise _runtime_error(400, "runtime_lora_unsupported")
        if self._handler.config.disaggregation_mode != DisaggregationMode.AGGREGATED:
            raise _runtime_error(400, "runtime_lora_unsupported")
        if not bool(getattr(self._handler.engine_args, "enable_lora", False)):
            raise _runtime_error(400, "runtime_lora_unsupported")

        served_names = {
            name
            for name in (
                getattr(self._handler, "_served_model_name", None),
                getattr(self._handler.engine_args, "model", None),
                *getattr(self._handler, "_served_model_aliases", ()),
            )
            if isinstance(name, str) and name
        }
        if metadata.base_model_name not in served_names:
            raise _runtime_error(400, "runtime_lora_identity_mismatch")

        allowed = {
            item.strip().lower()
            for item in os.environ.get("DYN_LORA_ALLOWED_SCHEMES", "").split(",")
            if item.strip()
        }
        manager = get_lora_manager()
        if (
            manager is None
            or metadata.scheme not in allowed
            or metadata.scheme not in manager.runtime_lora_schemes
        ):
            raise _runtime_error(400, "unsupported_lora_scheme")

    def _validate_resident(
        self,
        metadata: _RuntimeRequest,
        resident: RuntimeLoRAInfo,
    ) -> LoRARequest:
        loaded = self._handler._lora_state.loaded_loras.get(metadata.adapter_key)
        if (
            resident.full_identity_digest != metadata.identity_digest
            or resident.base_model_name != metadata.base_model_name
            or loaded is None
            or loaded.id != resident.id
            or loaded.path != resident.path
        ):
            raise _runtime_error(400, "runtime_lora_identity_mismatch")
        return _lora_request(resident)

    async def _resolve_validate_and_load(
        self,
        metadata: _RuntimeRequest,
        request_id: str,
    ) -> RuntimeLoRAInfo:
        manager = get_lora_manager()
        if manager is None:
            raise _runtime_error(400, "runtime_lora_unsupported")
        state = self._handler._lora_state
        if state.runtime_resolution_semaphore is None:
            state.runtime_resolution_semaphore = asyncio.Semaphore(
                _positive_int("DYN_LORA_MAX_CONCURRENT_RESOLUTIONS", 2)
            )
        if state.runtime_cache_guard is None:
            state.runtime_cache_guard = asyncio.Lock()

        timeout_seconds = _positive_int("DYN_LORA_RESOLVE_TIMEOUT_SECONDS", 300)
        max_download_bytes = _positive_int(
            "DYN_LORA_MAX_DOWNLOAD_BYTES", 10 * 1024 * 1024 * 1024
        )
        max_cache_bytes = _positive_int(
            "DYN_LORA_MAX_CACHE_BYTES", 100 * 1024 * 1024 * 1024
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        async with (
            state.runtime_resolution_semaphore,
            state.runtime_cache_guard,
        ):
            cache_bytes = self._cache_size(manager.cache_root, max_cache_bytes)
            if cache_bytes >= max_cache_bytes:
                raise _runtime_error(429, "lora_capacity_exceeded")
            context = ResolveContext(
                adapter_key=metadata.adapter_key,
                base_model_name=metadata.base_model_name,
                cache_root=manager.cache_root,
                deadline_monotonic=deadline,
                max_download_bytes=min(
                    max_download_bytes,
                    max_cache_bytes - cache_bytes,
                ),
                request_id=request_id,
            )
            try:
                resolved = await manager.resolve_runtime_lora(
                    source_uri=metadata.source_uri,
                    context=context,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                raise _runtime_error(504, "runtime_lora_resolve_timeout") from None
            except RuntimeLoRANotFoundError:
                raise _runtime_error(404, "lora_not_found") from None
            except RuntimeLoRAResolverUnavailableError:
                raise _runtime_error(503, "lora_resolver_unavailable") from None
            except RuntimeLoRAPluginError:
                raise _runtime_error(500, "lora_plugin_error") from None
            except RuntimeLoRAConfigurationError:
                raise _runtime_error(500, "lora_plugin_error") from None
            except Exception as exc:  # noqa: BLE001 - plugin trust boundary
                logger.warning(
                    "Runtime LoRA resolver failed for adapter %s (type=%s)",
                    metadata.adapter_key,
                    type(exc).__name__,
                )
                raise _runtime_error(500, "lora_plugin_error") from None

            local_path = self._validate_snapshot(
                resolved.local_path,
                manager.cache_root,
                max_download_bytes,
                metadata.base_model_name,
            )
            if self._cache_size(manager.cache_root, max_cache_bytes) > max_cache_bytes:
                raise _runtime_error(429, "lora_capacity_exceeded")
        return await self._load_snapshot(
            metadata,
            str(local_path),
            resolved.source_revision,
        )

    def _cache_size(self, cache_root: Path, stop_after: int) -> int:
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

    def _validate_safetensors(self, weights_path: Path) -> frozenset[str]:
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

    def _validate_snapshot(
        self,
        snapshot: Path,
        cache_root: Path,
        max_download_bytes: int,
        base_model_name: str,
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
                if stat.S_ISREG(mode):
                    if metadata.st_nlink != 1:
                        raise _runtime_error(422, "invalid_lora_adapter")
                    file_count += 1
                    total_bytes += metadata.st_size
                    if file_count > _MAX_FILES or total_bytes > max_download_bytes:
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
        max_rank = getattr(self._handler.engine_args, "max_lora_rank", None)
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank <= 0
            or (max_rank is not None and rank > max_rank)
        ):
            raise _runtime_error(422, "invalid_lora_adapter")

        compatible_bases = {
            name
            for name in (
                base_model_name,
                getattr(self._handler, "_served_model_name", None),
                getattr(self._handler.engine_args, "model", None),
                *getattr(self._handler, "_served_model_aliases", ()),
            )
            if isinstance(name, str) and name
        }
        if adapter_config.get("base_model_name_or_path") not in compatible_bases:
            raise _runtime_error(422, "invalid_lora_adapter")

        raw_targets = adapter_config.get("target_modules")
        targets = [raw_targets] if isinstance(raw_targets, str) else raw_targets
        if (
            not isinstance(targets, list)
            or not targets
            or any(
                not isinstance(target, str)
                or not target
                or len(target.encode("utf-8")) > 256
                or any(ord(char) < 0x20 or ord(char) == 0x7F for char in target)
                for target in targets
            )
        ):
            raise _runtime_error(422, "invalid_lora_adapter")
        tensor_names = self._validate_safetensors(weights_path)
        if any(not any(target in name for name in tensor_names) for target in targets):
            raise _runtime_error(422, "invalid_lora_adapter")
        return canonical_snapshot

    async def _load_snapshot(
        self,
        metadata: _RuntimeRequest,
        local_path: str,
        source_revision: str,
    ) -> RuntimeLoRAInfo:
        state = self._handler._lora_state
        lock = state.get_lock(metadata.adapter_key)
        async with lock:
            resident = state.runtime_loras.get(metadata.adapter_key)
            if resident is not None:
                self._validate_resident(metadata, resident)
                return resident
            if metadata.adapter_key in state.loaded_loras:
                raise _runtime_error(500, "runtime_lora_identity_collision")

            capacity_reserved = False
            try:
                async with self._handler._lora_capacity_guard:
                    max_runtime = _positive_int(
                        "DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS",
                        getattr(self._handler, "_lora_capacity", 4),
                    )
                    if len(state.runtime_loras) >= max_runtime:
                        raise _runtime_error(429, "lora_capacity_exceeded")
                    capacity = getattr(self._handler, "_lora_capacity", None)
                    if capacity is not None and len(state.loaded_loras) >= capacity:
                        raise _runtime_error(429, "lora_capacity_exceeded")
                    lora_id = self._allocate_lora_id(metadata.adapter_key)
                    state.runtime_reserved_ids[metadata.adapter_key] = lora_id
                    state.loaded_loras[metadata.adapter_key] = LoRAInfo(id=-1, path="")
                    capacity_reserved = True

                request = LoRARequest(
                    lora_name=metadata.adapter_key,
                    lora_int_id=lora_id,
                    lora_path=local_path,
                )
                try:
                    await self._handler.engine_client.add_lora(request)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - engine trust boundary
                    logger.warning(
                        "vLLM rejected runtime LoRA adapter %s (type=%s)",
                        metadata.adapter_key,
                        type(exc).__name__,
                    )
                    raise _runtime_error(500, "lora_load_failed") from None
                record = RuntimeLoRAInfo(
                    adapter_key=metadata.adapter_key,
                    full_identity_digest=metadata.identity_digest,
                    base_model_name=metadata.base_model_name,
                    source_revision=source_revision,
                    id=lora_id,
                    path=local_path,
                )
                state.loaded_loras[metadata.adapter_key] = LoRAInfo(
                    id=lora_id,
                    path=local_path,
                )
                state.runtime_loras[metadata.adapter_key] = record
                self._handler._engine_loaded_loras.add(metadata.adapter_key)
                return record
            finally:
                if capacity_reserved:
                    state.runtime_reserved_ids.pop(metadata.adapter_key, None)
                    current = state.loaded_loras.get(metadata.adapter_key)
                    if current is not None and current.id == -1:
                        state.loaded_loras.pop(metadata.adapter_key, None)

    def _allocate_lora_id(self, adapter_key: str) -> int:
        used = {
            info.id: name
            for name, info in self._handler._lora_state.loaded_loras.items()
            if info.id > 0
        }
        used.update(
            {
                lora_id: name
                for name, lora_id in self._handler._lora_state.runtime_reserved_ids.items()
            }
        )
        candidate = max(1, int(lora_name_to_id(adapter_key)))
        for _ in range(len(used) + 1):
            owner = used.get(candidate)
            if owner is None or owner == adapter_key:
                return candidate
            candidate = candidate % 0x7FFFFFFF + 1
        raise _runtime_error(500, "runtime_lora_identity_collision")
