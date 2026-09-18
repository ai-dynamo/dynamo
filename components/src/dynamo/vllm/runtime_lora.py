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
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

from vllm.lora.request import LoRARequest

from dynamo.common.lora.manager import LoRAInfo, get_lora_manager
from dynamo.common.lora.runtime import (
    ResolveContext,
    RuntimeLoRAConfigurationError,
    RuntimeLoRANotFoundError,
    RuntimeLoRAPluginError,
    RuntimeLoRAResolverUnavailableError,
)
from dynamo.common.utils.env import env_bool
from dynamo.llm import HttpError, ModelRuntimeConfig, WorkerType, lora_name_to_id

from .constants import DisaggregationMode
from .lora_lifecycle import run_lora_mutation
from .lora_state import RuntimeLoRAInfo
from .runtime_lora_validation import (
    cache_reservation,
    check_cache_limit,
    validate_snapshot,
)

_IDENTITY_DOMAIN = b"dynamo-runtime-lora-v1\0"
_RUNTIME_KEY_PREFIX = "dyn-lora-"
_RUNTIME_PROTOCOL_VERSION = 2
_MAX_BASE_BYTES = 512
_MAX_SOURCE_BYTES = 3072
_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*$")
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


@dataclass(frozen=True)
class RuntimeLoRASettings:
    max_registered_loras: int
    max_resident_runtime_loras: int
    max_download_bytes: int
    max_cache_bytes: int
    resolve_timeout_seconds: int
    max_concurrent_resolutions: int
    max_pending_runtime_keys: int
    max_pending_admissions: int
    pending_admission_timeout_seconds: int

    @classmethod
    def from_engine_args(cls, engine_args: Any) -> RuntimeLoRASettings:
        try:
            max_loras = int(getattr(engine_args, "max_loras", 4))
            max_cpu_loras = getattr(engine_args, "max_cpu_loras", None)
            max_registered = int(max_cpu_loras or max_loras)
            if max_loras <= 0 or max_registered <= 0:
                raise ValueError("vLLM LoRA capacity must be positive")
            max_resident = _positive_int(
                "DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", max_registered
            )
            max_download = _positive_int(
                "DYN_LORA_MAX_DOWNLOAD_BYTES", 10 * 1024 * 1024 * 1024
            )
            max_cache = _positive_int(
                "DYN_LORA_MAX_CACHE_BYTES", 100 * 1024 * 1024 * 1024
            )
            resolve_timeout = _positive_int("DYN_LORA_RESOLVE_TIMEOUT_SECONDS", 300)
            max_concurrent = _positive_int("DYN_LORA_MAX_CONCURRENT_RESOLUTIONS", 2)
            max_pending = _positive_int("DYN_LORA_MAX_PENDING_RUNTIME_KEYS", 64)
            max_pending_admissions = _positive_int(
                "DYN_LORA_MAX_PENDING_ADMISSIONS",
                int(getattr(engine_args, "max_num_seqs", 256)),
            )
            pending_admission_timeout = _positive_int(
                "DYN_LORA_PENDING_ADMISSION_TIMEOUT_SECONDS", 30
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeLoRAConfigurationError(str(exc)) from exc
        if max_resident > max_registered:
            raise RuntimeLoRAConfigurationError(
                "DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS cannot exceed vLLM max_cpu_loras"
            )
        if max_download > max_cache:
            raise RuntimeLoRAConfigurationError(
                "DYN_LORA_MAX_DOWNLOAD_BYTES cannot exceed DYN_LORA_MAX_CACHE_BYTES"
            )
        return cls(
            max_registered_loras=max_registered,
            max_resident_runtime_loras=max_resident,
            max_download_bytes=max_download,
            max_cache_bytes=max_cache,
            resolve_timeout_seconds=resolve_timeout,
            max_concurrent_resolutions=max_concurrent,
            max_pending_runtime_keys=max_pending,
            max_pending_admissions=max_pending_admissions,
            pending_admission_timeout_seconds=pending_admission_timeout,
        )


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
    ):
        raise _runtime_error(400, "invalid_lora_model_id")

    scheme, separator, remainder = source_uri.partition(":")
    scheme = scheme.lower()
    if not separator or not remainder or not _SCHEME_PATTERN.fullmatch(scheme):
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
    if not env_bool("DYN_LORA_RUNTIME_LOAD_ENABLED"):
        return
    if not env_bool("DYN_LORA_ENABLED") or not bool(
        getattr(handler_config.engine_args, "enable_lora", False)
    ):
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA loading requires vLLM LoRA support"
        )
    if worker_type != WorkerType.Aggregated:
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA protocol version 2 supports aggregated workers only"
        )
    if bool(getattr(handler_config, "use_vllm_tokenizer", False)):
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA protocol version 2 requires Dynamo's tokenized request path"
        )

    manager = get_lora_manager(configure_runtime=True)
    if manager is None or not manager.runtime_lora_schemes:
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA resolver did not advertise any allowed schemes"
        )
    settings = RuntimeLoRASettings.from_engine_args(handler_config.engine_args)
    handler_config.runtime_lora_settings = settings
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
        self._runtime_enabled = env_bool("DYN_LORA_ENABLED") and env_bool(
            "DYN_LORA_RUNTIME_LOAD_ENABLED"
        )
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._reconciliation_lock = asyncio.Lock()
        self._settings = (
            (
                getattr(handler.config, "runtime_lora_settings", None)
                or RuntimeLoRASettings.from_engine_args(handler.engine_args)
            )
            if self._runtime_enabled
            else None
        )

    @property
    def _active_settings(self) -> RuntimeLoRASettings:
        if self._settings is None:
            raise _runtime_error(400, "runtime_lora_unsupported")
        return self._settings

    @property
    def settings(self) -> RuntimeLoRASettings | None:
        return self._settings

    @property
    def enabled(self) -> bool:
        return self._runtime_enabled

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
                and (
                    routed_name in self._handler._lora_state.runtime_loras
                    or routed_name in self._handler._lora_state.runtime_load_digests
                    or routed_name not in self._handler._lora_state.loaded_loras
                )
            ):
                raise _runtime_error(400, "invalid_lora_model_id")
            return None
        if self._closed:
            raise _runtime_error(503, "lora_resolver_unavailable")
        self._validate_worker_capability(metadata)

        resident = await self._lease_resident(metadata, request_id)
        if resident is not None:
            return resident
        if self._closed:
            raise _runtime_error(503, "lora_resolver_unavailable")

        state = self._handler._lora_state
        task = state.runtime_load_tasks.get(metadata.adapter_key)
        if task is None:
            if (
                len(state.runtime_load_tasks)
                >= self._active_settings.max_pending_runtime_keys
            ):
                raise _runtime_error(429, "lora_capacity_exceeded")
            task = asyncio.create_task(
                self._resolve_validate_and_load(metadata, request_id)
            )
            state.runtime_load_tasks[metadata.adapter_key] = task
            state.runtime_load_digests[metadata.adapter_key] = metadata.identity_digest
            task.add_done_callback(partial(self._cleanup_task, metadata.adapter_key))
        elif (
            state.runtime_load_digests.get(metadata.adapter_key)
            != metadata.identity_digest
        ):
            raise _runtime_error(500, "runtime_lora_identity_collision")

        await asyncio.shield(task)
        resident = await self._lease_resident(metadata, request_id)
        if resident is None:
            raise _runtime_error(429, "lora_capacity_exceeded")
        return resident

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

    async def close(self) -> None:
        """Drain worker-owned resolver tasks before the engine shuts down."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_impl())
        cancelled, _result, error = await run_lora_mutation(self._close_task)
        if error is not None:
            raise error
        if cancelled:
            raise asyncio.CancelledError

    async def _close_impl(self) -> None:
        state = self._handler._lora_state
        for _adapter_key, timer in state.runtime_pending_admissions.values():
            timer.cancel()

        expiry_tasks = tuple(state.runtime_expiry_tasks)
        load_tasks = tuple(set(state.runtime_load_tasks.values()))
        for task in (*expiry_tasks, *load_tasks):
            task.cancel()
        if expiry_tasks or load_tasks:
            await asyncio.gather(*expiry_tasks, *load_tasks, return_exceptions=True)

        async with self._handler._lora_capacity_guard:
            for _adapter_key, timer in state.runtime_pending_admissions.values():
                timer.cancel()
            state.runtime_pending_admissions.clear()
            state.runtime_pending_leases.clear()
            state.runtime_expiry_tasks.difference_update(expiry_tasks)
            for adapter_key, task in tuple(state.runtime_load_tasks.items()):
                if task.done():
                    state.runtime_load_tasks.pop(adapter_key, None)
                    state.runtime_load_digests.pop(adapter_key, None)
            if state.runtime_active_leases:
                logger.warning(
                    "Runtime LoRA coordinator closed with %d active adapter leases",
                    sum(state.runtime_active_leases.values()),
                )

    def _validate_worker_capability(self, metadata: _RuntimeRequest) -> None:
        if not self._runtime_enabled:
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

        manager = get_lora_manager(configure_runtime=True)
        if manager is None or metadata.scheme not in manager.runtime_lora_schemes:
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
        self._handler._lora_state.runtime_loras.move_to_end(metadata.adapter_key)
        return _lora_request(resident)

    async def _lease_resident(
        self,
        metadata: _RuntimeRequest,
        request_id: str,
    ) -> LoRARequest | None:
        state = self._handler._lora_state
        while True:
            async with self._handler._lora_capacity_guard:
                if self._closed:
                    raise _runtime_error(503, "lora_resolver_unavailable")
                transition = state.runtime_eviction_events.get(metadata.adapter_key)
                if transition is None:
                    resident = state.runtime_loras.get(metadata.adapter_key)
                    if resident is None:
                        return None
                    request = self._validate_resident(metadata, resident)
                    self._record_pending_admission_locked(
                        metadata.adapter_key, request_id
                    )
                    return request
            await transition.wait()

    def _record_pending_admission_locked(
        self,
        adapter_key: str,
        request_id: str,
    ) -> None:
        state = self._handler._lora_state
        existing = state.runtime_pending_admissions.get(request_id)
        if existing is not None:
            existing_key, _timer = existing
            if existing_key != adapter_key:
                raise _runtime_error(500, "runtime_lora_identity_collision")
            return
        if (
            len(state.runtime_pending_admissions)
            >= self._active_settings.max_pending_admissions
        ):
            raise _runtime_error(429, "lora_capacity_exceeded")

        def expire() -> None:
            if self._closed:
                return
            task = asyncio.create_task(self.release_pending_admission(request_id))
            state.runtime_expiry_tasks.add(task)
            task.add_done_callback(state.runtime_expiry_tasks.discard)

        timer = asyncio.get_running_loop().call_later(
            self._active_settings.pending_admission_timeout_seconds,
            expire,
        )
        state.runtime_pending_admissions[request_id] = (
            adapter_key,
            timer,
        )
        pending = state.runtime_pending_leases
        pending[adapter_key] = pending.get(adapter_key, 0) + 1

    async def release_pending_admission(self, request_id: str) -> None:
        state = self._handler._lora_state
        async with self._handler._lora_capacity_guard:
            pending = state.runtime_pending_admissions.pop(request_id, None)
            if pending is None:
                return
            adapter_key, timer = pending
            timer.cancel()
            lease_count = state.runtime_pending_leases.get(adapter_key, 0)
            if lease_count <= 1:
                state.runtime_pending_leases.pop(adapter_key, None)
            else:
                state.runtime_pending_leases[adapter_key] = lease_count - 1

    @asynccontextmanager
    async def pending_admission_guard(self, request_id: str):
        """Release an unactivated pending admission on every request exit path."""
        try:
            yield
        finally:
            cancelled, _released, error = await run_lora_mutation(
                self.release_pending_admission(request_id)
            )
            if cancelled:
                raise asyncio.CancelledError
            if error is not None:
                raise error

    async def activate_pending_admission(
        self,
        request_id: str,
        lora_request: LoRARequest,
    ) -> LoRARequest:
        state = self._handler._lora_state
        async with self._handler._lora_capacity_guard:
            pending = state.runtime_pending_admissions.pop(request_id, None)
            if pending is None:
                raise _runtime_error(429, "lora_capacity_exceeded")
            adapter_key, timer = pending
            timer.cancel()
            pending_count = state.runtime_pending_leases.get(adapter_key, 0)
            if pending_count <= 1:
                state.runtime_pending_leases.pop(adapter_key, None)
            else:
                state.runtime_pending_leases[adapter_key] = pending_count - 1
            if adapter_key != lora_request.lora_name:
                raise _runtime_error(500, "runtime_lora_identity_collision")

            resident = state.runtime_loras.get(adapter_key)
            loaded = state.loaded_loras.get(adapter_key)
            if (
                resident is None
                or loaded is None
                or loaded.id != resident.id
                or loaded.path != resident.path
                or lora_request.lora_int_id != resident.id
                or lora_request.lora_path != resident.path
            ):
                raise _runtime_error(429, "lora_capacity_exceeded")
            active = state.runtime_active_leases
            active[adapter_key] = active.get(adapter_key, 0) + 1
            state.runtime_loras.move_to_end(adapter_key)
            return _lora_request(resident)

    async def release_active_admission(self, adapter_key: str) -> None:
        state = self._handler._lora_state
        async with self._handler._lora_capacity_guard:
            lease_count = state.runtime_active_leases.get(adapter_key, 0)
            if lease_count <= 1:
                state.runtime_active_leases.pop(adapter_key, None)
            else:
                state.runtime_active_leases[adapter_key] = lease_count - 1

    async def _resolve_validate_and_load(
        self,
        metadata: _RuntimeRequest,
        request_id: str,
    ) -> RuntimeLoRAInfo:
        manager = get_lora_manager(configure_runtime=True)
        if manager is None:
            raise _runtime_error(400, "runtime_lora_unsupported")
        state = self._handler._lora_state
        if state.runtime_resolution_semaphore is None:
            state.runtime_resolution_semaphore = asyncio.Semaphore(
                self._active_settings.max_concurrent_resolutions
            )

        try:
            await asyncio.wait_for(
                state.runtime_resolution_semaphore.acquire(),
                timeout=self._active_settings.resolve_timeout_seconds,
            )
        except TimeoutError:
            raise _runtime_error(429, "lora_capacity_exceeded") from None

        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._active_settings.resolve_timeout_seconds
            async with cache_reservation(
                state, manager.cache_root, self._active_settings
            ) as reservation:
                context = ResolveContext(
                    adapter_key=metadata.adapter_key,
                    base_model_name=metadata.base_model_name,
                    cache_root=manager.cache_root,
                    deadline_monotonic=deadline,
                    max_download_bytes=reservation,
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

                local_path = await asyncio.to_thread(
                    validate_snapshot,
                    resolved.local_path,
                    manager.cache_root,
                    self._active_settings.max_download_bytes,
                    getattr(self._handler.engine_args, "max_lora_rank", None),
                    (
                        metadata.base_model_name,
                        getattr(self._handler, "_served_model_name", None),
                        getattr(self._handler.engine_args, "model", None),
                        *getattr(self._handler, "_served_model_aliases", ()),
                    ),
                )
                await check_cache_limit(
                    state,
                    manager.cache_root,
                    self._active_settings.max_cache_bytes,
                )
        finally:
            state.runtime_resolution_semaphore.release()
        return await self._load_snapshot(
            metadata,
            str(local_path),
            resolved.source_revision,
        )

    def _reserve_runtime_lora_eviction(self, exclude_key: str) -> RuntimeLoRAInfo:
        state = self._handler._lora_state
        victim = next(
            (
                record
                for key, record in state.runtime_loras.items()
                if key != exclude_key
                and state.runtime_pending_leases.get(key, 0) == 0
                and state.runtime_active_leases.get(key, 0) == 0
            ),
            None,
        )
        if victim is None:
            raise _runtime_error(429, "lora_capacity_exceeded")

        if victim.adapter_key in state.runtime_eviction_events:
            raise _runtime_error(500, "runtime_lora_identity_collision")
        state.runtime_eviction_events[victim.adapter_key] = asyncio.Event()
        state.runtime_loras.pop(victim.adapter_key, None)
        state.loaded_loras.pop(victim.adapter_key, None)
        state.rollback_reserved_ids.add(victim.id)
        return victim

    def _finish_eviction_transitions_locked(
        self, victims: list[RuntimeLoRAInfo]
    ) -> None:
        state = self._handler._lora_state
        for victim in victims:
            transition = state.runtime_eviction_events.pop(victim.adapter_key, None)
            if transition is not None:
                transition.set()

    async def _finish_runtime_rollback(
        self,
        adapter_key: str,
        victims: list[RuntimeLoRAInfo],
        removed_victim_ids: set[int],
        new_lora_id: int | None = None,
    ) -> bool:
        cancelled, _result, error = await run_lora_mutation(
            self._rollback_runtime_load(
                adapter_key,
                victims,
                removed_victim_ids,
                new_lora_id,
            )
        )
        if error is not None:
            raise error
        return cancelled

    def _is_absent_engine_error(self, error: BaseException) -> bool:
        return isinstance(error, Exception) and getattr(
            self._handler,
            "_is_lora_not_loaded_error",
            lambda _error: False,
        )(error)

    async def _engine_lora_ids(self) -> set[int] | None:
        try:
            lora_ids = await self._handler.engine_client.list_loras()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - engine RPC boundary
            logger.warning(
                "Could not reconcile vLLM LoRA residency (type=%s)",
                type(exc).__name__,
            )
            return None
        if not isinstance(lora_ids, set) or any(
            isinstance(lora_id, bool) or not isinstance(lora_id, int) or lora_id <= 0
            for lora_id in lora_ids
        ):
            logger.warning("vLLM returned an invalid LoRA residency set")
            return None
        return lora_ids

    async def _engine_contains_lora(self, lora_id: int) -> bool:
        lora_ids = await self._engine_lora_ids()
        return lora_ids is not None and lora_id in lora_ids

    async def _reconcile_uncertain_engine_loras(self) -> None:
        """Bound one reconciliation pass to the currently uncertain engine IDs."""
        state = self._handler._lora_state
        async with self._reconciliation_lock:
            async with self._handler._lora_capacity_guard:
                uncertain = frozenset(state.uncertain_engine_lora_ids)
            if not uncertain:
                return

            engine_ids = await self._engine_lora_ids()
            if engine_ids is None:
                return
            cancellation_requested = False
            for lora_id in sorted(uncertain & engine_ids):
                cancelled, _removed, _error = await run_lora_mutation(
                    self._handler.engine_client.remove_lora(lora_id)
                )
                cancellation_requested = cancellation_requested or cancelled

            if uncertain & engine_ids:
                final_engine_ids = await self._engine_lora_ids()
                if final_engine_ids is None:
                    if cancellation_requested:
                        raise asyncio.CancelledError
                    return
            else:
                final_engine_ids = engine_ids

            confirmed_absent = uncertain - final_engine_ids
            async with self._handler._lora_capacity_guard:
                state.uncertain_engine_lora_ids.difference_update(confirmed_absent)
                state.rollback_reserved_ids.difference_update(confirmed_absent)
            if cancellation_requested:
                raise asyncio.CancelledError

    async def _remove_evicted_runtime_loras(
        self, victims: list[RuntimeLoRAInfo]
    ) -> tuple[set[int], bool, BaseException | None]:
        removed_ids: set[int] = set()
        cancellation_requested = False
        for victim in victims:
            cancelled, _removed, error = await run_lora_mutation(
                self._handler.engine_client.remove_lora(victim.id)
            )
            cancellation_requested = cancellation_requested or cancelled
            removed_ids.add(victim.id)
            if error is not None and not self._is_absent_engine_error(error):
                # The engine may have applied the removal before the response
                # failed. Force rollback to re-add this victim instead of
                # assuming it is still resident.
                logger.warning(
                    "vLLM could not evict runtime LoRA adapter %s (type=%s)",
                    victim.adapter_key,
                    type(error).__name__,
                )
                return removed_ids, cancellation_requested, error
            if cancellation_requested:
                break
        return removed_ids, cancellation_requested, None

    async def _rollback_runtime_load(
        self,
        adapter_key: str,
        victims: list[RuntimeLoRAInfo],
        removed_victim_ids: set[int],
        new_lora_id: int | None = None,
    ) -> None:
        state = self._handler._lora_state
        new_lora_uncertain = False
        restored_ids = {
            victim.id for victim in victims if victim.id not in removed_victim_ids
        }
        failed_restore_ids: set[int] = set()

        if new_lora_id is not None:
            _cancelled, _removed, error = await run_lora_mutation(
                self._handler.engine_client.remove_lora(new_lora_id)
            )
            if error is not None and not self._is_absent_engine_error(error):
                new_lora_uncertain = True
                logger.error(
                    "vLLM could not roll back runtime LoRA adapter %s (type=%s)",
                    adapter_key,
                    type(error).__name__,
                )

        for victim in victims:
            if victim.id not in removed_victim_ids:
                continue
            _cancelled, restored, error = await run_lora_mutation(
                self._handler.engine_client.add_lora(_lora_request(victim))
            )
            if error is None and (
                restored is True
                or (restored is False and await self._engine_contains_lora(victim.id))
            ):
                restored_ids.add(victim.id)
            else:
                failed_restore_ids.add(victim.id)
                logger.error(
                    "vLLM could not restore runtime LoRA adapter %s (type=%s)",
                    victim.adapter_key,
                    type(error).__name__,
                )

        async with self._handler._lora_capacity_guard:
            if new_lora_id is not None:
                if new_lora_uncertain:
                    state.rollback_reserved_ids.add(new_lora_id)
                    state.uncertain_engine_lora_ids.add(new_lora_id)
                else:
                    state.rollback_reserved_ids.discard(new_lora_id)
                    state.uncertain_engine_lora_ids.discard(new_lora_id)
                self._handler._engine_loaded_loras.discard(adapter_key)
            state.runtime_reserved_ids.pop(adapter_key, None)
            current = state.loaded_loras.get(adapter_key)
            if current is not None and current.id == -1:
                state.loaded_loras.pop(adapter_key, None)
            for victim in victims:
                if victim.id in restored_ids:
                    state.loaded_loras[victim.adapter_key] = LoRAInfo(
                        id=victim.id, path=victim.path
                    )
                    state.runtime_loras[victim.adapter_key] = victim
                    state.rollback_reserved_ids.discard(victim.id)
                    state.uncertain_engine_lora_ids.discard(victim.id)
                    self._handler._engine_loaded_loras.add(victim.adapter_key)
                elif victim.id in failed_restore_ids:
                    state.rollback_reserved_ids.add(victim.id)
                    state.uncertain_engine_lora_ids.add(victim.id)
                    self._handler._engine_loaded_loras.discard(victim.adapter_key)
            self._finish_eviction_transitions_locked(victims)

    async def _load_snapshot(
        self,
        metadata: _RuntimeRequest,
        local_path: str,
        source_revision: str,
    ) -> RuntimeLoRAInfo:
        state = self._handler._lora_state
        lock = state.get_lock(metadata.adapter_key)
        async with lock:
            await self._reconcile_uncertain_engine_loras()
            resident = state.runtime_loras.get(metadata.adapter_key)
            if resident is not None:
                self._validate_resident(metadata, resident)
                return resident
            if metadata.adapter_key in state.loaded_loras:
                raise _runtime_error(500, "runtime_lora_identity_collision")

            capacity_reserved = False
            engine_added = False
            evicted: list[RuntimeLoRAInfo] = []
            removed_victim_ids: set[int] = set()
            try:
                async with self._handler._lora_capacity_guard:
                    capacity_reserved = True
                    while (
                        len(state.runtime_loras)
                        + len(state.runtime_reserved_ids)
                        + len(state.uncertain_engine_lora_ids)
                        >= self._active_settings.max_resident_runtime_loras
                        or len(state.loaded_loras)
                        + len(state.uncertain_engine_lora_ids)
                        >= self._active_settings.max_registered_loras
                    ):
                        evicted.append(
                            self._reserve_runtime_lora_eviction(metadata.adapter_key)
                        )
                    lora_id = self._allocate_lora_id(metadata.adapter_key)
                    state.runtime_reserved_ids[metadata.adapter_key] = lora_id
                    state.loaded_loras[metadata.adapter_key] = LoRAInfo(id=-1, path="")

                (
                    removed_victim_ids,
                    eviction_cancelled,
                    eviction_error,
                ) = await self._remove_evicted_runtime_loras(evicted)
                if eviction_cancelled or eviction_error is not None:
                    capacity_reserved = False
                    rollback_cancelled = await self._finish_runtime_rollback(
                        metadata.adapter_key,
                        evicted,
                        removed_victim_ids,
                    )
                    if eviction_cancelled or rollback_cancelled:
                        raise asyncio.CancelledError
                    raise _runtime_error(429, "lora_capacity_exceeded") from None

                request = LoRARequest(
                    lora_name=metadata.adapter_key,
                    lora_int_id=lora_id,
                    lora_path=local_path,
                )
                add_cancelled, added, add_error = await run_lora_mutation(
                    self._handler.engine_client.add_lora(request)
                )
                engine_added = add_error is None and (
                    added is True
                    or (added is False and await self._engine_contains_lora(lora_id))
                )
                if add_cancelled or not engine_added:
                    capacity_reserved = False
                    rollback_cancelled = await self._finish_runtime_rollback(
                        metadata.adapter_key,
                        evicted,
                        removed_victim_ids,
                        lora_id,
                    )
                    engine_added = False
                    if (
                        add_cancelled
                        or rollback_cancelled
                        or isinstance(add_error, asyncio.CancelledError)
                    ):
                        raise asyncio.CancelledError
                    logger.warning(
                        "vLLM rejected runtime LoRA adapter %s (type=%s)",
                        metadata.adapter_key,
                        type(add_error).__name__ if add_error is not None else "false",
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
                async with self._handler._lora_capacity_guard:
                    state.loaded_loras[metadata.adapter_key] = LoRAInfo(
                        id=lora_id,
                        path=local_path,
                    )
                    state.runtime_loras[metadata.adapter_key] = record
                    state.runtime_loras.move_to_end(metadata.adapter_key)
                    state.runtime_reserved_ids.pop(metadata.adapter_key, None)
                    for victim in evicted:
                        state.rollback_reserved_ids.discard(victim.id)
                        self._handler._engine_loaded_loras.discard(victim.adapter_key)
                    self._finish_eviction_transitions_locked(evicted)
                    self._handler._engine_loaded_loras.add(metadata.adapter_key)
                    capacity_reserved = False
                    engine_added = False
                    return record
            finally:
                if capacity_reserved:
                    capacity_reserved = False
                    rollback_cancelled = await self._finish_runtime_rollback(
                        metadata.adapter_key,
                        evicted,
                        removed_victim_ids,
                        lora_id if engine_added else None,
                    )
                    if rollback_cancelled:
                        raise asyncio.CancelledError

    def _allocate_lora_id(self, adapter_key: str) -> int:
        try:
            return self._handler._lora_state.allocate_lora_id(
                adapter_key,
                lora_name_to_id(adapter_key),
            )
        except ValueError as exc:
            raise _runtime_error(500, "runtime_lora_identity_collision") from exc
