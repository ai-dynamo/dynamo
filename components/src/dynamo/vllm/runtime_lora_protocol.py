# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker request contract, settings, and capability admission for runtime LoRA."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dynamo.common.lora.manager import get_lora_manager
from dynamo.common.lora.runtime import RuntimeLoRAConfigurationError
from dynamo.common.utils.env import env_bool
from dynamo.llm import HttpError, ModelRuntimeConfig, WorkerType

_IDENTITY_DOMAIN = b"dynamo-runtime-lora-v1\0"
_RUNTIME_KEY_PREFIX = "dyn-lora-"
_RUNTIME_PROTOCOL_VERSION = 2
_MAX_BASE_BYTES = 512
_MAX_SOURCE_BYTES = 3072
_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*$")


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
