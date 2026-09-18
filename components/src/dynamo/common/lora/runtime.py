# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral request-time LoRA resolver protocol."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

RUNTIME_LORA_PROTOCOL_VERSION = 2
_MAX_SOURCE_REVISION_BYTES = 512
_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*$")


class RuntimeLoRAError(RuntimeError):
    """Base error for request-time LoRA resolution."""


class RuntimeLoRAConfigurationError(RuntimeLoRAError):
    """The configured resolver does not satisfy Dynamo's protocol."""


class RuntimeLoRANotFoundError(RuntimeLoRAError):
    """No configured resolver produced a snapshot for the source URI."""


class RuntimeLoRAResolverUnavailableError(RuntimeLoRAError):
    """The backing provider is temporarily unavailable."""


class RuntimeLoRAPluginError(RuntimeLoRAError):
    """A resolver violated the runtime protocol."""


@dataclass(frozen=True)
class ResolveContext:
    adapter_key: str
    base_model_name: str
    cache_root: Path
    deadline_monotonic: float
    max_download_bytes: int
    request_id: str


@dataclass(frozen=True)
class ResolvedLoRA:
    local_path: Path
    source_revision: str
    size_bytes: int | None = None
    metadata: Mapping[str, str] | None = None


class RuntimeLoRAResolverProtocol(Protocol):
    protocol_version: int

    @property
    def schemes(self) -> frozenset[str]: ...

    async def resolve(
        self, *, source_uri: str, context: ResolveContext
    ) -> ResolvedLoRA | None: ...


def _import_object(reference: str) -> Any:
    if not reference or reference.isspace():
        raise RuntimeLoRAConfigurationError(
            "DYN_LORA_DOWNLOADER_PLUGIN must be a Python import reference"
        )

    if ":" in reference:
        module_name, object_name = reference.rsplit(":", 1)
        if not module_name or not object_name:
            raise RuntimeLoRAConfigurationError(
                "plugin reference must use module:object or module.object"
            )
        module = importlib.import_module(module_name)
        try:
            return getattr(module, object_name)
        except AttributeError as exc:
            raise RuntimeLoRAConfigurationError(
                f"plugin object {object_name!r} was not found"
            ) from exc

    module_name, separator, object_name = reference.rpartition(".")
    if not separator:
        raise RuntimeLoRAConfigurationError(
            "plugin reference must use module:object or module.object"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise RuntimeLoRAConfigurationError(
            f"plugin object {object_name!r} was not found"
        ) from exc


def _normalize_resolver(candidate: Any) -> RuntimeLoRAResolverProtocol:
    if (
        inspect.isclass(candidate)
        or callable(candidate)
        and not hasattr(candidate, "resolve")
    ):
        candidate = candidate()

    resolve = getattr(candidate, "resolve", None)
    if resolve is None or not callable(resolve):
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA resolver must define async resolve()"
        )
    if not inspect.iscoroutinefunction(resolve):
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA resolver resolve() must be async"
        )

    protocol_version = getattr(candidate, "protocol_version", None)
    if protocol_version != RUNTIME_LORA_PROTOCOL_VERSION:
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA resolver protocol_version must be 2"
        )

    raw_schemes = getattr(candidate, "schemes", None)
    if not isinstance(raw_schemes, (set, frozenset)) or not raw_schemes:
        raise RuntimeLoRAConfigurationError(
            "runtime LoRA resolver schemes must be a non-empty set"
        )
    for scheme in raw_schemes:
        if not isinstance(scheme, str) or not _SCHEME_PATTERN.fullmatch(scheme):
            raise RuntimeLoRAConfigurationError(
                "runtime LoRA resolver scheme names must be lowercase URI schemes"
            )

    return candidate


def load_runtime_lora_resolver(reference: str) -> RuntimeLoRAResolverProtocol:
    """Load and validate one resolver instance, class, or zero-argument factory."""

    try:
        candidate = _import_object(reference)
        return _normalize_resolver(candidate)
    except RuntimeLoRAConfigurationError:
        raise
    except Exception as exc:
        raise RuntimeLoRAConfigurationError(
            "failed to initialize runtime LoRA resolver"
        ) from exc


class RuntimeLoRAResolverChain:
    """Validated resolver chain restricted to an operator allowlist."""

    def __init__(
        self,
        resolvers: list[RuntimeLoRAResolverProtocol],
        allowed_schemes: set[str] | frozenset[str],
    ) -> None:
        if not resolvers:
            raise RuntimeLoRAConfigurationError(
                "at least one runtime LoRA resolver is required"
            )
        normalized_allowed = frozenset(
            scheme.strip().lower() for scheme in allowed_schemes
        )
        if not normalized_allowed or any(
            not _SCHEME_PATTERN.fullmatch(scheme) for scheme in normalized_allowed
        ):
            raise RuntimeLoRAConfigurationError(
                "DYN_LORA_ALLOWED_SCHEMES must contain valid URI schemes"
            )

        declared = frozenset(
            scheme for resolver in resolvers for scheme in resolver.schemes
        )
        enabled = normalized_allowed & declared
        if not enabled:
            raise RuntimeLoRAConfigurationError(
                "no operator-allowed scheme is declared by the runtime LoRA resolver"
            )

        self._resolvers = tuple(resolvers)
        self.schemes = enabled

    async def resolve(
        self, *, source_uri: str, context: ResolveContext
    ) -> ResolvedLoRA:
        scheme = urlsplit(source_uri).scheme.lower()
        if scheme not in self.schemes:
            raise RuntimeLoRANotFoundError("runtime LoRA source is not supported")

        loop = asyncio.get_running_loop()
        remaining = context.deadline_monotonic - loop.time()
        if remaining <= 0:
            raise TimeoutError("runtime LoRA resolution deadline exceeded")

        async with asyncio.timeout(remaining):
            for resolver in self._resolvers:
                if scheme not in resolver.schemes:
                    continue
                try:
                    result = await resolver.resolve(
                        source_uri=source_uri,
                        context=context,
                    )
                except asyncio.CancelledError:
                    raise
                except RuntimeLoRAError:
                    raise
                except Exception as exc:
                    raise RuntimeLoRAPluginError(
                        "runtime LoRA resolver failed"
                    ) from exc
                if result is None:
                    continue
                if not isinstance(result, ResolvedLoRA):
                    raise RuntimeLoRAPluginError(
                        "runtime LoRA resolver returned an invalid result"
                    )
                if not isinstance(result.local_path, Path):
                    raise RuntimeLoRAPluginError(
                        "runtime LoRA resolver returned an invalid local_path"
                    )
                if (
                    not isinstance(result.source_revision, str)
                    or not result.source_revision
                    or len(result.source_revision.encode("utf-8"))
                    > _MAX_SOURCE_REVISION_BYTES
                    or any(
                        ord(character) < 0x20 or ord(character) == 0x7F
                        for character in result.source_revision
                    )
                ):
                    raise RuntimeLoRAPluginError(
                        "runtime LoRA resolver returned an invalid source_revision"
                    )
                if result.size_bytes is not None and (
                    isinstance(result.size_bytes, bool)
                    or not isinstance(result.size_bytes, int)
                    or result.size_bytes < 0
                ):
                    raise RuntimeLoRAPluginError(
                        "runtime LoRA resolver returned an invalid size_bytes"
                    )
                return result

        raise RuntimeLoRANotFoundError("runtime LoRA source was not found")
