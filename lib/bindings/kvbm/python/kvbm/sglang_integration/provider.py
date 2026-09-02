# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-memory provider seam for the SGLang KVBM integration."""

from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class HostRegionRequest:
    """Requested rcommu data-region geometry."""

    requested_bytes: int
    bytes_per_block: int
    alignment: int
    manager_namespace: bytes
    tp_rank: int
    dp_rank: int | None
    attn_dp_rank: int
    attn_cp_rank: int


class AttachedHostRegion(Protocol):
    """Provisional region returned by an owner-backed provider."""

    def data_ptr(self) -> int:
        ...

    def nbytes(self) -> int:
        ...

    def activate(self) -> None:
        ...

    def abort(self) -> None:
        ...

    def close(self) -> None:
        ...


class HostMemoryProvider(Protocol):
    def attach(
        self, request: HostRegionRequest, cuda_device: int
    ) -> AttachedHostRegion:
        ...


_provider: HostMemoryProvider | None = None
_provider_lock = threading.Lock()


def register_host_memory_provider(provider: HostMemoryProvider) -> None:
    """Register the process-wide owner-backed host-memory provider."""
    global _provider
    with _provider_lock:
        if _provider is not None and _provider is not provider:
            raise RuntimeError(
                "A different KVBM host-memory provider is already registered."
            )
        _provider = provider


def get_host_memory_provider() -> HostMemoryProvider:
    """Resolve a registered provider or load its zero-argument env factory."""
    global _provider
    with _provider_lock:
        if _provider is not None:
            return _provider

    factory_path = os.environ.get("DYN_KVBM_HOST_MEMORY_PROVIDER")
    if not factory_path or ":" not in factory_path:
        raise RuntimeError(
            "dynamo_kvbm requires an owner-backed host-memory provider. "
            "Call register_host_memory_provider(...) or set "
            "DYN_KVBM_HOST_MEMORY_PROVIDER=module:factory."
        )
    module_name, factory_name = factory_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    provider = factory()

    with _provider_lock:
        if _provider is None:
            _provider = provider
        return _provider
