# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Scheduler Connector implementation for vLLM.

This connector uses minimal scheduler-specific implementations that provide
no-op responses, used for scheduler integration testing without KV transfer.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import torch
from typing_extensions import override
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.kv_cache_manager import KVCacheConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from kvbm.v2.vllm.config import extract_vllm_config_for_kvbm

# Import our minimal scheduler connector implementations
from .leader import SchedulerConnectorLeader
from .worker import SchedulerConnectorWorker

EngineId = str


class DynamoSchedulerConnectorMetadata(KVConnectorMetadata):
    """Minimal metadata container for scheduler connector."""

    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


class DynamoConnector(KVConnectorBase_V1):
    """
    Dynamo Scheduler Connector that uses minimal no-op implementations.

    This connector is specifically for scheduler integration testing and
    provides no actual KV transfer functionality.
    """

    _scheduler: Optional[SchedulerConnectorLeader]
    _worker: Optional[SchedulerConnectorWorker]

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional[KVCacheConfig] = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None

        # Get extra config from vLLM's KVTransferConfig (if available)
        # This dict gets serialized to JSON and merged with env/file config in Rust
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        extra_config = (
            getattr(kv_transfer_config, "kv_connector_extra_config", {})
            if kv_transfer_config
            else {}
        )

        # Serialize to JSON and pass to Rust (empty dict = use defaults)
        kvbm_override_config = json.dumps(extra_config) if extra_config else None

        kvbm_config = extract_vllm_config_for_kvbm(vllm_config)

        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = SchedulerConnectorLeader(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                kvbm_config=kvbm_config,
                kvbm_override_config=kvbm_override_config,
            )
            self._worker = None
        elif role == KVConnectorRole.WORKER:
            self._worker = SchedulerConnectorWorker(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                kvbm_config=kvbm_config,
                kvbm_override_config=kvbm_override_config,
            )
            self._scheduler = None
        else:
            raise ValueError(
                f"Invalid KVConnectorRole: {role}. Must be SCHEDULER or WORKER."
            )

    # Scheduler/Leader methods

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """Always returns (0, False) - no external tokens available."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """No-op since we never have external tokens."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build step metadata for workers."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")

        data = self._scheduler.build_connector_meta(scheduler_output)
        return DynamoSchedulerConnectorMetadata(data)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Never delays block freeing - returns (False, None)."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return self._scheduler.request_finished(request, block_ids)

    # added in v0.11
    def update_connector_output(self, connector_output: KVConnectorOutput):
        """No-op - no state updates needed."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        self._scheduler.update_connector_output(connector_output)

    # added in v0.11
    def take_events(self):
        """Returns empty tuple - no events."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return ()

    # added in v0.11
    def get_finished_count(self):
        """Returns None - no async operations tracked."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return self._scheduler.get_finished_count()

    # added in v0.11.1
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, "KVConnectorHandshakeMetadata"]
    ) -> None:
        """No-op - handshake metadata not used."""
        if self._scheduler is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        self._scheduler.set_xfer_handshake_metadata(metadata)

    # added in v0.11
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        """Returns None - no specific layout required."""
        return None

    # added in v0.11
    @classmethod
    def build_kv_connector_stats(cls, data=None):
        """Returns None - no custom stats."""
        return cls._build_kv_connector_stats_impl(data)

    @staticmethod
    def _build_kv_connector_stats_impl(data=None):
        return None

    # added in v0.11.1
    @classmethod
    def build_prom_metrics(
        cls, vllm_config, metric_types, labelnames, per_engine_labelvalues
    ):
        """Returns None - no Prometheus metrics."""
        return None

    # Worker methods

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """
        KVBM prefers a single cross-layer allocation so the device-side
        layout becomes `FullyContiguousLayout`, letting the transfer planner
        do per-block transfers across all layers in one operation.

        Override at runtime via `KVBM_PREFER_FULLY_CONTIGUOUS_BLOCKS={false,0,no,n}`
        to force the per-layer `register_kv_caches` path (useful for A/B
        comparisons, or to work around a backend whose physical layout
        doesn't match `FullyContiguousLayout`'s schema). Default is `true`.

        Override precedence (highest first):
          1. Env var `KVBM_PREFER_FULLY_CONTIGUOUS_BLOCKS={true,false}`.
             Note: vLLM strips parent env when spawning EngineCore, so this
             channel does NOT work for the connector running inside disagg
             EngineCore subprocesses. Use the JSON config channel instead.
          2. JSON config
             `kv_transfer_config.kv_connector_extra_config.default.prefer_fully_contiguous_blocks`
             (bool). This is the channel that survives the EngineCore spawn
             and is the canonical way for tests / launch scripts to pin
             FC vs LW per run.
          3. Default `True` (prefer fully-contiguous).

        Even with this `True`, vLLM still falls back to the per-layer
        `register_kv_caches` path when the backend doesn't support
        `get_kv_cache_stride_order(include_num_layers_dimension=True)` or
        for MLA models.
        """
        override = os.getenv("KVBM_PREFER_FULLY_CONTIGUOUS_BLOCKS", "").lower()
        if override in ("false", "0", "no", "n", "off"):
            return False
        if override in ("true", "1", "yes", "y", "on"):
            return True

        # JSON-config override. Survives the EngineCore subprocess spawn that
        # strips env vars; the canonical channel for test/launch overrides.
        try:
            extra = (
                getattr(self._vllm_config, "kv_transfer_config", None)
                and getattr(
                    self._vllm_config.kv_transfer_config,
                    "kv_connector_extra_config",
                    None,
                )
                or {}
            )
            json_pref = (extra.get("default") or {}).get(
                "prefer_fully_contiguous_blocks"
            )
        except Exception:
            json_pref = None
        if isinstance(json_pref, bool):
            print(
                f"[KVBM] prefer_cross_layer_blocks={json_pref}: forced via "
                f"kv_connector_extra_config.default.prefer_fully_contiguous_blocks"
            )
            return json_pref
        if isinstance(json_pref, str):
            jp = json_pref.lower()
            if jp in ("true", "1", "yes", "y", "on"):
                print(
                    "[KVBM] prefer_cross_layer_blocks=True: forced via "
                    "kv_connector_extra_config.default.prefer_fully_contiguous_blocks"
                )
                return True
            if jp in ("false", "0", "no", "n", "off"):
                print(
                    "[KVBM] prefer_cross_layer_blocks=False: forced via "
                    "kv_connector_extra_config.default.prefer_fully_contiguous_blocks"
                )
                return False

        # Default: prefer fully-contiguous (current-branch default). vLLM
        # guards FC-ineligible backends by falling back to the per-layer
        # `register_kv_caches` path.
        return True

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches - no-op for scheduler connector."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        self._worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type,
    ) -> None:
        """Register a single cross-layer KV cache tensor — delegates to worker."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        self._worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    @override
    def bind_connector_metadata(
        self, connector_metadata: DynamoSchedulerConnectorMetadata
    ) -> None:
        """Bind connector metadata."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        # Must call super() to set _connector_metadata so has_connector_metadata() returns True
        # This is required for save_kv_layer to be called during the forward pass
        assert isinstance(connector_metadata.metadata, bytes)
        if self._worker.bind_connector_metadata(connector_metadata.metadata):
            super().bind_connector_metadata(connector_metadata)

    @override
    def clear_connector_metadata(self) -> None:
        """Clear connector metadata."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        super().clear_connector_metadata()
        self._worker.clear_connector_metadata()

    @override
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading KV cache - no-op for scheduler connector."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        self._worker.start_load_kv(forward_context, **kwargs)

    @override
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer load - no-op."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        self._worker.wait_for_layer_load(layer_name)

    @override
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Save KV layer - no-op for scheduler connector."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        self._worker.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    @override
    def wait_for_save(self):
        """Wait for save - no-op."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        self._worker.wait_for_save()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get finished request IDs - always returns (None, None)."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        return self._worker.get_finished(finished_req_ids)

    # added in v0.11
    def set_host_xfer_buffer_ops(self, copy_operation):
        """No-op - not needed for scheduler connector."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        pass

    # added in v0.11
    def shutdown(self):
        """No-op - no resources to cleanup."""
        # Note: shutdown can be called on both SCHEDULER and WORKER roles
        pass

    # added in v0.11
    def get_kv_connector_stats(self):
        """Returns None - no stats collected."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        return None

    # added in v0.11.1
    def get_block_ids_with_load_errors(self) -> set[int]:
        """Returns empty set - no load errors tracked."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        return self._worker.get_block_ids_with_load_errors()

    # added in v0.11.1
    def get_handshake_metadata(self):
        """Returns None - no handshake metadata."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        return self._worker.get_handshake_metadata()
