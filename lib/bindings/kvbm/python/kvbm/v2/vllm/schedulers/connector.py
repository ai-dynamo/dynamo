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
        Decide whether vLLM should allocate a single cross-layer KV cache
        (FC, registered through `register_cross_layers_kv_cache`) or one
        tensor per layer (LW, registered through `register_kv_caches`).

        Default ("auto"):

        1. Hybrid models with multiple distinct attention backends are
           rejected here with a clear log. KVBM v2 does NOT currently
           support hybrid models in either path — `register_kv_caches`
           also bails on `len(self._attn_backends) != 1` and `len(groups)
           != 1`. Returning False here just routes the failure into LW
           where the `NotImplementedError` is the authoritative source
           of truth; the FC path's `register_cross_layers_kv_cache` also
           rejects multi-group inputs defensively. Hybrid kv_cache_groups
           on a single backend cannot be detected here (kv_cache_config
           is not yet built when this property is read) and will surface
           as the LW `NotImplementedError`.

        2. Otherwise: probe the single backend with
           `dim_probe.select_fc_variant`. Return True iff it maps to one
           of `FullyContiguousLayout`'s supported orderings (OperationalNHD,
           OperationalHND, Universal). Anything else (FlashInfer HND,
           MLA, missing cross-layer stride support) returns False so vLLM
           uses LW for the single-backend case.

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
          3. Auto-detect (the body below).
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

        # Auto: enumerate backends and use the shared `select_fc_for_model`
        # helper so the FC eligibility contract has one definition (and
        # one set of tests) shared between this connector and any future
        # caller.
        try:
            from kvbm.v2.vllm.dim_probe import (
                FC_INELIGIBLE_BACKEND_NO_MATCH,
                FC_INELIGIBLE_HYBRID_BACKENDS,
                FC_INELIGIBLE_NO_BACKENDS,
                select_fc_for_model,
            )
            from vllm.distributed.kv_transfer.kv_connector.utils import (
                get_current_attn_backends,
            )

            backends = list(get_current_attn_backends(self._vllm_config))
        except Exception as e:
            print(
                f"[KVBM] prefer_cross_layer_blocks auto-detect failed to "
                f"enumerate backends ({type(e).__name__}: {e}); "
                f"defaulting to per-layer (LW) registration."
            )
            return False

        variant, reason = select_fc_for_model(backends)
        if variant is not None:
            print(
                f"[KVBM] prefer_cross_layer_blocks=True: backend "
                f"{backends[0].__name__} maps to FC variant {variant.value}."
            )
            return True

        if reason == FC_INELIGIBLE_NO_BACKENDS:
            # Unusual: no static_forward_context entries. Defer to LW.
            print(
                "[KVBM] prefer_cross_layer_blocks=False: "
                "get_current_attn_backends returned no backends."
            )
        elif reason == FC_INELIGIBLE_HYBRID_BACKENDS:
            backend_names = [b.__name__ for b in backends]
            print(
                f"[KVBM] prefer_cross_layer_blocks=False: model has "
                f"{len(backends)} distinct attention backends "
                f"({backend_names}); hybrid models are not supported in "
                f"either FC or LW paths. Registration will fail with a "
                f"NotImplementedError from register_kv_caches."
            )
        elif reason == FC_INELIGIBLE_BACKEND_NO_MATCH:
            print(
                f"[KVBM] prefer_cross_layer_blocks=False: backend "
                f"{backends[0].__name__} has no compatible FC variant "
                f"(probably HND-with-Outer-before-HeadCount, MLA, or "
                f"missing cross-layer stride support). Set "
                f"KVBM_PREFER_FULLY_CONTIGUOUS_BLOCKS=true to override "
                f"(will fail at registration if truly incompatible)."
            )
        return False

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
