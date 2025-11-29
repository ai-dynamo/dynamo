# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Scheduler Connector implementation for vLLM.

This connector uses minimal scheduler-specific implementations that provide
no-op responses, used for scheduler integration testing without KV transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from typing_extensions import override
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self._leader = SchedulerConnectorLeader(
                vllm_config=vllm_config, engine_id=self.engine_id
            )
            self._worker = None
        elif role == KVConnectorRole.WORKER:
            self._worker = SchedulerConnectorWorker(
                vllm_config=vllm_config, engine_id=self.engine_id
            )
            self._leader = None
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
        if self._leader is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return self._leader.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """No-op since we never have external tokens."""
        if self._leader is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        self._leader.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build minimal connector metadata (empty bytes)."""
        if self._leader is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        data = self._leader.build_connector_meta(scheduler_output)
        return DynamoSchedulerConnectorMetadata(data)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Never delays block freeing - returns (False, None)."""
        if self._leader is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return self._leader.request_finished(request, block_ids)

    # added in v0.11
    def update_connector_output(self, connector_output: KVConnectorOutput):
        """No-op - no state updates needed."""
        if self._leader is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        pass

    # added in v0.11
    def take_events(self):
        """Returns empty tuple - no events."""
        if self._leader is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return ()

    # added in v0.11
    def get_finished_count(self):
        """Returns None - no async operations tracked."""
        if self._leader is None:
            raise RuntimeError("Cannot call scheduler methods on WORKER role")
        return None

    # added in v0.11
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        """Returns None - no specific layout required."""
        return None

    # added in v0.11
    @classmethod
    def build_kv_connector_stats(cls, data=None):
        """Returns None - no custom stats."""
        return None

    # Worker methods

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches - no-op for scheduler connector."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        self._worker.register_kv_caches(kv_caches)

    def bind_connector_metadata(
        self, connector_metadata: DynamoSchedulerConnectorMetadata
    ) -> None:
        """Bind connector metadata - no-op."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        assert isinstance(connector_metadata.metadata, bytes)
        self._worker.bind_connector_metadata(connector_metadata.metadata)

    def clear_connector_metadata(self) -> None:
        """Clear connector metadata - no-op."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
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
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        pass

    # added in v0.11
    def get_kv_connector_stats(self):
        """Returns None - no stats collected."""
        if self._worker is None:
            raise RuntimeError("Cannot call worker methods on SCHEDULER role")
        return None
