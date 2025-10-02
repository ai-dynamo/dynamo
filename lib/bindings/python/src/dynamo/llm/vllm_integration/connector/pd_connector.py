# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector
from vllm.v1.core.sched.output import SchedulerOutput

from dynamo.llm.vllm_integration.connector.dynamo_connector import DynamoConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


@dataclass
class PdConnectorMetadata(MultiKVConnectorMetadata):
    pass


class PdConnector(MultiConnector):
    """
    A wrapper for using KV offloading Connectors (e.g. KVBM or LMCache) and NIXL Connector for PD disaggregated serving.

    The current logic is:
    - The first connector must be KVBM or LMCache and would be used by prefill worker to offload and onboard KV blocks.
    - The second connector must be NIXL and will be used by decode worker to get KV blocks from prefill worker.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        if len(self._connectors) != 2:
            raise ValueError(
                f"PdConnector requires exactly two connectors (got {len(self._connectors)})"
            )
        if not isinstance(self._connectors[0], (DynamoConnector, LMCacheConnectorV1)):
            raise TypeError(
                f"Expected first connector to be DynamoConnector or LMCacheConnector, "
                f"got {type(self._connectors[0]).__name__}"
            )
        if not isinstance(self._connectors[1], NixlConnector):
            raise TypeError(
                f"Expected second connector to be NixlConnector, "
                f"got {type(self._connectors[1]).__name__}"
            )

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: PdConnectorMetadata) -> None:
        assert isinstance(connector_metadata, PdConnectorMetadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(connector_metadata.extra_async_saves)
        for c, cm in zip(self._connectors, connector_metadata.metadata):
            c.bind_connector_metadata(cm)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get the number of matched tokens for the request using Dynamo Connector (KVBM).
        """
        return self._connectors[0].get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update the state after allocation using Dynamo Connector (KVBM) and Nixl Connector.
        """
        empty_blocks = blocks.new_empty()
        # allocate blocks for KV offloading connector to onboard KV blocks
        self._connectors[0].update_state_after_alloc(
            request, blocks, num_external_tokens
        )
        # no need to allocate any blocks for NIXL connector since this is in prefill worker side
        # and it only needs to wait for decode worker to pull its data.
        self._connectors[1].update_state_after_alloc(request, empty_blocks, 0)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> PdConnectorMetadata:
        metadata = PdConnectorMetadata(
            metadata=tuple(
                c.build_connector_meta(scheduler_output) for c in self._connectors
            )
        )
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata
