# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Type

from kvbm.v2.vllm.schedulers.connector import DynamoConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)

try:
    # vLLM >= 0.19.1: nixl_connector module became the nixl package.
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnector
except ImportError:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnector,
    )

from vllm.v1.core.sched.output import SchedulerOutput

_LMCacheConnectorV1: Optional[Type] = None
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )

    _LMCacheConnectorV1 = LMCacheConnectorV1
except ImportError:
    pass

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class PdConnectorMetadata(MultiKVConnectorMetadata):
    pass


@dataclass
class PdHandshakeMetadata(KVConnectorHandshakeMetadata):
    """Composite handshake metadata for KVBM and NIXL child connectors."""

    dynamo_metadata: Optional[KVConnectorHandshakeMetadata]
    nixl_metadata: Optional[KVConnectorHandshakeMetadata]


class PdConnector(MultiConnector):
    """
    Compose a KV offload connector with NIXL for P/D disaggregated serving.

    The first child is KVBM (or LMCache when available) and handles local
    prefix matching/offload/onboard. The second child is NIXL and handles the
    prefill-to-decode transfer.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        if len(self._connectors) != 2:
            raise ValueError(
                f"PdConnector requires exactly two connectors (got {len(self._connectors)})"
            )

        allowed_first_types: list[Type] = [DynamoConnector]
        if _LMCacheConnectorV1 is not None:
            allowed_first_types.append(_LMCacheConnectorV1)

        if not isinstance(self._connectors[0], tuple(allowed_first_types)):
            allowed_names = ["DynamoConnector"]
            if _LMCacheConnectorV1 is not None:
                allowed_names.append("LMCacheConnectorV1")
            raise TypeError(
                f"Expected first connector to be {' or '.join(allowed_names)}, "
                f"got {type(self._connectors[0]).__name__}"
            )
        if not isinstance(self._connectors[1], NixlConnector):
            raise TypeError(
                f"Expected second connector to be NixlConnector, "
                f"got {type(self._connectors[1]).__name__}"
            )

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Route composite handshake metadata to the child that produced it."""
        dynamo_meta: dict[int, KVConnectorHandshakeMetadata] = {}
        nixl_meta: dict[int, KVConnectorHandshakeMetadata] = {}

        for rank, composite in metadata.items():
            if isinstance(composite, PdHandshakeMetadata):
                if composite.dynamo_metadata is not None:
                    dynamo_meta[rank] = composite.dynamo_metadata
                if composite.nixl_metadata is not None:
                    nixl_meta[rank] = composite.nixl_metadata
            else:
                # Backwards compatibility for NIXL-only handshakes.
                nixl_meta[rank] = composite

        if dynamo_meta:
            self._connectors[0].set_xfer_handshake_metadata(dynamo_meta)
        if nixl_meta:
            self._connectors[1].set_xfer_handshake_metadata(nixl_meta)

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """Collect child handshake metadata without leaking it across connectors."""
        dynamo_metadata = self._connectors[0].get_handshake_metadata()
        nixl_metadata = self._connectors[1].get_handshake_metadata()

        if dynamo_metadata is None and nixl_metadata is None:
            return None

        return PdHandshakeMetadata(
            dynamo_metadata=dynamo_metadata,
            nixl_metadata=nixl_metadata,
        )

    def bind_connector_metadata(self, connector_metadata: PdConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata, PdConnectorMetadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(connector_metadata.extra_async_saves)
        for child, child_metadata in zip(self._connectors, connector_metadata.metadata):
            child.bind_connector_metadata(child_metadata)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        return self._connectors[0].get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        empty_blocks = blocks.new_empty()
        self._connectors[0].update_state_after_alloc(
            request, blocks, num_external_tokens
        )
        self._connectors[1].update_state_after_alloc(request, empty_blocks, 0)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> PdConnectorMetadata:
        metadata = PdConnectorMetadata(
            metadata=tuple(
                child.build_connector_meta(scheduler_output)
                for child in self._connectors
            )
        )
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata
