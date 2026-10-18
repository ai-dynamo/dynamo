# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Type

from kvbm.vllm_integration.connector.dynamo_connector import DynamoConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput

# Keep legacy KVBM-only deployments importable on vLLM releases that predate
# OffloadingConnector. KVCR deployments require the pinned adapter release.
_OffloadingConnector: Optional[Type] = None
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    _OffloadingConnector = OffloadingConnector
except ImportError:
    pass

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnector,
        NixlHandshakePayload,
    )
except ModuleNotFoundError as exc:
    if exc.name != "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector":
        raise
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        NixlConnector,
        NixlHandshakePayload,
    )

# Optional import for LMCache support
_LMCacheConnectorV1: Optional[Type] = None
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )

    _LMCacheConnectorV1 = LMCacheConnectorV1
except ImportError:
    pass

# Optional import for FlexKV support
_FlexKVConnectorV1: Optional[Type] = None
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector import (
        FlexKVConnectorV1,
    )

    _FlexKVConnectorV1 = FlexKVConnectorV1
except ImportError:
    pass

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class PdConnectorMetadata(MultiKVConnectorMetadata):
    pass


class PdConnector(MultiConnector):
    """
    A wrapper for a prefix offload connector and NIXL P/D handoff.

    The current logic is:
    - The first connector is KVBM, LMCache, FlexKV, or OffloadingConnector
      (including its KVCR secondary tier). It loads reusable prefill prefixes
      and stores computed blocks on either worker role.
    - The second connector is NIXL. A request carrying do_remote_prefill uses
      this child exclusively for loading the ordinary P/D handoff.
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

        # Build allowed types for first connector
        allowed_first_types: list[Type] = [DynamoConnector]
        if _OffloadingConnector is not None:
            allowed_first_types.append(_OffloadingConnector)
        if _LMCacheConnectorV1 is not None:
            allowed_first_types.append(_LMCacheConnectorV1)
        if _FlexKVConnectorV1 is not None:
            allowed_first_types.append(_FlexKVConnectorV1)

        if not isinstance(self._connectors[0], tuple(allowed_first_types)):
            allowed_names = ["DynamoConnector"]
            if _OffloadingConnector is not None:
                allowed_names.append("OffloadingConnector")
            if _LMCacheConnectorV1 is not None:
                allowed_names.append("LMCacheConnectorV1")
            if _FlexKVConnectorV1 is not None:
                allowed_names.append("FlexKVConnectorV1")
            raise TypeError(
                f"Expected first connector to be {' or '.join(allowed_names)}, "
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

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Propagate handshake metadata to child connectors.

        This is required for NIXL connector to start its handshake listener
        which decode workers connect to for KV transfer coordination.
        """
        for c in self._connectors:
            c.set_xfer_handshake_metadata(metadata)

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata from the NIXL connector.

        This metadata is used for out-of-band connector handshake between
        P/D workers. The NIXL connector (second connector) provides this
        metadata so decode workers can connect for KV transfer coordination.

        Returns:
            NixlHandshakePayload from the NIXL connector, or None if not available.
        """
        # Get handshake metadata from the NIXL connector (second connector)
        nixl_connector = self._connectors[1]
        metadata = nixl_connector.get_handshake_metadata()
        if metadata is not None and not isinstance(metadata, NixlHandshakePayload):
            raise TypeError(
                f"Expected NixlHandshakePayload from NIXL connector, "
                f"got {type(metadata).__name__}"
            )
        return metadata

    def bind_connector_metadata(self, connector_metadata: PdConnectorMetadata) -> None:
        # Must call super() to set _connector_metadata so has_connector_metadata() returns True
        # This is required for save_kv_layer to be called during the forward pass
        super().bind_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata, PdConnectorMetadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(connector_metadata.extra_async_saves)
        for c, cm in zip(self._connectors, connector_metadata.metadata):
            c.bind_connector_metadata(cm)

    def get_kv_connector_kv_cache_events(self) -> "KVConnectorKVEvents | None":
        """Forward worker events from the single prefix inventory provider.

        Scheduler-side events (including KVCR ready/removed residency) are
        already forwarded by MultiConnector.take_events. NIXL owns handoff
        completion, not prefix inventory, so it is not an event provider here.
        """
        return self._connectors[0].get_kv_connector_kv_cache_events()

    def take_events(self):
        """Forward ready/removed prefix residency from the offload scheduler."""
        yield from self._connectors[0].take_events()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Choose one loader according to the request phase, before lookup.

        NIXL may clear do_remote_prefill after allocation. Retain the selected
        child until MultiConnector's request-finished cleanup removes it.
        """
        connector_index = self._load_connector_index(request)
        return self._connectors[connector_index].get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def _load_connector_index(self, request: "Request") -> int:
        params = request.kv_transfer_params
        is_handoff = isinstance(params, Mapping) and params.get("do_remote_prefill")
        return self._requests_to_connector.setdefault(
            request.request_id, 1 if is_handoff else 0
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Give one loader the external prefix; both children see real blocks.

        The offload child needs decode allocations to retain generated KV even
        when NIXL loaded the prompt. Passing zero external tokens prevents it
        from claiming the same destination blocks for an independent load.
        """
        connector_index = self._load_connector_index(request)
        for index, connector in enumerate(self._connectors):
            connector.update_state_after_alloc(
                request, blocks, num_external_tokens if index == connector_index else 0
            )

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
