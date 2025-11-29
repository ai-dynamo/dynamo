# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal scheduler connector leader implementation for testing.

This is a barebones implementation that returns minimal/no-op responses,
used specifically for scheduler integration testing without actual KV transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

from .connector import DynamoSchedulerConnectorMetadata


class SchedulerConnectorLeader:
    """
    Minimal scheduler connector leader that returns no-op responses.

    This connector is used for scheduler integration where no actual
    KV transfer is needed. All methods return minimal valid responses.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        """Initialize the scheduler connector leader."""
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        print(f"SchedulerConnectorLeader initialized with engine_id: {engine_id}")

        try:
            from kvbm.libkvbm_py3 import ConnectorMetadataBuilder  # type: ignore

            self._metadata_builder = ConnectorMetadataBuilder()
        except ImportError:
            self._metadata_builder = None

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Always returns (0, False) indicating no external tokens available.

        Returns:
            (0, False): No external tokens, no async loading
        """
        return (0, False)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """
        No-op since we never have external tokens.

        This should never be called with num_external_tokens > 0.
        """
        if num_external_tokens > 0:
            print(
                f"Warning: update_state_after_alloc called with {num_external_tokens} "
                f"external tokens for request {request.request_id}, but scheduler "
                "connector always returns 0 external tokens"
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build minimal connector metadata.

        Returns:
            Serialized bytes containing pending metadata events.
        """
        _ = scheduler_output  # Reserved for future use
        if self._metadata_builder is None:
            return DynamoSchedulerConnectorMetadata(b"")
        return DynamoSchedulerConnectorMetadata(b"")

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Never delays block freeing.

        Returns:
            (False, None): Don't delay block freeing, no KV transfer params
        """
        return (False, None)
