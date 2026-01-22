# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler connector leader implementation for v2 vLLM integration.

This implementation delegates to the Rust PyKvConnectorLeader when available,
falling back to stub responses when the Rust bindings are not loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorOutput

from .connector import DynamoSchedulerConnectorMetadata


class SchedulerConnectorLeader:
    """
    Scheduler connector leader that delegates to Rust implementation.

    When the Rust bindings (kvbm._core.v2.KvConnectorLeader) are available,
    this class delegates all operations to the Rust implementation which
    provides proper slot state tracking and transfer coordination.

    When bindings are unavailable, falls back to stub responses for testing.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        """Initialize the scheduler connector leader."""
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self._rust_leader = None

        # Try to load Rust bindings
        try:
            import kvbm

            if kvbm.v2.is_available():
                from ..config import extract_vllm_config_for_kvbm

                kvbm_config = extract_vllm_config_for_kvbm(vllm_config)
                self._rust_leader = kvbm.v2.KvConnectorLeader(engine_id, kvbm_config)
                print(
                    f"SchedulerConnectorLeader initialized with Rust backing, engine_id: {engine_id}"
                )
            else:
                raise ImportError("kvbm v2 feature not available")
        except (ImportError, Exception) as e:
            print(
                f"SchedulerConnectorLeader initialized in stub mode (no Rust backing): {e}"
            )
            print(f"Engine ID: {engine_id}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Get the number of new matched tokens for the given request.

        Returns:
            (Option<int>, bool): Number of matched tokens (None if evaluating),
                                 and whether async loading is in progress.
        """
        if self._rust_leader is not None:
            return self._rust_leader.get_num_new_matched_tokens(
                request.request_id, num_computed_tokens
            )
        # Stub: no external tokens available
        return (0, False)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """
        Update state after block allocation for onboarding.

        Called when scheduler allocates blocks for a request with matched tokens.
        """
        if self._rust_leader is not None:
            block_ids = list(blocks.get_unhashed_block_ids())
            self._rust_leader.update_state_after_alloc(
                request.request_id, block_ids, num_external_tokens
            )
        elif num_external_tokens > 0:
            print(
                f"Warning: update_state_after_alloc called with {num_external_tokens} "
                f"external tokens for request {request.request_id}, but no Rust backing"
            )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        """
        Build connector metadata for the forward pass.

        Returns:
            Serialized metadata bytes for workers.
        """
        # TODO: Wire up to Rust build_connector_metadata when SchedulerOutput
        # conversion is implemented
        return DynamoSchedulerConnectorMetadata(b"")

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Notify leader that a request has finished.

        Returns:
            (is_pending, connector_metadata):
            - is_pending=True: Don't free blocks yet (transfer in progress)
            - is_pending=False: Safe to free blocks
        """
        if self._rust_leader is not None:
            is_pending = self._rust_leader.request_finished(
                request.request_id, block_ids
            )
            return (is_pending, None)
        # Stub: never delay block freeing
        return (False, None)

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """
        Process worker reports of finished transfers.

        Called by the scheduler after workers report get_finished() results.
        Updates slot state based on completed onboarding/offloading operations.
        """
        if self._rust_leader is None:
            return

        # Process finished offloading (sends)
        if connector_output.finished_sending:
            for req_id in connector_output.finished_sending:
                try:
                    self._rust_leader.handle_finished_offload(req_id)
                except Exception as e:
                    print(f"Error handling finished offload for {req_id}: {e}")

        # Process finished onboarding (receives)
        if connector_output.finished_recving:
            for req_id in connector_output.finished_recving:
                try:
                    self._rust_leader.handle_finished_onboard(req_id)
                except Exception as e:
                    print(f"Error handling finished onboard for {req_id}: {e}")

    def set_xfer_handshake_metadata(self, metadata) -> None:
        """Set transfer handshake metadata from workers."""
        # TODO: Wire up to Rust if needed for distributed coordination
        pass
