# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal scheduler connector worker implementation for testing.

This is a barebones implementation that provides no-op responses,
used specifically for scheduler integration testing without actual KV transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from vllm.model_executor.models.utils import extract_layer_index

from ..config import extract_vllm_config_for_kvbm

# Import our local block builder and config extractor
# from dynamo._core import SchedulerWorker
# from kvbm.v2.vllm.config import extract_vllm_config_for_kvbm


if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext


class SchedulerConnectorWorker:
    """
    Minimal scheduler connector worker that provides no-op implementations.

    This connector is used for scheduler integration where no actual
    KV transfer is needed. All methods are no-ops or return minimal responses.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        """Initialize the scheduler connector worker."""
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.scheduler_worker = None
        print(f"SchedulerConnectorWorker initialized with engine_id: {engine_id}")

        # Extract vLLM config for Dynamo
        self.kvbm_config = extract_vllm_config_for_kvbm(vllm_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register KV caches - builds local blocks without leader sync.

        This creates device blocks locally from the provided tensors
        without requiring any network setup or synchronization.
        """
        if not kv_caches:
            print("Warning: register_kv_caches called with empty kv_caches")
            return

        print(
            f"SchedulerConnectorWorker.register_kv_caches called with {len(kv_caches)} layers"
        )

        # Sort tensors by layer index to ensure correct ordering
        ordered_kv_caches = sorted(
            kv_caches.items(), key=lambda item: extract_layer_index(item[0])
        )

        # Extract tensors in order
        tensors = [tensor for _, tensor in ordered_kv_caches]

        # Get first tensor to extract common properties
        first_tensor = tensors[0]
        shape = first_tensor.shape

        # Validate all tensors have same shape
        if not all(t.shape == shape for t in tensors):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # Extract parameters
        # TODO: Assume the block dimension is within the first 2. This will break if you're doing something weird
        # Need to get the proper rank in case of multi-node.
        num_device_blocks = max(shape[0], shape[1])

        # Register KV caches with the scheduler worker
        try:
            success = self.scheduler_worker.register_kv(tensors, num_device_blocks)
            if success:
                print("Successfully registered KV caches with SchedulerWorker")
                print(f"  - Num device blocks: {num_device_blocks}")
                print(f"  - Num layers: {len(tensors)}")
            else:
                print("Warning: register_kv returned False")

        except Exception as e:
            print(f"Failed to register KV caches: {e}")
            raise

    def bind_connector_metadata(self, data: bytes) -> None:
        """
        Bind connector metadata - no-op.

        Since our leader returns empty bytes, this is always a no-op.
        """
        pass

    def clear_connector_metadata(self) -> None:
        """
        Clear connector metadata - no-op.
        """
        pass

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading KV cache - no-op.

        No KV loading needed for scheduler connector.
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Save KV layer - no-op.

        No KV saving needed for scheduler connector.
        """
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op - no async loading."""
        pass

    def wait_for_save(self) -> None:
        """No-op - no async saving."""
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Get finished request IDs.

        Since request_finished() always returns False (never delays block freeing),
        we just acknowledge the finished requests but don't return any as finished
        for KV transfer purposes.

        Returns:
            (None, None): No finished sends/receives
        """
        # Just acknowledge the finished requests
        # Since our leader's request_finished() always returns False,
        # these requests have already had their blocks freed
        if len(finished_req_ids) > 0:
            print(
                f"SchedulerConnectorWorker.get_finished() acknowledging {len(finished_req_ids)} finished requests"
            )

        return (None, None)
