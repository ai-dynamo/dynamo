# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler connector worker implementation for v2 vLLM integration.

This implementation delegates to the Rust PyConnectorWorker when available,
providing proper FinishedState tracking for transfer coordination.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

import torch
from vllm.model_executor.models.utils import extract_layer_index

from ..config import extract_vllm_config_for_kvbm

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext


class SchedulerConnectorWorker:
    """
    Scheduler connector worker that delegates to Rust implementation.

    When the Rust bindings (kvbm._core.v2.ConnectorWorker) are available,
    this class delegates transfer-related operations to the Rust implementation
    which provides proper FinishedState tracking for onboarding/offloading.

    When bindings are unavailable, falls back to stub responses for testing.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        """Initialize the scheduler connector worker."""
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self._rust_worker = None
        print(f"SchedulerConnectorWorker initialized with engine_id: {engine_id}")

        # Extract vLLM config for Dynamo
        self.kvbm_config = extract_vllm_config_for_kvbm(vllm_config)
        self._slots: dict[str, dict[str, str]] = {}

        # Check if Rust bindings are available
        import kvbm

        if kvbm.v2.is_available():
            # Note: ConnectorWorker requires a KvbmRuntime which needs Nova setup.
            # For now, we'll initialize the Rust worker lazily during register_kv_caches
            # when we have access to the actual tensors.
            print(
                "SchedulerConnectorWorker: Rust bindings available, will initialize on register_kv_caches"
            )
        else:
            print(
                "SchedulerConnectorWorker: Rust bindings not available, using stub mode"
            )

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

        if first_tensor.device.type != "cuda":
            raise NotImplementedError("Only CUDA tensors are supported for now.")

        device_index = first_tensor.device.index
        device_properties = torch.cuda.get_device_properties(device_index)
        device_uuid = device_properties.uuid

        print(
            f"Connector Worker: {self.kvbm_config.rank()} of {self.kvbm_config.world_size()} using Device {device_index} ({device_uuid})"
        )

        # Validate all tensors have same shape
        if not all(t.shape == shape for t in tensors):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # Extract parameters
        num_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        inferred_gpu_blocks = max(shape[0], shape[1])
        assert (
            inferred_gpu_blocks == num_gpu_blocks
        ), "Number of device blocks does not match the number of GPU blocks in the configuration"

        # Try to initialize Rust worker if bindings are available
        try:
            # Create runtime and worker
            # Note: This requires Nova to be set up properly.
            # For now, we're just documenting the flow - actual initialization
            # requires runtime environment setup.
            pass  # TODO: Wire up when KvbmRuntime initialization is ready

        except Exception as e:
            print(f"Could not initialize Rust worker: {e}")

    def bind_connector_metadata(self, data: bytes) -> None:
        """
        Bind connector metadata by applying the serialized updates.
        """
        if not data:
            return

        payload = json.loads(data.decode("utf-8"))

        for create in payload.get("slot_creates", []):
            request_id = create["request_id"]
            self._slots.setdefault(request_id, {}).update(
                {"create_event": create.get("create_event", "")}
            )
            print(f"[worker] slot create received for {request_id}")

        for delete in payload.get("slot_deletes", []):
            request_id = delete["request_id"]
            if self._slots.pop(request_id, None) is not None:
                print(f"[worker] slot delete received for {request_id}")

        for fence in payload.get("forward_events", []):
            request_id = fence["request_id"]
            slot = self._slots.setdefault(request_id, {})
            slot["forward_events"] = json.dumps(fence.get("worker_events", []))

    def clear_connector_metadata(self) -> None:
        """
        Clear connector metadata - no-op.
        """
        self._slots.clear()

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

        When Rust worker is available, delegates to its FinishedState tracker
        which returns completed onboarding and offloading request IDs.

        Returns:
            (finished_sending, finished_recving): Sets of request IDs that have
            completed offloading and onboarding respectively. None if no
            completed operations of that type.
        """
        # Delegate to Rust worker if available
        if self._rust_worker is not None:
            try:
                return self._rust_worker.get_finished()
            except Exception as e:
                print(f"Error calling Rust get_finished(): {e}")
                return (None, None)

        # Stub behavior: acknowledge finished requests but don't return any
        # as finished for KV transfer purposes
        if len(finished_req_ids) > 0:
            print(
                f"SchedulerConnectorWorker.get_finished() acknowledging {len(finished_req_ids)} finished requests"
            )

        return (None, None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Returns empty set - no load errors tracked."""
        return set()

    def get_handshake_metadata(self):
        """Returns None - no handshake metadata."""
        return None
