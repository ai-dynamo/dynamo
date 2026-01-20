# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TRTLLM v2 Connector Worker implementation.

This connector uses the v2 runtime with Nova for distributed communication,
replacing the v1 pattern of RustKvConnectorWorker + DistributedRuntime.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import kvbm
import torch

# Import v2 bindings (requires v2 feature)
if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
    ConnectorWorker = kvbm.v2.ConnectorWorker
else:
    KvbmRuntime = None
    ConnectorWorker = None

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# Conditional import - only fail when class is instantiated, not at module load
try:
    from tensorrt_llm._torch.pyexecutor.kv_cache_connector import KvCacheConnectorWorker

    _TRTLLM_AVAILABLE = True
except ImportError:
    KvCacheConnectorWorker = object  # Stub base class
    _TRTLLM_AVAILABLE = False

from kvbm.v2.common import NovaPeerMetadata

logger = logging.getLogger(__name__)


class DynamoKVBMConnectorWorker(KvCacheConnectorWorker):
    """
    TRTLLM v2 Connector Worker using KvbmRuntime + Nova.

    This replaces the v1 pattern of RustKvConnectorWorker + DistributedRuntime
    with the v2 KvbmRuntime which uses Nova for distributed communication.
    """

    def __init__(
        self,
        llm_args: "TorchLlmArgs",
        kvbm_override_config: Optional[str] = None,
    ):
        # Check dependencies before super().__init__
        if not _TRTLLM_AVAILABLE:
            raise ImportError(
                "DynamoKVBMConnectorWorkerV2 requires tensorrt_llm. "
                "Install tensorrt_llm to use this connector."
            )

        super().__init__(llm_args)

        # Check that v2 feature is available
        if not kvbm.v2.is_available():
            raise ImportError(
                "DynamoKVBMConnectorWorkerV2 requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )

        config = {
            "worker": {
                "nixl": {"backends": {"UCX": {}}},
                "tokio": {"worker_threads": 2},
            }
        }
        self.kvbm_override_config = config

        mappings = self._llm_args.parallel_config.to_mapping()
        self.rank = mappings.rank

        # Build KvbmRuntime with Nova (replaces DRT)
        print(f"Building KvbmRuntime with config: {json.dumps(config)}")
        self.runtime = KvbmRuntime.build_worker(json.dumps(config))

        # Create the Rust ConnectorWorker that handles NIXL registration
        print(f"Creating ConnectorWorker with runtime: {self.runtime}")
        self.worker = ConnectorWorker(self.runtime)

        # Store peer info for handshake
        instance_id, worker_addr = self.runtime.peer_info()
        self._handshake_metadata = NovaPeerMetadata(
            instance_id=instance_id,
            worker_address=worker_addr,
        )

        # Forward pass callable support
        self.event = torch.cuda.Event()
        self.use_forward_pass_callable = True

        # Layer events for save_kv_layer
        self.layer_events: List[torch.cuda.Event] = []
        self._num_layers: int = 0

        logger.info(
            f"DynamoKVBMConnectorWorkerV2 initialized with Nova instance: {instance_id.hex()[:8]}..."
        )
        logger.info(f"  rank: {self.rank}")

    def _callable_object(self) -> callable:
        """Create callable for forward pass completion hook."""
        assert self.worker is not None, "Expected worker to be initialized"
        assert self.event is not None, "Expected event to be initialized"

        def callback():
            self.event.record()
            self.event.synchronize()
            # Note: v2 worker handles offload via save_kv_layer
            # This callback is for compatibility with the forward pass hook pattern

        return callback

    def register_forward_pass_callable(self) -> callable:
        """
        Register a callable object which will be called at the end of the forward pass.
        """
        self.use_forward_pass_callable = True
        return self._callable_object()

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        """
        Register the KV cache tensors to the worker.

        TRTLLM passes a single contiguous tensor with shape:
        [num_blocks, num_layers, 2, num_heads, block_size, head_size]

        We need to convert this to a list of per-layer tensors for the v2 API.

        Args:
            kv_cache_tensor: The contiguous KV cache tensor.
        """
        logger.info(
            f"DynamoKVBMConnectorWorkerV2.register_kv_caches starting on rank {self.rank}"
        )

        num_device_blocks = kv_cache_tensor.shape[0]
        num_cache_layers = kv_cache_tensor.shape[1]
        page_size = self._llm_args.kv_cache_config.tokens_per_block
        device_id = kv_cache_tensor.device.index
        dtype_width_bytes = kv_cache_tensor.dtype.itemsize

        self._num_layers = num_cache_layers

        # Create layer events
        self.layer_events = [
            torch.cuda.Event(enable_timing=False, interprocess=False)
            for _ in range(num_cache_layers)
        ]

        # Record events on current stream
        current_stream = torch.cuda.current_stream(device_id)
        for event in self.layer_events:
            event.record(current_stream)

        # Convert single contiguous tensor to list of per-layer tensors
        # Each layer tensor has shape: [num_blocks, 2, num_heads, block_size, head_size]
        tensors = [kv_cache_tensor[:, i] for i in range(num_cache_layers)]

        # Register with v2 API
        self.worker.register_kv_caches(
            tensors,
            num_device_blocks,
            page_size,
            dtype_width_bytes,
        )

        logger.info("[KVBM] KV caches registered (v2 mode)")
        logger.info(f"  - Num device blocks: {num_device_blocks}")
        logger.info(f"  - Num layers: {num_cache_layers}")
        logger.info(f"  - Page size: {page_size}")
        logger.info(f"  - Dtype width bytes: {dtype_width_bytes}")
        logger.info(f"  - Device ID: {device_id}")

    def bind_connector_meta(self, metadata: object) -> None:
        """
        Set the connector metadata from the scheduler.

        Args:
            metadata: The connector metadata (bytes).
        """
        super().bind_connector_meta(metadata)
        if isinstance(metadata, bytes):
            self.worker.bind_connector_metadata(metadata)

    def clear_connector_meta(self) -> None:
        """
        Clear the connector metadata.
        """
        print(f"Clearing connector metadata on rank {self.rank}")
        self.worker.clear_connector_metadata()
        super().clear_connector_meta()

    def start_load_kv(self, stream: torch.cuda.Stream) -> None:
        """
        Begin loading the KV cache in preparation for the next forward pass.
        """
        self.worker.start_load_kv()

    def wait_for_save(self, stream: torch.cuda.Stream) -> None:
        """
        Block until all synchronous saving operations are complete.
        """
        # v2 handles this internally
        pass

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        """
        Wait for a layer to finish being loaded.

        Args:
            layer_idx: The index of the layer to wait for.
            stream: The stream the forward pass is being executed on.
        """
        stream_handle = stream.cuda_stream
        self.worker.wait_for_layer_load(layer_idx, stream_handle)

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        """
        Begin saving the KV cache for a layer.

        Args:
            layer_idx: The index of the layer to save.
            stream: The stream the forward pass is being executed on.
        """
        stream_handle = stream.cuda_stream
        print(f"Saving KV layer {layer_idx} on stream {stream_handle}")
        self.worker.save_kv_layer(layer_idx, stream_handle)

    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Get the requests that have finished loading and saving.

        Args:
            finished_gen_req_ids: Request IDs that have finished generating.
            started_loading_req_ids: Request IDs that have started loading.

        Returns:
            Tuple of (finished_saving_ids, finished_loading_ids).
        """
        # v2 API returns (Optional[set[str]], Optional[set[str]])
        finished_sending, finished_recving = self.worker.get_finished()

        # Convert to list[int] for TRTLLM compatibility
        finished_saving = []
        finished_loading = []

        if finished_sending:
            for req_id in finished_sending:
                try:
                    finished_saving.append(int(req_id))
                except ValueError:
                    pass

        if finished_recving:
            for req_id in finished_recving:
                try:
                    finished_loading.append(int(req_id))
                except ValueError:
                    pass

        return (finished_saving, finished_loading)

    def get_handshake_metadata(self) -> NovaPeerMetadata:
        """
        Return Nova peer info for leader to connect.

        Returns:
            NovaPeerMetadata containing instance_id and worker_address bytes.
        """
        return self._handshake_metadata
