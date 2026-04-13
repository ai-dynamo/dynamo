# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM KV-connector worker wired to the Rust ConnectorWorker.

The Python class adapts vLLM's KVConnectorBase_V1 worker-side hooks to
the Rust `ConnectorWorker` in `lib/kvbm-connector/src/connector/worker/`,
which performs the four runtime actions documented at the top of that
crate: intra-pass onboard (G2→G1 per-layer H2D with CudaEvent sync on
the torch compute stream), inter-pass onboard driven by the leader via
the VeloWorkerService/DirectWorker, forward-pass completion notification
back to the leader, and direct layer-wise offload.

On worker bring-up the Python side builds a `KvbmRuntime` (Velo messenger
+ tokio) and a `ConnectorWorker` over it, exports its Velo peer info as
`VeloPeerMetadata` for the leader's `set_xfer_handshake_metadata`, and
defers NIXL registration until `register_kv_caches` — the actual NIXL
bind happens later, when the leader's `initialize_workers()` RPC drives
`configure_layouts` with final G2/G3 block counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import kvbm
import torch
from kvbm.v2.vllm import KvbmVllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.model_executor.models.utils import extract_layer_index

# Import KvbmRuntime and ConnectorWorker from Rust bindings (requires v2 feature)
if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
    ConnectorWorker = kvbm.v2.ConnectorWorker
else:
    KvbmRuntime = None
    ConnectorWorker = None

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheConfig


@dataclass
class VeloPeerMetadata(KVConnectorHandshakeMetadata):
    """
    Velo peer info exported by a worker for the leader handshake.

    The two fields map directly to `velo::PeerInfo { instance_id, worker_address }`
    on the Rust side. The leader consumes these in `register_worker` and
    registers the worker with the Velo messenger so subsequent RPCs
    (layout config, connector metadata, offload acks) can be routed.

    Attributes:
        instance_id: 16-byte UUID identifying the worker's Velo instance.
        worker_address: JSON-serialized `velo::WorkerAddress` of the worker peer.
    """

    instance_id: bytes  # 16-byte UUID
    worker_address: bytes  # JSON-serialized velo::WorkerAddress


class SchedulerConnectorWorker:
    """
    vLLM KV-connector worker backed by the Rust ConnectorWorker.

    Responsibilities on the worker side:
    - Build a `KvbmRuntime` + `ConnectorWorker` and export `VeloPeerMetadata`
      for the leader handshake.
    - Register vLLM's KV cache tensors with NIXL/UCX (deferred — the actual
      NIXL bind happens when the leader's init flow resolves G2/G3 block counts).
    - Drive the per-forward-pass dance on every layer:
        * `start_load_kv` → if the bound metadata carries an intra-pass
          onboard request, launch the per-layer H2D on a dedicated stream
          and record a CudaEvent per layer.
        * `wait_for_layer_load(layer_i, torch_stream)` → insert a
          `cuStreamWaitEvent` on the torch compute stream so attention for
          that layer cannot start before its KV slots are populated.
        * `save_kv_layer(layer_i, torch_stream)` → on the final layer,
          arm the forward-pass completion event the leader is waiting on.
    - Surface finished request IDs and failed-onboarding block IDs back
      to vLLM for state cleanup.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kvbm_config: KvbmVllmConfig,
        kv_cache_config: KVCacheConfig,
        **kwargs,
    ):
        """Initialize the scheduler connector worker."""
        # Check that v2 feature is available
        if not kvbm.v2.is_available():
            raise ImportError(
                "SchedulerConnectorWorker requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )

        self.vllm_config = vllm_config
        self.kvbm_config = kvbm_config
        self.vllm_kv_cache_config = kv_cache_config
        self.kvbm_override_config = kwargs.get("kvbm_override_config", None)
        self.device_id = None

        # Events
        # Map of layer name to onboarding event
        # This is used for intra-pass onboarding
        self.layer_onboarding_events = {}
        self.layer_offloading_events = {}

        # Build KvbmRuntime (Velo messenger + tokio)
        self.runtime = KvbmRuntime.build_worker(self.kvbm_override_config)

        # Create the Rust ConnectorWorker that handles NIXL registration
        self.worker = ConnectorWorker(self.runtime)

        # Store peer info for handshake
        instance_id, worker_addr = self.runtime.peer_info()
        self._handshake_metadata = VeloPeerMetadata(
            instance_id=instance_id,
            worker_address=worker_addr,
        )

        # Will be set during register_kv_caches
        self._num_device_blocks: Optional[int] = None
        self._num_layers: int = 0
        self._last_layer_name: Optional[str] = None

        print(
            f"SchedulerConnectorWorker initialized with Velo instance: {instance_id.hex()[:8]}...",
            flush=True,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register KV caches with NIXL for RDMA transfers.

        This registers the KV cache tensors with NIXL via the UCX backend,
        enabling remote GPU-to-GPU transfers.
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
        self.ordered_kv_caches = ordered_kv_caches

        # Create a mapping of layer name to layer index
        self.layer_name_to_index = {
            item[0]: i for i, item in enumerate(ordered_kv_caches)
        }

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
        # For NHD layout: [2 (K/V), num_blocks, block_size, num_heads, head_size]
        # For HND layout: [2 (K/V), num_blocks, num_heads, block_size, head_size]
        num_device_blocks = max(shape[0], shape[1])
        page_size = self.vllm_config.cache_config.block_size
        dtype_width_bytes = self.kvbm_config.cache_dtype_bytes()

        config_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        if num_device_blocks != config_gpu_blocks:
            print(
                f"Warning: num_device_blocks from tensor ({num_device_blocks}) "
                f"!= config.num_gpu_blocks ({config_gpu_blocks}). "
                f"Using tensor-derived value."
            )

        # Phase 2A: Register KV caches with NIXL via Rust binding
        # This caches tensor state for deferred NIXL registration
        # The actual NIXL registration happens when the leader triggers
        # initialization via bind_connector_metadata()
        self.worker.register_kv_caches(
            tensors,
            num_device_blocks,
            page_size,
            dtype_width_bytes,
        )

        # Store device block count and last layer name for later use
        self._num_device_blocks = num_device_blocks
        self._num_layers = len(tensors)

        # Get the last layer name from the ordered list
        self._last_layer_name = ordered_kv_caches[-1][0] if ordered_kv_caches else None
        print(
            f"[DEBUG] register_kv_caches: _last_layer_name set to: {self._last_layer_name}"
        )

        print("[KVBM] KV caches registered (deferred mode)")
        print(f"  - Num device blocks: {num_device_blocks}")
        print(f"  - Num layers: {len(tensors)}")
        print(f"  - Page size: {page_size}")
        print(f"  - Dtype width bytes: {dtype_width_bytes}")
        print(f"  - Shape: {shape}")
        print("[KVBM] Waiting for leader to trigger initialization...")

    def bind_connector_metadata(self, data: bytes) -> bool:
        """
        Bind connector metadata from the leader.

        Returns:
            True if metadata should be bound, False otherwise.
        """
        return self.worker.bind_connector_metadata(data)

    def clear_connector_metadata(self) -> None:
        """
        Release the per-iteration connector metadata held by the Rust worker.

        Called after every forward pass. Drops the currently-bound metadata
        and resets the intra-pass onboard / forward-pass-completion flags
        so the next `bind_connector_metadata` starts from a clean state.
        """
        self.worker.clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading KV cache

        If the bound metadata dictates that we should
        """
        self.worker.start_load_kv()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Notify the Rust worker that vLLM just finished attending `layer_name`.

        Always callable — the Rust side returns early unless this layer is
        the last one and the bound metadata carries a forward-pass completion
        event. When both conditions hold, the worker records a CudaEvent on
        the current torch stream and spawns an async task that waits on it
        and then fires the Velo active message back to the leader — that
        message is the precondition the leader's offload engine is waiting
        on before reading from the freshly-written G1 blocks.
        """
        layer_index = self.layer_name_to_index[layer_name]

        # Get the current CUDA stream handle
        stream = torch.cuda.current_stream()
        stream_handle = stream.cuda_stream

        # Call Rust - returns early if no action needed for this layer
        self.worker.save_kv_layer(layer_index, stream_handle)

    def wait_for_layer_load(
        self,
        layer_name: str,
    ) -> None:
        """
        Wait for a specific layer's KV cache load to complete.

        If intra-pass onboarding was triggered, this inserts a cudaStreamWaitEvent
        on the current torch stream to synchronize with the layer's onboard completion.
        """
        layer_index = self.layer_name_to_index[layer_name]

        # Get the current CUDA stream handle
        stream = torch.cuda.current_stream()
        stream_handle = stream.cuda_stream

        # Call Rust - returns early if no intra-pass onboarding is active
        self.worker.wait_for_layer_load(layer_index, stream_handle)

    def wait_for_save(self) -> None:
        """
        Intentional Python-side no-op.

        The forward-pass completion event armed in `save_kv_layer` is
        awaited asynchronously on the Rust side via the Velo messenger,
        so there is nothing for vLLM's synchronous `wait_for_save` hook
        to block on here.
        """
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
        # print(
        #     f"SchedulerConnectorWorker.get_finished called with {len(finished_req_ids)} finished requests"
        # )
        return self.worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Returns empty set - no load errors tracked."""
        return self.worker.get_failed_onboarding()

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata:
        """
        Return this worker's Velo peer info for the leader handshake.

        Returns:
            `VeloPeerMetadata` carrying the worker's Velo `instance_id`
            (16-byte UUID) and JSON-serialized `velo::WorkerAddress`. The
            leader consumes these in `register_worker` to register us with
            its Velo messenger.
        """
        return self._handshake_metadata
