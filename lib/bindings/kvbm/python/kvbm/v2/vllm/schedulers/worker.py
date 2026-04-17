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

from typing import TYPE_CHECKING, Optional

import kvbm
import torch
from kvbm.v2.connector.base import KvbmConnectorWorkerBase, VeloPeerMetadata
from kvbm.v2.vllm import KvbmVllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.model_executor.models.utils import extract_layer_index

if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
else:
    KvbmRuntime = None

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheConfig

# Re-export VeloPeerMetadata so existing imports from this module still work.
__all__ = ["VeloPeerMetadata", "SchedulerConnectorWorker"]


class SchedulerConnectorWorker(KvbmConnectorWorkerBase):
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
        kv_cache_config: "KVCacheConfig",
        **kwargs,
    ):
        super().__init__()

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

        self.layer_onboarding_events = {}
        self.layer_offloading_events = {}

        runtime = KvbmRuntime.build_worker(self.kvbm_override_config)
        self._init_worker(runtime)

        self._num_device_blocks: Optional[int] = None
        self._num_layers: int = 0
        self._last_layer_name: Optional[str] = None
        # Populated during register_kv_caches; maps layer name → zero-based index.
        self.layer_name_to_index: dict[str, int] = {}

        instance_id = self._handshake_metadata.instance_id
        print(
            f"SchedulerConnectorWorker initialized with Velo instance: {instance_id.hex()[:8]}...",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

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

        ordered_kv_caches = sorted(
            kv_caches.items(), key=lambda item: extract_layer_index(item[0])
        )
        self.layer_name_to_index = {
            item[0]: i for i, item in enumerate(ordered_kv_caches)
        }

        tensors = [tensor for _, tensor in ordered_kv_caches]
        first_tensor = tensors[0]
        shape = first_tensor.shape

        if not all(t.shape == shape for t in tensors):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

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

        self._worker.register_kv_caches(
            tensors,
            num_device_blocks,
            page_size,
            dtype_width_bytes,
        )

        self._num_device_blocks = num_device_blocks
        self._num_layers = len(tensors)
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

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor = None,
        attn_metadata: "AttentionMetadata" = None,
        **kwargs,
    ) -> None:
        """
        Notify the Rust worker that vLLM just finished attending `layer_name`.

        Always callable — the Rust side returns early unless this layer is
        the last one and the bound metadata carries a forward-pass completion
        event. When both conditions hold, the worker records a CudaEvent on
        the current torch stream and spawns an async task that fires the Velo
        active message back to the leader.
        """
        layer_index = self.layer_name_to_index[layer_name]
        stream_handle = torch.cuda.current_stream().cuda_stream
        self._worker.save_kv_layer(layer_index, stream_handle)

    # ------------------------------------------------------------------
    # vLLM-specific signature overrides
    # ------------------------------------------------------------------

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._worker.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Wait for a specific layer's KV cache load to complete.

        Inserts a cudaStreamWaitEvent on the current torch stream to
        synchronise with the layer's onboard completion event.
        """
        layer_index = self.layer_name_to_index[layer_name]
        stream_handle = torch.cuda.current_stream().cuda_stream
        self._worker.wait_for_layer_load(layer_index, stream_handle)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self._worker.get_finished()

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata:
        """
        Return this worker's Velo peer info for the leader handshake.

        Returns:
            ``VeloPeerMetadata`` carrying the worker's Velo ``instance_id``
            (16-byte UUID) and JSON-serialised ``velo::WorkerAddress``.
        """
        return self._handshake_metadata
