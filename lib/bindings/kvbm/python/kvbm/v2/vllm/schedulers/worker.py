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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import kvbm
import torch
from kvbm.v2.vllm import KvbmVllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.model_executor.models.utils import extract_layer_index


@dataclass
class KvTensorLayout:
    """Semantic description of the KV cache tensor layout.

    Python derives this from VllmConfig (which has the model architecture)
    so that Rust doesn't have to guess from raw tensor shapes.

    For MLA models, outer_dim and inner_dim are set explicitly because Rust's
    shape-based inference cannot distinguish the fused KV latent axis from the
    block/page axis. For standard attention the fields are None, which tells
    Rust to fall back to its own contiguity-based inference (already correct).

    Mirrors the v1 helper in
    lib/bindings/kvbm/python/kvbm/vllm_integration/connector_worker.py.
    """

    outer_dim: Optional[int]  # None = let Rust infer; 1 for MLA
    inner_dim: Optional[int]  # None = let Rust infer; head_size for MLA

    @classmethod
    def from_vllm_config(
        cls, shape: "torch.Size", use_mla: bool = False
    ) -> "KvTensorLayout":
        if use_mla:
            # MLA tensors are 3D: [num_blocks, page_size, head_size].
            # No outer_dim axis — K and V are fused into a single latent.
            return cls(outer_dim=1, inner_dim=int(shape[-1]))
        # Standard attention: Rust already infers outer_dim/inner_dim correctly
        # from tensor shape. Don't guess here — the block dimension can be at
        # position 0 or 1 depending on the attention backend.
        return cls(outer_dim=None, inner_dim=None)


# Import KvbmRuntime and ConnectorWorker from Rust bindings (requires v2 feature)
if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
    ConnectorWorker = kvbm.v2.ConnectorWorker
else:
    KvbmRuntime = None
    ConnectorWorker = None

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
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

        # Derive layout semantics from the vLLM model config so Rust doesn't
        # have to guess the outer_dim/inner_dim from potentially-ambiguous
        # tensor shapes (MLA caches are 3D without a K/V axis).
        use_mla = getattr(self.vllm_config.model_config, "use_mla", False)
        layout = KvTensorLayout.from_vllm_config(shape, use_mla)

        # Extract parameters.
        # Standard attention: [2 (K/V), num_blocks, ...] or [num_blocks, 2, ...]
        #   — block dim is whichever of shape[0]/shape[1] is larger.
        # MLA: [num_blocks, page_size, head_size] — block dim is always axis 0.
        num_device_blocks = shape[0] if use_mla else max(shape[0], shape[1])
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
            outer_dim=layout.outer_dim,
            inner_dim=layout.inner_dim,
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

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type["AttentionBackend"],
    ) -> None:
        """
        Register a single cross-layer KV cache tensor with NIXL.

        Called by vLLM when `prefer_cross_layer_blocks` is True and the
        backend supports a uniform layout. The tensor is the single contiguous
        allocation produced by `allocate_uniform_kv_caches` — its logical
        shape is `[num_layers, ...]` permuted by the backend's stride order.
        We require the physical byte layout to be
        `[num_blocks, num_layers, outer_dim=2, page_size, num_kv_heads, head_size]`
        (the last two collapse into `inner_dim` for KVBM). Both FlashAttention
        NHD and FlashInfer/Triton NHD produce this physical layout despite
        differing logical orderings; FP8 NHD variants that interleave the
        heads dimension are rejected with a clear error.
        """
        print(
            f"[KVBM] register_cross_layers_kv_cache: shape={tuple(kv_cache.shape)}, "
            f"dtype={kv_cache.dtype}, device={kv_cache.device}, "
            f"backend={attn_backend.__name__}",
            flush=True,
        )

        # FullyContiguousLayout requires the physical byte layout to be
        # `[num_blocks, num_layers, K/V, page_size, num_kv_heads, head_size]`
        # (the last two collapse into inner_dim). Backends order their
        # per-layer logical shape differently — FlashAttention NHD returns
        # `(2, num_blocks, block_size, h, d)` (K/V first) while FlashInfer
        # and Triton NHD return `(num_blocks, 2, block_size, h, d)` (blocks
        # first). After vLLM prepends a num_layers axis, the *logical*
        # position of num_blocks/K/V therefore differs by backend, but the
        # *physical* layout we need is the same. Probe the per-layer shape
        # with distinct markers to discover which logical axis each
        # dimension occupies, then assert the stride_order permutation
        # lands every axis where FullyContiguousLayout expects it.
        try:
            stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
        except (AttributeError, NotImplementedError) as e:
            raise RuntimeError(
                f"KVBM cross-layer registration requires "
                f"{attn_backend.__name__}.get_kv_cache_stride_order"
                f"(include_num_layers_dimension=True); got {type(e).__name__}: {e}"
            ) from e

        # Markers: each value is unique across the per-layer shape so we
        # can identify every axis via list.index(). The "2" marker is the
        # K/V outer dim itself — every other marker is non-2 to avoid
        # collisions. `block_size` must be a multiple of 16 (FA, FlashInfer,
        # Triton all enforce this in their get_kv_cache_shape probe), so we
        # use 1024 / 32 / 7 / 64 to stay valid across backends.
        BLOCK_MARKER, BSIZE_MARKER, KVHEADS_MARKER, HEAD_MARKER = 1024, 32, 7, 64
        per_layer_shape = tuple(
            attn_backend.get_kv_cache_shape(
                num_blocks=BLOCK_MARKER,
                block_size=BSIZE_MARKER,
                num_kv_heads=KVHEADS_MARKER,
                head_size=HEAD_MARKER,
            )
        )

        def _logical_axis(marker: int) -> int:
            # +1 because vLLM prepends num_layers at logical position 0.
            try:
                return per_layer_shape.index(marker) + 1
            except ValueError as e:
                raise RuntimeError(
                    f"KVBM cross-layer registration cannot locate marker "
                    f"{marker} in {attn_backend.__name__} per-layer shape "
                    f"{per_layer_shape}"
                ) from e

        num_blocks_log = _logical_axis(BLOCK_MARKER)
        kv_log = _logical_axis(2)  # the literal K/V outer dim
        block_size_log = _logical_axis(BSIZE_MARKER)
        heads_log = _logical_axis(KVHEADS_MARKER)
        head_size_log = _logical_axis(HEAD_MARKER)

        def _physical_axis(logical: int) -> int:
            return stride_order.index(logical)

        expected = {
            "num_blocks @ physical 0": _physical_axis(num_blocks_log) == 0,
            "num_layers @ physical 1": _physical_axis(0) == 1,
            "K/V        @ physical 2": _physical_axis(kv_log) == 2,
            "block_size @ physical 3": _physical_axis(block_size_log) == 3,
            "heads+head_size @ physical {4,5}": {
                _physical_axis(heads_log),
                _physical_axis(head_size_log),
            }
            == {4, 5},
        }
        if not all(expected.values()):
            failed = [name for name, ok in expected.items() if not ok]
            raise RuntimeError(
                f"KVBM cross-layer registration requires physical byte order "
                f"[num_blocks, num_layers, K/V, page_size, heads, head_size]; "
                f"{attn_backend.__name__} stride_order={stride_order} over "
                f"per-layer shape {per_layer_shape} fails: {failed}. "
                f"Use a NHD-compatible backend (FLASH_ATTN, FLASHINFER, "
                f"TRITON_ATTN) or disable cross-layer support."
            )

        if not kv_cache.is_contiguous() or kv_cache.storage_offset() != 0:
            raise RuntimeError(
                f"KVBM cross-layer tensor must be contiguous with offset 0; "
                f"got is_contiguous={kv_cache.is_contiguous()}, "
                f"storage_offset={kv_cache.storage_offset()}"
            )

        shape = tuple(kv_cache.shape)
        if len(shape) < 5:
            raise RuntimeError(
                f"KVBM cross-layer tensor must have at least 5 dims "
                f"(num_blocks, num_layers, outer, page_size, inner...); got shape={shape}"
            )

        num_blocks, num_layers, outer_dim, page_size = shape[:4]
        inner_dim = math.prod(shape[4:])

        # Cross-checks against vLLM config — catch a mismatch between the
        # tensor we got and what vLLM thinks it allocated.
        config_block_size = self.vllm_config.cache_config.block_size
        if page_size != config_block_size:
            raise RuntimeError(
                f"KVBM cross-layer page_size mismatch: tensor has {page_size}, "
                f"vllm_config.cache_config.block_size is {config_block_size}"
            )
        if outer_dim != 2:
            raise RuntimeError(
                f"KVBM cross-layer expects outer_dim=2 (K/V split); got {outer_dim}. "
                f"MLA backends should not reach this path."
            )

        config_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        if num_blocks != config_gpu_blocks:
            print(
                f"[KVBM] Warning: cross-layer num_blocks from tensor ({num_blocks}) "
                f"!= config.num_gpu_blocks ({config_gpu_blocks}); using tensor value."
            )

        dtype_width_bytes = self.kvbm_config.cache_dtype_bytes()

        self.worker.register_cross_layers_kv_cache(
            kv_cache,
            num_blocks,
            num_layers,
            outer_dim,
            page_size,
            inner_dim,
            dtype_width_bytes,
        )

        # Downstream save_kv_layer / wait_for_layer_load identify the layer
        # by its name. With cross-layer we have no per-layer tensor dict to
        # extract the name order from, so we read it from the kv_cache_config.
        # Single group is guaranteed by use_uniform_kv_cache.
        groups = self.vllm_kv_cache_config.kv_cache_groups
        assert (
            len(groups) == 1
        ), f"cross-layer registration expects exactly one KV cache group; got {len(groups)}"
        layer_names = list(groups[0].layer_names)
        assert (
            len(layer_names) == num_layers
        ), f"layer_names count ({len(layer_names)}) != num_layers ({num_layers})"

        self.layer_name_to_index = {name: i for i, name in enumerate(layer_names)}
        self._num_device_blocks = num_blocks
        self._num_layers = num_layers
        self._last_layer_name = layer_names[-1] if layer_names else None

        print(
            "[KVBM] Cross-layer KV cache registered (deferred mode):"
            f" num_blocks={num_blocks} num_layers={num_layers}"
            f" outer_dim={outer_dim} page_size={page_size} inner_dim={inner_dim}"
            f" dtype_width_bytes={dtype_width_bytes}",
            flush=True,
        )
        print("[KVBM] Waiting for leader to trigger initialization...", flush=True)

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
