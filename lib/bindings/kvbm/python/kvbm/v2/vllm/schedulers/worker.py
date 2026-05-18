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

# ---------------------------------------------------------------------------
# Per-layer KV axis discovery
# ---------------------------------------------------------------------------
#
# Both registration paths need to identify named axes in the KV cache tensor:
# - LW (`register_kv_caches`) needs `inner_dim` (the packed `heads*head_size`
#   tail), which depends on where the backend places those axes.
# - FC (`register_cross_layers_kv_cache`) needs to verify that the backend's
#   stride-order permutation produces the byte layout
#   `[num_blocks, num_layers, K/V, page_size, heads, head_size]` that
#   `FullyContiguousLayout` assumes.
#
# Both can be served by probing `attn_backend.get_kv_cache_shape(...)` with
# distinct marker values and reading back each axis's logical position.
# FC additionally consults `get_kv_cache_stride_order(include_num_layers=True)`
# to verify the physical permutation; LW just needs the positions.

# Unique per-axis probe values. `block_size` must be a multiple of 16 (FA,
# FlashInfer, Triton all enforce this in `get_kv_cache_shape`); the others
# must be != 2 so the K/V dim (always literally 2) is unambiguous.
_PROBE_BLOCKS = 1024
_PROBE_PAGE_SIZE = 32
_PROBE_KVHEADS = 7
_PROBE_HEAD_SIZE = 64
_KV_DIM_VALUE = 2  # K/V axis is always size 2 in standard attention


@dataclass(frozen=True)
class KvAxisMap:
    """Logical positions of each named axis in a per-layer KV cache shape.

    Indices are into the per-layer shape returned by
    `attn_backend.get_kv_cache_shape(...)`. For cross-layer tensors (where
    vLLM prepends a `num_layers` axis at logical position 0), every index
    here should be shifted by +1 when comparing against a stride_order
    that was obtained with `include_num_layers_dimension=True`.
    """

    blocks: int  # axis carrying num_blocks
    kv: int  # K/V outer axis (always size 2 for standard attention)
    page_size: int  # axis carrying block_size
    heads: int  # axis carrying num_kv_heads
    head_size: int  # axis carrying head_size


def probe_per_layer_axes(attn_backend: type) -> KvAxisMap:
    """Discover the logical position of each named axis in a backend's
    per-layer KV cache shape.

    Calls `attn_backend.get_kv_cache_shape(num_blocks=1024, block_size=32,
    num_kv_heads=7, head_size=64)` and locates each marker. Used by both
    the LW path (to compute `inner_dim` from the *actual* heads/head_size
    positions, not a hardcoded `shape[3:]` assumption) and the FC path
    (to validate the stride-order permutation).

    Standard attention only — MLA's per-layer shape has no K/V axis and is
    handled separately by callers.
    """
    per_layer_shape = tuple(
        attn_backend.get_kv_cache_shape(
            num_blocks=_PROBE_BLOCKS,
            block_size=_PROBE_PAGE_SIZE,
            num_kv_heads=_PROBE_KVHEADS,
            head_size=_PROBE_HEAD_SIZE,
        )
    )

    def _find(marker: int, name: str) -> int:
        try:
            return per_layer_shape.index(marker)
        except ValueError as e:
            raise RuntimeError(
                f"KVBM cannot locate {name} (marker={marker}) in "
                f"{attn_backend.__name__} per-layer shape {per_layer_shape}"
            ) from e

    return KvAxisMap(
        blocks=_find(_PROBE_BLOCKS, "num_blocks"),
        kv=_find(_KV_DIM_VALUE, "K/V"),
        page_size=_find(_PROBE_PAGE_SIZE, "block_size"),
        heads=_find(_PROBE_KVHEADS, "num_kv_heads"),
        head_size=_find(_PROBE_HEAD_SIZE, "head_size"),
    )


@dataclass
class KvTensorLayout:
    """Semantic description of the KV cache tensor layout.

    Python derives this from VllmConfig (which has the model architecture)
    so that Rust does not need to guess from raw tensor shapes.

    For both MLA and standard attention, dimensions are computed explicitly
    from vLLM config and passed down to Rust. Tensor shape is only used for
    sanity validation (e.g., block-axis presence / packed-tail checks).

    Mirrors the v1 helper in
    lib/bindings/kvbm/python/kvbm/vllm_integration/connector_worker.py.
    """

    outer_dim: Optional[int]
    inner_dim: Optional[int]

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: "VllmConfig",
        shape: "torch.Size",
        num_device_blocks: int,
        use_mla: bool = False,
        attn_backend: Optional[type] = None,
    ) -> "KvTensorLayout":
        """Resolve `outer_dim` / `inner_dim` from vLLM config + tensor shape.

        When `attn_backend` is supplied, `probe_per_layer_axes` is used to
        locate the heads / head_size axes — robust against backends that
        permute the per-layer tail. When it's omitted (the current LW call
        site, since vLLM v1 doesn't carry the backend on the
        `register_kv_caches` callback), we fall back to the documented
        `shape[3:]` assumption that heads and head_size are the last two
        axes; this is true for FA / FlashInfer / Triton today and is
        validated by the config cross-check below.
        """
        if use_mla:
            # MLA tensors are 3D: [num_blocks, page_size, head_size].
            # No outer_dim axis — K and V are fused into a single latent.
            return cls(outer_dim=1, inner_dim=int(shape[-1]))
        # Standard attention:
        # - outer_dim is K/V axis = 2
        # - inner_dim is per-rank kv feature width
        #
        # vLLM API surfaces can differ by version/model family:
        # `get_total_num_kv_heads()` may be global or already per-rank.
        # Resolve this robustly by checking which config interpretation matches
        # the observed packed tail in tensor shape.
        total_kv_heads = int(vllm_config.model_config.get_total_num_kv_heads())
        head_size = int(vllm_config.model_config.get_head_size())
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        if tp_size <= 0:
            raise ValueError(f"Invalid tensor_parallel_size={tp_size}")
        outer_dim = 2
        if head_size <= 0 or total_kv_heads <= 0:
            raise ValueError(
                "Invalid standard attention inner_dim derived from config: "
                f"total_num_kv_heads={total_kv_heads}, head_size={head_size}"
            )

        # Sanity checks against actual tensor shape so we fail loudly if vLLM
        # changes layout semantics.
        if len(shape) < 4:
            raise ValueError(
                "Standard attention KV tensor must have >=4 dims; "
                f"got shape={tuple(shape)}"
            )
        if int(shape[0]) < num_device_blocks and int(shape[1]) < num_device_blocks:
            raise ValueError(
                "Cannot locate num_device_blocks in standard KV tensor shape; "
                f"shape[:2]={tuple(shape[:2])}, num_device_blocks={num_device_blocks}"
            )

        # inner_dim from tensor shape: prefer probe-derived axis positions
        # when the backend is available, otherwise use the shape[3:] tail.
        if attn_backend is not None:
            axes = probe_per_layer_axes(attn_backend)
            inferred_inner_from_shape = int(shape[axes.heads]) * int(
                shape[axes.head_size]
            )
        else:
            inferred_inner_from_shape = int(math.prod(shape[3:]))

        config_inner_candidates = [total_kv_heads * head_size]
        if total_kv_heads % tp_size == 0:
            config_inner_candidates.append((total_kv_heads // tp_size) * head_size)

        matched_inner = None
        for candidate in config_inner_candidates:
            if candidate == inferred_inner_from_shape:
                matched_inner = candidate
                break

        if matched_inner is None:
            raise ValueError(
                "KV layout mismatch between vLLM config and tensor shape: "
                f"config_inner_candidates={config_inner_candidates}, "
                f"shape_inner_dim={inferred_inner_from_shape}, "
                f"shape={tuple(shape)}, total_num_kv_heads={total_kv_heads}, "
                f"head_size={head_size}, tp_size={tp_size}"
            )
        inner_dim = matched_inner

        return cls(outer_dim=outer_dim, inner_dim=inner_dim)


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

        # Extract parameters.
        # Prefer vLLM config's GPU block count; fall back to tensor-derived only
        # when config is unavailable (older bring-up edge cases).
        use_mla = getattr(self.vllm_config.model_config, "use_mla", False)
        config_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        if config_gpu_blocks is None or config_gpu_blocks <= 0:
            num_device_blocks = int(shape[0] if use_mla else max(shape[0], shape[1]))
            print(
                "Warning: cache_config.num_gpu_blocks unavailable; "
                f"falling back to tensor-derived num_device_blocks={num_device_blocks}"
            )
        else:
            num_device_blocks = int(config_gpu_blocks)

        # Derive explicit layout semantics from vLLM model config + tensor shape.
        layout = KvTensorLayout.from_vllm_config(
            self.vllm_config, shape, num_device_blocks, use_mla
        )

        page_size = self.vllm_config.cache_config.block_size
        dtype_width_bytes = self.kvbm_config.cache_dtype_bytes()
        if num_device_blocks <= 0:
            raise ValueError(f"Invalid num_device_blocks={num_device_blocks}")
        if page_size <= 0:
            raise ValueError(f"Invalid page_size={page_size}")
        if dtype_width_bytes <= 0:
            raise ValueError(f"Invalid dtype_width_bytes={dtype_width_bytes}")

        if shape[0] < num_device_blocks and shape[1] < num_device_blocks:
            raise ValueError(
                "Cannot locate num_device_blocks in KV tensor shape; "
                f"shape[:2]={tuple(shape[:2])}, num_gpu_blocks={num_device_blocks}"
            )

        # Strong startup contract:
        # bytes_per_block from layout math must equal bytes_per_block implied by the
        # concrete tensor storage. This catches subtle TP/layout drift early.
        layer_total_bytes_from_shape = int(math.prod(shape)) * dtype_width_bytes
        if layer_total_bytes_from_shape % num_device_blocks != 0:
            raise ValueError(
                "KV tensor total bytes is not divisible by num_device_blocks: "
                f"shape={tuple(shape)}, dtype_width_bytes={dtype_width_bytes}, "
                f"layer_total_bytes={layer_total_bytes_from_shape}, "
                f"num_device_blocks={num_device_blocks}"
            )
        bytes_per_block_from_shape = layer_total_bytes_from_shape // num_device_blocks
        bytes_per_block_from_layout = (
            int(layout.outer_dim)
            * page_size
            * int(layout.inner_dim)
            * dtype_width_bytes
        )
        if bytes_per_block_from_shape != bytes_per_block_from_layout:
            raise ValueError(
                "KV bytes_per_block mismatch between tensor shape and derived layout: "
                f"shape={tuple(shape)}, "
                f"bytes_per_block_from_shape={bytes_per_block_from_shape}, "
                f"bytes_per_block_from_layout={bytes_per_block_from_layout}, "
                f"outer_dim={layout.outer_dim}, inner_dim={layout.inner_dim}, "
                f"page_size={page_size}, dtype_width_bytes={dtype_width_bytes}, "
                f"num_device_blocks={num_device_blocks}"
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
        # per-layer logical shape differently — FA NHD returns
        # `(2, num_blocks, block_size, h, d)` (K/V first) while FlashInfer
        # and Triton NHD return `(num_blocks, 2, block_size, h, d)` (blocks
        # first). After vLLM prepends a num_layers axis, the *logical*
        # position of num_blocks/K/V therefore differs by backend, but the
        # *physical* layout we need is the same. Use the shared probe to
        # discover per-layer axis positions, shift by +1 for the prepended
        # num_layers axis, then assert the stride-order permutation lands
        # every axis where FullyContiguousLayout expects it.
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

        axes = probe_per_layer_axes(attn_backend)

        # vLLM prepends num_layers at logical position 0, so every per-layer
        # axis position bumps by 1 in the cross-layer logical order.
        num_blocks_log = axes.blocks + 1
        kv_log = axes.kv + 1
        block_size_log = axes.page_size + 1
        heads_log = axes.heads + 1
        head_size_log = axes.head_size + 1

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
                f"per-layer axes {axes} fails: {failed}. "
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
