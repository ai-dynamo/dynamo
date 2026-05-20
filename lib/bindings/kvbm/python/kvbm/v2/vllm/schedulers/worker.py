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
from kvbm.v2.vllm.dim_probe import (
    KvBlockLayout,
    KvDim,
    build_dim_layout_from_tensor,
    derive_block_layout,
    select_fc_variant,
)
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)

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

        # Cache the deduplicated AttentionBackend classes for this model.
        # Sentinel-probed in `register_kv_caches` to derive `KvDimLayout` /
        # `KvBlockLayout`. Mirrors NIXL's pattern at
        # `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py:316`.
        self._attn_backends = get_current_attn_backends(vllm_config)
        if not self._attn_backends:
            # `get_current_attn_backends` is supposed to fall back to
            # `get_attn_backend(...)` when `static_forward_context` is empty
            # (`utils.py:873-886`); an empty result here indicates an
            # unexpected vLLM init order — fail loudly so register_kv_caches
            # doesn't blow up later with a confusing IndexError.
            raise RuntimeError(
                "get_current_attn_backends(vllm_config) returned an empty "
                "list — vLLM static_forward_context appears to be empty at "
                "connector init time. File a bug."
            )

        # Build layer_name → tensor_index from `kv_cache_config.kv_cache_tensors`,
        # not from `extract_layer_index(name)` over the kv_caches dict. The
        # vLLM model runner registers tensors keyed by layer name; multiple
        # logical layers may share one cache tensor (`shared_by`), and the
        # tensor list we hand to Rust must be ordered by `kv_cache_tensors`
        # index — so save_kv_layer / wait_for_layer_load lookups stay
        # consistent with the per-tensor index. Built once here so cross-
        # layer mode (M2) shares the same source of truth.
        self.layer_name_to_index: dict[str, int] = {}
        for i, kct in enumerate(kv_cache_config.kv_cache_tensors):
            for name in kct.shared_by:
                self.layer_name_to_index[name] = i

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

        Drives the labelled-axis path: probe the per-layer
        `AttentionBackend.get_kv_cache_shape(...)` with sentinel values to
        learn each axis's role (`Block`, `Outer`, `Page`, `HeadCount`,
        `HeadSize`/`Payload`), classify NHD vs HND from
        `get_kv_cache_stride_order(False)`, validate every tensor's
        `shape()` and `numel()` against the labelled sizes, then hand
        labelled `(dims, sizes, block_layout)` to Rust. Replaces the
        previous shape-inference codepath.
        """
        if not kv_caches:
            print("Warning: register_kv_caches called with empty kv_caches")
            return

        kct_list = self.vllm_kv_cache_config.kv_cache_tensors
        groups = self.vllm_kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise NotImplementedError(
                f"hybrid kv_cache_groups not supported (found {len(groups)} "
                f"groups); KVBM v2 currently assumes a single uniform group"
            )
        if len(self._attn_backends) != 1:
            raise NotImplementedError(
                f"per-layer-divergent attn_backends not supported "
                f"(found {len(self._attn_backends)}); KVBM v2 currently "
                f"requires a single backend across all attention layers"
            )

        backend = self._attn_backends[0]
        use_mla = bool(getattr(self.vllm_config.model_config, "use_mla", False))
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        # Tensor list ordered by `kv_cache_tensors` index — same source as
        # `layer_name_to_index` (built in __init__). Multiple layer names
        # may share the same physical tensor (`shared_by`); we register the
        # tensor once, indexed by its kv_cache_tensors position.
        tensors: list[torch.Tensor] = []
        for kct in kct_list:
            # Any name in shared_by points to the same underlying tensor.
            tensors.append(kv_caches[kct.shared_by[0]])

        # All tensors must share a shape — same restriction as before, but
        # the error now points at the offending layer index.
        first_shape = tuple(tensors[0].shape)
        for i, t in enumerate(tensors[1:], start=1):
            if tuple(t.shape) != first_shape:
                raise NotImplementedError(
                    f"hybrid models with per-layer-divergent shapes are not "
                    f"supported yet (tensor {i} has shape {tuple(t.shape)}, "
                    f"tensor 0 has shape {first_shape})"
                )

        # Probe the backend with sentinels and bind the labels to the
        # ACTUAL tensor shape — not `kv_cache_spec.block_size` /
        # `num_gpu_blocks` — so we sidestep vLLM's `kernel_block_size !=
        # spec.block_size` case (`kv_connector_model_runner_mixin.py:235-238`).
        dims, sizes = build_dim_layout_from_tensor(
            backend,
            tensor_shape=first_shape,
            cache_dtype_str=cache_dtype_str,
            use_mla=use_mla,
        )
        block_layout: KvBlockLayout = derive_block_layout(backend, dims)

        # Per-layer fast-fail in Python so the error fires before crossing
        # the FFI boundary. (The Rust side checks the same invariants.)
        expected_numel = math.prod(sizes)
        for layer_name, kct, tensor in zip(
            (kct.shared_by[0] for kct in kct_list), kct_list, tensors
        ):
            if tuple(tensor.shape) != tuple(sizes):
                raise RuntimeError(
                    f"layer {layer_name}: tensor.shape {tuple(tensor.shape)} "
                    f"!= probed-and-labelled sizes {tuple(sizes)}"
                )
            if tensor.numel() != expected_numel:
                raise RuntimeError(
                    f"layer {layer_name}: tensor.numel() {tensor.numel()} "
                    f"!= product of labelled sizes {expected_numel}"
                )

        # Find the Block axis size — the authoritative `num_device_blocks`
        # value. Cross-checked against `kv_cache_config` for visibility but
        # the layout's value wins (matches the previous
        # `tensor-derived value wins over config` behaviour).
        block_idx = dims.index(KvDim.Block)
        num_device_blocks = sizes[block_idx]

        config_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        if config_gpu_blocks is not None and num_device_blocks != config_gpu_blocks:
            print(
                f"Warning: num_device_blocks from labelled tensor "
                f"({num_device_blocks}) != config.num_gpu_blocks "
                f"({config_gpu_blocks}). Using labelled value."
            )

        dtype_width_bytes = self.kvbm_config.cache_dtype_bytes()

        # Hand off to Rust. Strings (not pyclass enums) cross the FFI;
        # `kvbm-py3` parses them via `parse_kv_dim` / `parse_kv_block_layout`.
        self.worker.register_kv_caches(
            tensors,
            num_device_blocks,
            dtype_width_bytes,
            [d.value for d in dims],
            list(sizes),
            block_layout.value,
        )

        self._num_device_blocks = num_device_blocks
        self._num_layers = len(tensors)
        # Last "layer name" = a representative name for the trailing tensor.
        # save_kv_layer's "is this the last layer?" check on the Rust side
        # already keys off layer index, so this is just for logging.
        self._last_layer_name = kct_list[-1].shared_by[0] if kct_list else None

        print(
            f"[KVBM] KV caches registered (deferred mode): "
            f"backend={backend.__name__}, "
            f"cache_dtype_str={cache_dtype_str}, use_mla={use_mla}, "
            f"dims={[d.value for d in dims]}, sizes={list(sizes)}, "
            f"block_layout={block_layout.value}, "
            f"num_device_blocks={num_device_blocks}, "
            f"num_layers={self._num_layers}, "
            f"dtype_bytes={dtype_width_bytes}"
        )
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

        Probes `attn_backend` with `dim_probe.probe_kv_dim_layout(...,
        include_num_layers=True)` for per-axis labels and classifies
        NHD/HND with `derive_block_layout(..., include_num_layers=True)`,
        then hands labelled `(dims, sizes, block_layout)` to Rust. The
        physical byte layout `[num_blocks, num_layers, K/V, page_size,
        heads, head_size]` that `FullyContiguousLayout` assumes is
        asserted defensively here (labels alone don't prove the physical
        permutation is what FC expects); FP8 NHD variants that
        interleave the heads dimension are rejected with a clear error.
        """
        print(
            f"[KVBM] register_cross_layers_kv_cache: shape={tuple(kv_cache.shape)}, "
            f"dtype={kv_cache.dtype}, device={kv_cache.device}, "
            f"backend={attn_backend.__name__}",
            flush=True,
        )

        # Hybrid-model guards. Mirror the per-layer path's
        # `register_kv_caches` checks (raise NotImplementedError, not
        # assert) so any failure has the same shape and message style
        # regardless of which registration path vLLM picked. KVBM v2 does
        # not support hybrid models in either path; this is the FC-side
        # rejection.
        groups = self.vllm_kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise NotImplementedError(
                f"hybrid kv_cache_groups not supported in cross-layer "
                f"registration (found {len(groups)} groups); KVBM v2 "
                f"currently assumes a single uniform group"
            )
        if len(self._attn_backends) != 1:
            raise NotImplementedError(
                f"per-layer-divergent attn_backends not supported in "
                f"cross-layer registration (found {len(self._attn_backends)}); "
                f"KVBM v2 currently requires a single backend across all "
                f"attention layers"
            )

        if not kv_cache.is_contiguous() or kv_cache.storage_offset() != 0:
            raise RuntimeError(
                f"KVBM cross-layer tensor must be contiguous with offset 0; "
                f"got is_contiguous={kv_cache.is_contiguous()}, "
                f"storage_offset={kv_cache.storage_offset()}"
            )

        shape = tuple(kv_cache.shape)
        use_mla = bool(getattr(self.vllm_config.model_config, "use_mla", False))
        if use_mla:
            raise RuntimeError(
                "KVBM cross-layer (fully-contiguous) registration does not "
                "support MLA backends — the FC layout assumes K/V outer "
                "split (outer_dim=2)."
            )
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        # Probe with `include_num_layers=True` — returns labels in the
        # PHYSICAL byte order that matches vLLM's contiguous FC tensor
        # shape (the probe applies the cross-layer stride permutation
        # internally). Sizes therefore pair index-for-index with
        # `kv_cache.shape`.
        dims, sizes = build_dim_layout_from_tensor(
            attn_backend,
            tensor_shape=shape,
            cache_dtype_str=cache_dtype_str,
            use_mla=False,
            include_num_layers=True,
        )

        # Match the backend's physical cross-layer layout to one of
        # `FullyContiguousLayout`'s supported orderings (OperationalNHD,
        # OperationalHND, Universal). `None` means the backend's natural
        # FC byte order doesn't match any variant we know how to build —
        # the caller should have routed to the per-layer (LW) path via
        # `prefer_cross_layer_blocks=False`.
        block_layout: Optional[KvBlockLayout] = select_fc_variant(
            attn_backend, cache_dtype_str=cache_dtype_str
        )
        if block_layout is None:
            raise RuntimeError(
                f"KVBM cross-layer registration: backend "
                f"{attn_backend.__name__} cross-layer physical layout {dims} "
                f"matches no supported FullyContiguousLayout variant "
                f"(OperationalNHD, OperationalHND, Universal). "
                f"Set KVBM_PREFER_FULLY_CONTIGUOUS_BLOCKS=false to use the "
                f"per-layer (LW) path."
            )

        # Defensive cross-check: the probe-derived NHD/HND classification
        # should agree with the FC variant's NHD/HND character. A
        # disagreement indicates a programming error in dim_probe.
        derived = derive_block_layout(attn_backend, dims, include_num_layers=True)
        if (
            derived != KvBlockLayout.Unknown
            and block_layout != KvBlockLayout.Universal
            and derived != block_layout
        ):
            raise RuntimeError(
                f"KVBM cross-layer: dim_probe internal mismatch — "
                f"select_fc_variant returned {block_layout.value} but "
                f"derive_block_layout returned {derived.value} for "
                f"{attn_backend.__name__} dims={dims}. "
                f"File a bug against kvbm.v2.vllm.dim_probe."
            )

        # Block axis size from labels is authoritative; cross-check vLLM
        # config for visibility.
        block_idx = dims.index(KvDim.Block)
        num_device_blocks = sizes[block_idx]
        config_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        if config_gpu_blocks is not None and num_device_blocks != config_gpu_blocks:
            print(
                f"[KVBM] Warning: cross-layer num_device_blocks from labelled tensor "
                f"({num_device_blocks}) != config.num_gpu_blocks "
                f"({config_gpu_blocks}); using labelled value."
            )

        num_layers = sizes[dims.index(KvDim.Layer)]
        dtype_width_bytes = self.kvbm_config.cache_dtype_bytes()

        self.worker.register_cross_layers_kv_cache(
            kv_cache,
            num_device_blocks,
            dtype_width_bytes,
            [d.value for d in dims],
            list(sizes),
            block_layout.value,
        )

        # Downstream save_kv_layer / wait_for_layer_load identify the layer
        # by its name. With cross-layer we have no per-layer tensor dict to
        # extract the name order from, so we read it from the kv_cache_config.
        # Group count was already validated above; just pull the layer names.
        layer_names = list(groups[0].layer_names)
        if len(layer_names) != num_layers:
            raise RuntimeError(
                f"cross-layer layer_names count ({len(layer_names)}) "
                f"!= labelled num_layers ({num_layers})"
            )

        self.layer_name_to_index = {name: i for i, name in enumerate(layer_names)}
        self._num_device_blocks = num_device_blocks
        self._num_layers = num_layers
        self._last_layer_name = layer_names[-1] if layer_names else None

        print(
            "[KVBM] Cross-layer KV cache registered (deferred mode): "
            f"backend={attn_backend.__name__}, "
            f"dims={[d.value for d in dims]}, sizes={list(sizes)}, "
            f"block_layout={block_layout.value}, "
            f"num_device_blocks={num_device_blocks}, num_layers={num_layers}, "
            f"dtype_width_bytes={dtype_width_bytes}",
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
