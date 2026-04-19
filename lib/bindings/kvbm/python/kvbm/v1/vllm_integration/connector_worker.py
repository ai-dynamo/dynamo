# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from kvbm.v1.utils import is_dyn_runtime_enabled
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext


# from kvbm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
# from kvbm.vllm_integration.rust import BlockManager
# from kvbm.vllm_integration.rust import (
#     KvConnectorMetadata as RustKvConnectorMetadata,
#     KvConnectorWorker as RustKvConnectorWorker,
# )

from kvbm.v1.vllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker


@dataclass
class KvTensorLayout:
    """Semantic description of the KV cache tensor layout.

    Python derives this from VllmConfig (which has the model architecture)
    so that Rust does not need to guess from raw tensor shapes.

    For MLA models, outer_dim and inner_dim are set explicitly because Rust's
    shape-based inference cannot distinguish the fused KV latent axis from the
    block/page axis.  For standard attention the fields are None, which tells
    Rust to fall back to its own contiguity-based inference (already correct).
    """

    outer_dim: Optional[int]  # None = let Rust detect; 1 for MLA
    inner_dim: Optional[int]  # None = let Rust detect; head_size for MLA

    @classmethod
    def from_vllm_config(
        cls, vllm_config: "VllmConfig", shape: "torch.Size", use_mla: bool = False
    ) -> "KvTensorLayout":
        if use_mla:
            # MLA tensors are 3D: [num_blocks, page_size, head_size]
            # No outer_dim axis — K and V are fused into a single latent.
            return cls(outer_dim=1, inner_dim=shape[-1])
        else:
            # Standard attention: Rust already infers outer_dim and inner_dim
            # correctly from tensor strides/contiguity.  Don't guess here —
            # the block dimension can be at position 0 or 1 depending on the
            # attention backend, so shape[1] is not reliably outer_dim.
            return cls(outer_dim=None, inner_dim=None)


DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


class KvConnectorWorker:
    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        drt: Optional[object] = kwargs.get("drt")

        if drt is None and is_dyn_runtime_enabled():
            drt = DistributedRuntime.detached()
        else:
            drt = None

        self.drt = drt

        self.vllm_config = vllm_config
        self._connector = RustKvConnectorWorker(self.drt, engine_id)

    # Worker

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """

        cache_config = self.vllm_config.cache_config

        # Create ordered list of (layer_name, tensor) tuples sorted by layer index
        ordered_kv_caches = [
            (layer_name, tensor)
            for layer_name, tensor in sorted(
                kv_caches.items(), key=lambda item: extract_layer_index(item[0])
            )
        ]

        events = [
            torch.cuda.Event(enable_timing=False, interprocess=False)
            for _ in range(len(ordered_kv_caches))
        ]

        # events are lazy, if we don't record them once here, the raw handles we pass to rust will be null
        for event in events:
            event.record(torch.cuda.current_stream())

        raw_event_handles = [event.cuda_event for event in events]

        self.events = {
            layer_name: event
            for (layer_name, _tensor), event in zip(ordered_kv_caches, events)
        }

        # Get first tensor to extract common properties
        first_tensor = ordered_kv_caches[0][1]
        shape = first_tensor.shape

        # Validate all tensors have same shape
        if not all(t.shape == shape for t in kv_caches.values()):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        page_size = cache_config.block_size
        use_mla = getattr(self.vllm_config.model_config, "use_mla", False)

        # MLA tensors are [num_blocks, page_size, head_size] — block dim is always axis 0.
        # Standard attention tensors are [outer_dim, num_blocks, ...] or [num_blocks, outer_dim, ...]
        # — block dim is whichever axis is larger, which is unambiguous as long as
        # num_blocks >> outer_dim (always true in practice).
        num_device_blocks = shape[0] if use_mla else max(shape[0], shape[1])
        device_id = first_tensor.device.index

        layout = KvTensorLayout.from_vllm_config(self.vllm_config, shape, use_mla)

        # Determine cache dtype
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self.vllm_config.model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Register with connector using ordered data
        self._connector.register_kv_caches(
            num_device_blocks,
            page_size,
            device_id,
            kv_cache_dtype.itemsize,
            ordered_kv_caches,
            raw_event_handles,
            outer_dim=layout.outer_dim,
            inner_dim=layout.inner_dim,
        )

    def bind_connector_metadata(self, data: bytes) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector.bind_connector_metadata(data)

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._connector.clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

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
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.events[layer_name].record(torch.cuda.current_stream())
        self._connector.save_kv_layer(layer_name, kv_layer)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        # finished_ids = [id for id in finished_req_ids]
        # return set(sending_ids), set(receiving_ids)
        return self._connector.get_finished(finished_req_ids)
