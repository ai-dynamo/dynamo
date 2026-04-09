# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from kvbm.utils import is_dyn_runtime_enabled
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

logger = logging.getLogger(__name__)

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

from kvbm.vllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker

DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


def _recover_raw_tensor(state_tensors: "list[torch.Tensor]") -> "torch.Tensor":
    """Recover the flat raw_tensor backing a list of Mamba state tensors.

    vLLM's HMA mode creates Mamba state tensors as strided views
    (``torch.as_strided``) into a single flat ``int8`` buffer.  The
    ``raw_tensor`` attribute is *not* set on these views in vLLM 0.16.0,
    so we reconstruct it from the shared ``untyped_storage()``.
    """
    raw = getattr(state_tensors[0], "raw_tensor", None)
    if raw is not None:
        return raw
    storage = state_tensors[0].untyped_storage()
    num_blocks = state_tensors[0].shape[0]
    device = state_tensors[0].device
    raw = torch.empty([], dtype=torch.uint8, device=device).set_(storage)
    return raw.view(num_blocks, -1)


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
        self.mamba_layer_names: set[str] = set()

    # Worker

    def register_kv_caches(
        self, kv_caches: dict[str, "torch.Tensor | list[torch.Tensor]"]
    ):
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

        # Log tensor types, shapes, and byte sizes for each layer.
        # Hybrid models (e.g. Nemotron) pass list[Tensor] for Mamba/SSM
        # layers instead of a single Tensor.
        attention_layer_count = 0
        mamba_layer_count = 0
        for layer_name, value in ordered_kv_caches:
            if isinstance(value, torch.Tensor):
                attention_layer_count += 1
                byte_size_per_block = (
                    value.element_size()
                    * (value.nelement() // value.shape[0])
                )
                logger.debug(
                    "register_kv_caches: layer=%s Tensor shape=%s bspb=%d",
                    layer_name,
                    list(value.shape),
                    byte_size_per_block,
                )
            elif isinstance(value, list):
                mamba_layer_count += 1
                raw_tensor = _recover_raw_tensor(value)
                raw_byte_size = (
                    raw_tensor.element_size()
                    * (raw_tensor.nelement() // raw_tensor.shape[0])
                )
                logger.debug(
                    "register_kv_caches: layer=%s list[Tensor] len=%d "
                    "raw_shape=%s bspb=%d",
                    layer_name,
                    len(value),
                    list(raw_tensor.shape),
                    raw_byte_size,
                )
            else:
                logger.warning(
                    "register_kv_caches: layer=%s UNEXPECTED type=%s",
                    layer_name,
                    type(value).__name__,
                )

        logger.debug(
            "register_kv_caches: total_layers=%d "
            "attention_layers=%d mamba_layers=%d",
            len(ordered_kv_caches),
            attention_layer_count,
            mamba_layer_count,
        )

        # Normalize list[Tensor] (Mamba/SSM layers) to the raw_tensor that
        # backs them.  In HMA mode (mamba_cache_mode="all") each Mamba layer's
        # cache is a list of strided views (conv_state, ssm_state) into a
        # single raw_tensor that shares the same block layout as attention KV
        # caches.  The block manager only needs the raw_tensor.
        normalized_kv_caches: list[tuple[str, torch.Tensor]] = []
        mamba_layer_names: set[str] = set()
        for layer_name, value in ordered_kv_caches:
            if isinstance(value, list):
                raw_tensor = _recover_raw_tensor(value)
                mamba_layer_names.add(layer_name)
                normalized_kv_caches.append((layer_name, raw_tensor))
            else:
                normalized_kv_caches.append((layer_name, value))
        ordered_kv_caches = normalized_kv_caches
        self.mamba_layer_names = mamba_layer_names

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

        # Determine num_device_blocks from an attention tensor (>= 3 dims)
        # whose block dimension is reliably max(shape[0], shape[1]).
        # Mamba raw_tensors (2 dims) can't be used because shape[1] may be
        # the flat state size, not num_blocks.
        first_tensor = ordered_kv_caches[0][1]
        num_device_blocks = None
        ref_shape = None
        for _name, t in ordered_kv_caches:
            if t.dim() >= 3:
                ref_shape = t.shape
                num_device_blocks = max(t.shape[0], t.shape[1])
                break
        if num_device_blocks is None:
            ref_shape = first_tensor.shape
            num_device_blocks = first_tensor.shape[0]
        shape = ref_shape

        def _byte_size_per_block(t: torch.Tensor) -> int:
            return (t.element_size() * t.nelement()) // num_device_blocks

        expected_bspb = _byte_size_per_block(first_tensor)
        for layer_name, tensor in ordered_kv_caches:
            actual_bspb = _byte_size_per_block(tensor)
            if actual_bspb != expected_bspb:
                raise ValueError(
                    f"Layer {layer_name}: byte_size_per_block={actual_bspb} "
                    f"differs from expected={expected_bspb}. All layers must "
                    "have uniform byte-size-per-block for the block manager."
                )
        page_size = cache_config.block_size
        device_id = first_tensor.device.index

        # Determine cache dtype
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self.vllm_config.model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Register with connector using ordered data.
        # Pass attention_layer_count so Rust triggers offloading after the
        # last attention layer's save_kv_layer call (Mamba/SSM layers never
        # call save_kv_layer).
        self._connector.register_kv_caches(
            num_device_blocks,
            page_size,
            device_id,
            kv_cache_dtype.itemsize,
            ordered_kv_caches,
            raw_event_handles,
            active_save_layers=attention_layer_count,
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

    def wait_for_save(self) -> None:
        """Block until all save operations complete.

        For hybrid models (Nemotron), the forward pass has trailing Mamba
        layers after the last attention layer.  save_kv_layer defers the
        offload operations; this method records a CUDA event *after* all
        layers have executed and passes it to Rust to sync and enqueue.

        For pure-attention models this is a no-op because offloading was
        already triggered in save_kv_layer.
        """
        if not self.mamba_layer_names:
            return
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream())
        self._connector.wait_for_save(event.cuda_event)

    def enqueue_flush_ops(self, ops_json: bytes) -> None:
        """Forward flush-on-finish offload operations from the leader.

        The leader creates these operations in mark_as_finished for blocks
        that were fully filled during the final iteration.  They must be
        enqueued on the worker side so the scheduler can match them with
        the leader-side transfer request.
        """
        self._connector.enqueue_flush_ops(ops_json)

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
