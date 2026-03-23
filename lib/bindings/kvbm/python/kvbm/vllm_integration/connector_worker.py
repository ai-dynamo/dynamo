# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from kvbm.utils import is_dyn_runtime_enabled
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

from kvbm.vllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker

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

        # Get first tensor to detect device type/id for event handling.
        first_tensor = ordered_kv_caches[0][1]
        self.device_type = first_tensor.device.type
        self.device_id = first_tensor.device.index if self.device_type=="cuda" else (first_tensor.device.index or 0)

        if self.device_type == "cuda":
            events = [
                torch.cuda.Event(enable_timing=False, interprocess=False)
                for _ in range(len(ordered_kv_caches))
            ]
            # Events are lazy: record once so raw handles are valid.
            for event in events:
                event.record(torch.cuda.current_stream(self.device_id))
            raw_event_handles = [event.cuda_event for event in events]
        elif self.device_type == "xpu":
            events = [torch.xpu.Event(enable_timing=False) for _ in range(len(ordered_kv_caches))]
            for event in events:
                event.record(torch.xpu.current_stream(self.device_id))
            # Pass raw SYCL event handles for Rust-side XPU waiting.
            raw_event_handles = [event.sycl_event for event in events]
        elif self.device_type == "hpu":
            events = [torch.hpu.Event(enable_timing=False) for _ in range(len(ordered_kv_caches))]
            for event in events:
                event.record(torch.hpu.current_stream(self.device_id))
            # Pass raw Synapse event handles for Rust-side HPU waiting.
            raw_event_handles = [event.hpu_event for event in events]
        else:
            raise NotImplementedError(f"Unsupported KV cache device type: {self.device_type}")

        self.events = {
            layer_name: event
            for (layer_name, _tensor), event in zip(ordered_kv_caches, events)
        }

        # Get first tensor to extract common properties
        shape = first_tensor.shape

        # Validate all tensors have same shape
        if not all(t.shape == shape for t in kv_caches.values()):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # Extract parameters
        # TODO: Assume the block dimension is within the first 2. This will break if you're doing something weird like having 1 or 2 device blocks.
        num_device_blocks = max(shape[0], shape[1])
        page_size = cache_config.block_size
        device_id = self.device_id

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
        if self.device_type == "cuda":
            self.events[layer_name].record(torch.cuda.current_stream(self.device_id))
        elif self.device_type == "xpu":
            self.events[layer_name].record(torch.xpu.current_stream(self.device_id))
            # Host wait for XPU path until Rust-side XPU event wait is available.
            print('buke before sync in save_kv_layer, device_id:', self.device_id)
            self.events[layer_name].synchronize()
        elif self.device_type == "hpu":
            self.events[layer_name].record(torch.hpu.current_stream(self.device_id))
            # Host wait for HPU path until Rust-side HPU event wait is available.
            self.events[layer_name].synchronize()
        else:
            raise NotImplementedError(f"Unsupported KV cache device type: {self.device_type}")
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
