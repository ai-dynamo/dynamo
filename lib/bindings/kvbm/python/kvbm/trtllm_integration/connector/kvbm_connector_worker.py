# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

# Keeping this import is important because it runs the code in nixl’s __init__.py
# to set up the Nixl plugin path.
import nixl  # noqa: F401
import torch
from kvbm.trtllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker
from kvbm.utils import is_dyn_runtime_enabled
from tensorrt_llm import logger
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import KvCacheConnectorWorker
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime


class DynamoKVBMConnectorWorker(KvCacheConnectorWorker):
    def _callable_object(self) -> callable:
        assert (
            self._connector is not None
        ), "Expected cache connector worker to have non-None _connector obj"
        assert (
            self.event is not None
        ), "Expected cache connector worker to have non-None event obj"

        def callback():
            self.event.record()
            self.event.synchronize()
            self._connector.execute_offload_operations()

        return callback

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        drt: Optional[object] = None
        if is_dyn_runtime_enabled():
            drt = DistributedRuntime.detached()

        self.drt = drt

        mappings = self._llm_args.parallel_config.to_mapping()
        self.rank = mappings.rank

        self._connector = RustKvConnectorWorker(self.drt, str(self.rank))
        self.event = torch.cuda.Event()

        # Default to old way of processing offload
        self.use_forward_pass_callable = False

    def register_forward_pass_callable(self) -> callable:
        """
        Register a callable object which will be called at the
        end of the forward pass.
        """
        self.use_forward_pass_callable = True
        return self._callable_object()

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        """
        Register the KV cache tensors to the worker.
        This can be used for something like NIXL registration.
        Args:
            kv_cache_tensor: The contiguous KV cache tensor.
        """
        logger.info(
            f"KvConnectorWorker started registering the kv caches on rank {self.rank}"
        )

        num_device_blocks = kv_cache_tensor.shape[0]
        page_size = self._llm_args.kv_cache_config.tokens_per_block
        device_id = kv_cache_tensor.device.index
        kv_cache_dtype = kv_cache_tensor.dtype

        num_cache_layers = kv_cache_tensor.shape[1]
        self.events = [
            torch.cuda.Event(enable_timing=False, interprocess=False)
            for _ in range(num_cache_layers)
        ]

        for event in self.events:
            event.record(torch.cuda.current_stream(device_id))

        raw_event_handles = [event.cuda_event for event in self.events]

        self._connector.register_kv_caches(
            num_device_blocks,
            page_size,
            device_id,
            kv_cache_dtype.itemsize,
            kv_cache_tensor,
            raw_event_handles,
        )

    def bind_connector_meta(self, metadata: object):
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            metadata (bytes): the connector metadata.
        """
        super().bind_connector_meta(metadata)
        self._connector.bind_connector_meta(metadata)

    def start_load_kv(self, stream: torch.cuda.Stream):
        """
        Begin loading the KV cache in preparation for the next forward pass.
        Specific blocks to transfer are indicated by the scheduler's metadata.
        """
        self._connector.start_load_kv()

    def wait_for_save(self, stream: torch.cuda.Stream):
        """
        Block until all synchronous saving operations are complete. Called at the end of the forward pass.
        """
        pass

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Wait for a layer to finish being loaded before proceeding with the forward pass on the layer.
        Note: This function is called immediately before the layer's work is enqueued into the stream.
        Args:
            layer_idx: The index of the layer to wait for.
            stream: The stream the forward pass is being executed on.
        """
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Begin saving the KV cache for a layer.
        Note: This function is called immediately after the layer's work is enqueued into the stream.
        Args:
            layer_idx: The index of the layer to save.
            stream: The stream the forward pass is being executed on.
        """
        if not self.use_forward_pass_callable:
            self.events[layer_idx].record(stream)
            self._connector.save_kv_layer(layer_idx)

    def get_finished(
        self, finished_gen_req_ids: list[int], started_loading_req_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Get the requests that have finished loading and saving.
        Args:
            finished_gen_req_ids: The IDs of the requests that have finished generating tokens, and are now asynchronously saving.
            started_loading_req_ids: The IDs of the requests that have started asynchronously loading.
        Returns:
            The IDs of the requests that have finished saving.
            The IDs of the requests that have finished loading.
        Note: IDs may only be returned from this call after they've been provided in the `finished_gen_req_ids` and `started_loading_req_ids` arguments.
        Additionally, the runtime will only take action based on these returned IDs once they've been returned by ALL workers. This allows some workers to take longer than others to complete the operations.
        """
        return self._connector.get_finished(
            finished_gen_req_ids, started_loading_req_ids
        )
