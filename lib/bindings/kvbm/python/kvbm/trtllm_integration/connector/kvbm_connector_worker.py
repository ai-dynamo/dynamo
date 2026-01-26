# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from kvbm.trtllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker
from kvbm.utils import is_dyn_runtime_enabled
from tensorrt_llm import logger
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import KvCacheConnectorWorker
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime


def _get_mpi_info() -> Tuple[Optional[int], Optional[int]]:
    """Get MPI rank and world_size if MPI is initialized.

    Returns:
        Tuple of (rank, world_size), or (None, None) if MPI is not available/initialized.
    """
    try:
        from mpi4py import MPI

        if MPI.Is_initialized():
            comm = MPI.COMM_WORLD
            return comm.Get_rank(), comm.Get_size()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to get MPI info: {e}")
    return None, None


def _create_kvbm_nccl_comm(rank: int, world_size: int) -> int:
    """Create a dedicated NCCL communicator for KVBM using MPI for bootstrap.

    This function creates an NCCL communicator that is separate from any other
    communicators (e.g., TRT-LLM's). The bootstrap uses MPI to distribute the
    unique ID from rank 0 to all other ranks.

    Args:
        rank: This process's rank (0 to world_size-1)
        world_size: Total number of ranks

    Returns:
        The raw ncclComm_t pointer as an integer

    Raises:
        ImportError: If mpi4py or NcclBootstrap is not available
        RuntimeError: If NCCL initialization fails
    """
    from mpi4py import MPI

    try:
        from kvbm._core import NcclBootstrap
    except ImportError:
        raise ImportError(
            "NcclBootstrap not available. "
            "Make sure kvbm was built with the 'nccl' feature enabled."
        )

    comm = MPI.COMM_WORLD

    # Rank 0 generates unique ID
    if rank == 0:
        bootstrap = NcclBootstrap.generate(world_size)
        bootstrap_data = bootstrap.serialize()
    else:
        bootstrap_data = None

    # Broadcast bootstrap data to all ranks
    bootstrap_data = comm.bcast(bootstrap_data, root=0)

    # Non-rank-0 deserializes the data
    if rank != 0:
        bootstrap = NcclBootstrap.deserialize(bootstrap_data)

    # All ranks collectively initialize (must be called together)
    # This is a blocking collective operation
    nccl_comm_ptr = bootstrap.init_communicator(rank)

    logger.info(f"KVBM: Rank {rank} created dedicated NCCL communicator")
    return nccl_comm_ptr


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
            # Non-blocking: passes event to Rust for async polling
            self._connector.submit_offload_on_event(self.event.cuda_event)
            # Returns immediately - no CPU blocking

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
