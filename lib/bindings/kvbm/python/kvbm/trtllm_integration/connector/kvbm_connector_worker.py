# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Tuple

import torch
from kvbm.trtllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker
from kvbm.utils import is_dyn_runtime_enabled, nvtx_annotate
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


def _create_kvbm_nccl_comm(rank: int, world_size: int):
    """Create a dedicated NCCL communicator for KVBM using MPI for bootstrap.

    This function creates an NCCL communicator that is separate from any other
    communicators (e.g., TRT-LLM's). The bootstrap uses MPI to distribute the
    unique ID from rank 0 to all other ranks.

    Args:
        rank: This process's rank (0 to world_size-1)
        world_size: Total number of ranks

    Returns:
        NcclCommRef: Owning reference; pass to the worker so the comm is
        kept alive and destroyed when the worker is done.

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
    logger.info(
        f"KVBM: Rank {rank} entering bcast (data_len={len(bootstrap_data) if bootstrap_data else 0})"
    )
    bootstrap_data = comm.bcast(bootstrap_data, root=0)
    logger.info(
        f"KVBM: Rank {rank} received bootstrap data (len={len(bootstrap_data)})"
    )

    # Non-rank-0 deserializes the data
    if rank != 0:
        bootstrap = NcclBootstrap.deserialize(bootstrap_data)

    logger.info(f"KVBM: Rank {rank} bootstrap world_size={bootstrap.world_size()}")

    # TRT-LLM MPI launch exposes all GPUs to each rank but manages device
    # assignment internally. torch.cuda.current_device() defaults to 0 for
    # all ranks, which causes ncclCommInitRank to fail (all ranks on same
    # device). Explicitly set the device to match the MPI rank.
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError(
            "KVBM NCCL MLA mode requires at least one visible CUDA device."
        )
    device_id = rank % device_count
    torch.cuda.set_device(device_id)
    logger.info(
        f"KVBM: Rank {rank} set to CUDA device {device_id} "
        f"(device_count={device_count})"
    )

    logger.info(f"KVBM: Rank {rank} waiting at MPI barrier " "before ncclCommInitRank")
    comm.Barrier()
    logger.info(f"KVBM: Rank {rank} passed barrier, " "calling ncclCommInitRank")

    # All ranks collectively initialize (must be called together).
    # This is a blocking collective operation; returns owning NcclCommRef.
    nccl_comm_ref = bootstrap.init_communicator(rank)

    logger.info(f"KVBM: Rank {rank} created dedicated NCCL communicator")
    return nccl_comm_ref


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

        # NCCL replicated mode for MLA support - controlled by feature flag
        # Set DYN_KVBM_NCCL_MLA_MODE=true to enable NCCL broadcast optimization for MLA models
        nccl_rank, nccl_world_size, nccl_comm_ref = None, None, None
        enable_nccl_mla = os.environ.get("DYN_KVBM_NCCL_MLA_MODE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        if enable_nccl_mla:
            logger.info("KVBM NCCL MLA mode enabled via DYN_KVBM_NCCL_MLA_MODE")
            nccl_rank, nccl_world_size = _get_mpi_info()
        else:
            logger.info(
                "KVBM NCCL MLA mode disabled. Set DYN_KVBM_NCCL_MLA_MODE=true to enable "
                "NCCL broadcast optimization for MLA models (e.g., DeepSeek)."
            )

        if enable_nccl_mla and nccl_rank is not None and nccl_world_size is not None:
            try:
                nccl_comm_ref = _create_kvbm_nccl_comm(nccl_rank, nccl_world_size)
                logger.info(
                    f"KVBM MLA support: NCCL broadcast optimization enabled. "
                    f"Rank {nccl_rank}/{nccl_world_size}: only rank 0 loads "
                    f"from G2/G3 storage, then broadcasts to all GPUs."
                )
            except ImportError:
                logger.warning(
                    "KVBM MLA support: NCCL not compiled. Using worker-level "
                    "replication (each GPU loads independently). For optimal "
                    "broadcast-based replication, rebuild with: "
                    "cargo build -p kvbm --features nccl"
                )
                nccl_rank, nccl_world_size, nccl_comm_ref = None, None, None
            except Exception as e:
                logger.warning(
                    "KVBM MLA support: _create_kvbm_nccl_comm failed (nccl_rank=%s, "
                    "nccl_world_size=%s). MLA broadcast disabled; using worker-level "
                    "replication (each GPU loads independently). Error: %s",
                    nccl_rank,
                    nccl_world_size,
                    e,
                )
                nccl_rank, nccl_world_size, nccl_comm_ref = None, None, None
        elif enable_nccl_mla:
            logger.info(
                "KVBM: MPI not available, using standard sharded mode. "
                "For NCCL replicated mode, ensure MPI is initialized."
            )
        # else: NCCL MLA mode disabled, no additional logging needed

        self._connector = RustKvConnectorWorker(
            self.drt,
            str(self.rank),
            nccl_rank=nccl_rank,
            world_size=nccl_world_size,
            nccl_comm_ref=nccl_comm_ref,
        )
        self.event = torch.cuda.Event()

        # Default to old way of processing offload
        self.use_forward_pass_callable = False

    @nvtx_annotate(category="worker")
    def register_forward_pass_callable(self) -> callable:
        """
        Register a callable object which will be called at the
        end of the forward pass.
        """
        self.use_forward_pass_callable = True
        return self._callable_object()

    @nvtx_annotate(category="worker")
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

    @nvtx_annotate(category="worker")
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

    @nvtx_annotate(category="worker")
    def start_load_kv(self, stream: torch.cuda.Stream):
        """
        Begin loading the KV cache in preparation for the next forward pass.
        Specific blocks to transfer are indicated by the scheduler's metadata.
        """
        self._connector.start_load_kv()

    @nvtx_annotate(category="worker")
    def wait_for_save(self, stream: torch.cuda.Stream):
        """
        Block until all synchronous saving operations are complete. Called at the end of the forward pass.
        """
        pass

    @nvtx_annotate(category="worker")
    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Wait for a layer to finish being loaded before proceeding with the forward pass on the layer.
        Note: This function is called immediately before the layer's work is enqueued into the stream.
        Args:
            layer_idx: The index of the layer to wait for.
            stream: The stream the forward pass is being executed on.
        """
        pass

    @nvtx_annotate(category="worker")
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

    @nvtx_annotate(category="worker")
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
