# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional

class NcclBootstrap:
    """
    NCCL bootstrap for creating dedicated KVBM communicators.

    This class provides methods to generate, serialize, deserialize,
    and initialize NCCL communicators for KVBM's replicated mode.

    Usage pattern:
    1. Rank 0: Call `NcclBootstrap.generate(world_size)` to create a new unique ID
    2. Rank 0: Call `serialize()` and broadcast to other ranks via MPI
    3. Other ranks: Call `NcclBootstrap.deserialize(bytes)` to reconstruct
    4. All ranks: Call `init_communicator(rank)` collectively to create the comm
    """

    @staticmethod
    def generate(world_size: int) -> "NcclBootstrap":
        """
        Generate a new unique ID for NCCL communicator initialization.
        This should only be called on rank 0.

        Parameters:
        -----------
        world_size: int
            The total number of ranks that will participate

        Returns:
        --------
        NcclBootstrap
            A new NcclBootstrap instance
        """
        ...

    def serialize(self) -> bytes:
        """
        Serialize the bootstrap data for distribution to other ranks.

        Returns:
        --------
        bytes
            The serialized bootstrap data (136 bytes)
        """
        ...

    @staticmethod
    def deserialize(data: bytes) -> "NcclBootstrap":
        """
        Deserialize bootstrap data received from rank 0.

        Parameters:
        -----------
        data: bytes
            The serialized bootstrap data (136 bytes)

        Returns:
        --------
        NcclBootstrap
            A new NcclBootstrap instance
        """
        ...

    def init_communicator(self, rank: int) -> int:
        """
        Initialize the NCCL communicator.

        IMPORTANT: This is a collective operation!
        All ranks must call this function together with matching parameters.
        The function will block until all ranks have called it.

        Parameters:
        -----------
        rank: int
            This rank's ID (0 to world_size-1)

        Returns:
        --------
        int
            The raw ncclComm_t pointer as an integer
        """
        ...

    def world_size(self) -> int:
        """
        Get the world size for this bootstrap.

        Returns:
        --------
        int
            The world size
        """
        ...


class KvbmWorker:
    """
    A KVBM worker that handles block transfers.
    """

    def __init__(
        self,
        num_device_blocks: int,
        page_size: int,
        tensors: List[Any],
        device_id: int = 0,
        dtype_width_bytes: int = 2,
        drt: Optional[Any] = None,
        layout_blocking: bool = False,
        device_layout_type: Optional[Any] = None,
        host_layout_type: Optional[Any] = None,
        disk_layout_type: Optional[Any] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        nccl_comm_ptr: Optional[int] = None,
    ) -> None:
        """
        Create a KvbmWorker instance.

        Parameters:
        -----------
        num_device_blocks: int
            Number of device blocks to manage
        page_size: int
            Page size for blocks
        tensors: List[Any]
            List of tensor objects (e.g., torch.Tensor)
        device_id: int
            CUDA device ID, defaults to 0
        dtype_width_bytes: int
            Data type width in bytes, defaults to 2 (fp16)
        drt: Optional[Any]
            Distributed runtime, if applicable
        layout_blocking: bool
            Whether to block on layout initialization, defaults to False
        device_layout_type: Optional[Any]
            Layout type for device blocks
        host_layout_type: Optional[Any]
            Layout type for host blocks
        disk_layout_type: Optional[Any]
            Layout type for disk blocks
        rank: Optional[int]
            Rank for replicated mode (None = sharded mode)
        world_size: Optional[int]
            World size for replicated mode
        nccl_comm_ptr: Optional[int]
            Raw ncclComm_t pointer for replicated mode (from NcclBootstrap)
        """
        ...

class Layer:
    """
    A KV cache block layer
    """

    ...

    def __dlpack__(self, stream: Optional[Any] = None, max_version: Optional[Any] = None, dl_device: Optional[Any] = None, copy: Optional[bool] = None) -> Any:
        """
        Get a dlpack capsule of the layer
        """
        ...

    def __dlpack_device__(self) -> Any:
        """
        Get the dlpack device of the layer
        """
        ...

class Block:
    """
    A KV cache block
    """

    ...

    def __len__(self) -> int:
        """
        Get the number of layers in the list
        """
        ...

    def __getitem__(self, index: int) -> Layer:
        """
        Get a layer by index
        """
        ...

    def __iter__(self) -> 'Block':
        """
        Get an iterator over the layers
        """
        ...

    def __next__(self) -> Block:
        """
        Get the next layer in the iterator
        """
        ...

    def to_list(self) -> List[Layer]:
        """
        Get a list of layers
        """
        ...

    def __dlpack__(self, stream: Optional[Any] = None, max_version: Optional[Any] = None, dl_device: Optional[Any] = None, copy: Optional[bool] = None) -> Any:
        """
        Get a dlpack capsule of the block
        Exception raised if the block is not contiguous
        """
        ...

    def __dlpack_device__(self) -> Any:
        """
        Get the dlpack device of the block
        """
        ...

class BlockList:
    """
    A list of KV cache blocks
    """

    ...

    def __len__(self) -> int:
        """
        Get the number of blocks in the list
        """
        ...

    def __getitem__(self, index: int) -> Block:
        """
        Get a block by index
        """
        ...

    def __iter__(self) -> 'BlockList':
        """
        Get an iterator over the blocks
        """
        ...

    def __next__(self) -> Block:
        """
        Get the next block in the iterator
        """
        ...

    def to_list(self) -> List[Block]:
        """
        Get a list of blocks
        """
        ...

class BlockManager:
    """
    A KV cache block manager
    """

    def __init__(
        self,
        worker_id: int,
        num_layer: int,
        page_size: int,
        inner_dim: int,
        dtype: Optional[str] = None,
        host_num_blocks: Optional[int] = None,
        device_num_blocks: Optional[int] = None,
        device_id: int = 0
    ) -> None:
        """
        Create a `BlockManager` object

        Parameters:
        -----------
        worker_id: int
            The worker ID for this block manager
        num_layer: int
            Number of layers in the model
        page_size: int
            Page size for blocks
        inner_dim: int
            Inner dimension size
        dtype: Optional[str]
            Data type (e.g., 'fp16', 'bf16', 'fp32'), defaults to 'fp16' if None
        host_num_blocks: Optional[int]
            Number of host blocks to allocate, None means no host blocks
        device_num_blocks: Optional[int]
            Number of device blocks to allocate, None means no device blocks
        device_id: int
            CUDA device ID, defaults to 0
        """
        ...

    def allocate_host_blocks_blocking(self, count: int) -> BlockList:
        """
        Allocate a list of host blocks (blocking call)

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

    async def allocate_host_blocks(self, count: int) -> BlockList:
        """
        Allocate a list of host blocks

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

    def allocate_device_blocks_blocking(self, count: int) -> BlockList:
        """
        Allocate a list of device blocks (blocking call)

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

    async def allocate_device_blocks(self, count: int) -> BlockList:
        """
        Allocate a list of device blocks

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

class KvbmCacheManager:
    """
    A KV cache manager for VLLM
    """

    def __init__(self, block_manager: BlockManager) -> None:
        ...


class KvbmRequest:
    """
    A request for KV cache
    """

    def __init__(self, request_id: int, tokens: List[int], block_size: int) -> None:
        ...


class SchedulerOutput:
    """
    Scheduler output containing information about scheduled requests.
    """

    new_requests: List[Any]
    cached_requests: List[Any]
    num_scheduled_tokens: dict

    def __init__(self) -> None:
        ...


class PyTrtllmKvConnectorWorker:
    """
    TensorRT-LLM KV connector worker for KVBM integration.

    This class handles KV cache operations on the worker side for TRT-LLM,
    including registration of KV caches, offloading, and loading operations.
    """

    def __init__(
        self,
        py_drt: Optional[Any],
        trtllm_rank: str,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        nccl_comm_ptr: Optional[int] = None,
    ) -> None:
        """
        Create a PyTrtllmKvConnectorWorker instance.

        Parameters:
        -----------
        py_drt: Optional[Any]
            The distributed runtime object (DistributedRuntime)
        trtllm_rank: str
            The TRT-LLM rank identifier
        rank: Optional[int]
            Rank for replicated mode (None = sharded mode).
            Required when running in DEP (Data Expert Parallelism) mode.
        world_size: Optional[int]
            World size for replicated mode.
            Required when running in DEP mode.
        nccl_comm_ptr: Optional[int]
            Raw ncclComm_t pointer for replicated mode.
            Can be obtained from NcclBootstrap.init_communicator().
            Required when running in DEP mode.
        """
        ...

    def register_kv_caches(
        self,
        num_device_blocks: int,
        page_size: int,
        device_id: int,
        dtype_width_bytes: int,
        kv_cache_tensor: Any,
        raw_event_handles: List[int],
    ) -> None:
        """
        Register KV cache tensors with the connector worker.

        Parameters:
        -----------
        num_device_blocks: int
            Number of device blocks to manage
        page_size: int
            Page size for blocks
        device_id: int
            CUDA device ID
        dtype_width_bytes: int
            Data type width in bytes (e.g., 2 for fp16)
        kv_cache_tensor: Any
            The KV cache tensor (torch.Tensor)
        raw_event_handles: List[int]
            List of raw CUDA event handles
        """
        ...

    def bind_connector_meta(self, metadata: bytes) -> None:
        """
        Bind connector metadata from the leader.

        Parameters:
        -----------
        metadata: bytes
            Serialized connector metadata
        """
        ...

    def execute_offload_operations(self) -> None:
        """
        Execute pending offload operations.
        """
        ...

    def save_kv_layer(self, layer_idx: int) -> None:
        """
        Save a KV cache layer.

        Parameters:
        -----------
        layer_idx: int
            Index of the layer to save
        """
        ...

    def start_load_kv(self) -> None:
        """
        Start loading KV cache data.
        """
        ...

    def get_finished(
        self,
        finished_gen_req_ids: List[int],
        started_loading_req_ids: List[int],
    ) -> tuple:
        """
        Get finished offloading and onboarding request IDs.

        Parameters:
        -----------
        finished_gen_req_ids: List[int]
            List of request IDs that have finished generation
        started_loading_req_ids: List[int]
            List of request IDs that have started loading

        Returns:
        --------
        tuple
            A tuple of (finished_offloading, finished_onboarding) request ID lists
        """
        ...

    def submit_offload_on_event(self, event: int) -> None:
        """
        Submit offload operations to be executed when the given event completes.

        Parameters:
        -----------
        event: int
            Raw CUDA event handle
        """
        ...


class PyTrtllmKvConnectorLeader:
    """
    TensorRT-LLM KV connector leader for KVBM integration.

    This class handles KV cache coordination on the leader side for TRT-LLM,
    including slot management, token matching, and metadata building.
    """

    def __init__(
        self,
        worker_id: int,
        drt: Optional[Any],
        page_size: int,
        leader: Any,
        consolidator_trtllm_endpoint: Optional[str] = None,
        consolidator_output_endpoint: Optional[str] = None,
    ) -> None:
        """
        Create a PyTrtllmKvConnectorLeader instance.

        Parameters:
        -----------
        worker_id: int
            The worker ID for this leader
        drt: Optional[Any]
            The distributed runtime object (currently unused)
        page_size: int
            Page size for blocks
        leader: Any
            The KVBM leader object (PyKvbmLeader)
        consolidator_trtllm_endpoint: Optional[str]
            TRT-LLM consolidator endpoint
        consolidator_output_endpoint: Optional[str]
            Output consolidator endpoint
        """
        ...

    def get_num_new_matched_tokens(
        self,
        request_id: str,
        request_num_tokens: int,
        num_computed_tokens: int,
    ) -> tuple:
        """
        Get the number of newly matched tokens for a request.

        Parameters:
        -----------
        request_id: str
            The request identifier
        request_num_tokens: int
            Total number of tokens in the request
        num_computed_tokens: int
            Number of already computed tokens

        Returns:
        --------
        tuple
            A tuple of (num_matched_tokens, is_complete)
        """
        ...

    def update_state_after_alloc(
        self,
        request_id: str,
        block_ids: List[int],
        context_current_position: int,
    ) -> None:
        """
        Update state after block allocation.

        Parameters:
        -----------
        request_id: str
            The request identifier
        block_ids: List[int]
            List of allocated block IDs
        context_current_position: int
            Current context position
        """
        ...

    def build_connector_metadata(self, scheduler_output: SchedulerOutput) -> bytes:
        """
        Build connector metadata from scheduler output.

        Parameters:
        -----------
        scheduler_output: SchedulerOutput
            The scheduler output

        Returns:
        --------
        bytes
            Serialized connector metadata
        """
        ...

    def request_finished(self, request_id: str, block_ids: List[int]) -> bool:
        """
        Mark a request as finished.

        Parameters:
        -----------
        request_id: str
            The request identifier
        block_ids: List[int]
            List of block IDs used by the request

        Returns:
        --------
        bool
            True if the request was successfully marked as finished
        """
        ...

    def has_slot(self, request_id: str) -> bool:
        """
        Check if a slot exists for the given request.

        Parameters:
        -----------
        request_id: str
            The request identifier

        Returns:
        --------
        bool
            True if a slot exists
        """
        ...

    def create_slot(self, request: KvbmRequest, tokens: List[int]) -> None:
        """
        Create a slot for a request.

        Parameters:
        -----------
        request: KvbmRequest
            The KVBM request object
        tokens: List[int]
            List of tokens for the request
        """
        ...
