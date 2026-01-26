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
