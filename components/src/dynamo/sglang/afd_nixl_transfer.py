# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL-based Activation Transfer for AFD (Attention-FFN Disaggregation).

This module provides zero-copy RDMA-based activation transfer between
Attention and FFN workers using NVIDIA's NIXL (NVIDIA Inter-Xchange Library).

Key Features:
- Zero-copy RDMA transfer for activation tensors
- Pre-registered tensor descriptors for low-latency communication
- Async transfer with completion tracking
- Support for multi-GPU topologies

Reference: https://arxiv.org/abs/2601.21351
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# NIXL connector for RDMA transfers
# This module is provided by the Dynamo runtime
try:
    import dynamo.nixl_connect as connect
    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False
    logging.warning(
        "dynamo.nixl_connect not available. "
        "AFD will use simulated transfer mode."
    )


@dataclass
class AFDTransferConfig:
    """Configuration for NIXL-based activation transfer.
    
    Attributes:
        backend: NIXL backend to use (UCX, GDS_MT, POSIX)
        buffer_size: Size of transfer buffer in bytes
        num_buffers: Number of pre-allocated buffers
        enable_checksum: Enable transfer checksum verification
        timeout_ms: Transfer timeout in milliseconds
    """
    backend: str = "UCX"
    buffer_size: int = 64 * 1024 * 1024  # 64MB default
    num_buffers: int = 8
    enable_checksum: bool = False
    timeout_ms: int = 5000


class AFDActivationBuffer:
    """Pre-allocated buffer for activation transfer.
    
    This class manages pre-registered GPU memory for zero-copy
    activation transfer between Attention and FFN workers.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int, int],  # (batch, seq, hidden)
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """Initialize the activation buffer.
        
        Args:
            shape: Tensor shape (batch, seq, hidden)
            dtype: Data type for the tensor
            device: Device to allocate on
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device
        
        # Allocate GPU tensor
        self._tensor = torch.empty(shape, dtype=dtype, device=device)
        self._numpy_view = None  # Lazy initialization
        
        # NIXL descriptor (set during registration)
        self._descriptor: Optional[Any] = None
        self._is_registered = False
        
        logging.debug(
            f"AFDActivationBuffer created - shape={shape}, "
            f"dtype={dtype}, device={device}"
        )
    
    @property
    def tensor(self) -> torch.Tensor:
        """Get the underlying PyTorch tensor."""
        return self._tensor
    
    @property
    def numpy(self) -> np.ndarray:
        """Get NumPy view of the tensor (CPU)."""
        if self._numpy_view is None:
            self._numpy_view = self._tensor.cpu().numpy()
        return self._numpy_view
    
    def register_with_nixl(self, connector: "connect.Connector") -> None:
        """Register this buffer with NIXL for zero-copy transfer.
        
        Args:
            connector: NIXL connector instance
        """
        if not NIXL_AVAILABLE:
            logging.warning("NIXL not available, skipping registration")
            return
        
        if self._is_registered:
            return
        
        # Register tensor with NIXL
        # This creates a descriptor that can be used for RDMA transfers
        self._descriptor = connector.register_tensor(self._tensor)
        self._is_registered = True
        
        logging.debug(f"AFDActivationBuffer registered with NIXL")
    
    def unregister(self) -> None:
        """Unregister buffer from NIXL."""
        self._is_registered = False
        self._descriptor = None


class AFDNixlTransferManager:
    """Manages NIXL-based activation transfers for AFD.
    
    This class provides:
    1. Connection management between Attention and FFN workers
    2. Buffer registration and management
    3. Zero-copy activation transfer
    4. Transfer completion tracking
    """
    
    def __init__(
        self,
        config: AFDTransferConfig,
        is_attention_worker: bool = True,
    ):
        """Initialize the transfer manager.
        
        Args:
            config: Transfer configuration
            is_attention_worker: True if this is an Attention worker
        """
        self.config = config
        self.is_attention_worker = is_attention_worker
        
        # NIXL connector
        self._connector: Optional["connect.Connector"] = None
        self._is_initialized = False
        
        # Buffer pool
        self._buffers: List[AFDActivationBuffer] = []
        self._free_buffers: asyncio.Queue = asyncio.Queue()
        
        # Pending transfers
        self._pending_transfers: Dict[str, asyncio.Future] = {}
        
        logging.info(
            f"AFDNixlTransferManager created - "
            f"backend={config.backend}, is_attention={is_attention_worker}"
        )
    
    async def initialize(self, runtime_endpoint: str) -> None:
        """Initialize the NIXL connector.
        
        Args:
            runtime_endpoint: Endpoint for the FFN worker (Attention) or
                             Attention worker (FFN)
        """
        if not NIXL_AVAILABLE:
            logging.warning(
                "NIXL not available. Using simulated transfer mode."
            )
            self._is_initialized = True
            return
        
        # Create NIXL connector
        self._connector = connect.Connector(
            backend=self.config.backend,
            buffer_size=self.config.buffer_size,
            num_buffers=self.config.num_buffers,
        )
        
        # Initialize connection to remote worker
        await self._connector.initialize(runtime_endpoint)
        
        self._is_initialized = True
        logging.info(
            f"AFD NIXL connector initialized - endpoint={runtime_endpoint}"
        )
    
    def allocate_buffer(
        self,
        shape: Tuple[int, int, int],
        dtype: torch.dtype = torch.float16,
    ) -> AFDActivationBuffer:
        """Allocate a transfer buffer.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            
        Returns:
            AFDActivationBuffer instance
        """
        buffer = AFDActivationBuffer(shape, dtype)
        
        if self._connector is not None:
            buffer.register_with_nixl(self._connector)
        
        return buffer
    
    async def send_activation(
        self,
        request_id: str,
        activation: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send activation tensor to FFN worker (Attention worker only).
        
        Args:
            request_id: Unique request identifier
            activation: Activation tensor to send
            metadata: Additional metadata
        """
        if not self._is_initialized:
            raise RuntimeError("Transfer manager not initialized")
        
        if not self.is_attention_worker:
            raise RuntimeError("Only Attention workers can send activations")
        
        # Create future for completion tracking
        future = asyncio.Future()
        self._pending_transfers[request_id] = future
        
        if NIXL_AVAILABLE and self._connector is not None:
            # NIXL zero-copy transfer
            await self._connector.send_tensor(
                request_id,
                activation,
                metadata=metadata,
            )
        else:
            # Simulated transfer
            await asyncio.sleep(0.001)  # Simulate network latency
            logging.debug(f"Simulated transfer for {request_id}")
    
    async def receive_activation(
        self,
        request_id: str,
        buffer: AFDActivationBuffer,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Receive activation tensor from Attention worker (FFN worker only).
        
        Args:
            request_id: Expected request ID
            buffer: Buffer to receive into
            
        Returns:
            Tuple of (activation tensor, metadata)
        """
        if not self._is_initialized:
            raise RuntimeError("Transfer manager not initialized")
        
        if self.is_attention_worker:
            raise RuntimeError("Only FFN workers can receive activations")
        
        if NIXL_AVAILABLE and self._connector is not None:
            # NIXL zero-copy receive
            result = await self._connector.receive_tensor(
                request_id,
                buffer.tensor,
            )
            return buffer.tensor, result.metadata
        else:
            # Simulated receive
            await asyncio.sleep(0.001)
            return buffer.tensor, {}
    
    async def send_result(
        self,
        request_id: str,
        result: torch.Tensor,
    ) -> None:
        """Send FFN result back to Attention worker (FFN worker only).
        
        Args:
            request_id: Request identifier
            result: FFN output tensor
        """
        if not self._is_initialized:
            raise RuntimeError("Transfer manager not initialized")
        
        if self.is_attention_worker:
            raise RuntimeError("Only FFN workers can send results")
        
        if NIXL_AVAILABLE and self._connector is not None:
            await self._connector.send_tensor(
                f"{request_id}_result",
                result,
            )
        else:
            await asyncio.sleep(0.001)
    
    async def receive_result(
        self,
        request_id: str,
        buffer: AFDActivationBuffer,
    ) -> torch.Tensor:
        """Receive FFN result (Attention worker only).
        
        Args:
            request_id: Request identifier
            buffer: Buffer to receive into
            
        Returns:
            FFN output tensor
        """
        if not self._is_initialized:
            raise RuntimeError("Transfer manager not initialized")
        
        if not self.is_attention_worker:
            raise RuntimeError("Only Attention workers can receive results")
        
        if NIXL_AVAILABLE and self._connector is not None:
            await self._connector.receive_tensor(
                f"{request_id}_result",
                buffer.tensor,
            )
            return buffer.tensor
        else:
            await asyncio.sleep(0.001)
            return buffer.tensor
    
    def complete_transfer(self, request_id: str) -> None:
        """Mark a transfer as complete (called by completion handler).
        
        Args:
            request_id: The completed request ID
        """
        if request_id in self._pending_transfers:
            future = self._pending_transfers.pop(request_id)
            if not future.done():
                future.set_result(True)
    
    async def shutdown(self) -> None:
        """Shutdown the transfer manager."""
        # Cancel pending transfers
        for future in self._pending_transfers.values():
            if not future.done():
                future.cancel()
        self._pending_transfers.clear()
        
        # Cleanup connector
        if self._connector is not None:
            await self._connector.shutdown()
            self._connector = None
        
        # Cleanup buffers
        for buffer in self._buffers:
            buffer.unregister()
        self._buffers.clear()
        
        self._is_initialized = False
        logging.info("AFD NIXL transfer manager shutdown")


class AFDTransferStats:
    """Statistics for AFD activation transfers."""
    
    def __init__(self):
        self.total_transfers = 0
        self.total_bytes = 0
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0
        self.min_latency_ms = float('inf')
    
    def record_transfer(self, num_bytes: int, latency_ms: float) -> None:
        """Record a transfer."""
        self.total_transfers += 1
        self.total_bytes += num_bytes
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_transfers == 0:
            return 0.0
        return self.total_latency_ms / self.total_transfers
    
    @property
    def throughput_gbps(self) -> float:
        """Calculate throughput in Gbps."""
        if self.total_latency_ms == 0:
            return 0.0
        total_bits = self.total_bytes * 8
        total_seconds = self.total_latency_ms / 1000
        return (total_bits / total_seconds) / 1e9
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transfers": self.total_transfers,
            "total_bytes": self.total_bytes,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "throughput_gbps": self.throughput_gbps,
        }
