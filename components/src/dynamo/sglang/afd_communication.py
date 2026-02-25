# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD (Attention-FFN Disaggregation) Communication Protocol.

This module implements the communication layer between Attention and FFN workers
in AFD disaggregated serving mode.

Architecture: r Attention instances -> 1 shared FFN instance

The communication protocol uses:
1. Zero-copy activation transfer (NIXL or similar RDMA-based transport)
2. Microbatch pipelining for overlapping communication and computation
3. Asynchronous request-response pattern for FFN computation

Reference: https://arxiv.org/abs/2601.21351
"""

import asyncio
import logging
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Note: NIXL (NVIDIA Inter-Xchange Library) integration will be added
# when the library becomes available. For now, we use a placeholder
# that simulates the communication pattern.


class AFDMessageType(Enum):
    """Message types for AFD communication."""

    # Attention -> FFN
    ACTIVATION_TRANSFER = "activation_transfer"
    BATCH_ACTIVATE = "batch_activate"

    # FFN -> Attention
    ACTIVATION_RESULT = "activation_result"
    BATCH_RESULT = "batch_result"

    # Control messages
    SYNC_REQUEST = "sync_request"
    SYNC_ACK = "sync_ack"
    HEARTBEAT = "heartbeat"


@dataclass
class AFDActivationBatch:
    """Batch of activations from Attention worker to FFN worker.

    Attributes:
        request_id: Unique identifier for this batch
        layer_idx: Index of the transformer layer
        activations: Hidden states from attention layer
        attention_mask: Attention mask for the batch
        position_ids: Position IDs for the batch
        metadata: Additional metadata (temperature, top_p, etc.)
    """

    request_id: str
    layer_idx: int
    activations: np.ndarray  # Shape: [batch_size, seq_len, hidden_dim]
    attention_mask: Optional[np.ndarray] = None
    position_ids: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def serialize(self) -> bytes:
        """Serialize the activation batch for transmission.

        Format:
        - request_id_len (4 bytes)
        - request_id (variable)
        - layer_idx (4 bytes)
        - activations_shape (12 bytes: 3 x 4 bytes)
        - activations_data (variable)
        - attention_mask_shape (12 bytes, optional)
        - attention_mask_data (variable, optional)
        - position_ids_shape (12 bytes, optional)
        - position_ids_data (variable, optional)
        - metadata_json_len (4 bytes)
        - metadata_json (variable)
        """
        parts = []

        # Request ID
        request_id_bytes = self.request_id.encode("utf-8")
        parts.append(struct.pack("I", len(request_id_bytes)))
        parts.append(request_id_bytes)

        # Layer index
        parts.append(struct.pack("I", self.layer_idx))

        # Activations
        shape = self.activations.shape
        parts.append(struct.pack("III", *shape))
        parts.append(self.activations.tobytes())

        # Attention mask (optional)
        if self.attention_mask is not None:
            mask_shape = self.attention_mask.shape
            parts.append(struct.pack("III", *mask_shape))
            parts.append(self.attention_mask.tobytes())
        else:
            parts.append(struct.pack("III", 0, 0, 0))

        # Position IDs (optional)
        if self.position_ids is not None:
            pos_shape = self.position_ids.shape
            parts.append(struct.pack("II", *pos_shape[:2]))
            parts.append(self.position_ids.tobytes())
        else:
            parts.append(struct.pack("II", 0, 0))

        # Metadata
        import json

        metadata_json = json.dumps(self.metadata or {})
        metadata_bytes = metadata_json.encode("utf-8")
        parts.append(struct.pack("I", len(metadata_bytes)))
        parts.append(metadata_bytes)

        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "AFDActivationBatch":
        """Deserialize activation batch from bytes."""
        offset = 0

        # Request ID
        request_id_len = struct.unpack_from("I", data, offset)[0]
        offset += 4
        request_id = data[offset : offset + request_id_len].decode("utf-8")
        offset += request_id_len

        # Layer index
        layer_idx = struct.unpack_from("I", data, offset)[0]
        offset += 4

        # Activations
        shape = struct.unpack_from("III", data, offset)
        offset += 12
        activations_size = shape[0] * shape[1] * shape[2] * 4  # float32
        activations = np.frombuffer(data[offset : offset + activations_size], dtype=np.float32).reshape(
            shape
        )
        offset += activations_size

        # Attention mask
        mask_shape = struct.unpack_from("III", data, offset)
        offset += 12
        if mask_shape[0] > 0:
            mask_size = mask_shape[0] * mask_shape[1] * mask_shape[2]
            attention_mask = np.frombuffer(data[offset : offset + mask_size], dtype=np.int64).reshape(
                mask_shape
            )
            offset += mask_size
        else:
            attention_mask = None

        # Position IDs
        pos_shape = struct.unpack_from("II", data, offset)
        offset += 8
        if pos_shape[0] > 0:
            pos_size = pos_shape[0] * pos_shape[1] * 4
            position_ids = np.frombuffer(data[offset : offset + pos_size], dtype=np.int32).reshape(
                pos_shape
            )
            offset += pos_size
        else:
            position_ids = None

        # Metadata
        import json

        metadata_len = struct.unpack_from("I", data, offset)[0]
        offset += 4
        metadata_json = data[offset : offset + metadata_len].decode("utf-8")
        metadata = json.loads(metadata_json)

        return cls(
            request_id=request_id,
            layer_idx=layer_idx,
            activations=activations,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata=metadata,
        )


@dataclass
class AFDFFNResult:
    """Result from FFN computation.

    Attributes:
        request_id: Unique identifier matching the activation batch
        output: FFN output hidden states
        finish_reason: Optional finish reason if generation is complete
    """

    request_id: str
    output: np.ndarray  # Shape: [batch_size, seq_len, hidden_dim]
    finish_reason: Optional[str] = None

    def serialize(self) -> bytes:
        """Serialize the FFN result for transmission."""
        parts = []

        # Request ID
        request_id_bytes = self.request_id.encode("utf-8")
        parts.append(struct.pack("I", len(request_id_bytes)))
        parts.append(request_id_bytes)

        # Output
        shape = self.output.shape
        parts.append(struct.pack("III", *shape))
        parts.append(self.output.tobytes())

        # Finish reason
        finish_reason_bytes = (self.finish_reason or "").encode("utf-8")
        parts.append(struct.pack("I", len(finish_reason_bytes)))
        parts.append(finish_reason_bytes)

        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "AFDFFNResult":
        """Deserialize FFN result from bytes."""
        offset = 0

        # Request ID
        request_id_len = struct.unpack_from("I", data, offset)[0]
        offset += 4
        request_id = data[offset : offset + request_id_len].decode("utf-8")
        offset += request_id_len

        # Output
        shape = struct.unpack_from("III", data, offset)
        offset += 12
        output_size = shape[0] * shape[1] * shape[2] * 4  # float32
        output = np.frombuffer(data[offset : offset + output_size], dtype=np.float32).reshape(shape)
        offset += output_size

        # Finish reason
        finish_reason_len = struct.unpack_from("I", data, offset)[0]
        offset += 4
        finish_reason = data[offset : offset + finish_reason_len].decode("utf-8") or None

        return cls(
            request_id=request_id,
            output=output,
            finish_reason=finish_reason,
        )


class AFDCommunicationManager:
    """Manages communication between Attention and FFN workers.

    This class provides:
    1. Connection management for FFN worker endpoint
    2. Activation batch serialization and transfer
    3. Result aggregation from FFN worker
    4. Microbatch pipelining for overlapping communication/computation

    Note: This is a placeholder implementation. The actual implementation
    will use NIXL (NVIDIA Inter-Xchange Library) for zero-copy RDMA transfers.
    """

    def __init__(
        self,
        ffn_endpoint: Optional[str] = None,
        attention_ratio: int = 1,
        microbatch_size: int = 256,
        sync_timeout_ms: int = 1000,
    ):
        """Initialize the communication manager.

        Args:
            ffn_endpoint: Endpoint for FFN worker (format: namespace.component.endpoint)
            attention_ratio: Number of attention workers per FFN worker
            microbatch_size: Size of microbatches for pipelining
            sync_timeout_ms: Timeout for synchronization operations
        """
        self.ffn_endpoint = ffn_endpoint
        self.attention_ratio = attention_ratio
        self.microbatch_size = microbatch_size
        self.sync_timeout_ms = sync_timeout_ms

        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._connection = None
        self._running = False

        logging.info(
            f"AFD Communication Manager initialized - "
            f"ffn_endpoint={ffn_endpoint}, attention_ratio={attention_ratio}"
        )

    async def connect(self) -> None:
        """Establish connection to FFN worker.

        Note: In production, this will use NIXL for RDMA connection setup.
        """
        if not self.ffn_endpoint:
            logging.warning("No FFN endpoint configured, AFD communication disabled")
            return

        # TODO: Implement actual NIXL connection
        logging.info(f"Connecting to FFN worker at {self.ffn_endpoint}")
        self._running = True

    async def disconnect(self) -> None:
        """Disconnect from FFN worker."""
        self._running = False
        self._connection = None
        logging.info("Disconnected from FFN worker")

    async def send_activation_batch(self, batch: AFDActivationBatch) -> str:
        """Send activation batch to FFN worker.

        Args:
            batch: The activation batch to send

        Returns:
            Request ID for tracking the result
        """
        if not self._running:
            raise RuntimeError("Communication manager not connected")

        # Create future for result
        future = asyncio.Future()
        self._pending_requests[batch.request_id] = future

        # TODO: Implement actual NIXL zero-copy transfer
        logging.debug(
            f"Sending activation batch {batch.request_id} - "
            f"layer={batch.layer_idx}, shape={batch.activations.shape}"
        )

        return batch.request_id

    async def receive_ffn_result(self, request_id: str, timeout_ms: Optional[int] = None) -> AFDFFNResult:
        """Receive FFN computation result.

        Args:
            request_id: The request ID to wait for
            timeout_ms: Optional timeout override

        Returns:
            The FFN computation result

        Raises:
            asyncio.TimeoutError: If timeout expires
        """
        if request_id not in self._pending_requests:
            raise ValueError(f"Unknown request ID: {request_id}")

        timeout = (timeout_ms or self.sync_timeout_ms) / 1000
        future = self._pending_requests[request_id]

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        finally:
            del self._pending_requests[request_id]

    def handle_ffn_result(self, result: AFDFFNResult) -> None:
        """Handle incoming FFN result (called by receiver thread).

        Args:
            result: The FFN computation result
        """
        if result.request_id in self._pending_requests:
            future = self._pending_requests[result.request_id]
            if not future.done():
                future.set_result(result)
            logging.debug(f"Received FFN result for {result.request_id}")
        else:
            logging.warning(f"Received result for unknown request: {result.request_id}")


class AFDMicrobatchPipeline:
    """Manages microbatch pipelining for AFD.

    Microbatch pipelining allows overlapping activation transfer
    with FFN computation, improving overall throughput.

    Pipeline stages:
    1. Attention computation (on Attention worker)
    2. Activation transfer (network)
    3. FFN computation (on FFN worker)
    4. Result transfer (network)
    """

    def __init__(
        self,
        communication_manager: AFDCommunicationManager,
        num_stages: int = 4,
        batch_size: int = 256,
    ):
        """Initialize the microbatch pipeline.

        Args:
            communication_manager: The AFD communication manager
            num_stages: Number of pipeline stages
            batch_size: Size of each microbatch
        """
        self.comm_manager = communication_manager
        self.num_stages = num_stages
        self.batch_size = batch_size

        self._pipeline_queue: asyncio.Queue = asyncio.Queue(maxsize=num_stages)
        self._running = False

        logging.info(
            f"AFD Microbatch Pipeline initialized - "
            f"stages={num_stages}, batch_size={batch_size}"
        )

    async def start(self) -> None:
        """Start the pipeline processing loop."""
        self._running = True
        logging.info("AFD Pipeline started")

    async def stop(self) -> None:
        """Stop the pipeline processing loop."""
        self._running = False
        logging.info("AFD Pipeline stopped")

    async def submit_attention_output(
        self,
        request_id: str,
        layer_idx: int,
        activations: np.ndarray,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Submit attention output for FFN processing.

        Args:
            request_id: Unique request identifier
            layer_idx: Transformer layer index
            activations: Attention layer output
            metadata: Additional metadata

        Returns:
            Pipeline ticket for tracking
        """
        # Split into microbatches
        total_tokens = activations.shape[0] * activations.shape[1]
        num_microbatches = (total_tokens + self.batch_size - 1) // self.batch_size

        tickets = []
        for i in range(num_microbatches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_tokens)

            microbatch_id = f"{request_id}_mb{i}"
            batch = AFDActivationBatch(
                request_id=microbatch_id,
                layer_idx=layer_idx,
                activations=activations.flatten()[start_idx:end_idx].reshape(
                    1, end_idx - start_idx, activations.shape[-1]
                ),
                metadata=metadata,
            )

            ticket = await self.comm_manager.send_activation_batch(batch)
            tickets.append(ticket)

        # Return combined ticket
        return f"{request_id}_pipeline"

    async def get_ffn_output(self, pipeline_ticket: str, timeout_ms: Optional[int] = None) -> np.ndarray:
        """Get FFN output for a pipeline ticket.

        Args:
            pipeline_ticket: The pipeline ticket from submit_attention_output
            timeout_ms: Optional timeout override

        Returns:
            The combined FFN output
        """
        request_id = pipeline_ticket.rsplit("_pipeline", 1)[0]

        # Wait for all microbatch results
        # TODO: Implement proper microbatch tracking
        timeout = (timeout_ms or self.comm_manager.sync_timeout_ms) / 1000

        # Placeholder: return dummy result
        logging.debug(f"Waiting for FFN output for {request_id}")
        await asyncio.sleep(0.001)  # Simulate network latency

        # Return placeholder
        return np.zeros((1, 1, 4096), dtype=np.float32)
