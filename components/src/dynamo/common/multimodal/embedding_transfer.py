# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import logging
import math
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from queue import Queue
from typing import Any, Awaitable, List, Optional

import msgspec
import torch
from pydantic import BaseModel
from safetensors import torch as safetensors_torch

from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.runtime import run_async

logger = logging.getLogger(__name__)
NIXL_WRITE_LEASE_GUARD_SECONDS = 5.0
NIXL_WRITE_RETRY_INITIAL_SECONDS = 0.01
NIXL_WRITE_RETRY_MAX_SECONDS = 1.0


def _load_nixl_api():
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as exc:
        raise RuntimeError(
            "NIXL is required for NIXL embedding transfer; install nixl "
            "to use NIXL write-based multimodal embedding transfer."
        ) from exc

    return nixl_agent, nixl_agent_config


def _load_nixl_connect():
    try:
        import dynamo.nixl_connect as nixl_connect
    except ImportError as exc:
        raise RuntimeError(
            "NIXL is required for NIXL embedding transfer; install "
            "dynamo.nixl_connect to use NIXL read-based transfers."
        ) from exc

    return nixl_connect


def torch_dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype object.

    Args:
        dtype_str: String representation of torch dtype (e.g., "torch.float32")

    Returns:
        Corresponding torch.dtype object

    Example:
        >>> dtype = EncodeHelper.get_torch_dtype_from_string("torch.bfloat16")
        >>> # Result: torch.bfloat16
    """
    dtype = getattr(torch, dtype_str.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_str!r}")
    return dtype


def torch_dtype_to_string(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


# Opaque object to the caller, different implementation may carry
# different information (e.g. local file path vs nixl metadata)
class TransferRequest(BaseModel):
    """
    Data class for transfer requests containing necessary information for embedding transfer.
    """

    embeddings_shape: List[int]
    embedding_dtype_str: str
    serialized_request: Any


class AbstractEmbeddingReceiver(ABC):
    """
    Abstract base class for a receiver of precomputed embeddings from the encode worker.
    """

    @abstractmethod
    async def receive_embeddings(
        self, request: TransferRequest
    ) -> tuple[int, torch.Tensor]:
        """
        Abstract method to receive precomputed embeddings for a given request ID.

        Args:
            request: The TransferRequest object containing information to receive embeddings.

        Returns:
            A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
            Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
        """
        pass

    @abstractmethod
    def release_tensor(self, tensor_id: int) -> None:
        """
        Abstract method to indicate that the tensor associated with the ID is no longer in use.
        Args:
            tensor_id: The ID of the tensor to release.
        """
        pass

    async def cancel_embeddings(self, request: TransferRequest) -> None:
        """Best-effort cancellation before a transfer buffer is received."""

        del request


class AbstractEmbeddingSender(ABC):
    """
    Abstract base class for a sender of precomputed embeddings to the downstream worker.
    """

    @abstractmethod
    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
    ) -> tuple[TransferRequest, Awaitable[None]]:
        """
        Abstract method to send precomputed embeddings for a given request ID.

        Args:
            embeddings: A torch.Tensor of the embeddings to send.
            stage_embeddings: A boolean indicating whether the embeddings should be staged for the transfer,
            if True, the embeddings may be used as transfer buffer and must not be released until the return future is completed.
        Returns:
            A tuple containing the TransferRequest object and an awaitable that can be awaited to indicate the send is completed.
        """
        pass

    async def aclose(self) -> None:
        """Release sender-owned background resources."""


class LocalEmbeddingSender(AbstractEmbeddingSender):
    """
    Sender that saves embeddings to a local file and sends the file path as the serialized request.
    """

    def __init__(self):
        self.sender_id = uuid.uuid4().hex
        self.embedding_counter = 0

    def save_embeddings_to_file(
        self, embedding_key: str, embeddings: torch.Tensor
    ) -> str:
        """
        Save the embeddings to a local file and return the file path.

        Args:
            embedding_key: A unique key for the embeddings.
            embeddings: A torch.Tensor of the embeddings to save.
        Returns:
            The file path where the embeddings are saved.
        """
        fd, tensor_path = tempfile.mkstemp(
            prefix=f"encoder_cache.{embedding_key}.", suffix=".safetensors"
        )
        os.close(fd)
        tensors = {"ec_cache": embeddings.cpu()}
        safetensors_torch.save_file(
            tensors,
            tensor_path,
        )
        return tensor_path

    @_nvtx.annotate("mm:local:send_embeddings", color="magenta")
    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
    ) -> tuple[TransferRequest, Awaitable[None]]:
        """
        Send precomputed embeddings for a given request ID.

        Args:
            embeddings: A torch.Tensor of the embeddings to send.
            stage_embeddings: A boolean indicating whether the embeddings should be staged for the transfer,
            if True, the embeddings may be used as transfer buffer and must not be released until the return future is completed.
        Returns:
            A tuple containing the TransferRequest object and an awaitable that can be awaited to indicate the send is completed.
        """
        # Implementation to send embeddings to the downstream worker
        # This could involve publishing to a message queue or making an API call
        embedding_key = f"{self.sender_id}_{self.embedding_counter}"
        self.embedding_counter += 1
        tensor_path = await asyncio.to_thread(
            self.save_embeddings_to_file,
            embedding_key,
            embeddings,
        )
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return (
            TransferRequest(
                embeddings_shape=list(embeddings.shape),
                embedding_dtype_str=torch_dtype_to_string(embeddings.dtype),
                serialized_request=tensor_path,
            ),
            fut,
        )


class LocalEmbeddingReceiver(AbstractEmbeddingReceiver):
    """
    Receiver that reads embeddings from a local file path provided in the serialized request.
    """

    def __init__(self):
        super().__init__()
        self.received_tensors = {}
        self.tensor_id_counter = 0

    @_nvtx.annotate("mm:local:receive_embeddings", color="magenta")
    async def receive_embeddings(
        self, request: TransferRequest
    ) -> tuple[int, torch.Tensor]:
        """
        Receive precomputed embeddings for a given request ID.

        Args:
            request: The TransferRequest object containing information to receive embeddings for.

        Returns:
            A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
            Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
        """
        tensor_path = request.serialized_request
        tensors = await asyncio.to_thread(safetensors_torch.load_file, tensor_path)
        embedding_tensor = tensors["ec_cache"]
        tensor_id = self.tensor_id_counter
        self.tensor_id_counter += 1
        self.received_tensors[tensor_id] = tensor_path
        return tensor_id, embedding_tensor

    def release_tensor(self, tensor_id: int) -> None:
        """
        Indicate that the tensor associated with the ID is no longer in use.

        Args:
            tensor_id: The ID of the tensor to release.
        """
        if tensor_id in self.received_tensors:
            file_path = self.received_tensors[tensor_id]
            os.remove(file_path)  # Clean up the local file
            del self.received_tensors[tensor_id]

    async def cancel_embeddings(self, request: TransferRequest) -> None:
        tensor_path = request.serialized_request
        if isinstance(tensor_path, str):
            try:
                await asyncio.to_thread(os.remove, tensor_path)
            except FileNotFoundError:
                pass


class MonolithicCounter:
    """
    A simple counter implementation for generating unique IDs.
    """

    def __init__(self):
        self.counter = 0

    def get_next_id(self) -> int:
        current_id = self.counter
        self.counter += 1
        return current_id


class RingBuffer:
    """
    A ring buffer implementation for managing memory allocation.
    Uses a circular buffer pattern to efficiently reuse memory without wrapped-around allocations.
    When insufficient space remains at the end, allocation restarts from the beginning.
    """

    BufferId = int

    def __init__(self, buffer_size):
        self.buffer_tensor = torch.zeros(buffer_size, dtype=torch.int8)
        # Index tracking for the ring buffer, when
        # free_start_idx < allocated_start_idx, the allocation has been wrapped around,
        # so the allocation request should be rejected if the requested size is larger
        # than the remaining space before allocated_start_idx.
        self.free_start_idx = 0
        self.allocated_start_idx = 0
        self.buffer_size = buffer_size
        self.end_idx = buffer_size
        self.wrapped_around = False

        # Track allocated buffers and their release state,
        # keeping released range in 'freed_list' for simpler monotonical buffer release
        self.freed_list = {}
        self.allocated_buffer_id_to_range = {}
        # For generate buffer IDs
        self.id_counter = MonolithicCounter()

    def __repr__(self):
        return f"RingBuffer(size={self.buffer_size}, free_start_idx={self.free_start_idx}, allocated_start_idx={self.allocated_start_idx}, wrapped_around={self.wrapped_around}, freed_list={self.freed_list}, allocated_buffers={self.allocated_buffer_id_to_range})"

    def _flush_freed_list(self):
        allocated_end = self.freed_list.pop(self.allocated_start_idx, None)
        while allocated_end is not None:
            self.allocated_start_idx = allocated_end
            if self.allocated_start_idx == self.end_idx:
                self.allocated_start_idx = 0
                self.wrapped_around = False
            allocated_end = self.freed_list.pop(self.allocated_start_idx, None)
        # No allocated buffer, reset indices. Important as the ring buffer doesn't
        # support non-contiguous allocation, this make sure the next allocation can
        # use the full buffer.
        if not self.allocated_buffer_id_to_range:
            self.free_start_idx = 0
            self.allocated_start_idx = 0
            self.wrapped_around = False

    def get_buffer(self, size):
        """
        Get a buffer of given size in the form of 1D tensor with dtype int8,
        the buffer is owned by the RingBuffer instance.
        The returned ID will be used for releasing the buffer after use, as
        an indicator that the buffer can be reused for future allocation.

        Args:
            size: The size of the buffer to allocate.

        Returns:
            A tuple containing the buffer ID and the allocated tensor, or None if allocation fails.
        """
        # [gluo TODO] raise exception as there is no way to satisfy the request.
        # Can not allocate for sure
        if size > self.buffer_size:
            return None, None
        # Sanity clean up freed list
        self._flush_freed_list()

        # If the allocation will go over end boundary, simply try allocate from the start
        if self.free_start_idx + size > self.end_idx:
            # Not enough space even after wrap around, reject the allocation early
            # so we don't mark the remaining space "used"
            if self.allocated_start_idx < size:
                return None, None
            # add artificial entry to freed_list to treat the remaining space to be
            # allocated and released.
            self.freed_list[self.free_start_idx] = self.end_idx
            self.free_start_idx = 0
            self.wrapped_around = True
        start_idx = self.free_start_idx
        end_idx = start_idx + size

        # Check availability of the buffer, if the allocation overlaps with allocated buffer,
        # return None for the caller to retry later after some buffers are released.
        if self.wrapped_around and end_idx > self.allocated_start_idx:
            return None, None

        # book-keep allocations
        buffer_id = self.id_counter.get_next_id()
        self.allocated_buffer_id_to_range[buffer_id] = (start_idx, end_idx)
        self.free_start_idx = end_idx

        return buffer_id, self.buffer_tensor[start_idx:end_idx]

    def release_buffer(self, buffer_id):
        start_end = self.allocated_buffer_id_to_range.pop(buffer_id, None)
        if start_end is not None:
            self.freed_list[start_end[0]] = start_end[1]
            self._flush_freed_list()


class NixlTransferRequest(BaseModel):
    """
    A TransferRequest subclass that includes additional fields specific to NIXL-based embedding transfer.
    """

    sender_agent_id: str
    # metadata of the given agent ID, can be None if
    # sender determines that the receiver already connected to the sender.
    agent_metadata: Optional[str]
    # The ID of the tensor to be written
    tensor_id: int
    tensor_size: int
    # Wall-clock lease for an emitted descriptor. A receiver must reject an
    # expired lease before allocating or advertising a target buffer.
    expires_at_unix: float | None = None


class NixlWriteEmbeddingSender(AbstractEmbeddingSender):
    """NIXL WRITE-based implementation of the embedding sender interface.

    Designed for scenarios where the sender transmits dynamically allocated
    tensors. Because these tensors allocation is external to the sender,
    NIXL memory registration will perform on each send request. The receiver
    will manage a pre-allocated buffer, so its NIXL metadata is consistent once
    initialized. In such acenarios, let sender initiate the WRITE operations requires
    minimal metadata exchange.

    Protocol:
        1. Record the receiver NIXL metadata, this is done:
            * Implicitly through the first transfer request as fallback if the metadata
              hasn't been recorded.
            * [REMOVED] Explicitly through add_agent() API before calling send_embeddings().
              The receiver provides get_agent_metadata() API to return its NIXL metadata.
              This complicates the implementation and add extra responsiblity on the caller side,
              will revisit the necessity if metadata exchange overhead is significant.
        2. The sender prepares the embeddings and produces a TransferRequest
           containing sender contact and tensor metadata (shape, dtype, size, etc).
        3. The receiver responds with (optional) receiver contact, target tensor
           metadata (buffer address, device, etc) and done signal through NIXL notification.
        4. The sender performs a NIXL WRITE to push the data into the
           receiver's buffer.
    """

    def __init__(self):
        # NIXL agent setup
        nixl_agent, nixl_agent_config = _load_nixl_api()
        self.sender_id = f"sender_{str(uuid.uuid4())}"
        self.nixl_agent = nixl_agent(
            self.sender_id, nixl_agent_config(num_threads=8, capture_telemetry=True)
        )
        self.remote_agents = {}
        self.agent_metadata = self.nixl_agent.get_agent_metadata()
        self.agent_metadata_b64 = base64.b64encode(self.agent_metadata).decode("utf-8")

        # tracker for the prepared embeddings
        self.transfer_tracker = {}
        self.transfer_created_at = {}
        # A failed/cancelled in-flight WRITE must retain its source tensor and
        # memory registration until NIXL reports a terminal handle state.
        self.transfer_failures: dict[int, BaseException] = {}
        # Keep a bounded record of retired IDs so delayed receiver handshakes
        # are answered rather than silently stranding their advertised buffer.
        self.retired_transfer_ids: OrderedDict[int, float] = OrderedDict()
        self.retired_transfer_limit = 4096
        # Terminal notifications release receiver-owned target buffers. Retain
        # transiently failed sends so an idle poll can retry them.
        self.pending_terminal_notifications: OrderedDict[
            tuple[str, int], str
        ] = OrderedDict()
        # The receiver stops accepting this descriptor at its serialized
        # expiry. Keep the sender available a little longer so a handshake
        # emitted just before that cutoff can still receive a terminal reply.
        self.responder_lease_expirations: dict[int, float] = {}
        self.pending_write_requests: OrderedDict[
            tuple[str, int, int], tuple
        ] = OrderedDict()
        self.pending_write_retry_after: dict[tuple[str, int, int], float] = {}
        self.pending_write_retry_attempts: dict[tuple[str, int, int], int] = {}
        self.inflight_transfers: dict[int, list[Any]] = {}
        self._closing = False

        # Track dynamically registered descriptors for cleanup,
        # there can be case of the same tensor being requested to be transferred multiple times,
        # we want to avoid duplicated registration or early deregistration while other transfer
        # of the tensor is still in-flight, so we track the inflight transfer with respect to
        # the actual tensor buffer and only deregister after all transfers of the same tensor is completed.
        self.registered_descs = {}

        self.id_counter = MonolithicCounter()

        # Background transfer task..
        # Create a queue hinting whether the sender is expecting future transfer
        self.transfer_queue: asyncio.Queue[str] = asyncio.Queue()
        self._state_update_task = asyncio.create_task(self._state_update())
        self.transfer_timeout = 60  # seconds, can be tuned based on expected transfer time and network condition

    def __del__(self):
        state_update_task = getattr(self, "_state_update_task", None)
        if state_update_task is not None:
            state_update_task.cancel()

    async def aclose(self) -> None:
        state_update_task = getattr(self, "_state_update_task", None)
        if state_update_task is None:
            return
        self._closing = True
        shutdown_error = RuntimeError("NIXL WRITE sender is shutting down")
        for tensor_id in list(self.transfer_tracker):
            if tensor_id in self.inflight_transfers:
                self.transfer_failures.setdefault(tensor_id, shutdown_error)
            else:
                self._complete_transfer(tensor_id, shutdown_error)
        self.transfer_queue.put_nowait("shutdown")

        # Keep the progress engine alive until every potentially active remote
        # WRITE is terminal, every staged future is settled, registrations are
        # released, and receiver target-buffer acknowledgements are flushed.
        while (
            self.transfer_tracker
            or self.inflight_transfers
            or self.pending_terminal_notifications
            or self.responder_lease_expirations
            or self.pending_write_requests
        ):
            if state_update_task.done():
                await state_update_task
                raise RuntimeError(
                    "NIXL WRITE progress task exited before shutdown drained"
                )
            await asyncio.sleep(0.001)

        if self.registered_descs:
            raise RuntimeError(
                "NIXL WRITE shutdown drained transfers but retained registrations"
            )

        self._state_update_task = None
        state_update_task.cancel()
        try:
            await state_update_task
        except asyncio.CancelledError:
            pass

    async def _state_update(self):
        """Long-running async task that processes transfer requests."""
        inflight_transfers = self.inflight_transfers
        scheduled_transfer_task = None
        while True:
            try:
                # Receiver handshakes arrive through NIXL rather than this local
                # queue, so even the idle loop must poll independently.
                if scheduled_transfer_task is None:
                    try:
                        scheduled_transfer_task = await asyncio.wait_for(
                            self.transfer_queue.get(), timeout=0.01
                        )
                    except TimeoutError:
                        pass

                # check if write is requested, initiate the write
                write_requests = self._get_receiver_handshakes()
                for (
                    remote_agent_id,
                    remote_agent_metadata,
                    tensor_id,
                    (target_buffer, target_byte_size, target_device_id, target_mem_str),
                    write_done_id,
                ) in write_requests:
                    request_key = (remote_agent_id, tensor_id, write_done_id)
                    if time.perf_counter() < self.pending_write_retry_after.get(
                        request_key, 0.0
                    ):
                        continue
                    # Just in time add remote agent if not added
                    if remote_agent_id not in self.remote_agents:
                        if len(remote_agent_metadata) == 0:
                            self._defer_pending_write_request(
                                request_key,
                                "Received NIXL WRITE notification from unknown "
                                f"agent {remote_agent_id} without metadata",
                            )
                            # Keep the consumed handshake and its responder lease
                            # retryable. The guarded lease eventually makes it safe
                            # to retire if first-contact metadata never arrives.
                            continue
                        try:
                            self.remote_agents[
                                remote_agent_id
                            ] = self.nixl_agent.add_remote_agent(remote_agent_metadata)
                        except Exception:
                            self._defer_pending_write_request(
                                request_key,
                                f"Failed to add NIXL WRITE receiver {remote_agent_id}",
                                exc_info=True,
                            )
                            continue

                    if tensor_id in self.transfer_failures:
                        # Cancellation after the target was advertised must
                        # explicitly release that receiver-owned buffer. The
                        # source descriptor is safe to retire because no WRITE
                        # handle has been initialized.
                        self._queue_terminal_notification(
                            remote_agent_id, write_done_id, "ERR"
                        )
                        self._release_pending_write_request(request_key)
                        continue

                    if tensor_id not in self.transfer_tracker:
                        if tensor_id in self.retired_transfer_ids:
                            logger.debug(
                                "Rejecting late write request for retired tensor_id %s",
                                tensor_id,
                            )
                        else:
                            logger.warning(
                                "Rejecting write request for unknown tensor_id %s",
                                tensor_id,
                            )
                        self._queue_terminal_notification(
                            remote_agent_id, write_done_id, "ERR"
                        )
                        self._release_pending_write_request(request_key)
                        continue

                    # Build the transfer transactionally. Failures before a
                    # handle exists cannot have started a WRITE and can be
                    # acknowledged immediately. Once a handle exists, retain
                    # it and its registration until NIXL reports terminal.
                    source_tensor, source_desc, _ = self.transfer_tracker[tensor_id]
                    done_signal = str(write_done_id).encode()
                    try:
                        target_desc = self.nixl_agent.get_xfer_descs(
                            [
                                (
                                    target_buffer,
                                    target_byte_size,
                                    target_device_id,
                                ),
                            ],
                            mem_type=target_mem_str,
                        )
                        xfer_handle = self.nixl_agent.initialize_xfer(
                            "WRITE",
                            source_desc,
                            target_desc,
                            remote_agent_id,
                            done_signal,
                        )
                    except Exception as exc:
                        self._queue_terminal_notification(
                            remote_agent_id, write_done_id, "ERR"
                        )
                        self._release_pending_write_request(request_key)
                        self._complete_transfer(tensor_id, exc)
                        continue

                    inflight_transfers[tensor_id] = [
                        xfer_handle,
                        time.perf_counter(),
                        remote_agent_id,
                        write_done_id,
                    ]
                    self._release_pending_write_request(request_key)
                    try:
                        self.nixl_agent.transfer(xfer_handle, done_signal)
                    except Exception as exc:
                        # transfer() may have submitted work before raising.
                        # Poll the handle to terminal before releasing memory or
                        # acknowledging the receiver's target buffer.
                        self.transfer_failures[tensor_id] = exc

                # check inflight transfer state, if completed, get another task to match
                # remaining transfers count
                # use list() to create a copy of the dict items since the dict will be modified in the loop
                now_time = time.perf_counter()
                for tensor_id, error in list(self.transfer_failures.items()):
                    if tensor_id not in self.transfer_tracker:
                        self.transfer_failures.pop(tensor_id, None)
                    elif tensor_id not in inflight_transfers:
                        # No NIXL handle owns this registration yet, so retiring
                        # an unclaimed descriptor is immediately safe.
                        self._complete_transfer(tensor_id, error)
                for tensor_id, created_at in list(self.transfer_created_at.items()):
                    if tensor_id in inflight_transfers:
                        continue
                    if now_time - created_at <= self.transfer_timeout:
                        continue
                    logger.warning(
                        "Prepared tensor_id %s was not claimed within %s seconds",
                        tensor_id,
                        self.transfer_timeout,
                    )
                    self._complete_transfer(
                        tensor_id,
                        TimeoutError(
                            "embedding transfer was not claimed before expiry"
                        ),
                    )
                for tensor_id, (
                    xfer_handle,
                    start_time,
                    remote_agent_id,
                    write_done_id,
                ) in list(inflight_transfers.items()):
                    state = self.nixl_agent.check_xfer_state(xfer_handle)
                    if state == "ERR":
                        logger.error(f"Transfer failed for tensor_id {tensor_id}")
                    elif state == "DONE":
                        logger.debug(
                            f"Send completed for tensor_id {tensor_id}, total wait time: {now_time - start_time:.2f} seconds"
                        )
                    else:
                        # still in-flight, check again later
                        if now_time - start_time > self.transfer_timeout:
                            if tensor_id not in self.transfer_failures:
                                logger.warning(
                                    f"Transfer for tensor_id {tensor_id} exceeded the "
                                    f"{self.transfer_timeout} second timeout; retaining "
                                    "registered memory until NIXL reaches a terminal state"
                                )
                                self.transfer_failures[tensor_id] = TimeoutError(
                                    "embedding transfer timed out"
                                )
                        continue
                    # The receiver owns the target buffer and must not recycle it
                    # until the WRITE handle is terminal.  The normal NIXL done
                    # signal only covers successful transfers, so send an explicit
                    # terminal acknowledgement for both DONE and ERR.
                    self._queue_terminal_notification(
                        remote_agent_id, write_done_id, state
                    )
                    transfer_error = self.transfer_failures.pop(tensor_id, None)
                    if transfer_error is None and state == "ERR":
                        transfer_error = RuntimeError(
                            f"NIXL WRITE failed for tensor_id {tensor_id}"
                        )
                    self._complete_transfer(tensor_id, transfer_error)
                    inflight_transfers.pop(tensor_id)
                    try:
                        scheduled_transfer_task = self.transfer_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        if inflight_transfers:
                            logger.error(
                                f"Unexpected no scheduled transfer request, while there are still {len(inflight_transfers)} inflight transfers"
                            )
                            # Continue the loop to check the state of remaining inflight transfers
                            continue
                        logger.debug("No pending transfer task in the queue.")
                        scheduled_transfer_task = None
                        break

                if not self.transfer_tracker and not inflight_transfers:
                    # Queue entries are wake-up hints, not a transfer ledger. A
                    # cancellation or unclaimed-descriptor expiry may retire the
                    # last transfer without passing through the DONE branch above.
                    # Drain stale hints before returning to the blocking wait.
                    while True:
                        try:
                            self.transfer_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    scheduled_transfer_task = None

                self._flush_terminal_notifications()
                self._expire_responder_leases()

                # short pause to yield control and allow cancellation
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in state update loop: {e}")
                await asyncio.sleep(1)  # Backoff on error to prevent tight error loop

    def _get_receiver_handshakes(self):
        write_requests = []
        notifs = self.nixl_agent.get_new_notifs()
        for remote_agent_id, notifs in notifs.items():
            for notif in notifs:
                decoded = msgspec.msgpack.decode(notif)
                if (
                    isinstance(decoded, (list, tuple))
                    and len(decoded) == 2
                    and decoded[0] == "cancel"
                ):
                    tensor_id = decoded[1]
                    has_advertised_target = any(
                        request_key[1] == tensor_id
                        for request_key in self.pending_write_requests
                    )
                    if not has_advertised_target:
                        self.responder_lease_expirations.pop(tensor_id, None)
                    self.transfer_failures.setdefault(
                        tensor_id,
                        RuntimeError("embedding transfer cancelled by receiver"),
                    )
                    continue
                (
                    tensor_id,
                    (target_buffer, target_byte_size, target_device_id, target_mem_str),
                    write_done_id,
                    remote_agent_metadata,
                ) = decoded
                write_request = (
                    # receiver contact
                    remote_agent_id,
                    remote_agent_metadata,
                    # source tensor
                    tensor_id,
                    # target tensor
                    # (note byte size can be retrieved from source tensor)
                    (
                        target_buffer,
                        target_byte_size,
                        target_device_id,
                        target_mem_str,
                    ),
                    # done signal
                    write_done_id,
                )
                request_key = (remote_agent_id, tensor_id, write_done_id)
                if self.pending_write_requests.get(request_key) != write_request:
                    self.pending_write_retry_after.pop(request_key, None)
                    self.pending_write_retry_attempts.pop(request_key, None)
                self.pending_write_requests[request_key] = write_request
        write_requests.extend(self.pending_write_requests.values())
        return write_requests

    def _release_pending_write_request(self, request_key: tuple[str, int, int]) -> None:
        self.pending_write_requests.pop(request_key, None)
        self.pending_write_retry_after.pop(request_key, None)
        self.pending_write_retry_attempts.pop(request_key, None)
        self.responder_lease_expirations.pop(request_key[1], None)

    def _defer_pending_write_request(
        self,
        request_key: tuple[str, int, int],
        message: str,
        *,
        exc_info: bool = False,
    ) -> None:
        attempts = self.pending_write_retry_attempts.get(request_key, 0) + 1
        self.pending_write_retry_attempts[request_key] = attempts
        max_exponent = math.ceil(
            math.log2(NIXL_WRITE_RETRY_MAX_SECONDS / NIXL_WRITE_RETRY_INITIAL_SECONDS)
        )
        delay = min(
            NIXL_WRITE_RETRY_INITIAL_SECONDS * (2 ** min(attempts - 1, max_exponent)),
            NIXL_WRITE_RETRY_MAX_SECONDS,
        )
        self.pending_write_retry_after[request_key] = time.perf_counter() + delay
        logger.warning(
            "%s; retaining handshake and retrying in %.3f seconds",
            message,
            delay,
            exc_info=exc_info,
        )

    def _queue_terminal_notification(
        self, remote_agent_id: str, write_done_id: int, state: str
    ) -> None:
        key = (remote_agent_id, write_done_id)
        self.pending_terminal_notifications[key] = state
        self.pending_terminal_notifications.move_to_end(key)
        self._flush_terminal_notifications()

    def _flush_terminal_notifications(self) -> None:
        for (remote_agent_id, write_done_id), state in list(
            self.pending_terminal_notifications.items()
        ):
            try:
                self.nixl_agent.send_notif(
                    remote_agent_id,
                    notif_msg=msgspec.msgpack.encode(
                        ("terminal", write_done_id, state)
                    ),
                )
            except Exception:
                logger.warning(
                    "Failed to send terminal acknowledgement for receiver "
                    "%s transfer %s; retaining it for retry",
                    remote_agent_id,
                    write_done_id,
                    exc_info=True,
                )
                continue
            self.pending_terminal_notifications.pop(
                (remote_agent_id, write_done_id), None
            )

    def _expire_responder_leases(self) -> None:
        now = time.time()
        for tensor_id, expires_at in list(self.responder_lease_expirations.items()):
            if now >= expires_at:
                # An observed handshake owns its receiver buffer until an
                # in-flight handle or terminal acknowledgement takes over.
                # In particular, do not discard the only copy merely because
                # first-contact remote-agent setup is temporarily failing.
                if any(key[1] == tensor_id for key in self.pending_write_requests):
                    continue
                self.responder_lease_expirations.pop(tensor_id, None)
                self.retired_transfer_ids.pop(tensor_id, None)

    def _record_retired_transfer(self, tensor_id: int) -> None:
        self.retired_transfer_ids[tensor_id] = time.perf_counter()
        self.retired_transfer_ids.move_to_end(tensor_id)
        while len(self.retired_transfer_ids) > self.retired_transfer_limit:
            self.retired_transfer_ids.popitem(last=False)

    def _complete_transfer(self, tensor_id, error: BaseException | None = None):
        transfer_info = self.transfer_tracker.pop(tensor_id, None)
        self.transfer_created_at.pop(tensor_id, None)
        self.transfer_failures.pop(tensor_id, None)
        if transfer_info is not None:
            self._record_retired_transfer(tensor_id)
            # Clean up registered memory after transfer completion
            embeddings, _, fut = transfer_info
            desc_key = (embeddings.data_ptr(), embeddings.get_device())
            self.registered_descs[desc_key][1] -= 1
            if self.registered_descs[desc_key][1] == 0:
                self.nixl_agent.deregister_memory(self.registered_descs[desc_key][0])
                del self.registered_descs[desc_key]
            # Future can be 'done' if the embeddings is not external
            # (send_embeddings with stage_embeddings=False)
            if not fut.done():
                if error is None:
                    fut.set_result(None)
                else:
                    fut.set_exception(error)

    async def send_embeddings(
        self,
        embeddings: torch.Tensor,
        stage_embeddings: bool = False,
    ) -> tuple[TransferRequest, asyncio.Future]:
        """
        Send precomputed embeddings.

        Args:
            embeddings: A torch.Tensor of the embeddings to send.
            stage_embeddings: A boolean indicating whether the embeddings should be staged for the transfer,
            if True, the embeddings may be used as transfer buffer and must not be released until the return future is completed.
        Returns:
            A tuple containing the TransferRequest object and an awaitable that can be awaited to indicate the send is completed.
        """
        if self._closing:
            raise RuntimeError("NIXL WRITE sender is shutting down")
        tensor_id = self.id_counter.get_next_id()
        expires_at_unix = time.time() + self.transfer_timeout
        responder_expires_at_unix = expires_at_unix + NIXL_WRITE_LEASE_GUARD_SECONDS
        fut = asyncio.get_event_loop().create_future()
        if not stage_embeddings:
            embeddings = embeddings.clone().detach()
            fut.set_result(None)

        # In case the same embedding tensor is sent multiple times,
        # we want to avoid potential issues with duplicated NIXL memory registration.
        desc_key = (embeddings.data_ptr(), embeddings.get_device())
        if desc_key not in self.registered_descs:
            registered_desc = self.nixl_agent.register_memory(embeddings)
            self.registered_descs[desc_key] = [registered_desc, 1]
        else:
            self.registered_descs[desc_key][1] += 1

        desc = self.nixl_agent.get_xfer_descs(embeddings)
        # use tracker to also extend lifecycle of transfer-related objects
        self.transfer_tracker[tensor_id] = (embeddings, desc, fut)
        self.transfer_created_at[tensor_id] = time.perf_counter()
        self.responder_lease_expirations[tensor_id] = responder_expires_at_unix
        self.transfer_queue.put_nowait("task_indicator")

        request = TransferRequest(
            embeddings_shape=list(embeddings.shape),
            embedding_dtype_str=torch_dtype_to_string(embeddings.dtype),
            serialized_request=NixlTransferRequest(
                sender_agent_id=self.sender_id,
                agent_metadata=self.agent_metadata_b64,
                tensor_id=tensor_id,
                tensor_size=embeddings.nbytes,
                expires_at_unix=expires_at_unix,
            ).model_dump_json(),
        )
        return request, fut


class NixlWriteEmbeddingReceiver(AbstractEmbeddingReceiver):
    """
    Counter part of 'NixlWriteEmbeddingSender', see 'NixlWriteEmbeddingSender' for details.
    The receiver manages a ring buffer for sender to write the embeddings into, and respond
    to the sender's transfer request with the buffer information for the WRITE transfer.
    """

    def __init__(self, buffer_size=2 * 8 * 1024 * 1024 * 256 * 2):
        # the default buffer_size is the product of:
        # 2 (typical dtype size float16)
        # 8 * 1024 (typical embedding hidden size for Qwen-VL)
        # 256 * 1024 (1024 count of 256 mm token item)
        # 2 (extra copies) = 8 GB memory
        # ring buffer without wrapped around allocation, i.e. will allocate from
        # start if the last remaining buffer is not enough
        self.ring_buffer = RingBuffer(buffer_size)
        self.transfer_tensor = self.ring_buffer.buffer_tensor

        # NIXL agent setup
        nixl_agent, nixl_agent_config = _load_nixl_api()
        self.receiver_id = f"receiver_{str(uuid.uuid4())}"
        self.nixl_agent = nixl_agent(
            self.receiver_id, nixl_agent_config(num_threads=8, capture_telemetry=True)
        )
        self.remote_agents = {}
        self.reg_descs = self.nixl_agent.register_memory(self.transfer_tensor)
        self.agent_metadata = self.nixl_agent.get_agent_metadata()

        self.id_counter = MonolithicCounter()
        self.to_buffer_id = {}
        # A receive coroutine can be cancelled after advertising its target
        # buffer.  Keep such buffers quarantined until the sender confirms that
        # the remote WRITE handle is terminal.
        self._quarantine_tasks: set[asyncio.Task] = set()

    def _pop_terminal_state(self, sender_agent_id: str, tensor_id: int):
        self.nixl_agent.update_notifs()
        sender_notifs = self.nixl_agent.notifs.get(sender_agent_id, [])
        done_signal = str(tensor_id).encode()
        terminal_state = None
        for notif in list(sender_notifs):
            if notif == done_signal:
                # The explicit terminal acknowledgement below supersedes the
                # success-only NIXL notification.  Discard it to avoid buildup.
                sender_notifs.remove(notif)
                continue
            try:
                decoded = msgspec.msgpack.decode(notif)
            except Exception:
                continue
            if (
                isinstance(decoded, (list, tuple))
                and len(decoded) == 3
                and decoded[0] == "terminal"
                and decoded[1] == tensor_id
            ):
                sender_notifs.remove(notif)
                terminal_state = decoded[2]
                break
        return terminal_state

    async def _wait_for_terminal_state(
        self,
        sender_agent_id: str,
        tensor_id: int,
        timeout: float | None,
    ) -> str:
        start_time = time.perf_counter()
        while True:
            state = self._pop_terminal_state(sender_agent_id, tensor_id)
            if state in {"DONE", "ERR"}:
                return state
            if timeout is not None and time.perf_counter() - start_time > timeout:
                raise TimeoutError(
                    "Timeout while waiting for transfer completion for "
                    f"tensor_id {tensor_id} for more than {timeout} seconds"
                )
            await asyncio.sleep(0.001)

    def _quarantine_buffer(
        self,
        buffer_id: int,
        sender_agent_id: str,
        tensor_id: int,
    ) -> None:
        async def wait_and_release():
            try:
                await self._wait_for_terminal_state(
                    sender_agent_id, tensor_id, timeout=None
                )
            except asyncio.CancelledError:
                # At process shutdown there is no future request to protect.
                # During normal operation, never recycle a target whose remote
                # WRITE may still be active.
                raise
            except Exception:
                logger.exception(
                    "Failed while quarantining NIXL WRITE buffer %s", buffer_id
                )
                return
            self.ring_buffer.release_buffer(buffer_id)

        task = asyncio.create_task(wait_and_release())
        self._quarantine_tasks.add(task)
        task.add_done_callback(self._quarantine_tasks.discard)

    def _notify_cancel(self, nixl_request: NixlTransferRequest) -> None:
        self.nixl_agent.send_notif(
            nixl_request.sender_agent_id,
            notif_msg=msgspec.msgpack.encode(("cancel", nixl_request.tensor_id)),
        )

    async def receive_embeddings(
        self, request: TransferRequest, receive_timeout=60
    ) -> tuple[int, torch.Tensor]:
        """
        Receive precomputed embeddings for a given request ID.

        Args:
            request: The TransferRequest object containing information to receive embeddings for.
            receive_timeout: Maximum time to wait for the transfer to complete before raising a TimeoutError.
            The timeout will be applied separately for waiting for available buffer and waiting for transfer completion.

        Returns:
            A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
            Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
        """
        nixl_request = NixlTransferRequest.model_validate_json(
            request.serialized_request
        )
        if (
            nixl_request.expires_at_unix is not None
            and time.time() >= nixl_request.expires_at_unix
        ):
            raise TimeoutError("NIXL WRITE descriptor lease expired")
        embeddings_shape = request.embeddings_shape
        if not embeddings_shape or any(
            dimension <= 0 for dimension in embeddings_shape
        ):
            raise ValueError(
                f"Embedding shape must contain only positive dimensions: {embeddings_shape}"
            )
        embeddings_dtype = torch_dtype_from_string(request.embedding_dtype_str)
        expected_tensor_size = (
            math.prod(embeddings_shape)
            * torch.empty((), dtype=embeddings_dtype).element_size()
        )
        if nixl_request.tensor_size <= 0:
            raise ValueError("Embedding tensor_size must be positive")
        if nixl_request.tensor_size != expected_tensor_size:
            raise ValueError(
                "Embedding tensor_size does not match shape and dtype: "
                f"got {nixl_request.tensor_size}, expected {expected_tensor_size}"
            )
        if nixl_request.sender_agent_id not in self.remote_agents:
            if nixl_request.agent_metadata is None:
                raise ValueError(
                    f"Missing agent metadata for new sender {nixl_request.sender_agent_id}"
                )
            self.remote_agents[
                nixl_request.sender_agent_id
            ] = self.nixl_agent.add_remote_agent(
                base64.b64decode(nixl_request.agent_metadata)
            )

        # Allocate tensor to be written into.
        start_time = time.perf_counter()
        while True:
            buffer_id, transfer_tensor = self.ring_buffer.get_buffer(
                nixl_request.tensor_size
            )
            if transfer_tensor is not None:
                break

            # No available buffer, wait for a short period and retry.
            # The receiver side should have concurrent work on other
            # allocated buffer and release them in a timely manner,
            # so the wait time should not be long.
            #
            # NOTE This approach can result in deadlock due to
            # the current usage of the receiver:
            # The case of concurrent requests may request 2 buffer in order,
            # if all request get the first buffer and exhaust the ring buffer,
            # then no request can get the second buffer and proceed.
            # On raising the timeout error from this function, the caller must
            # release all previously allocated tensor of the request to unblock
            # other requests, and retry the request after some delay to avoid
            # repeated deadlock.
            # [gluo WIP] provide an API for batch allocation so some requests can
            # proceed.
            if time.perf_counter() - start_time > receive_timeout:
                raise TimeoutError("Timeout while waiting for available buffer.")
            if (
                nixl_request.expires_at_unix is not None
                and time.time() >= nixl_request.expires_at_unix
            ):
                raise TimeoutError("NIXL WRITE descriptor lease expired")
            await asyncio.sleep(0.005)
        handshake_sent = False
        terminal_state = None
        try:
            if (
                nixl_request.expires_at_unix is not None
                and time.time() >= nixl_request.expires_at_unix
            ):
                raise TimeoutError("NIXL WRITE descriptor lease expired")
            # View as tensor matching the source tensor.  Keep every operation
            # after allocation under this cleanup guard.
            embedding_tensor = transfer_tensor.view(dtype=embeddings_dtype).view(
                embeddings_shape
            )
            # Request for transfer
            tensor_id = self.id_counter.get_next_id()
            notif_msg = msgspec.msgpack.encode(
                (
                    nixl_request.tensor_id,
                    (
                        transfer_tensor.data_ptr(),
                        nixl_request.tensor_size,
                        # torch returns -1 for CPU device, need to normalized there
                        max(transfer_tensor.get_device(), 0),
                        "cuda"
                        if str(transfer_tensor.device).startswith("cuda")
                        else "cpu",
                    ),
                    tensor_id,
                    # side channel handshake fallback for receiver API consistency,
                    # this will increase message size for the first few transfers before handshake
                    self.agent_metadata if nixl_request.agent_metadata else b"",
                )
            )
            self.nixl_agent.send_notif(
                nixl_request.sender_agent_id, notif_msg=notif_msg
            )
            handshake_sent = True

            # Await an explicit terminal acknowledgement.  Unlike the native
            # done signal, this is emitted for both successful and failed
            # transfers, so cancellation can safely quarantine the target.
            start_time = time.perf_counter()
            terminal_state = await self._wait_for_terminal_state(
                nixl_request.sender_agent_id,
                tensor_id,
                timeout=receive_timeout,
            )
            if terminal_state == "ERR":
                raise RuntimeError(f"NIXL WRITE failed for tensor_id {tensor_id}")
        except BaseException:
            if not handshake_sent or terminal_state is not None:
                self.ring_buffer.release_buffer(buffer_id)
            else:
                try:
                    self._notify_cancel(nixl_request)
                except Exception:
                    logger.warning(
                        "Failed to notify sender about cancelled NIXL WRITE",
                        exc_info=True,
                    )
                self._quarantine_buffer(
                    buffer_id,
                    nixl_request.sender_agent_id,
                    tensor_id,
                )
            raise
        logger.debug(
            f"Transfer completed for tensor_id {tensor_id}, total wait time: {time.perf_counter() - start_time:.2f} seconds"
        )

        self.to_buffer_id[tensor_id] = buffer_id
        return tensor_id, embedding_tensor

    async def cancel_embeddings(self, request: TransferRequest) -> None:
        nixl_request = NixlTransferRequest.model_validate_json(
            request.serialized_request
        )
        if nixl_request.sender_agent_id not in self.remote_agents:
            if nixl_request.agent_metadata is None:
                return
            self.remote_agents[
                nixl_request.sender_agent_id
            ] = self.nixl_agent.add_remote_agent(
                base64.b64decode(nixl_request.agent_metadata)
            )
        self._notify_cancel(nixl_request)

    def release_tensor(self, tensor_id: int) -> None:
        """
        Indicate that the tensor associated with the ID is no longer in use.

        Args:
            tensor_id: The ID of the tensor to release.
        """
        buffer_id = self.to_buffer_id.pop(tensor_id)
        self.ring_buffer.release_buffer(buffer_id)


class NixlReadEmbeddingSender(AbstractEmbeddingSender):
    """NIXL READ based embedding transfer sender.

    Uses nixl_connect.Connector which now natively provides a shared singleton
    Connection (NIXL agent) and reference-counted Remote agent lifecycle.
    """

    def __init__(self):
        self._nixl_connect = _load_nixl_connect()
        self.connector = self._nixl_connect.Connector()

    @_nvtx.annotate("mm:nixl:send_embeddings", color="magenta")
    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
    ) -> tuple[TransferRequest, Awaitable[None]]:
        """
        Send precomputed embeddings.

        Args:
            embeddings: A torch.Tensor of the embeddings to send.
            stage_embeddings: A boolean indicating whether the embeddings should be staged for the transfer,
            if True, the embeddings may be used as transfer buffer and must not be released until the return future is completed.
            if False, the sender will copy the embeddings.
        Returns:
            A tuple containing the TransferRequest object and an awaitable that can be awaited to indicate the send is completed.
        """
        if stage_embeddings:
            transfer_buf = embeddings
        else:
            transfer_buf = embeddings.clone().detach()
        with _nvtx.annotate("mm:nixl:create_descriptor", color="pink"):
            descriptor = self._nixl_connect.Descriptor(transfer_buf)
        with _nvtx.annotate("mm:nixl:create_readable", color="pink"):
            try:
                readable_op = await self.connector.create_readable(descriptor)
            except Exception as exc:
                # If NIXL registration fails for a device tensor, fall back to CPU staging.
                if not transfer_buf.device.type == "cpu":
                    logger.warning(
                        "NIXL registration failed for %s tensor, falling back "
                        "to CPU staging: %s",
                        transfer_buf.device.type,
                        exc,
                    )
                    transfer_buf = transfer_buf.cpu()
                    descriptor = self._nixl_connect.Descriptor(transfer_buf)
                    readable_op = await self.connector.create_readable(descriptor)
                else:
                    raise
        request = TransferRequest(
            embeddings_shape=list(embeddings.shape),
            embedding_dtype_str=torch_dtype_to_string(embeddings.dtype),
            serialized_request=readable_op.metadata().model_dump(),
        )
        return request, readable_op.wait_for_completion()


class NixlReadEmbeddingReceiver(AbstractEmbeddingReceiver):
    """NIXL READ based embedding transfer receiver.

    Uses nixl_connect.Connector which now natively provides a shared singleton
    Connection (NIXL agent) and reference-counted Remote agent lifecycle.
    """

    def __init__(
        self,
        embedding_hidden_size: int = 8 * 1024,
        max_item_mm_token: int = 1024,
        max_items: int = 1024,
    ) -> None:
        super().__init__()
        self._nixl_connect = _load_nixl_connect()
        self.connector = self._nixl_connect.Connector()
        self.tensor_id_counter = 0
        self.aggregated_op_create_time = 0
        self.aggregated_op_wait_time = 0
        self.warmedup_descriptors: Queue[Any] = Queue()
        self.inuse_descriptors: dict[int, tuple[Any, bool]] = {}
        connection = run_async(self.connector._create_connection)
        # Create descriptor for our allocated tensor
        for _ in range(max_items):
            encodings_tensor = torch.zeros(
                max_item_mm_token * embedding_hidden_size, dtype=torch.int8
            )
            descriptor = self._nixl_connect.Descriptor(encodings_tensor)
            descriptor.register_with_connector(connection)
            self.warmedup_descriptors.put(descriptor)

    @_nvtx.annotate("mm:nixl:receive_embeddings", color="magenta")
    async def receive_embeddings(
        self, request: TransferRequest
    ) -> tuple[int, torch.Tensor]:
        """
        Receive precomputed embeddings for a given request ID.

        Args:
            request: The TransferRequest object containing information to receive embeddings for.

        Returns:
            A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
            Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
        """
        # Extract dynamic shape, metadata, and auxiliary data
        embeddings_shape = request.embeddings_shape
        embeddings_dtype = torch_dtype_from_string(request.embedding_dtype_str)
        readable_metadata = self._nixl_connect.RdmaMetadata.model_validate(
            request.serialized_request
        )

        original_descriptor_size = None
        if self.warmedup_descriptors.empty():
            logger.debug(
                "No warmed up descriptors available, creating a temporary one for transfer."
            )
            encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)
            descriptor = self._nixl_connect.Descriptor(encodings_tensor)
            dynamic_descriptor = True
        else:
            descriptor = self.warmedup_descriptors.get()
            # Slide view of pre-allocated tensor
            original_descriptor_size = descriptor._data_size
            tensor_size_bytes = embeddings_dtype.itemsize * math.prod(embeddings_shape)
            descriptor._data_size = tensor_size_bytes
            assert descriptor._data_ref is not None
            encodings_tensor = (
                descriptor._data_ref[:tensor_size_bytes]
                .view(dtype=embeddings_dtype)
                .view(embeddings_shape)
            )
            dynamic_descriptor = False

        with _nvtx.annotate("mm:nixl:begin_read", color="pink"):
            # Create read operation to read from EncodeHandler
            read_op = await self.connector.begin_read(readable_metadata, descriptor)
        with _nvtx.annotate("mm:nixl:wait_completion", color="pink"):
            # Wait for the read operation to complete
            await read_op.wait_for_completion()
        logging.debug(
            f"Successfully read embeddings via NIXL: {encodings_tensor.shape}"
        )
        if original_descriptor_size is not None:
            descriptor._data_size = original_descriptor_size
        tensor_id = self.tensor_id_counter
        self.tensor_id_counter += 1
        self.inuse_descriptors[tensor_id] = (descriptor, dynamic_descriptor)
        return tensor_id, encodings_tensor

    def release_tensor(self, tensor_id: int) -> None:
        """
        Indicate that the tensor associated with the ID is no longer in use.

        Args:
            tensor_id: The ID of the tensor to release.
        """
        if tensor_id in self.inuse_descriptors:
            descriptor, dynamic_descriptor = self.inuse_descriptors[tensor_id]
            # Only put back to warmedup_descriptors if it's not dynamically created, as dynamic ones
            # may have varied shapes and putting them back may cause shape mismatch for future receive operations.
            if not dynamic_descriptor:
                self.warmedup_descriptors.put(descriptor)
            del self.inuse_descriptors[tensor_id]
