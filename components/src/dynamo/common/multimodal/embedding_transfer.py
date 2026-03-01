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
from queue import Queue
from typing import Any, List, Optional

import msgpack
import torch
from nixl._api import nixl_agent, nixl_agent_config
from pydantic import BaseModel
from safetensors import torch as safetensors_torch

import dynamo.nixl_connect as nixl_connect

logger = logging.getLogger(__name__)


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
    return getattr(torch, dtype_str.removeprefix("torch."), torch.float32)


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
    def release_tensor(self, tensor_id: int):
        """
        Abstract method to indicate that the tensor associated with the ID is no longer in use.
        Args:
            tensor_id: The ID of the tensor to release.
        """
        pass


class AbstractEmbeddingSender(ABC):
    """
    Abstract base class for a sender of precomputed embeddings to the downstream worker.
    """

    @abstractmethod
    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
    ) -> tuple[TransferRequest, asyncio.Future]:
        """
        Abstract method to send precomputed embeddings for a given request ID.

        Args:
            embeddings: A torch.Tensor of the embeddings to send.
            stage_embeddings: A boolean indicating whether the embeddings should be staged for the transfer,
            if True, the embeddings may be used as transfer buffer and must not be released until the return future is completed.
        Returns:
            A tuple containing the TransferRequest object and a future that can be awaited to indicate the send is completed.
        """
        pass


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

    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
    ) -> tuple[TransferRequest, asyncio.Future]:
        """
        Send precomputed embeddings for a given request ID.

        Args:
            embeddings: A torch.Tensor of the embeddings to send.
            stage_embeddings: A boolean indicating whether the embeddings should be staged for the transfer,
            if True, the embeddings may be used as transfer buffer and must not be released until the return future is completed.
        Returns:
            A tuple containing the TransferRequest object and a future that can be awaited to indicate the send is completed.
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

    def release_tensor(self, tensor_id: int):
        """
        Indicate that the tensor associated with the ID is no longer in use.

        Args:
            tensor_id: The ID of the tensor to release.
        """
        if tensor_id in self.received_tensors:
            file_path = self.received_tensors[tensor_id]
            os.remove(file_path)  # Clean up the local file
            del self.received_tensors[tensor_id]


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
        self.sender_id = f"sender_{str(uuid.uuid4())}"
        self.nixl_agent = nixl_agent(
            self.sender_id, nixl_agent_config(num_threads=8, capture_telemetry=True)
        )
        self.remote_agents = {}
        self.agent_metadata = self.nixl_agent.get_agent_metadata()
        self.agent_metadata_b64 = base64.b64encode(self.agent_metadata).decode("utf-8")

        # tracker for the prepared embeddings
        self.transfer_tracker = {}

        # Track dynamically registered descriptors for cleanup,
        # there can be case of the same tensor being requested to be transferred multiple times,
        # we want to avoid duplicated registration or early deregistration while other transfer
        # of the tensor is still in-flight, so we track the inflight transfer with respect to
        # the actual tensor buffer and only deregister after all transfers of the same tensor is completed.
        self.registered_descs = {}

        self.id_counter = MonolithicCounter()

        # Background transfer task..
        # Create a queue hinting whether the sender is expecting future transfer
        self.transfer_queue = asyncio.Queue()
        self._state_update_task = asyncio.create_task(self._state_update())

    def __del__(self):
        self._state_update_task.cancel()

    async def _state_update(self):
        """Long-running async task that processes transfer requests."""
        inflight_transfers = {}
        scheduled_transfer_task = None
        while True:
            try:
                # If there is no scheduled transfer task, blocking wait for
                # a new transfer request because no state needs to be updated.
                if scheduled_transfer_task is None:
                    scheduled_transfer_task = await self.transfer_queue.get()

                # check if write is requested, initiate the write
                write_requests = self._get_receiver_handshakes()
                for (
                    remote_agent_id,
                    remote_agent_metadata,
                    tensor_id,
                    (target_buffer, target_byte_size, target_device_id, target_mem_str),
                    write_done_id,
                ) in write_requests:
                    # Just in time add remote agent if not added
                    if remote_agent_id not in self.remote_agents:
                        if len(remote_agent_metadata) == 0:
                            logger.error(
                                f"Received transfer notification from unknown agent {remote_agent_id} without metadata, cannot add remote agent for transfer"
                            )
                            # Can't proceed with the transfer without receiver metadata,
                            # mark the transfer as completed to unblock the sender.
                            self._complete_transfer(tensor_id)
                            continue
                        self.remote_agents[
                            remote_agent_id
                        ] = self.nixl_agent.add_remote_agent(remote_agent_metadata)

                    # initiate NIXL WRITE transfer
                    source_tensor, source_desc, _ = self.transfer_tracker[tensor_id]
                    target_desc = self.nixl_agent.get_xfer_descs(
                        [
                            (target_buffer, target_byte_size, target_device_id),
                        ],
                        mem_type=target_mem_str,
                    )
                    done_signal = str(write_done_id).encode()
                    xfer_handle = self.nixl_agent.initialize_xfer(
                        "WRITE",
                        source_desc,
                        target_desc,
                        remote_agent_id,
                        done_signal,
                    )
                    self.nixl_agent.transfer(xfer_handle, done_signal)

                    inflight_transfers[tensor_id] = (
                        xfer_handle,
                        time.perf_counter(),
                    )

                # check inflight transfer state, if completed, get another task to match
                # remaining transfers count
                # use list() to create a copy of the dict items since the dict will be modified in the loop
                for tensor_id, (
                    xfer_handle,
                    start_time,
                ) in list(inflight_transfers.items()):
                    state = self.nixl_agent.check_xfer_state(xfer_handle)
                    if state == "ERR":
                        logger.error(f"Transfer failed for tensor_id {tensor_id}")
                    elif state == "DONE":
                        logger.debug(
                            f"Send completed for tensor_id {tensor_id}, total wait time: {time.perf_counter() - start_time:.2f} seconds"
                        )
                    else:
                        # still in-flight, check again later
                        continue
                    # NOTE future is set with result None in "ERR" and "DONE", so the sender will not
                    # be able to distinguish failure with success, we can consider
                    # adding more explicit failure signal in the future if needed.
                    self._complete_transfer(tensor_id)
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
                (
                    tensor_id,
                    (target_buffer, target_byte_size, target_device_id, target_mem_str),
                    write_done_id,
                    remote_agent_metadata,
                ) = msgpack.unpackb(notif)
                write_requests.append(
                    (
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
                )
        return write_requests

    def _complete_transfer(self, tensor_id):
        transfer_info = self.transfer_tracker.pop(tensor_id, None)
        if transfer_info is not None:
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
                fut.set_result(None)

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
            A tuple containing the TransferRequest object and a future that can be awaited to indicate the send is completed.
        """
        tensor_id = self.id_counter.get_next_id()
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
        self.transfer_queue.put_nowait("task_indicator")

        request = TransferRequest(
            embeddings_shape=list(embeddings.shape),
            embedding_dtype_str=torch_dtype_to_string(embeddings.dtype),
            serialized_request=NixlTransferRequest(
                sender_agent_id=self.sender_id,
                agent_metadata=self.agent_metadata_b64,
                tensor_id=tensor_id,
                tensor_size=embeddings.nbytes,
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
        self.receiver_id = f"receiver_{str(uuid.uuid4())}"
        self.nixl_agent = nixl_agent(
            self.receiver_id, nixl_agent_config(num_threads=8, capture_telemetry=True)
        )
        self.remote_agents = {}
        self.reg_descs = self.nixl_agent.register_memory(self.transfer_tensor)
        self.agent_metadata = self.nixl_agent.get_agent_metadata()

        self.id_counter = MonolithicCounter()
        self.to_buffer_id = {}

    async def receive_embeddings(
        self, request: TransferRequest, allocation_timeout=10
    ) -> tuple[int, torch.Tensor]:
        """
        Receive precomputed embeddings for a given request ID.

        Args:
            request: The TransferRequest object containing information to receive embeddings for.

        Returns:
            A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
            Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
        """
        nixl_request = NixlTransferRequest.model_validate_json(
            request.serialized_request
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
            if time.perf_counter() - start_time > allocation_timeout:
                raise TimeoutError("Timeout while waiting for available buffer.")
            await asyncio.sleep(0.005)
        # view as tensor matching the source tensor..
        embeddings_shape = request.embeddings_shape
        embeddings_dtype = torch_dtype_from_string(request.embedding_dtype_str)
        embedding_tensor = transfer_tensor.view(dtype=embeddings_dtype).view(
            embeddings_shape
        )

        # Request for transfer
        tensor_id = self.id_counter.get_next_id()
        notif_msg = msgpack.packb(
            (
                nixl_request.tensor_id,
                (
                    transfer_tensor.data_ptr(),
                    nixl_request.tensor_size,
                    # torch returns -1 for CPU device, need to normalized there
                    max(transfer_tensor.get_device(), 0),
                    "cuda" if str(transfer_tensor.device).startswith("cuda") else "cpu",
                ),
                tensor_id,
                # side channel handshake fallback for receiver API consistency,
                # this will increase message size for the first few transfers before handshake
                self.agent_metadata if nixl_request.agent_metadata else b"",
            )
        )
        self.nixl_agent.send_notif(nixl_request.sender_agent_id, notif_msg=notif_msg)

        # await for write notification
        start = time.perf_counter()
        done_signal = str(tensor_id).encode()
        found = False
        logged = False
        while not found:
            # parse notifications to find done signal, we can't use 'check_remote_xfer_done' API
            # because it match requested string pattern in substring of the notifications instead
            # of exact match, which is not what we want, i.e. for two done signal "1" and "11",
            # 'check_remote_xfer_done("1")' will return True for both signal and "11" will be cleared
            # as a result, leading the subsequent 'check_remote_xfer_done("1")' returns False.
            notifs = self.nixl_agent.update_notifs()
            if nixl_request.sender_agent_id in notifs:
                for notif in notifs[nixl_request.sender_agent_id]:
                    if notif == done_signal:
                        self.nixl_agent.notifs[nixl_request.sender_agent_id].remove(
                            notif
                        )
                        found = True
                        break

            await asyncio.sleep(0.001)
            # Waited for too long without transfer completion, log for debugging
            timeout = 3
            if (time.perf_counter() - start) > timeout and not logged:
                logger.debug(
                    f"still waiting for transfer completion for tensor_id {tensor_id} for more than {timeout} seconds"
                )
                logged = True
        logger.debug(
            f"Transfer completed for tensor_id {tensor_id}, total wait time: {time.perf_counter() - start:.2f} seconds"
        )

        self.to_buffer_id[tensor_id] = buffer_id
        return tensor_id, embedding_tensor

    def release_tensor(self, tensor_id: int):
        """
        Indicate that the tensor associated with the ID is no longer in use.

        Args:
            tensor_id: The ID of the tensor to release.
        """
        buffer_id = self.to_buffer_id.pop(tensor_id)
        self.ring_buffer.release_buffer(buffer_id)


class PersistentConnector(nixl_connect.Connector):
    """A persistent NIXL connector that can be shared across multiple send/receive operations."""

    def __init__(self):
        super().__init__()
        self._connection = None

    async def _create_connection(self) -> nixl_connect.Connection:
        """
        Private method to create a new connection.
        """
        if self._connection is None:
            self._connection = nixl_connect.Connection(self, 1)
            await self._connection.initialize()
        return self._connection


# Overwrite the remote release method to prevent deregistering the remote agent on each release,
# with persistent connection, all operations will be initiated on the same agent-pair, if not
# avoiding the deregisteration, the inflight operations will be teminated.
def remote_release_overwrite(self) -> None:
    pass


nixl_connect.Remote._release = remote_release_overwrite


class NixlReadEmbeddingSender(AbstractEmbeddingSender):
    """
    Initial implementation of NIXL READ based transfer. This implementation uses
    a monkey-patched version of 'nixl_connect' wrapper library to persist
    connection (agent registration) and descriptors across multiple send operations
    to avoid the overhead of repeated connection setup and teardown.
    NOTE This implementation or the use of 'nixl_connect' needs to be revisited as
    the benchmarking result is unexpectedly slow. Keeping it now for completeness,
    i.e. provide NIXL WRITE based and READ based transfer classes.
    """

    def __init__(self):
        self.connector = PersistentConnector()

    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
    ) -> tuple[TransferRequest, asyncio.Future]:
        """
        Send precomputed embeddings.

        Args:
            embeddings: A torch.Tensor of the embeddings to send.
            stage_embeddings: A boolean indicating whether the embeddings should be staged for the transfer,
            if True, the embeddings may be used as transfer buffer and must not be released until the return future is completed.
            if False, the sender will copy the embeddings.
        Returns:
            A tuple containing the TransferRequest object and a future that can be awaited to indicate the send is completed.
        """
        if stage_embeddings:
            transfer_buf = embeddings
        else:
            transfer_buf = embeddings.clone().detach()
        descriptor = nixl_connect.Descriptor(transfer_buf)
        readable_op = await self.connector.create_readable(descriptor)

        request = TransferRequest(
            embeddings_shape=list(embeddings.shape),
            embedding_dtype_str=torch_dtype_to_string(embeddings.dtype),
            serialized_request=readable_op.metadata().model_dump(),
        )
        return request, readable_op.wait_for_completion()


class NixlReadEmbeddingReceiver(AbstractEmbeddingReceiver):
    """
    Counter part of 'NixlReadEmbeddingSender', see 'NixlReadEmbeddingSender' for details.
    Initial implementation of another usage of NIXL connect library that persists
    connection (agent registration) and descriptors (memory registration) across multiple send operations
    to avoid the overhead of repeated connection setup and teardown.
    [gluo FIXME] This implementation requires more memory allocation and somewhat rigid, should move away
    from connect library so we can have single descriptor and chunk for transfer on demand, similarly to
    KV cache transfer. We may worry less on memory fragmentation as the memory can be released for next
    transfer as soon as the embedding has passed to the framework (NEED TO VERIFY: framework will copy) and
    can simply loop around the large buffer.
    """

    def __init__(
        self, embedding_hidden_size=8 * 1024, max_item_mm_token=1024, max_items=1024
    ):
        super().__init__()
        self.connector = PersistentConnector()
        self.tensor_id_counter = 0
        self.aggregated_op_create_time = 0
        self.aggregated_op_wait_time = 0
        self.warmedup_descriptors = Queue()
        self.inuse_descriptors = {}
        # Handle both sync and async contexts
        try:
            asyncio.get_running_loop()  # Check if we're in async context
            # If we're in an async context, we need to run the connection creation in a separate thread to avoid blocking the event loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                connection = pool.submit(
                    asyncio.run, self.connector._create_connection()
                ).result(timeout=10)
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            connection = asyncio.run(self.connector._create_connection())
        # Create descriptor for our allocated tensor
        for _ in range(max_items):
            encodings_tensor = torch.zeros(
                max_item_mm_token * embedding_hidden_size, dtype=torch.int8
            )
            descriptor = nixl_connect.Descriptor(encodings_tensor)
            descriptor.register_with_connector(connection)
            self.warmedup_descriptors.put(descriptor)

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
        readable_metadata = nixl_connect.RdmaMetadata.model_validate(
            request.serialized_request
        )

        original_descriptor_size = None
        if self.warmedup_descriptors.empty():
            logger.debug(
                "No warmed up descriptors available, creating a temporary one for transfer."
            )
            encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)
            descriptor = nixl_connect.Descriptor(encodings_tensor)
            dynamic_descriptor = True
        else:
            descriptor = self.warmedup_descriptors.get()
            # Slide view of pre-allocated tensor
            original_descriptor_size = descriptor._data_size
            tensor_size_bytes = embeddings_dtype.itemsize * math.prod(embeddings_shape)
            descriptor._data_size = tensor_size_bytes
            encodings_tensor = (
                descriptor._data_ref[:tensor_size_bytes]
                .view(dtype=embeddings_dtype)
                .view(embeddings_shape)
            )
            dynamic_descriptor = False

        # Create read operation to read from EncodeHandler
        read_op = await self.connector.begin_read(readable_metadata, descriptor)
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

    def release_tensor(self, tensor_id: int):
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
