# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import logging
import math
import os
import pickle
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, List, Optional

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


class NixlEmbeddingSender(AbstractEmbeddingSender):
    """
    The EmbeddingSender implementation of current usage of NIXL connect library,
    which creates a new NIXL connection for each send operation. Only implemented here
    for reference and should not be used due to overhead discovered in practice.

    Note that the sender will initiate the transfer so the workflow is
    1) (pre-request) Receiver sends its metadata to sender with agent metadata contains
       the information of its registered memory buffer for transfer.
    2) (request) Sender prepares embeddings for transfer and return TransferRequest regarding
       where to send notification and the size of the embedding.
    3) (request) Receiver send notification with buffer ID and destination address to sender
    4) (request) Sender initiates the transfer after receiving the notification

    """

    def __init__(self):
        # NIXL agent setup
        self.sender_id = f"sender_{str(uuid.uuid4())}"
        self.nixl_agent = nixl_agent(
            self.sender_id, nixl_agent_config(num_threads=8, capture_telemetry=True)
        )
        self.remote_agents = {}
        self.handshaked_receivers = set()
        self.agent_metadata = self.nixl_agent.get_agent_metadata()
        self.agent_metadata_b64 = base64.b64encode(self.agent_metadata).decode("utf-8")

        self.transfer_tracker = {}
        # Track dynamically registered descriptors for cleanup,
        # there can be case of the same tensor being requested to be transferred multiple times,
        # we want to avoid duplicated registration or early deregistration while other transfer
        # of the tensor is still in-flight, so we track the inflight transfer with respect to
        # the actual tensor buffer and only deregister after all transfers of the same tensor is completed.
        self.registered_descs = {}

        self.id_counter = MonolithicCounter()
        # Create a queue for hinting if there is future transfer
        self.transfer_queue = asyncio.Queue()
        self._state_update_task = asyncio.create_task(self._state_update())

    async def _state_update(self):
        """Long-running async task that processes transfer requests."""
        transfer_handlers = {}
        transfer_task = None
        while True:
            try:
                # Wait for transfer requests with timeout to allow periodic checks
                if transfer_task is None:
                    transfer_task = await self.transfer_queue.get()

                # check if write is requested, initiate the write
                notifs = self.nixl_agent.get_new_notifs()
                for remote_agent_id, notifs in notifs.items():
                    self.handshaked_receivers.add(remote_agent_id)
                    for notif in notifs:
                        (
                            tensor_id,
                            (dest_buffer, dest_device_id, dest_mem_str),
                            write_done_id,
                            remote_agent_metadata,
                        ) = pickle.loads(notif)
                        if remote_agent_id not in self.remote_agents:
                            if len(remote_agent_metadata) == 0:
                                logger.error(
                                    f"Received transfer notification from unknown agent {remote_agent_id} without metadata, cannot add remote agent for transfer"
                                )
                                continue
                            # This means the sender has not handshaked with the receiver before, add remote agent for future transfer
                            self.remote_agents[
                                remote_agent_id
                            ] = self.nixl_agent.add_remote_agent(remote_agent_metadata)
                        transfer_info = self.transfer_tracker[tensor_id]
                        remote_memory_info = [
                            (dest_buffer, transfer_info[0].nbytes, dest_device_id),
                        ]
                        remote_desc = self.nixl_agent.get_xfer_descs(
                            remote_memory_info, mem_type=dest_mem_str
                        )
                        done_signal = str(write_done_id).encode()
                        xfer_handle = self.nixl_agent.initialize_xfer(
                            "WRITE",
                            transfer_info[1],
                            remote_desc,
                            remote_agent_id,
                            done_signal,
                        )
                        self.nixl_agent.transfer(xfer_handle, done_signal)
                        transfer_handlers[tensor_id] = (
                            xfer_handle,
                            remote_agent_id,
                            done_signal,
                            time.perf_counter(),
                        )

                # check inflight transfer state, if completed, get another task
                done_id = []
                for tensor_id, (
                    xfer_handle,
                    remote_agent_id,
                    done_signal,
                    start_time,
                ) in list(transfer_handlers.items()):
                    state = self.nixl_agent.check_xfer_state(xfer_handle)
                    if state == "ERR":
                        logger.error(f"Transfer failed for tensor_id {tensor_id}")
                        transfer_handlers.pop(tensor_id)
                        self.transfer_tracker.pop(tensor_id, None)
                    elif state == "DONE":
                        logger.debug(
                            f"Send completed for tensor_id {tensor_id}, total wait time: {time.perf_counter() - start_time:.2f} seconds"
                        )
                        done_id.append(tensor_id)
                        transfer_handlers.pop(tensor_id)
                        transfer_info = self.transfer_tracker.pop(tensor_id, None)
                        if transfer_info is not None:
                            # Clean up registered memory after transfer completion
                            embeddings, _, fut = transfer_info
                            desc_key = (embeddings.data_ptr(), embeddings.get_device())
                            self.registered_descs[desc_key][1] -= 1
                            if self.registered_descs[desc_key][1] == 0:
                                self.nixl_agent.deregister_memory(
                                    self.registered_descs[desc_key][0]
                                )
                                del self.registered_descs[desc_key]
                            # Future can be done if the embeddings is not external
                            if not fut.done():
                                fut.set_result(None)
                        try:
                            transfer_task = self.transfer_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            logger.debug("No pending transfer task in the queue.")
                            transfer_task = None
                            break

                # short pause to yield control and allow cancellation
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in state update loop: {e}")
                await asyncio.sleep(1)  # Backoff on error to prevent tight error loop

    def __del__(self):
        self._state_update_task.cancel()

    async def add_agent(self, remote_agent_id, remote_agent_metadata):
        """
        Add a remote agent for transfer based on the metadata provided by the receiver.

        Args:
            remote_agent_metadata: The metadata of the remote agent to add.
        """
        if remote_agent_id not in self.remote_agents:
            self.remote_agents[remote_agent_id] = self.nixl_agent.add_remote_agent(
                base64.b64decode(remote_agent_metadata)
            )

    async def send_embeddings(
        self,
        embeddings: torch.Tensor,
        stage_embeddings: bool = False,
        remote_agent_id: Optional[str] = None,
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

        # track the NIXL descriptors for future transfer
        desc_key = (embeddings.data_ptr(), embeddings.get_device())
        if desc_key not in self.registered_descs:
            # [NOTE] registeration can be time consuming, see e2e benchmark for the difference.
            registered_desc = self.nixl_agent.register_memory(embeddings)
            self.registered_descs[desc_key] = [registered_desc, 1]
        else:
            self.registered_descs[desc_key][1] += 1

        desc = self.nixl_agent.get_xfer_descs(embeddings)
        self.transfer_tracker[tensor_id] = (embeddings, desc, fut)
        self.transfer_queue.put_nowait("task_indicator")

        request = TransferRequest(
            embeddings_shape=list(embeddings.shape),
            embedding_dtype_str=torch_dtype_to_string(embeddings.dtype),
            serialized_request=NixlTransferRequest(
                sender_agent_id=self.sender_id,
                agent_metadata=self.agent_metadata_b64
                if remote_agent_id is None
                or remote_agent_id not in self.handshaked_receivers
                else None,
                tensor_id=tensor_id,
                tensor_size=embeddings.nbytes,
            ).model_dump_json(),
        )
        return request, fut


class NixlEmbeddingReceiver(AbstractEmbeddingReceiver):
    """
    The EmbeddingReceiver implementation of current usage of NIXL connect library,
    which creates a new NIXL connection for each send operation. Only implemented here
    for reference and should not be used due to overhead discovered in practice.
    """

    def __init__(self, buffer_size=2 * 8 * 1024 * 1024 * 256 * 2):
        # buffer_size is product of:
        # 2 (typical dtype size float16)
        # 8 * 1024 (typical embedding hidden size for Qwen-VL)
        # 256 * 1024 (1024 count of 256 mm token item)
        # 2 (extra copies) = 8 GB memory
        # ring buffer imple without wrapped around allocation, i.e. will allocate from
        # start if the last remaining buffer is not enough
        self.ring_buffer = RingBuffer(buffer_size)
        self.transfer_tensor = self.ring_buffer.buffer_tensor

        self.receiver_id = f"receiver_{str(uuid.uuid4())}"
        self.nixl_agent = nixl_agent(
            self.receiver_id, nixl_agent_config(num_threads=8, capture_telemetry=True)
        )
        self.remote_agents = {}
        self.reg_descs = self.nixl_agent.register_memory(self.transfer_tensor)

        self.agent_metadata = self.nixl_agent.get_agent_metadata()
        self.agent_metadata_b64 = base64.b64encode(self.agent_metadata).decode("utf-8")

        self.id_counter = MonolithicCounter()
        self.to_buffer_id = {}

    def get_agent_metadata(self):
        return self.receiver_id, self.agent_metadata_b64

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

        # Extract dynamic shape, metadata, and auxiliary data
        embeddings_shape = request.embeddings_shape
        embeddings_dtype = torch_dtype_from_string(request.embedding_dtype_str)
        while True:
            buffer_id, transfer_tensor = self.ring_buffer.get_buffer(
                nixl_request.tensor_size
            )
            if transfer_tensor is not None:
                break

            # [gluo FIXME] This approach will results in deadlock due to
            # the current usage:
            # concurrent requests may request 2 buffer in order,
            # if all request get the first buffer and exhaust the ring buffer,
            # then no request can get the second buffer and proceed.
            # Must provide an API for batch allocation so some requests can
            # proceed.
            #
            # No available buffer, wait for a short period and retry.
            # The receiver side should have concurrent work on other
            # allocated buffer and release them in a timely manner,
            # so the wait time should not be long.
            await asyncio.sleep(0.005)
        embedding_tensor = transfer_tensor.view(dtype=embeddings_dtype).view(
            embeddings_shape
        )

        # Request for transfer
        tensor_id = self.id_counter.get_next_id()
        notif_msg = pickle.dumps(
            (
                nixl_request.tensor_id,
                (
                    transfer_tensor.data_ptr(),
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
        logged = False
        done_signal = str(tensor_id).encode()
        # 'check_remote_xfer_done' will find occurence of the message in substring which is not
        # what we want, we want exact match, need to parse by ourselves
        found = False
        while not found:
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


class NixlPersistentEmbeddingSender(AbstractEmbeddingSender):
    """
    Initial implementation of another usage of NIXL connect library that persists
    connection (agent registration) and descriptors across multiple send operations
    to avoid the overhead of repeated connection setup and teardown.
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


class NixlPersistentEmbeddingReceiver(AbstractEmbeddingReceiver):
    """
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
