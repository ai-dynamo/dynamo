# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import math
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, List

import torch
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


class NixlEmbeddingSender(AbstractEmbeddingSender):
    """
    The EmbeddingSender implementation of current usage of NIXL connect library,
    which creates a new NIXL connection for each send operation. Only implemented here
    for reference and should not be used due to overhead discovered in practice.
    """

    def __init__(self):
        self.connector = nixl_connect.Connector()

    async def send_embeddings(
        self, embeddings: torch.Tensor, stage_embeddings: bool = False
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

        descriptor = nixl_connect.Descriptor(embeddings.cpu())
        readable_op = await self.connector.create_readable(descriptor)

        request = TransferRequest(
            embeddings_shape=list(embeddings.shape),
            embedding_dtype_str=torch_dtype_to_string(embeddings.dtype),
            serialized_request=readable_op.metadata().model_dump(),
        )
        return request, readable_op.wait_for_completion()


class NixlEmbeddingReceiver(AbstractEmbeddingReceiver):
    """
    The EmbeddingReceiver implementation of current usage of NIXL connect library,
    which creates a new NIXL connection for each send operation. Only implemented here
    for reference and should not be used due to overhead discovered in practice.
    """

    def __init__(self):
        super().__init__()
        self.connector = nixl_connect.Connector()
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
        # Extract dynamic shape, metadata, and auxiliary data
        embeddings_shape = request.embeddings_shape
        embeddings_dtype = torch_dtype_from_string(request.embedding_dtype_str)
        readable_metadata = nixl_connect.RdmaMetadata.model_validate(
            request.serialized_request
        )

        encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)

        # Create descriptor for our allocated tensor
        descriptor = nixl_connect.Descriptor(encodings_tensor)

        # Create read operation to read from EncodeHandler
        read_op = await self.connector.begin_read(readable_metadata, descriptor)
        with read_op:
            # Wait for the read operation to complete
            await read_op.wait_for_completion()
            logging.debug(
                f"Successfully read embeddings via NIXL: {encodings_tensor.shape}"
            )
        tensor_id = self.tensor_id_counter
        self.tensor_id_counter += 1
        return tensor_id, encodings_tensor

    def release_tensor(self, tensor_id: int):
        """
        Indicate that the tensor associated with the ID is no longer in use.

        Args:
            tensor_id: The ID of the tensor to release.
        """
        # receiver doesn't hold the embedding
        pass


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
        # If not staging embedding and embedding is on CPU, we explicitly copy
        # the tensor as torch.Tensor.cpu() will return original tensor if it's already on CPU
        if not stage_embeddings and not embeddings.is_cuda:
            embeddings_cpu = embeddings.clone().detach()
        else:
            embeddings_cpu = embeddings.cpu()
        descriptor = nixl_connect.Descriptor(embeddings_cpu)
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
        self, embedding_hidden_size=8 * 1024, max_item_mm_token=1024, max_items=50
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
            tensor_size_bytes = embeddings_dtype.itemsize * math.prod(embeddings_shape)
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
