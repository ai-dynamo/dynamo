# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import logging
import socket
import uuid
import zlib
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch must be installed to use this module. Please install PyTorch, ex: 'pip install torch'."
    ) from e

try:
    import nixl._api as nixl_api
    import nixl._bindings as nixl_bindings
except ImportError as e:
    raise ImportError(
        "NIXL Python bindings must be installed to use this module. Please install NIXL, ex: 'pip install nixl'."
    ) from e

logger = logging.getLogger(__name__)

try:
    import cupy as array_module
    from cupy_backends.cuda.api.runtime import CUDARuntimeError

    logger.info("dynamo.nixl_connect: Utilizing CuPy to enable GPU acceleration.")
except ImportError:
    try:
        import numpy as array_module

        logger.warning(
            "dynamo.nixl_connect: Failed to load CuPy for GPU acceleration, utilizing numpy to provide CPU based operations."
        )
    except ImportError as e:
        raise ImportError("Numpy or CuPy must be installed to use this module.") from e


class AbstractOperation(ABC):
    """
    Abstract base class for awaitable NIXL based RDMA operations.
    """

    def __init__(
        self,
        connector: Connector,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
        remote_descriptors: Optional[Descriptor | list[Descriptor]],
        notification_key: Optional[str],
    ) -> None:
        if not isinstance(connector, Connector):
            raise TypeError(
                "Argument `connector` must be `dynamo.nixl_connect.Connector`."
            )
        if (
            operation_kind is not OperationKind.READ
            and operation_kind is not OperationKind.WRITE
        ):
            raise ValueError(
                "Argument `operation_kind` must be either `READ` or `WRITE`."
            )
        if not (
            isinstance(local_descriptors, (Descriptor, list))
            or (
                isinstance(local_descriptors, list)
                and all(isinstance(d, Descriptor) for d in local_descriptors)
            )
        ):
            raise TypeError(
                "Argument `local_descriptors` must be `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`."
            )
        if remote_descriptors is not None and not (
            isinstance(remote_descriptors, Descriptor)
            or (
                isinstance(remote_descriptors, list)
                and all(isinstance(d, Descriptor) for d in remote_descriptors)
            )
        ):
            raise TypeError(
                "Argument `remote_descriptors` must be `dynamo.nixl_connect.Descriptor`, `list[dynamo.nixl_connect.Descriptor]`, or `None`."
            )
        if isinstance(local_descriptors, list) and len(local_descriptors) == 0:
            raise ValueError("Argument `local_descriptors` must not be an empty list.")
        if (
            remote_descriptors is not None
            and isinstance(remote_descriptors, list)
            and len(remote_descriptors) == 0
        ):
            raise ValueError("Argument `remote_descriptors` must not be an empty list.")

        notification_key = (
            str(uuid.uuid4()) if notification_key is None else notification_key
        )
        if not isinstance(notification_key, str):
            raise TypeError("Argument `notification_key` must be `str` or `None`.")
        if len(notification_key) == 0:
            raise ValueError("Argument `notification_key` must not be an empty string.")

        self._notification_key: str = (
            "" if notification_key is None else notification_key
        )
        self._connector: Connector = connector
        self._operation_kind: OperationKind = operation_kind
        self._local_desc_list: Descriptor | list[Descriptor] = local_descriptors
        self._local_desc_tlist: Optional[list[tuple[int, int, int]]] = None
        self._local_device_kind: DeviceKind = DeviceKind.UNSPECIFIED
        self._remote_desc_list: Optional[Descriptor | list[Descriptor]] = (
            None if remote_descriptors is None else remote_descriptors
        )
        self._remote_desc_tlist: Optional[list[tuple[int, int, int]]] = None
        self._remote_device_kind: DeviceKind = DeviceKind.UNSPECIFIED

        # Register local descriptors with NIXL.
        # Note: Only local descriptors should be registered with NIXL,
        if isinstance(local_descriptors, list):
            for d in local_descriptors:
                d.register_memory(self._connector)
        else:
            local_descriptors.register_memory(self._connector)

        # Record local descriptors.
        device_kind, desc_tlist = self._create_desc_tlist(local_descriptors)
        self._local_desc_tlist = desc_tlist
        self._local_device_kind = device_kind

        # Record remote descriptors when provided.
        if remote_descriptors is not None:
            device_kind, desc_tlist = self._create_desc_tlist(remote_descriptors)
            self._remote_desc_tlist = desc_tlist
            self._remote_device_kind = device_kind

    def __del__(self) -> None:
        self._release()

    def __enter__(self) -> AbstractOperation:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._release()

    def _release(self) -> None:
        pass

    @property
    def connector(self) -> Connector:
        """
        Gets the local associated with this operation.
        """
        return self._connector

    @property
    def operation_kind(self) -> OperationKind:
        """
        Gets the kind of operation.
        """
        return self._operation_kind

    @abstractmethod
    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        raise NotImplementedError("Abstract method not implemented by derived class.")

    # Private Methods

    def _create_desc_tlist(
        self,
        descriptors: Descriptor | list[Descriptor],
    ) -> tuple[DeviceKind, list[tuple[int, int, int]]]:
        """
        Helper function to create a list of tuples (ptr, size, device) from descriptors.
        """
        descriptor_tuples: list[tuple[int, int, int]] = []
        device_kind: DeviceKind = DeviceKind.UNSPECIFIED
        if isinstance(descriptors, list):
            device_kind = descriptors[0].device.kind
            for desc in descriptors:
                if device_kind != desc.device.kind:
                    raise ValueError(
                        "All local descriptors must have the same memory type."
                    )
                descriptor_tuples.append((desc.ptr, desc.size, desc.device.id))
        else:
            device_kind = descriptors.device.kind
            descriptor_tuples.append(
                (descriptors.ptr, descriptors.size, descriptors.device.id)
            )
        return (device_kind, descriptor_tuples)


class ActiveOperation(AbstractOperation):
    """
    Abstract class for active operations that initiates a NIXL based RDMA transfer based `RdmaMetadata`
    provided by the remote worker's corresponding `PassiveOperation`.
    """

    def __init__(
        self,
        remote: Remote,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
        remote_descriptors: Descriptor | list[Descriptor],
        notification_key: str,
    ) -> None:
        if not isinstance(remote, Remote) or remote._connector is None:
            raise TypeError(
                "Argument `remote` must be valid `dynamo.nixl_connect.Remote`."
            )
        if not isinstance(operation_kind, OperationKind):
            raise TypeError(
                "Argument `operation_kind` must `dynamo.nixl_connect.OperationKind`."
            )
        if (
            operation_kind is not OperationKind.READ
            and operation_kind is not OperationKind.WRITE
        ):
            raise ValueError(
                "Argument `operation_kind` must be either `READ` or `WRITE`."
            )
        if not (
            isinstance(local_descriptors, Descriptor)
            or (
                isinstance(local_descriptors, list)
                and all(isinstance(d, Descriptor) for d in local_descriptors)
            )
        ):
            raise TypeError(
                "Argument `local_descriptors` must be `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`."
            )
        if not (
            isinstance(remote_descriptors, Descriptor)
            or (
                isinstance(remote_descriptors, list)
                and all(isinstance(d, Descriptor) for d in remote_descriptors)
            )
        ):
            raise TypeError(
                "Argument `remote_descriptors` must be `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`."
            )

        # Unpack single descriptors from lists if they are provided as single descriptors.
        if isinstance(local_descriptors, list) and len(local_descriptors) == 1:
            local_descriptors = local_descriptors[0]
        if isinstance(remote_descriptors, list) and len(remote_descriptors) == 1:
            remote_descriptors = remote_descriptors[0]

        if isinstance(local_descriptors, list) != isinstance(remote_descriptors, list):
            raise ValueError(
                "Both `local_descriptors` and `remote_descriptors` must be either lists or single descriptors."
            )
        # Ensure that the descriptors are of the same size here to avoid confusing errors from NIXL.
        if isinstance(local_descriptors, list) and isinstance(remote_descriptors, list):
            if len(local_descriptors) != len(remote_descriptors):
                raise ValueError(
                    f"When `local_descriptors` and `remote_descriptors` are lists, they must have the same length. {len(local_descriptors)} != {len(remote_descriptors)}."
                )
            for i in range(len(local_descriptors)):
                if local_descriptors[i].size != remote_descriptors[i].size:
                    raise ValueError(
                        f"Descriptor length mismatch: `local_descriptors` and `remote_descriptors` descriptor at {i} must have the same size. {local_descriptors[i].size} != {remote_descriptors[i].size}."
                    )
        elif (
            isinstance(local_descriptors, Descriptor)
            and isinstance(remote_descriptors, Descriptor)
        ) and local_descriptors.size != remote_descriptors.size:
            raise ValueError(
                f"Local and remote descriptors must be the same size. {local_descriptors.size} != {remote_descriptors.size}."
            )
        if not isinstance(notification_key, str):
            raise TypeError("Argument `notification_key` must be `str`.")
        if len(notification_key) == 0:
            raise ValueError("Argument `notification_key` must not be an empty string.")

        self._remote = remote
        self._status = OperationStatus.UNINITIALIZED

        super().__init__(
            remote.connector,
            operation_kind,
            local_descriptors,
            remote_descriptors,
            notification_key,
        )
        # Quick check to ensure remote descriptors are not None to make static analysis happy.
        if self._local_desc_tlist is None or self._remote_desc_tlist is None:
            raise RuntimeError("NIXL descriptor list(s) not bound to operation.")

        self._local_xfer_descs: Optional[nixl_bindings.nixlXferDList] = None
        self._remote_xfer_descs: Optional[nixl_bindings.nixlXferDList] = None
        self._xfer_hndl: Optional[nixl_api.nixl_xfer_handle] = None

        self._local_xfer_descs = self._connector._nixl.get_xfer_descs(
            descs=self._local_desc_tlist,
            mem_type=str(self._local_device_kind),
        )
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created local NIXL transfer descriptors: {self._local_xfer_descs}"
        )
        self._remote_xfer_descs = self._connector._nixl.get_xfer_descs(
            descs=self._remote_desc_tlist,
            mem_type=str(self._remote_device_kind),
        )
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created remote NIXL transfer descriptors: {self._remote_xfer_descs}"
        )
        self._xfer_hndl = self._connector._nixl.initialize_xfer(
            operation=str(operation_kind),
            local_descs=self._local_xfer_descs,
            remote_descs=self._remote_xfer_descs,
            remote_agent=self._remote.name,
            notif_msg=self._notification_key.encode("utf-8"),
        )
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created NIXL transfer handle: {self._xfer_hndl}"
        )

    def __del__(self) -> None:
        super().__del__()
        self._release()

    def __enter__(self) -> ActiveOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        match self.status:
            case OperationStatus.IN_PROGRESS | OperationStatus.INITIALIZED:
                self._status = OperationStatus.CANCELLED

        self._release()

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"operation_kind={self._operation_kind}, "
            f"local_descriptors={self._local_desc_list}, "
            f"remote_descriptors={self._remote_desc_list}, "
            f"notification_key='{self._notification_key}', "
            f"remote='{self._remote.name}', "
            f"status='{self._status}'"
            f")"
        )

    def _release(self) -> None:
        """
        Private method to release resources.
        """
        error: Optional[Exception] = None

        if self._xfer_hndl is not None:
            try:
                logger.debug(
                    f"dynamo.nixl_connect.{self.__class__.__name__}: NIXL transfer handle {self._xfer_hndl} released."
                )
                self._connector._nixl.release_xfer_handle(self._xfer_hndl)
            except Exception as e:
                logger.error(
                    f"dynamo.nixl_connect.{self.__class__.__name__}: Failed to release resources: {e}"
                )
                error = e
            finally:
                self._xfer_hndl = None

        try:
            super()._release()
        except Exception as e:
            logger.error(
                f"dynamo.nixl_connect.{self.__class__.__name__}: Failed to release base class resources: {e}"
            )
            if error is not None:
                e.__cause__ = error
            error = e

        if error is not None:
            raise error

    def _cancel_(self) -> None:
        if self._xfer_hndl is None:
            return
        if self.status == OperationStatus.ERRORED:
            raise RuntimeError("Operation is errored, unable to cancel the operation.")

        logger.info(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Cancellation requested for operation {{ kind={self._operation_kind}, remote='{self._remote.name}', status={self._status} }}."
        )

        # NIXL will cancel the transfer if it is in progress when the handle is released.
        self._connector._nixl.release_xfer_handle(self._xfer_hndl)
        self._status = OperationStatus.CANCELLED
        self._xfer_hndl = None

    async def _wait_for_completion_(self) -> None:
        # Loop until the operation is no longer in progress (or "initialized"),
        # yielding control to the event loop to allow other operations to run.
        iteration_count = 0
        while True:
            if iteration_count & 10 == 0:
                logger.debug(
                    f"dynamo.nixl_connect.{self.__class__.__name__}: Waiting for operation {{ kind={self._operation_kind}, remote='{self._remote.name}', duration={iteration_count / 10}s }}."
                )
            match self.status:
                # "in progress" or "initialized" means the operation is ongoing.
                case OperationStatus.INITIALIZED:
                    await asyncio.sleep(0.1)
                case OperationStatus.IN_PROGRESS:
                    await asyncio.sleep(0.1)
                # Any other state indicates completion or error.
                case _:
                    return

    @abstractmethod
    def cancel(self) -> None:
        """
        Cancels the operation.
        No affect if the operation has already completed or errored, or has been cancelled.
        """
        raise NotImplementedError("Abstract method not implemented by derived class.")

    @property
    def remote(self) -> Remote:
        """
        Gets the remote worker associated with this operation.
        """
        return self._remote

    @property
    def status(self) -> OperationStatus:
        """
        Gets the status of the operation.
        """
        # Early return if the operation is already complete, errored, or cancelled.
        match self._status:
            case OperationStatus.COMPLETE | OperationStatus.ERRORED | OperationStatus.CANCELLED:
                return self._status

        if self._xfer_hndl is None:
            raise RuntimeError("NIXL transfer handle is invalid.")

        old_status = self._status

        if self._status == OperationStatus.UNINITIALIZED:
            state = self._connector._nixl.transfer(
                self._xfer_hndl,
                self._notification_key.encode("utf-8"),
            )
            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: NIXL reported transfer state: {state}"
            )
            if state == "ERR":
                self._status = OperationStatus.ERRORED
            elif state == "DONE":
                self._status = OperationStatus.COMPLETE
            else:
                self._status = OperationStatus.INITIALIZED
        else:
            state = self._connector._nixl.check_xfer_state(self._xfer_hndl)
            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: NIXL reported transfer state: {state}"
            )
            if state == "ERR":
                self._status = OperationStatus.ERRORED
            elif state == "DONE":
                self._status = OperationStatus.COMPLETE
            else:
                self._status = OperationStatus.IN_PROGRESS

        if self._status != old_status:
            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: {{ remote: '{self._remote.name}' status: '{old_status}' => '{self._status}' }}."
            )

        return self._status


class Connector:
    """
    Core class for managing the connection between workers in a distributed environment.
    Use this class to create readable and writable operations, or read and write data to remote workers.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
    ) -> None:
        """
        Creates a new Connector instance.

        Parameters
        ----------
        worker_id : Optional[str], optional
            Unique identifier of the worker, defaults to a new UUID when `None`.

        Raises
        ------
        TypeError
            When `worker_id` is provided and not of type `uuid.UUID`.
        """
        worker_id = (
            worker_id if worker_id is not None else str(uuid.uuid4()).replace("-", "")
        )
        if not isinstance(worker_id, str) or len(worker_id) == 0:
            raise TypeError("Argument `worker_id` must be a non-empty `str` or `None`.")

        self._worker_id = worker_id
        self._is_initialized = False
        self._nixl = nixl_api.nixl_agent(self._worker_id)
        self._hostname = socket.gethostname()
        self._agent_metadata: Optional[bytes] = None

        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()}."
        )

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"worker_id='{self._worker_id}', "
            f"hostname={self._hostname}, "
            f"metadata=<{0 if self._agent_metadata is None else len(self._agent_metadata)} bytes>"
            ")"
        )

    def __str__(self) -> str:
        return self._worker_id

    @cached_property
    def is_cuda_available(self) -> bool:
        # Note: `cuda.is_available` initializes CUDA
        #       and can't be called when forking subprocesses
        #       care should be taken to only call it within
        #       subprocesses or use 'spawn'
        try:
            return array_module.cuda is not None and array_module.cuda.is_available()
        except CUDARuntimeError:
            return False

    @property
    def metadata(self) -> bytes:
        """
        Get the metadata of the worker.
        """
        return self._nixl.get_agent_metadata()

    @property
    def name(self) -> str | None:
        """
        Get the name of the worker.
        """
        return self._worker_id

    async def begin_read(
        self,
        remote_metadata: RdmaMetadata,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> ReadOperation:
        """
        Creates a read operation for fulfilling a remote readable operation.

        Parameters
        ----------
        remote_metadata : RdmaMetadata
            RDMA metadata from a remote worker that has created a readable operation.
        local_descriptors : Descriptor | list[Descriptor]
            Local descriptor(s) to receive data from the remote worker described by `remote_metadata`.

        Returns
        -------
        ReadOperation
            Awaitable read operation that can be used to transfer data from a remote worker.

        Raises
        ------
        TypeError
            When `remote_metadata` is not of type `RdmaMetadata`.
        TypeError
            When `local_descriptors` is not of type `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`.
        """
        if remote_metadata is None or not isinstance(remote_metadata, RdmaMetadata):
            raise TypeError("Argument `remote_metadata` must be `RdmaMetadata`.")
        if not (
            isinstance(local_descriptors, Descriptor)
            or (
                isinstance(local_descriptors, list)
                and all(isinstance(d, Descriptor) for d in local_descriptors)
            )
        ):
            raise TypeError(
                "Argument `local_descriptors` must be `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`."
            )
        if remote_metadata.operation_kind != OperationKind.READ.value:
            raise RuntimeError(
                "Cannot create a `dynamo.nixl_connect.ReadOperation` to read from a remote `dynamo.nixl_connect.WritableOperation`."
            )

        if not self._is_initialized:
            raise RuntimeError(
                "Connector not initialized. Call `initialize()` before calling this method."
            )

        op = ReadOperation(self, remote_metadata, local_descriptors)
        return op

    async def begin_write(
        self,
        local_descriptors: Descriptor | list[Descriptor],
        remote_metadata: RdmaMetadata,
    ) -> WriteOperation:
        """
        Creates a write operation for transferring data to a remote worker.

        Parameters
        ----------
        local_descriptors : Descriptor | list[Descriptor]
            Local descriptors of one or more data objects to be transferred to the remote worker.
        remote_metadata : RdmaMetadata
            Serialized request from a remote worker that has created a readable operation.
        """
        if remote_metadata is None or not isinstance(remote_metadata, RdmaMetadata):
            raise TypeError("Argument `remote_metadata` must be `RdmaMetadata`.")
        if not (
            isinstance(local_descriptors, Descriptor)
            or (
                isinstance(local_descriptors, list)
                and all(isinstance(d, Descriptor) for d in local_descriptors)
            )
        ):
            raise TypeError(
                "Argument `local_descriptors` must be `Descriptor` or `list[Descriptor]`."
            )
        if remote_metadata.operation_kind != OperationKind.WRITE:
            raise RuntimeError(
                "Cannot create a `WriteOperation` to write to a remote `ReadableOperation`."
            )
        if not isinstance(remote_metadata.nixl_metadata, str):
            raise TypeError("Argument `remote_metadata.nixl_metadata` must be `str`.")

        if not self._is_initialized:
            raise RuntimeError(
                "Connector not initialized. Call `initialize()` before calling this method."
            )

        op = WriteOperation(self, local_descriptors, remote_metadata)
        return op

    def create_readable(
        self,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> ReadableOperation:
        """
        Creates a readable operation for transferring data from a remote worker.

        Returns
        -------
        ReadableOperation
            A readable operation that can be used to transfer data from a remote worker.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Connector not initialized. Call `initialize()` before calling this method."
            )

        op = ReadableOperation(self, local_descriptors)
        return op

    def create_writable(
        self,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> WritableOperation:
        """
        Creates a writable operation for transferring data to a remote worker.

        Returns
        -------
        WritableOperation
            A writable operation that can be used to transfer data to a remote worker.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Connector not initialized. Call `initialize()` before calling this method."
            )

        op = WritableOperation(self, local_descriptors)
        return op

    async def initialize(self) -> None:
        # Only initialize the connector once.
        if self._is_initialized:
            return

        self._is_initialized = True
        # This method is a no-op for now, in the future it may be used to initialize the connector.
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Initialized {{ name: '{self._worker_id}' }} completed."
        )


class Descriptor:
    """
    Memory descriptor that ensures memory is registered w/ NIXL, used for transferring data between workers.
    """

    def __init__(
        self,
        data: torch.Tensor
        | tuple[array_module.ndarray, Device | str]
        | bytes
        | tuple[int, int, Device | str, Any],
    ) -> None:
        """
        Memory descriptor for transferring data between workers.

        Parameters
        ----------
        data : torch.Tensor | tuple[ndarray, Device|str] | bytes | tuple[int, int, Device|str, Any]
            The data to be transferred.

            When `torch.Tensor` is provided, the attributes of the tensor will be used to create the descriptor.

            When `tuple[ndarray, Device]` is provided, the tuple must contain:
            - `ndarray`: The CuPy or NumPy array to be transferred.
            - `Device`: Either a `dynamo.nixl_connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").

            When `bytes` is provided, the pointer and size derived from the bytes object and memory type will be assumed to be CPU.

            When `tuple[int, int, Device|str, Any]` is provided, the tuple must contain the following elements:
            - `int`: Pointer to the data in memory.
            - `int`: Size of the data in bytes.
            - `Device`: Either a `dynamo.nixl_connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").
            - `Any`: Optional reference to the data (e.g., the original tensor or bytes object).
                     This is useful for keeping a reference to the data in memory, but it is not required.

        Raises
        ------
        ValueError
            When `data` is `None`.
        TypeError
            When `data` is not a valid type (i.e., not `torch.Tensor`, `bytes`, or a valid tuple).
        TypeError
            When `data` is a tuple but the elements are not of the expected types (i.e., [`ndarray`, `Device|str`] OR [`int`, `int`, `Device|str`, `Any`]).
        """
        TYPE_ERROR_MESSAGE = "Argument `data` must be `torch.Tensor`, `tuple[ndarray, Device|str]`, `bytes`, or `tuple[int, int, Device|str, Any]`."
        if data is None:
            raise ValueError("Argument `data` cannot be `None`.")
        if not (
            isinstance(data, torch.Tensor)
            or isinstance(data, bytes)
            or isinstance(data, tuple)
        ):
            raise TypeError(TYPE_ERROR_MESSAGE)

        self._data_device: Device = Device("cpu")
        self._data_ptr: int = 0
        self._data_ref: Optional[Any] = None
        self._data_size: int = 0

        # Member fields for managing NIXL memory registration.
        # Note: ONLY local descriptors should be registered with NIXL,
        #      remote descriptors do not have a valid memory address and registration will fault.
        self._connector: Optional[Connector] = None
        self._nixl_hndl: Optional[nixl_bindings.nixlRegDList] = None

        # Initially `None` cached serialized descriptor reference, populated when `get_metadata()` is called.
        self._serialized: Optional[SerializedDescriptor] = None

        # Data is `torch.Tensor`.
        if isinstance(data, torch.Tensor):
            self._data_ptr = data.data_ptr()
            self._data_size = data.numel() * data.element_size()
            if data.is_cuda:
                self._data_device = Device((DeviceKind.CUDA, data.get_device()))
            self._data_ref = data

            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()} from `torch.Tensor`."
            )

        # Data is `tuple[ndarray, Device]`.
        elif (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[0], array_module.ndarray)
            and (isinstance(data[1], Device) or isinstance(data[1], str))
        ):
            if hasattr(data[0], "__array_interface__"):
                self._data_ptr = data[0].__array_interface__["data"][0]
            elif hasattr(data[0], "__cuda_array_interface__"):
                self._data_ptr = data[0].__cuda_array_interface__["data"][0]
            else:
                raise TypeError(
                    "Argument `data[0]` must be a `ndarray` with a valid array interface."
                )
            self._data_size = data[0].nbytes
            self._data_device = (
                data[1] if isinstance(data[1], Device) else Device(data[1])
            )
            self._data_ref = data[0]

            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()} from `tuple[ndarray, Device|str]`."
            )

        # Data is `bytes`.
        elif isinstance(data, bytes):
            self._data_ptr = id(data)
            self._data_size = len(data)
            self._data_ref = data

            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()} from `bytes`."
            )

        # Data is `tuple[int, int, Device, dtype, tuple, Any]`.
        elif (
            isinstance(data, tuple)
            and len(data) >= 2
            and isinstance(data[0], int)
            and isinstance(data[1], int)
        ):
            if len(data) >= 3 and not (
                isinstance(data[2], Device) or isinstance(data[2], str)
            ):
                raise TypeError(
                    "Argument `data` must be a `tuple[int, int, Device|str, Any]`."
                )

            self._data_ptr = data[0]
            self._data_size = data[1]
            if len(data) >= 3:
                self._data_device = (
                    data[2] if isinstance(data[2], Device) else Device(data[2])
                )
            self._data_ref = data[3] if len(data) >= 4 else None

            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()} from `tuple[int, int, Device|str, Any]`."
            )
        else:
            raise TypeError(TYPE_ERROR_MESSAGE)

    def __del__(self) -> None:
        if self._nixl_hndl is not None and self._connector is not None:
            # Unregister the memory with NIXL.
            self._connector._nixl.deregister_memory(self._nixl_hndl)
            self._nixl_hndl = None

        if self._data_ref is not None:
            # Release the reference to the data.
            del self._data_ref

        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Deleted {self.__repr__()}."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"

    def __str__(self) -> str:
        return f"ptr={hex(self._data_ptr)}, size={self._data_size}, device={self._data_device}"

    @property
    def device(self) -> Device:
        """
        Gets the device the of the descriptor.
        """
        return self._data_device

    @property
    def ptr(self) -> int:
        """
        Gets the pointer of the descriptor.
        """
        return self._data_ptr

    @property
    def size(self) -> int:
        """
        Gets the size of the descriptor.
        """
        return self._data_size

    @staticmethod
    def from_serialized(
        serialized: SerializedDescriptor,
    ) -> Descriptor:
        """
        Deserializes a `SerializedDescriptor` into a `Descriptor` object.

        Parameters
        ----------
        serialized : SerializedDescriptor
            The serialized descriptor to deserialize.

        Returns
        -------
        Descriptor
            The deserialized descriptor.
        """
        if not isinstance(serialized, SerializedDescriptor):
            raise TypeError("Argument `serialized` must be `SerializedDescriptor`.")

        return serialized.to_descriptor()

    def metadata(self) -> SerializedDescriptor:
        """
        Serializes the descriptor into a `SerializedDescriptor` object.
        """
        if self._serialized is None:
            self._serialized = SerializedDescriptor(
                device=f"{self._data_device}",
                ptr=self._data_ptr,
                size=self._data_size,
            )

        return self._serialized

    def register_memory(
        self,
        connector: Connector,
    ) -> None:
        """
        Registers the memory of the descriptor with NIXL.
        """
        if not isinstance(connector, Connector):
            raise TypeError(
                "Argument `connector` must be `dynamo.nixl_connect.Connector`."
            )
        if self._data_ptr == 0:
            raise ValueError("Cannot register memory with a null pointer.")

        if not (self._nixl_hndl is None and self._connector is None):
            return

        # Register the memory with NIXL.
        self._connector = connector
        if isinstance(self._data_ref, torch.Tensor):
            self._nixl_hndl = connector._nixl.register_memory(self._data_ref)
        else:
            mem_type = str(self._data_device.kind)
            reg_list = [
                (self._data_ptr, self._data_size, self._data_device.id, mem_type)
            ]
            self._nixl_hndl = connector._nixl.register_memory(reg_list, mem_type)

        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Registered {self.__repr__()} with NIXL."
        )


class Device:
    """
    Represents a device in the system.
    """

    def __init__(
        self,
        metadata: str | tuple[DeviceKind, int],
    ) -> None:
        if metadata is None:
            raise ValueError("Argument `metadata` cannot be `None`.")
        if (
            isinstance(metadata, tuple)
            and len(metadata) == 2
            and isinstance(metadata[0], DeviceKind)
            and isinstance(metadata[1], int)
        ):
            kind, device_id = metadata
        elif isinstance(metadata, str):
            metadata = metadata.strip().lower()
            if metadata.startswith("cuda") or metadata.startswith("gpu"):
                kind = DeviceKind.CUDA
                device_id = (
                    0 if metadata.find(":") == -1 else int(metadata.split(":")[1])
                )
            elif metadata.startswith("cpu") or metadata.startswith("host"):
                kind = DeviceKind.HOST
                device_id = 0
            else:
                raise ValueError(
                    "Argument `metadata` must be in the format 'cuda:<device_id>' or 'cpu'."
                )
        else:
            raise TypeError(
                "Argument `metadata` must be a `tuple[MemoryKind, int]` or a `str`."
            )

        self._device_id = device_id
        self._kind = kind

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kind={self._kind}, id={self._device_id})"

    def __str__(self) -> str:
        return (
            f"{self._kind}:{self._device_id}"
            if self._kind is DeviceKind.CUDA
            else f"{self._kind}"
        )

    @property
    def id(self) -> int:
        """
        Gets the device ID of the device.
        """
        return self._device_id

    @property
    def kind(self) -> DeviceKind:
        """
        Gets the memory kind of the device.
        """
        return self._kind


class DeviceKind(IntEnum):
    """
    Type of memory a descriptor has been allocated to.
    """

    UNSPECIFIED = 0

    HOST = 1
    """
    System (CPU) memory.
    """

    CUDA = 2
    """
    CUDA addressable device (GPU) memory.
    """

    def __str__(self) -> str:
        if self == DeviceKind.HOST:
            return "cpu"
        elif self == DeviceKind.CUDA:
            return "cuda"
        else:
            return "<invalid>"


class OperationKind(IntEnum):
    """
    Kind of an operation.
    """

    UNSPECIFIED = 0
    READ = 1
    WRITE = 2

    def __str__(self) -> str:
        if self == OperationKind.READ:
            return "READ"
        elif self == OperationKind.WRITE:
            return "WRITE"
        else:
            return "<invalid>"


class OperationStatus(IntEnum):
    """
    Status of an operation.
    """

    UNINITIALIZED = 0
    """The operation has not been initialized yet and is not in a valid state."""

    INITIALIZED = 1
    """The operation has been initialized and is ready to be processed."""

    IN_PROGRESS = 2
    """The operation has been initialized and is in-progress (not completed, errored, or cancelled)."""

    COMPLETE = 3
    """The operation has been completed successfully."""

    CANCELLED = 4
    """The operation has been cancelled by the user or system."""

    ERRORED = 5
    """The operation has encountered an error and cannot be completed."""

    def __str__(self) -> str:
        if self == OperationStatus.INITIALIZED:
            return "INIT"
        elif self == OperationStatus.IN_PROGRESS:
            return "PROC"
        elif self == OperationStatus.COMPLETE:
            return "DONE"
        elif self == OperationStatus.ERRORED:
            return "ERR"
        elif self == OperationStatus.CANCELLED:
            return "STOP"
        else:
            return "<invalid>"


class PassiveOperation(AbstractOperation):
    """
    Abstract class for common functionality of passive operations.
    """

    def __init__(
        self,
        connector: Connector,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        if (
            operation_kind is not OperationKind.READ
            and operation_kind is not OperationKind.WRITE
        ):
            raise ValueError(
                "Argument `operation_kind` must be either `READ` or `WRITE`."
            )

        self._status = OperationStatus.UNINITIALIZED

        super().__init__(
            connector,
            operation_kind,
            local_descriptors,
            None,
            None,
        )

        self._serialized_request: Optional[RdmaMetadata] = None
        self._status = OperationStatus.INITIALIZED

    def __del__(self) -> None:
        super().__del__()

    def __enter__(self) -> AbstractOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"operation_kind={self._operation_kind}, "
            f"local_descriptors={self._local_desc_list}, "
            f"notification_key='{self._notification_key}', "
            f"status='{self._status}'"
            f")"
        )

    async def _wait_for_completion_(self) -> None:
        # Loop until the operation is no longer in progress (or "initialized"),
        # yielding control to the event loop to allow other operations to run.
        while True:
            match self.status:
                # "in progress" or "initialized" means the operation is ongoing.
                case OperationStatus.INITIALIZED:
                    await asyncio.sleep(0.1)
                case OperationStatus.IN_PROGRESS:
                    await asyncio.sleep(0.1)
                # Any other state indicates completion or error.
                case _:
                    return

    def metadata(self, hex_encode: bool = False) -> RdmaMetadata:
        """
        Gets the request descriptor for the operation.
        """
        if self._serialized_request is None:
            # When we've not yet cached the serialized request, we need to generate one before returning it.
            # Handle both cases: multiple and single descriptors.
            if isinstance(self._local_desc_list, list):
                descriptors = [desc.metadata() for desc in self._local_desc_list]
            else:
                descriptors = [self._local_desc_list.metadata()]

            original_len = len(self._connector.metadata)
            nixl_metadata = self._connector.metadata
            nixl_metadata = zlib.compress(nixl_metadata, level=6)
            compressed_len = len(nixl_metadata)
            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: Compressed NIXL metadata from {original_len} bytes to {compressed_len} bytes."
            )
            if compressed_len > original_len:
                logger.warning(
                    f"dynamo.nixl_connect.{self.__class__.__name__}: Compressed NIXL metadata is larger than original ({compressed_len} > {original_len})."
                )

            if not hex_encode:
                encoded_metadata = base64.b64encode(nixl_metadata).decode("utf-8")
                encoded_metadata = "b64:" + encoded_metadata
            else:
                encoded_metadata = nixl_metadata.hex()
            self._serialized_request = RdmaMetadata(
                descriptors=descriptors,
                nixl_metadata=encoded_metadata,
                notification_key=self._notification_key,
                operation_kind=int(self._operation_kind),
            )

        return self._serialized_request

    @property
    def status(self) -> OperationStatus:
        """
        Gets the status of the operation.
        """
        # Early return if the operation is already complete, errored, or cancelled.
        match self._status:
            case OperationStatus.COMPLETE | OperationStatus.ERRORED | OperationStatus.CANCELLED:
                return self._status

        old_status = self._status

        # Query NIXL for any notifications.
        notifications = self._connector._nixl.update_notifs()

        if isinstance(notifications, dict):
            remote_state = OperationStatus.IN_PROGRESS
            logger.debug(
                f"dynamo.nixl_connect.{self.__class__.__name__}: NIXL reported notifications: {len(notifications)}."
            )

            for key, values in notifications.items():
                if not isinstance(values, list):
                    raise TypeError(
                        f"Expected `dict[str, list[bytes]]` from NIXL notification query; got {type(notifications)}."
                    )
                for value in values:
                    if not isinstance(value, bytes):
                        continue
                    notification_key = value.decode("utf-8")

                    # Once we've found the notification key, we know the operation is complete.
                    if notification_key == self._notification_key:
                        remote_state = OperationStatus.COMPLETE
                        break

            if remote_state == OperationStatus.COMPLETE:
                self._status = remote_state
                logger.debug(
                    f"dynamo.nixl_connect.{self.__class__.__name__}: {{ remote: '{self._connector.name}' status: '{old_status}' => '{self._status}' }}."
                )

        return self._status

    @abstractmethod
    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        raise NotImplementedError("Abstract method not implemented by derived class.")


class ReadOperation(ActiveOperation):
    """
    Operation that initiates an RDMA read operation to transfer data from a remote worker's `ReadableOperation`,
    as described by `remote_metadata`, to local buffers.
    """

    def __init__(
        self,
        connector: Connector,
        remote_metadata: RdmaMetadata,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        """
        Creates a new instance of `ReadOperation`, registers `local_descriptors` with NIXL,
        and begins an RDMA read operation which will transfer data described by `remote_metadata`
        to `local_descriptors`.

        Parameters
        ----------
        connector : Connector
            Connector instance to use for the operation.
        remote_metadata : RdmaMetadata
            Serialized request from the remote worker.
        local_descriptors : Descriptor | list[Descriptor]
            Local descriptor(s) to to receive the data from the remote worker.
        """
        if not isinstance(connector, Connector):
            raise TypeError(
                "Argument `connector` must be `dynamo.nixl_connect.Connector`."
            )
        if not isinstance(remote_metadata, RdmaMetadata):
            raise TypeError(
                "Argument `remote_metadata` must be `dynamo.nixl_connect.RdmaMetadata`."
            )
        if remote_metadata.operation_kind != OperationKind.READ.value:
            raise ValueError("Argument `remote_metadata` must be of kind `READ`.")

        remote = Remote(connector, remote_metadata.nixl_metadata)
        remote_descriptors = remote_metadata.to_descriptors()

        if not (
            isinstance(local_descriptors, Descriptor)
            or (
                isinstance(local_descriptors, list)
                and all(isinstance(d, Descriptor) for d in local_descriptors)
            )
        ):
            raise TypeError(
                "Argument `local_descriptors` must be `dynamo.nixl_connect.Descriptor`, `list[dynamo.nixl_connect.Descriptor]`."
            )

        super().__init__(
            remote,
            OperationKind.READ,
            local_descriptors,
            remote_descriptors,
            remote_metadata.notification_key,
        )
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()}"
        )

    def __del__(self) -> None:
        super().__del__()
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Deleted {self.__repr__()}"
        )

    def __enter__(self) -> ReadOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return super().__repr__()

    def cancel(self) -> None:
        """
        Cancels the operation.
        No affect if the operation has already completed or errored, or been cancelled.
        """
        super()._cancel_()

    def results(self) -> list[Descriptor]:
        """
        Gets the results of the operation.
        Returns a single descriptor if only one was requested, or a list of descriptors if multiple were requested.
        """
        if self._status != OperationStatus.COMPLETE:
            raise RuntimeError("Operation has not completed yet, cannot get results.")

        return (
            self._local_desc_list
            if isinstance(self._local_desc_list, list)
            else [self._local_desc_list]
        )

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()


class ReadableOperation(PassiveOperation):
    """
    Operation that can be awaited until a remote worker has completed a `ReadOperation`.
    """

    def __init__(
        self,
        connector: Connector,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        super().__init__(connector, OperationKind.READ, local_descriptors)
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()}"
        )

    def __del__(self) -> None:
        super().__del__()
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Deleted {self.__repr__()}"
        )

    def __enter__(self) -> ReadableOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return super().__repr__()

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()


class RdmaMetadata(BaseModel):
    """
    Pydantic serialization type for describing the passive side of a transfer.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )
    descriptors: List[SerializedDescriptor] = []
    nixl_metadata: str = ""
    notification_key: str = ""
    operation_kind: int = 0

    def to_descriptors(self) -> Descriptor | list[Descriptor]:
        """
        Deserializes the request descriptor into a `dynamo.nixl_connect.Descriptor` or list of `dynamo.nixl_connect.Descriptor` objects.
        """
        if len(self.descriptors) == 0:
            raise ValueError(
                "Request descriptor must contain at least one serialized descriptor."
            )
        if len(self.descriptors) == 1:
            return self.descriptors[0].to_descriptor()
        return [item.to_descriptor() for item in self.descriptors]

    @field_validator("operation_kind")
    @classmethod
    def validate_operation_kind(cls, v: int) -> int:
        if v < 1 or v > 3:
            raise TypeError(
                "Argument `operation_kind` must be an integer value of `dynamo.nixl_connect.OperationKind`."
            )
        return v


class Remote:
    """
    Identifies a remote NIXL enabled worker relative to a local NIXL enabled worker.
    """

    def __init__(
        self,
        connector: Connector,
        nixl_metadata: bytes | str,
    ) -> None:
        if not isinstance(connector, Connector):
            raise TypeError("Argument `local` must be `dynamo.nixl_connect.Connector`.")
        if not (isinstance(nixl_metadata, bytes) or isinstance(nixl_metadata, str)):
            raise TypeError("Argument `nixl_metadata` must be `bytes` or `str`.")
        if len(nixl_metadata) == 0:
            raise ValueError("Argument `nixl_metadata` cannot be empty.")

        self._connector = connector

        # When `nixl_metadata` is a string, it is assumed to have come from a remote worker
        # via a `RdmaMetadata` object and therefore can assumed be a b64-encoded, compressed
        # representation of the NIXL metadata.
        if isinstance(nixl_metadata, str):
            if nixl_metadata.startswith("b64:"):
                # Decode the b64-encoded string into bytes.
                nixl_metadata = base64.b64decode(nixl_metadata.lstrip("b64:"))
            else:
                # fallback for earlier versions of nixl connect
                nixl_metadata = bytes.fromhex(nixl_metadata)
            # Decompress the NIXL metadata.
            nixl_metadata = zlib.decompress(nixl_metadata)

        self._name = connector._nixl.add_remote_agent(nixl_metadata)
        if isinstance(self._name, bytes):
            self._name = self._name.decode("utf-8")

        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()}."
        )

    def __del__(self) -> None:
        self._release()

    def __enter__(self) -> Remote:
        """
        Context manager entry method. Returns the current instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Context manager exit method. Cleans up the instance.
        """
        self._release()

    def __repr__(self) -> str:
        return f"Remote(name={self._name}, connector={self._connector.name})"

    def __str__(self) -> str:
        return self._name

    def _release(self) -> None:
        """
        Private method for releasing NIXL resources. Not intended for public use.
        """
        # We have to unregister the remote agent from NIXL because we cannot know if the remote worker has updated its descriptors or not, and
        # NIXL will return an error if we attempt to register a remote agent with the same name but different descriptors (aka conn_info).
        self._connector._nixl.remove_remote_agent(self._name)
        logger.debug(
            f'dynamo.nixl_connect.{self.__class__.__name__}: Unregistered NIXL remote {{ name: "{self._name}" }}.'
        )

    @property
    def connector(self) -> Connector:
        """
        Gets the local connector associated with this remote worker.
        """
        return self._connector

    @property
    def name(self) -> str:
        """
        Gets the name of the remote worker.
        """
        return self._name


class SerializedDescriptor(BaseModel):
    """
    Pydantic serialization type for memory descriptors.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    device: str = "cpu"
    ptr: int = 0
    size: int = 0

    def to_descriptor(self) -> Descriptor:
        """
        Deserialize the serialized descriptor into a `Descriptor` object.
        """
        return Descriptor(data=(self.ptr, self.size, self.device, None))

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("Argument `device` must be `str`.")
        v = v.strip().lower()
        if not (v.startswith("cuda") or v == "cpu"):
            raise ValueError(
                "Argument `device` must be one of 'cpu' or 'cuda:<device_id>'."
            )
        return v

    @field_validator("ptr")
    @classmethod
    def validate_ptr(cls, v: int) -> int:
        if v == 0:
            raise ValueError("Argument `ptr` cannot be zero (aka `null` or `None`).")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                "Argument `size` must be an integer greater than or equal to zero."
            )
        return v


class WritableOperation(PassiveOperation):
    """
    Operation which can be awaited until written to by a `WriteOperation` from a remote worker.
    """

    def __init__(
        self,
        connector: Connector,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        """
        Creates a new instance of `WritableOperation`, registers the operation and descriptors w/ NIXL,
        and enables an RDMA write operation to occur.

        Parameters
        ----------
        connector : Connector
            Connector instance to use for the operation.
        local_descriptors : Descriptor | list[Descriptor]
            Descriptors to receive data from a remote worker.

        Raises
        TypeError
            When `local` is not a `dynamo.nixl_connect.Connector`.
        TypeError
            When `local_descriptors` is not a `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`.
        """
        super().__init__(connector, OperationKind.WRITE, local_descriptors)
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()}"
        )

    def __del__(self) -> None:
        super().__del__()
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Deleted {self.__repr__()}"
        )

    def __enter__(self) -> WritableOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return super().__repr__()

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()


class WriteOperation(ActiveOperation):
    """
    Awaitable write operation which initiates an RDMA write operation to a remote worker
    which provided a `RdmaMetadata` object from a `WritableOperation`.
    """

    def __init__(
        self,
        connector: Connector,
        local_descriptors: Descriptor | list[Descriptor],
        remote_metadata: RdmaMetadata,
    ) -> None:
        """
        Creates a new instance of `WriteOperation`, registers `local_descriptors` with NIXL,
        and begins an RDMA write operation which will transfer from `local_descriptors` to
        remote target(s) described by `remote_metadata`

        Parameters
        ----------
        connector : Connector
            Connector instance to use for the operation.
        local_descriptors : Descriptor | list[Descriptor]
            Local descriptor(s) to send from, to the remote worker.
        remote_metadata : RdmaMetadata
            Serialized request from the remote worker that describes the target(s) to send to.

        Raises
        TypeError
            When `connector` is not a `dynamo.nixl_connect.Connector`.
        TypeError
            When `remote_metadata` is not a `dynamo.nixl_connect.RdmaMetadata`.
        ValueError
            When `remote_metadata` is not of kind `WRITE`.
        ValueError
            When `remote_metadata.nixl_metadata` is not a non-empty `str`.
        TypeError
            When `local_descriptors` is not a `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`.
        """
        if not isinstance(connector, Connector):
            raise TypeError(
                "Argument `connector` must be `dynamo.nixl_connect.Connector`."
            )
        if not isinstance(remote_metadata, RdmaMetadata):
            raise TypeError(
                "Argument `remote_metadata` must be `dynamo.nixl_connect.RdmaMetadata`."
            )
        if remote_metadata.operation_kind != OperationKind.WRITE.value:
            raise ValueError("Argument `remote_metadata` must be of kind `WRITE`.")

        remote = Remote(connector, remote_metadata.nixl_metadata)
        remote_descriptors = remote_metadata.to_descriptors()

        super().__init__(
            remote,
            OperationKind.WRITE,
            local_descriptors,
            remote_descriptors,
            remote_metadata.notification_key,
        )
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Created {self.__repr__()}"
        )

    def __del__(self) -> None:
        super().__del__()
        logger.debug(
            f"dynamo.nixl_connect.{self.__class__.__name__}: Deleted {self.__repr__()}"
        )

    def __enter__(self) -> WriteOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return super().__repr__()

    def cancel(self) -> None:
        """
        Cancels the operation.
        No affect if the operation has already completed or errored, or has been cancelled.
        """
        super()._cancel_()

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()
