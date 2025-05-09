# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import nixl._api as nixl_api
import nixl._bindings as nixl_bindings
import numpy as np
import os
import signal
import socket
import torch
import utils.nixl as nixl_utils
import uuid
import zlib

from abc import ABC, abstractmethod
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.runtime import Backend, Client, Component, DistributedRuntime
from enum import IntEnum
from functools import cached_property
from io import BytesIO
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import core_schema
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cupy as array_module
    from cupy_backends.cuda.api.runtime import CUDARuntimeError
    logger.info("Utilizing cupy to enable GPU acceleration.")
except ImportError:
    try:
        import numpy as array_module
        logger.warning("Failed to load cupy for GPU acceleration, utilizing numpy to provide CPU based operations.")
    except ImportError as e:
        raise ImportError("Numpy or cupy must be installed to use this module.") from e


class AbstractOperation(ABC):
    """
    An abstract base class for waitable operations.
    """
    def __init__(
        self,
        local: Connector,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
        remote_descriptors: Optional[Descriptor | list[Descriptor]],
        notification_key: Optional[str],
    ) -> None:
        if not isinstance(local, Connector):
            raise TypeError("Argument `local` must be `dynamo.connect.Connector`.")
        if operation_kind is not OperationKind.READ and operation_kind is not OperationKind.WRITE:
            raise ValueError("Argument `operation_kind` must be either `READ` or `WRITE`.")
        if not isinstance(local_descriptors, (Descriptor, list)) or (isinstance(local_descriptors, list) and not all(isinstance(d, Descriptor) for d in local_descriptors)):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor` or `list[dynamo.connect.Descriptor]`.")
        if remote_descriptors is not None and not isinstance(remote_descriptors, (Descriptor, list)) or (isinstance(remote_descriptors, list) and not all(isinstance(d, Descriptor) for d in remote_descriptors)):
            raise TypeError("Argument `remote_descriptors` must be dynamo.connect.Descriptor`, `list[dynamo.connect.Descriptor]`, or `None`.")

        if isinstance(local_descriptors, list) and len(local_descriptors) == 0:
            raise ValueError("Argument `local_descriptors` must not be an empty list.")
        if remote_descriptors is not None and isinstance(remote_descriptors, list) and len(remote_descriptors) == 0:
            raise ValueError("Argument `remote_descriptors` must not be an empty list.")

        notification_key = str(uuid.uuid4()) if notification_key is None else notification_key
        if not isinstance(notification_key, str):
            raise TypeError("Argument `notification_key` must be `str` or `None`.")
        if len(notification_key) == 0:
            raise ValueError("Argument `notification_key` must not be an empty string.")

        self._notification_key: str = "" if notification_key is None else notification_key
        self._connector: Connector = local
        self._operation_kind: OperationKind = operation_kind
        self._local_descriptors: Descriptor | list[Descriptor] = local_descriptors
        self._local_dlist: Optional[list[tuple[int, int, int]]] = None
        self._local_memtype: DeviceKind = DeviceKind.UNSPECIFIED
        self._remote_descriptors: Optional[Descriptor | list[Descriptor]] = None if remote_descriptors is None else remote_descriptors
        self._remote_dlist: Optional[list[tuple[int, int, int]]] = None
        self._remote_memtype: DeviceKind = DeviceKind.UNSPECIFIED

        # Register local descriptors with NIXL.
        # Note: Only local descriptors should be registered with NIXL,
        if isinstance(local_descriptors, list):
            for d in local_descriptors:
                d.register_memory(self._connector)
        else:
            local_descriptors.register_memory(self._connector)

        # Record local descriptors.
        memtype, dtlist = self._create_dlist(local_descriptors)
        self._local_dlist = dtlist
        self._local_memtype = memtype

        # Record remote descriptors when provided.
        if remote_descriptors is not None:
            memtype, dtlist = self._create_dlist(remote_descriptors)
            self._remote_dlist = dtlist
            self._remote_memtype = memtype

    def __del__(self) -> None:
        self.__release__()

    def __enter__(self) -> AbstractOperation:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.__release__()

    def __release__(self) -> None:
        """
        Private method to release resources. Only to be called by `self`.
        """
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

    def _create_dlist(self, descriptors: Descriptor | list[Descriptor]) -> tuple[DeviceKind, list[tuple[int, int, int]]]:
        """
        Helper function to create a list of tuples (ptr, size, device) from descriptors.
        """
        dlist: list[tuple[int, int, int]] = []
        memtype: DeviceKind = DeviceKind.UNSPECIFIED
        if isinstance(descriptors, list):
            memtype = descriptors[0].device.kind
            for desc in descriptors:
                if memtype != desc.device.kind:
                    raise ValueError("All local descriptors must have the same memory type.")
                dlist.append((desc.ptr, desc.size, desc.device.id))
        else:
            memtype = descriptors.device.kind
            dlist.append((descriptors.ptr, descriptors.size, descriptors.device.id))
        return (memtype, dlist)


class ActiveOperation(AbstractOperation):
    """
    Abstract class for active operations that initiates a transfer and can be awaited.
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
            raise TypeError("Argument `remote` must be valid `dynamo.connect.RemoteAgent`.")
        if not isinstance(operation_kind, OperationKind):
            raise TypeError("Argument `operation_kind` must `dynamo.connect.OperationKind`.")
        if operation_kind is not OperationKind.READ and operation_kind is not OperationKind.WRITE:
            raise ValueError("Argument `operation_kind` must be either `READ` or `WRITE`.")
        if not (
            isinstance(local_descriptors, Descriptor)
            or (isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors))
        ):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor` or `list[dynamo.connect.Descriptor]`.")
        if not (
            isinstance(remote_descriptors, Descriptor)
            or (isinstance(remote_descriptors, list) and all(isinstance(d, Descriptor) for d in remote_descriptors))
        ):
            raise TypeError("Argument `remote_descriptors` must be `dynamo.connect.Descriptor` or `list[dynamo.connect.Descriptor]`.")

        # Unpack single descriptors from lists if they are provided as single descriptors.
        if isinstance(local_descriptors, list) and len(local_descriptors) == 1:
            local_descriptors = local_descriptors[0]
        if isinstance(remote_descriptors, list) and len(remote_descriptors) == 1:
            remote_descriptors = remote_descriptors[0]

        if isinstance(local_descriptors, list) and isinstance(remote_descriptors, list) and len(local_descriptors) != len(remote_descriptors):
            raise ValueError("When `local_descriptors` and `remote_descriptors` are lists, they must have the same length.")
        elif isinstance(local_descriptors, list) != isinstance(remote_descriptors, list):
            raise ValueError("Both `local_descriptors` and `remote_descriptors` must be either lists or single descriptors.")
        if not isinstance(notification_key, str):
            raise TypeError("Argument `notification_key` must be `str`.")
        if len(notification_key) == 0:
            raise ValueError("Argument `notification_key` must not be an empty string.")

        self._connector: Connector = remote._connector
        self._remote: Remote = remote
        self._status: OperationStatus = OperationStatus.UNINTIALIZED

        super().__init__(self._connector, operation_kind, local_descriptors, remote_descriptors, notification_key)
        # Quick check to ensure remote descriptors are not None to make static analysis happy.
        if self._local_dlist is None or self._remote_dlist is None:
            raise RuntimeError("NIXL descriptor list(s) not bound to operation.")

        self._local_xfer_descs: Optional[nixl_bindings.nixlXferDList] = None
        self._remote_xfer_descs: Optional[nixl_bindings.nixlXferDList] = None
        self._xfer_hndl: Optional[nixl_api.nixl_xfer_handle] = None

        self._local_xfer_descs = self._connector._nixl.get_xfer_descs(
            descs=self._local_dlist,
            mem_type=str(self._local_memtype),
        )
        logger.debug(f"Created local NIXL xfer descs: {self._local_xfer_descs}")
        self._remote_xfer_descs = self._connector._nixl.get_xfer_descs(
            descs=self._remote_dlist,
            mem_type=str(self._remote_memtype),
        )
        logger.debug(f"Created remote NIXL xfer descs: {self._remote_xfer_descs}")
        self._xfer_hndl = self._connector._nixl.initialize_xfer(
            operation=str(operation_kind),
            local_descs=self._local_xfer_descs,
            remote_descs=self._remote_xfer_descs,
            remote_agent=self._remote.name,
            notif_msg=self._notification_key,
        )
        logger.debug(f"Created NIXL transfer handle: {self._xfer_hndl}")

    def __del__(self) -> None:
        super().__del__()
        self.__release__()

    def __enter__(self) -> ActiveOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        match self.status:
            case OperationStatus.IN_PROGRESS | OperationStatus.INITIALIZED:
                self._status = OperationStatus.CANCELLED

        error : Optional[Exception] = None

        try:
            super().__exit__(exc_type, exc_value, traceback)
        except Exception as e:
            error = e

        try:
            self.__release__()
        except Exception as e:
            if error is not None:
                e.__cause__ = error
            error = e

        if error is not None:
            raise error

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"operation_kind={self._operation_kind}, "
            f"local_descriptors={self._local_descriptors}, "
            f"remote_descriptors={self._remote_descriptors}, "
            f"notification_key='{self._notification_key}', "
            f"remote='{self._remote.name}', "
            f"status='{self._status}'"
            f")"
        )

    def __release__(self) -> None:
        """
        Private method to release resources.
        """
        error: Optional[Exception] = None

        if self._xfer_hndl is not None:
            try:
                logger.debug(f"NIXL transfer handle {self._xfer_hndl} released.")
                self._connector._nixl.release_xfer_handle(self._xfer_hndl)
            except Exception as e:
                logger.error(f"Failed to release resources: {e}")
                error = e
            finally:
                self._xfer_hndl = None

        try:
            super().__release__()
        except Exception as e:
            logger.error(f"Failed to release WaitableOperation resources: {e}")
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
        logger.info(f"Cancellation requested for operation {{ kind={self._operation_kind}, remote='{self._remote.name}', status={self._status} }}.")

        # NIXL will cancel the transfer if it is in progress when the handle is released.
        self._connector._nixl.release_xfer_handle(self._xfer_hndl)
        self._status = OperationStatus.CANCELLED
        self._xfer_hndl = None

    async def _wait_for_completion_(self) -> None:
        # Loop until the operation is no longer in progress (or "initalized"),
        # yielding control to the event loop to allow other operations to run.
        iteration_count = 0
        while True:
            if iteration_count & 10 == 0:
                logger.debug(f"Waiting for operation {{ kind={self._operation_kind}, remote='{self._remote.name}', duration={iteration_count / 10}s }}.")
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
        Gets the remote agent associated with this operation.
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

        if self._status == OperationStatus.UNINTIALIZED:
            state = self._connector._nixl.transfer(self._xfer_hndl, self._notification_key)
            logger.debug(f"NIXL reported transfer state: {state}")
            if state == "ERR":
                self._status = OperationStatus.ERRORED
            elif state == "DONE":
                self._status = OperationStatus.COMPLETE
            else:
                self._status = OperationStatus.INITIALIZED
        else:
            state = self._connector._nixl.check_xfer_state(self._xfer_hndl)
            logger.debug(f"NIXL reported transfer state: {state}")
            if state == "ERR":
                self._status = OperationStatus.ERRORED
            elif state == "DONE":
                self._status = OperationStatus.COMPLETE
            else:
                self._status = OperationStatus.IN_PROGRESS

        if self._status != old_status:
            logger.debug(f"{self.__class__.__name__} {{ remote: '{self._remote.name}' status: '{old_status}' => '{self._status}' }}.")

        return self._status


class Connector:
    """
    Core class for managing the connection between agents in a distributed environment.
    Use this class to create readable and writable operations, or read and write data to remote agents.
    """
    def __init__(
        self,
        namespace: Optional[str] = None,
        runtime: Optional[DistributedRuntime] = None,
        worker_id: Optional[str] = None,
    ) -> None:
        """
        Creates a new Connector instance.

        Parameters
        ----------
        namespace : Optional[str], optional
            Dynamo namespace of the component, defaults to "dynamo" when `None`.
        runtime : Optional[DistributedRuntime], optional
            Reference the dynamo runtime used by the compenent, attempts to use the current runtime when `None`.
        worker_id : Optional[str], optional
            Unique identifier of the worker, defaults to a new UUID when `None`.

        Raises
        ------
        TypeError
            When `namespace` is provied and not of type 'str'.
        TypeError
            When `runtime` iis provied and not of type `dynamo.runtime.DistributedRuntime`.
        TypeError
            When `worker_id` is provied and not of type `uuid.UUID`.
        """
        namespace = "dynamo" if namespace is None else namespace
        if not isinstance(namespace, str):
            raise TypeError("Argument `namespace` must be `str` or `None`.")
        if dynamo_context is not None and "runtime" in dynamo_context:
            runtime = dynamo_context["runtime"] if runtime is None else runtime
        if not isinstance(runtime, DistributedRuntime) or runtime is None:
            raise TypeError("Argument `runtime` must be `dynamo.runtime.DistributedRuntime` or `None`.")
        worker_id = worker_id if worker_id is not None else str(uuid.uuid4())
        if not isinstance(worker_id, str) or len(worker_id) == 0:
            raise TypeError("Argument `worker_id` must be a non-empty `str` or `None`.")

        self._worker_id = worker_id
        self._is_initialized = False
        self._runtime = runtime
        self._namespace = namespace
        self._nixl = nixl_api.nixl_agent(self._worker_id)
        self._hostname = socket.gethostname()
        self._agent_metadata: Optional[bytes] = None

        logger.debug(f"Created {self.__repr__()}.")

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"worker_id='{self._worker_id}', "
            f"namespace={self._namespace}, "
            f"hostname={self._hostname}, "
            f"metadata=<{0 if self._agent_metadata is None else len(self._agent_metadata)} bytes>"
            ")"
        )

    def __str__(self) -> str:
        return self._worker_id

    @cached_property
    def is_cuda_available(self) -> bool:
        # Note: cuda.is_avalailable initializes cuda
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
        Get the metadata of the agent.
        """
        return  self._nixl.get_agent_metadata()

    @property
    def name(self) -> str|None:
        """
        Get the name of the agent.
        """
        return self._worker_id

    @property
    def namespace(self) -> str:
        """
        Get the namespace of the local.
        """
        return self._namespace

    @property
    def runtime(self) -> DistributedRuntime:
        """
        Get the runtime of the local.
        """
        if self._runtime is None:
            raise RuntimeError("Runtime is not set. This Connector was not initialized with a runtime.")
        return self._runtime

    def allocate_descriptor(
        self,
        size: int,
        device: Device | DeviceKind | str,
        dtype: array_module.dtype = array_module.float16
    ) -> Descriptor:
        """
        Allocates a new buffer (aka `bytes`) of the specified size on the specified device.
        The allocated buffer is created with `cupy` when available; otherwise with `numpy`.
        The bufffer is automatically registered with NIXL and returned as a `Descriptor` object.

        Parameters
        ----------
        size : int
            Size, measured in bytes, of the memory to allocate.
        device : Device | MemoryKind | str, optional
            Device to allocate the memory on. Can be a `dynamo.connect.Device`, `dynamo.connect.MemoryKind`, or a string representing the device type (e.g., "cuda" or "cpu").
            Defaults to `Device(MemoryKind.CPU, 0)`.
        dtype : array_module.dtype, optional
            Data type of the memory to allocate. Defaults to `array_module.float16`.

        Returns
        -------
        Descriptor
            A descriptor representing the allocated memory.

        Raises
        ------
        ValueError
            When `size` is not an integer greater than or equal to zero.
        TypeError
            When `device` is not a valid type (i.e., not `dynamo.connect.Device`, `dynamo.connect.MemoryKind`, or `str`).
        """
        if not isinstance(size, int) or size < 0:
            raise ValueError("Argument `size` must an integer greater than or equal to zero.")
        device = Device(DeviceKind.HOST, 0) if device is None else device
        if not (isinstance(device, Device) or isinstance(device, DeviceKind) or isinstance(device, str)):
            raise TypeError("Argument `device` must be `dynamo.connect.Device`, `dynamo.connect.MemoryKind`, or `str` in the format of \"cuda:<device_id>\" or \"cpu\".")

        if not isinstance(device, Device):
            if isinstance(device, str):
                device = Device(device)
            elif isinstance(device, DeviceKind):
                device = Device((device, 0))

        logger.debug(f"Allocating memory descriptor of size {size} bytes on device {device} with dtype {dtype}.")

        if device.kind is DeviceKind.CUDA and self.is_cuda_available:
            device_manager = array_module.cuda.Device(device.id)
        else:
            device_manager = contextlib.nullcontext()

        with device_manager:
            storage = array_module.empty(size, dtype=dtype)

        return Descriptor((storage, device))

    async def begin_read(
        self,
        remote_request: SerializedRequest,
        local_descriptors: Descriptor | list[Descriptor] | Device,
    ) -> ReadOperation:
        """
        Creates a read operation for fulfilling a remote readable operation.

        Parameters
        ----------
        remote_request : SerializedRequest
            Serialized request from a remote worker that has created a readable operation.
        local_descriptors : Descriptor | list[Descriptor] | MemoryKind
            One of more local descriptors of data objects to be transferred to the remote agent.
            When `MemoryKind` is provided, one or more descriptors will be created with the specified memory type and device ID.

        Returns
        -------
        ReadOperation
            Awaitable read operation that can be used to transfer data from a remote agent.
        """
        if remote_request is None or not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `SerializedRequest`.")
        if not (
            local_descriptors is isinstance(local_descriptors, Descriptor)
            or isinstance(local_descriptors, Device)
            or isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors)
        ):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor`, `list[dynamo.connect.Descriptor]`, or `dynamo.connect.Device`.")
        if remote_request.operation_kind != OperationKind.READ.value:
            raise RuntimeError("Cannot create a `dynamo.connect.ReadOperation` to read from a remote `dynamo.connect.WritableOperation`.")
        if not isinstance(remote_request.nixl_metadata, str):
            raise TypeError("Argument `remote_request.nixl_metadata` must be `str`.")

        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        # Read the remote agent metadata from the serialized request,
        # convert it back into bytes, and then decompress it.
        nixl_metadata = bytes.fromhex(remote_request.nixl_metadata)
        nixl_metadata = zlib.decompress(nixl_metadata)
        # Use the remote metadata to initialize the remote agent instance, and
        # use it to create/begin the read operation.
        remote_agent = await self._get_remote_agent(nixl_metadata)
        op = ReadOperation(remote_agent, remote_request, local_descriptors)
        return op

    async def begin_write(
        self,
        local_descriptors: Descriptor | list[Descriptor],
        remote_request: SerializedRequest,
    ) -> WriteOperation:
        """
        Creates a write operation for transferring data to a remote agent.

        Parameters
        ----------
        remote_request : SerializedRequest
            Serialized request from a remote worker that has created a readable operation.
        local_descriptors : Descriptor | list[Descriptor]
            Local descriptors of one or more data objects to be transferred to the remote agent.

        Returns
        -------
        WriteOperation
            _description_
        """
        if remote_request is None or not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `SerializedRequest`.")
        if local_descriptors is None or not (isinstance(local_descriptors, Descriptor) or (isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors))):
            raise TypeError("Argument `local_descriptors` must be `Descriptor` or `list[Descriptor]`.")
        if remote_request.operation_kind != OperationKind.WRITE:
            raise RuntimeError("Cannot create a `WriteOperation` to write to a remote `ReadableOperation`.")
        if not isinstance(remote_request.nixl_metadata, str):
            raise TypeError("Argument `remote_request.nixl_metadata` must be `str`.")

        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        # Read the remote agent metadata from the serialized request,
        # convert it back into bytes, and then decompress it.
        nixl_metadata = bytes.fromhex(remote_request.nixl_metadata)
        nixl_metadata = zlib.decompress(nixl_metadata)
        # Use the remote metadata to initialize the remote agent instance, and
        # use it to create/begin the write operation.
        remote_agent = await self._get_remote_agent(nixl_metadata)
        op = WriteOperation(remote_agent, local_descriptors, remote_request)
        return op

    def create_readable(
        self,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> ReadableOperation:
        """
        Creates a readable operation for transferring data from a remote agent.

        Returns
        -------
        ReadableOperation
            A readable operation that can be used to transfer data from a remote agent.
        """
        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        op = ReadableOperation(self, local_descriptors)
        return op

    def create_writable(
        self,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> WritableOperation:
        """
        Creates a writable operation for transferring data to a remote agent.

        Returns
        -------
        WritableOperation
            A writable operation that can be used to transfer data to a remote agent.
        """
        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        op = WritableOperation(self, local_descriptors)
        return op

    async def initialize(self) -> None:
        # Only initialize the connector once.
        if self._is_initialized:
            return

        self._is_initialized = True

        logger.debug(f"Created Connector {{ name: '{self._worker_id}', namespace '{self._namespace}' }} completed.")

    async def _get_remote_agent(self, nixl_metadata: bytes) -> Remote:
        """
        Gets a remote agent by its unqiue worker identifier.

        Parameters
        ----------
        nixl_metadata : bytes
            Opaque metadata of the remote agent from NIXL.

        Returns
        -------
        RemoteAgent
            The remote agent with the given unqiue worker identifier.
        """
        if not isinstance(nixl_metadata, bytes):
            raise TypeError("Argument `nixl_metadata` must be `bytes`.")
        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        remote_agent = Remote(self, nixl_metadata)
        logger.debug(f"Remote agent {{ name: '{remote_agent.name}' }} initialized and added to cache.")

        return remote_agent


class Descriptor:
    """
    A memory descriptor for transferring data between agents.
    """
    def __init__(self, data: torch.Tensor | tuple[array_module.ndarray, Device|str] | bytes | tuple[int, int, Device|str, Any]) -> None:
        """
        Memory descriptor for transferring data between agents.

        Parameters
        ----------
        data : torch.Tensor | tuple[ndarray, Device|str] | bytes | tuple[int, int, Device|str, Any]
            The data to be transferred.

            When `torch.Tensor` is provided, the attributes of the tensor will be used to create the descriptor.

            When `tuple[ndarray, Device]` is provided, the tuple must contain:
            - `ndarray`: The CuPy or NumPy array to be transferred.
            - `Device`: Either a `dynamo.connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").

            When `bytes` is provided, the pointer and size derived from the bytes object and memory type will be assumed to be CPU.

            When `tuple[int, int, Device|str, Any]` is provided, the tuple must contain the following elements:
            - `int`: Pointer to the data in memory.
            - `int`: Size of the data in bytes.
            - `Device`: Either a `dynamo.connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").
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
        if not (isinstance(data, torch.Tensor) or isinstance(data, bytes) or isinstance(data, tuple)):
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

        # Initially `None` cached serialized descriptor reference, populated when `to_serialized()` is called.
        self._serialized: Optional[SerializedDescriptor] = None

        # Data is `torch.Tensor`.
        if isinstance(data, torch.Tensor):
            self._data_ptr = data.data_ptr()
            self._data_size = data.numel() * data.element_size()
            if data.is_cuda:
                self._data_device = Device((DeviceKind.CUDA, data.get_device()))
            self._data_ref = data

            logger.debug(f"Created {self.__repr__()} from `torch.Tensor`.")

        # Data is `tuple[ndarray, Device]`.
        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], array_module.ndarray) and (isinstance(data[1], Device) or isinstance(data[1], str)):
            if hasattr(data[0], '__array_interface__'):
                self._data_ptr = data[0].__array_interface__['data'][0]
            elif hasattr(data[0], '__cuda_array_interface__'):
                self._data_ptr = data[0].__cuda_array_interface__['data'][0]
            else:
                raise TypeError("Argument `data[0]` must be a `ndarray` with a valid array interface.")
            self._data_size = data[0].nbytes
            self._data_device = data[1] if isinstance(data[1], Device) else Device(data[1])
            self._data_ref = data[0]

            logger.debug(f"Created {self.__repr__()} from `tuple[ndarray, Device|str]`.")

        # Data is `bytes`.
        elif isinstance(data, bytes):
            self._data_ptr = id(data)
            self._data_size = len(data)
            self._data_ref = data

            logger.debug(f"Created {self.__repr__()} from `bytes`.")

        # Data is `tuple[int, int, Device, dtype, tuple, Any]`.
        elif isinstance(data, tuple) and len(data) >= 2 and isinstance(data[0], int) and isinstance(data[1], int):
            if len(data) >= 3 and not (isinstance(data[2], Device) or isinstance(data[2], str)):
                raise TypeError("Argument `data` must be a `tuple[int, int, Device|str, Any]`.")

            self._data_ptr = data[0]
            self._data_size = data[1]
            if len(data) >= 3:
                self._data_device = data[2] if isinstance(data[2], Device) else Device(data[2])
            self._data_ref = data[3] if len(data) >=4 else None

            logger.debug(f"Created {self.__repr__()} from `tuple[int, int, Device|str, Any]`.")
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

        logger.debug(f"Deleted {self.__repr__()}.")

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
    def from_serialized(serialized: SerializedDescriptor) -> Descriptor:
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

    def register_memory(self, connector: Connector) -> None:
        """
        Registers the memory of the descriptor with NIXL.
        """
        if not isinstance(connector, Connector):
            raise TypeError("Argument `connector` must be `dynamo.connect.Connector`.")
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
            reg_list = [(self._data_ptr, self._data_size, self._data_device.id, mem_type)]
            self._nixl_hndl = connector._nixl.register_memory(reg_list, mem_type)

        logger.debug(f"Registered {self.__repr__()} with NIXL.")

    def to_serialized(self) -> SerializedDescriptor:
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


class Device:
    """
    Represents a device in the system.
    """
    def __init__(self, metadata: str|tuple[DeviceKind, int]) -> None:
        if metadata is None:
            raise ValueError("Argument `metadata` cannot be `None`.")
        if isinstance(metadata, tuple) and len(metadata) == 2 and isinstance(metadata[0], DeviceKind) and isinstance(metadata[1], int):
            kind, device_id = metadata
        elif isinstance(metadata, str):
            metadata = metadata.strip().lower()
            if metadata.startswith("cuda") or metadata.startswith("gpu"):
                kind = DeviceKind.CUDA
                device_id = 0 if metadata.find(":") == -1 else int(metadata.split(":")[1])
            elif metadata.startswith("cpu") or metadata.startswith("host"):
                kind = DeviceKind.HOST
                device_id = 0
            else:
                raise ValueError("Argument `metadata` must be in the format 'cuda:<device_id>' or 'cpu'.")
        else:
            raise TypeError("Argument `metadata` must be a `tuple[MemoryKind, int]` or a `str`.")


        self._device_id = device_id
        self._kind = kind

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kind={self._kind}, id={self._device_id})"

    def __str__(self) -> str:
        return f"{self._kind}:{self._device_id}" if self._kind is DeviceKind.CUDA else f"{self._kind}"

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
    CUDA = 2

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
    UNSPECIFIED = 0,
    READ = 1,
    WRITE = 2,

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
    UNINTIALIZED = 0,
    INITIALIZED = 1,
    IN_PROGRESS = 2,
    COMPLETE = 3,
    CANCELLED = 4,
    ERRORED = 5,

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
    An abstract class for passive operations that can be awaited.
    """
    def __init__(
        self,
        local: Connector,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        if operation_kind is not OperationKind.READ and operation_kind is not OperationKind.WRITE:
            raise ValueError("Argument `operation_kind` must be either `READ` or `WRITE`.")

        self._status = OperationStatus.UNINTIALIZED

        super().__init__(local, operation_kind, local_descriptors, None, None)

        self._serialized_request: Optional[SerializedRequest] = None
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
            f"local_descriptors={self._local_descriptors}, "
            f"notification_key='{self._notification_key}', "
            f"status='{self._status}'"
            f")"
        )

    async def _wait_for_completion_(self) -> None:
        # Loop until the operation is no longer in progress (or "initalized"),
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
            logger.debug(f"NIXL reported notifications: {len(notifications)}.")

            for key, values in notifications.items():
                if not isinstance(values, list):
                    raise TypeError(f"Expected `dict[str, list[bytes]]` from NIXL notification query; got {type(notifications)}.")
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
                logger.debug(f"{self.__class__.__name__} {{ remote: '{self._connector.name}' status: '{old_status}' => '{self._status}' }}.")

        return self._status

    def to_serialized(self) -> SerializedRequest:
        """
        Gets the request descriptor for the operation.
        """
        if self._serialized_request is None:
            # When we've not yet cached the serialized request, we need to generate one before returning it.
            # Handle both cases: multiple and single descriptors.
            if isinstance(self._local_descriptors, list):
                descriptors=[
                    desc.to_serialized() for desc in self._local_descriptors
                ]
            else:
                descriptors=[
                    self._local_descriptors.to_serialized()
                ]

            original_len = len(self._connector.metadata)
            nixl_metadata = self._connector.metadata
            nixl_metadata = zlib.compress(nixl_metadata, level=6)
            compressed_len = len(nixl_metadata)
            logger.debug(f"Compressed NIXL metadata from {original_len} bytes to {compressed_len} bytes.")
            if compressed_len > original_len:
                logger.warning(f"Compressed NIXL metadata is larger than original ({compressed_len} > {original_len}).")

            self._serialized_request = SerializedRequest(
                    descriptors=descriptors,
                    nixl_metadata=nixl_metadata.hex(),
                    notification_key=self._notification_key,
                    operation_kind=int(self._operation_kind),
                )

        return self._serialized_request

    @abstractmethod
    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        raise NotImplementedError("Abstract method not implemented by derived class.")


class ReadOperation(ActiveOperation):
    """
    Awaitable read operation.
    """
    def __init__(
        self,
        remote: Remote,
        remote_request: SerializedRequest,
        local_descriptors: Descriptor | list[Descriptor] | Device,
    ) -> None:
        if not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `dynamo.connect.RequestDescriptor`.")
        if remote_request.operation_kind != OperationKind.READ.value:
            raise ValueError("Argument `remote_request` must be of kind `READ`.")

        remote_descriptors = remote_request.to_descriptors()

        if not (
            isinstance(local_descriptors, Descriptor)
            or isinstance(local_descriptors, Device)
            or (isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors))
        ):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor`, `list[dynamo.connect.Descriptor]`, or `dynamo.connect.Device`.")

        connector = remote.connector

        if isinstance(local_descriptors, Device):
            logger.debug(f"Creating local allocation for remote descriptors on device {local_descriptors}.")
            if isinstance(remote_descriptors, list):
                local_descriptors = [connector.allocate_descriptor(rd.size, rd.device) for rd in remote_descriptors]
                pass # create local allocations and descriptors for each remote descriptor
            else:
                local_descriptors = connector.allocate_descriptor(remote_descriptors.size, remote_descriptors.device)
                pass # create a single local allocation and descriptor for the remote descriptor

        super().__init__(remote, OperationKind.READ, local_descriptors, remote_descriptors, remote_request.notification_key)

        logger.debug(f"Created {self.__repr__()}")

    def __del__(self) -> None:
        super().__del__()
        logger.debug(f"Deleted {self.__repr__()}")

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

        return self._local_descriptors if isinstance(self._local_descriptors, list) else [self._local_descriptors]

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()


class ReadableOperation(PassiveOperation):
    """
    Operation that awaits until read from.
    """
    def __init__(
        self,
        local: Connector,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        super().__init__(local, OperationKind.READ, local_descriptors)
        logger.debug(f"Created {self.__repr__()}")

    def __del__(self) -> None:
        super().__del__()
        logger.debug(f"Deleted {self.__repr__()}")

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


class Remote:
    """
    Identifies a remote NIXL agent relative to a local NIXL agent.
    """
    def __init__(self, connector: Connector, nixl_agent_metadata: bytes) -> None:
        if not isinstance(connector, Connector):
            raise TypeError("Argument `local` must be `dynamo.connect.Connector`.")
        if not isinstance(nixl_agent_metadata, bytes):
            raise TypeError("Argument `nixl_agent_metadata` must be `bytes`.")

        self._connector = connector
        self._name = connector._nixl.add_remote_agent(nixl_agent_metadata)
        if isinstance(self._name, bytes):
            self._name = self._name.decode("utf-8")

        logger.debug(f"Created {self.__repr__()}.")

    def __del__(self) -> None:
        self.__release__()

    def __enter__(self) -> Remote:
        """
        Context manager entry method. Returns the current instance.
        """
        return self

    def __exit__(self, exc_type: type[BaseException], exc_value: BaseException, traceback: Any) -> None:
        """
        Context manager exit method. Cleans up the instance.
        """
        self.__release__()

    def __repr__(self) -> str:
        return f"RemoteAgent(name={self._name}, connector={self._connector.name})"

    def __str__(self) -> str:
        return self._name

    def __release__(self) -> None:
        """
        Private method for releasing NIXL resources. Not intended for public use.
        """
        pass

    @property
    def connector(self) -> Connector:
        """
        Gets the local connector associated with this remote agent.
        """
        return self._connector

    @property
    def name(self) -> str:
        """
        Gets the name of the remote agent.
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
        return Descriptor(
            data=(self.ptr, self.size, self.device, None)
        )

    @field_validator("device")
    def validate_memtype(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("Argument `device` must be `str`.")
        v = v.strip().lower()
        if not (v.startswith("cuda") or v == "cpu"):
            raise ValueError("Argument `device` must be one of 'cpu' or 'cuda:<device_id>'.")
        return v

    @field_validator("ptr")
    def validate_ptr(cls, v: int) -> int:
        if v == 0:
            raise ValueError("Argument `ptr` cannot be zero (aka `null` or `None`).")
        return v

    @field_validator("size")
    def validate_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Argument `size` must be an integer greater than or equal to zero.")
        return v


class SerializedRequest(BaseModel):
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
        Deserializes the request descriptor into a `dynamo.connect.Descriptor` or list of `dynamo.connect.Descriptor` objects.
        """
        if len(self.descriptors) == 0:
            raise ValueError("Request descriptor must contain at least one serialized descriptor.")
        if len(self.descriptors) == 1:
            return self.descriptors[0].to_descriptor()
        return [item.to_descriptor() for item in self.descriptors]

    @field_validator("operation_kind")
    def validate_operation_kind(cls, v: int) -> int:
        if v < 1 or v > 3:
            raise TypeError("Argument `operation_kind` must be an integer value of `dynamo.connect.OperationKind`.")
        return v


class WritableOperation(PassiveOperation):
    """
    Operation which awaits until written to.
    """
    def __init__(
        self,
        local: Connector,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        super().__init__(local, OperationKind.WRITE, local_descriptors)

        logger.debug(f"Created {self.__repr__()}")

    def __del__(self) -> None:
        super().__del__()
        logger.debug(f"Deleted {self.__repr__()}")

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
    Awaitable write operation.
    """
    def __init__(
        self,
        remote: Remote,
        local_descriptors: Descriptor | list[Descriptor],
        remote_request: SerializedRequest,
    ) -> None:
        if not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `dynamo.connect.RequestDescriptor`.")
        if remote_request.operation_kind != OperationKind.WRITE.value:
            raise ValueError("Argument `remote_request` must be of kind `WRITE`.")

        remote_descriptors = remote_request.to_descriptors()

        super().__init__(remote, OperationKind.WRITE, local_descriptors, remote_descriptors, remote_request.notification_key)

        logger.debug(f"Created {self.__repr__()}")

    def __del__(self) -> None:
        super().__del__()
        logger.debug(f"Deleted {self.__repr__()}")

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
