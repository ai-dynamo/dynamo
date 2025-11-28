<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# dynamo.nixl_connect.Connector

Core class for managing the connection between workers in a distributed environment.
Use this class to create readable and writable operations, or read and write data to remote workers.

This class provides a "pythonic" interface using NIXL library to utilize GPU Direct RDMA accelerated, when available, data transfers between models hosted by different workers in a Dynamo graph.
The connector provides two methods of moving data between workers:

  - Preparing local memory to be written to by a remote worker.

  - Preparing local memory to be read by a remote worker.

In both cases, local memory is registered with the NIXL-based I/O subsystem via the [`Descriptor`](#descriptor) class and provided to the connector.
When RDMA is available, the connector then configures the RDMA subsystem to expose the memory for the requested operation and returns an operation control object;
otherwise the connector will select the best available RDMA alternative.
The operation control object, either a [`ReadableOperation`](readable_operation.md) or a [`WritableOperation`](writable_operation.md),
provides NIXL metadata ([RdmaMetadata](rdma_metadata.md)) via its `.metadata()` method, functionality to query the operation's current state, as well as the ability to cancel the operation prior to its completion.

The NIXL metadata must be provided to the remote worker expected to complete the operation.
The metadata contains required information (identifiers, keys, etc.) which enables the remote worker to interact with the provided memory.

> [!Warning]
> NIXL metadata contains a worker's address as well as security keys to access specific registered memory descriptors.
> This data provides direct memory access between workers, and should be considered sensitive and therefore handled accordingly.


## Example Usage

```python
    @async_on_start
    async def async_init(self):
      self.connector = dynamo.nixl_connect.Connector()
      await self.connector.initialize()
```

> [!Tip]
> See [`ReadOperation`](read_operation.md#example-usage), [`ReadableOperation`](readable_operation.md#example-usage),
> [`WritableOperation`](writable_operation.md#example-usage), and [`WriteOperation`](write_operation.md#example-usage)
> for additional examples.


## Methods

### `begin_read`

```python
async def begin_read(
    self,
    remote_metadata: RdmaMetadata,
    local_descriptors: Descriptor | list[Descriptor],
) -> ReadOperation:
```

Creates a [`ReadOperation`](read_operation.md) for transferring data from a remote worker.

To create the operation, the serialized request from a remote worker's [`ReadableOperation`](readable_operation.md)
along with a matching set of local memory descriptors which reference memory intended to receive data from the remote worker
must be provided.
The serialized request must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.

Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](read_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.

### `begin_write`

```python
async def begin_write(
    self,
    local_descriptors: Descriptor | list[Descriptor],
    remote_metadata: RdmaMetadata,
) -> WriteOperation:
```

Creates a [`WriteOperation`](write_operation.md) for transferring data to a remote worker.

To create the operation, the serialized request from a remote worker's [`WritableOperation`](writable_operation.md)
along with a matching set of local memory descriptors which reference memory to be transferred to the remote worker
must be provided.
The serialized request must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.

Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](write_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.

### `create_readable`

```python
def create_readable(
    self,
    local_descriptors: Descriptor | list[Descriptor],
) -> ReadableOperation:
```

Creates a [`ReadableOperation`](readable_operation.md) for transferring data to a remote worker.

To create the operation, a set of local memory descriptors must be provided that reference memory intended to be transferred to a remote worker.
Once created, the memory referenced by the provided descriptors becomes immediately readable by a remote worker with the necessary metadata.
The metadata required to access the memory referenced by the provided descriptors is accessible via the operation's `.metadata()` method.
Once acquired, the metadata needs to be provided to a remote worker via a secondary channel, most likely HTTP or TCP+NATS.

Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](readable_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.

### `create_writable`

```python
def create_writable(
    self,
    local_descriptors: Descriptor | list[Descriptor],
) -> WritableOperation:
```

Creates a [`WritableOperation`](writable_operation.md) for transferring data from a remote worker.

To create the operation, a set of local memory descriptors must be provided which reference memory intended to receive data from a remote worker.
Once created, the memory referenced by the provided descriptors becomes immediately writable by a remote worker with the necessary metadata.
The metadata required to access the memory referenced by the provided descriptors is accessible via the operation's `.metadata()` method.
Once acquired, the metadata needs to be provided to a remote worker via a secondary channel, most likely HTTP or TCP+NATS.

Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](writable_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.


## Properties

### `is_cuda_available`

```python
@cached_property
def is_cuda_available(self) -> bool:
```

Gets `True` when CUDA is available for the selected array module (most likely CuPy); otherwise `False`.

### `name`

```python
@property
def name(self) -> str | None:
```

Gets the Dynamo component name used by the connector.

### `namespace`

```python
@property
def namespace(self) -> str:
```

Gets the Dynamo namespace used by the connector.

### `runtime`

```python
def runtime(self) -> dynamo.runtime.DistributedRuntime:
```

Gets the Dynamo distributed runtime instance associated with the connector.

## Related Classes

  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation_status.md)
  - [RdmaMetadata](rdma_metadata.md)
  - [ReadOperation](read_operation.md)
  - [ReadableOperation](readable_operation.md)
  - [WritableOperation](writable_operation.md)
  - [WriteOperation](write_operation.md)
