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

# dynamo.connect.Connector

Core class for managing the connection between workers in a distributed environment.
Use this class to create readable and writable operations, or read and write data to remote workers.

This class is responsible for interfacing with the NIXL-based RDMA subsystem and providing a "Pythonic" interface
with which to utilize GPU Direct RDMA accelerated data transfers between models hosted by different workers in a Dynamo pipeline.
The connector provides two methods of moving data between workers:

  - Preparing local memory to be written to by a remote worker.

  - Preparing local memory to be read by a remote worker.

In both cases, local memory is registered with the NIXL-based RDMA subsystem via the [`Descriptor`](#descriptor) class and provided to the connector.
The connector then configures the RDMA subsystem to expose the memory for the requested operation and returns an operation control object.
The operation control object, either a [`ReadableOperation`](readable_operation.md) or a [`WritableOperation`](writable_operation.md),
provides RDMA metadata ([RdmaMetadata](rdma_metadata.md)) via its `.metadata()` method, functionality to query the operation's current state, as well as the ability to cancel the operation prior to its completion.

The RDMA metadata must be provided to the remote worker expected to complete the operation.
The metadata contains required information (identifiers, keys, etc.) which enables the remote worker to interact with the provided memory.

> [!Warning]
> RDMA metadata contains a worker's address as well as security keys to access specific registered memory descriptors.
> This data provides direct memory access between workers, and should be considered sensitive and therefore handled accordingly.


## Example Usage

```python
    @async_on_start
    async def async_init(self):
      runtime = dynamo_context["runtime"]

      self.connector = dynamo.connect.Connector(runtime=runtime)
      await self.connector.initialize()
```

> [!Tip]
> See [`ReadOperation`](read_operation.md#example-usage), [`ReadableOperation`](readable_operation.md#example-usage),
> [`WritableOperation`](writable_operation.md#example-usage), and [`WriteOperation`](write_operation.md#example-usage)
> for additional examples.


## Methods

### `begin_read`

Creates a [`ReadOperation`](read_operation.md) for transferring data from a remote worker.

To create the operation, the serialized request from a remote worker's [`ReadableOperation`](readable_operation.md)
along with a matching set of local memory descriptors which reference memory intended to receive data from the remote worker
must be provided.
The serialized request must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.

Disposal of the object will instruct the RDMA subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](read_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.

### `begin_write`

Creates a [`WriteOperation`](write_operation.md) for transferring data to a remote worker.

To create the operation, the serialized request from a remote worker's [`WritableOperation`](writable_operation.md)
along with a matching set of local memory descriptors which reference memory to be transferred to the remote worker
must be provided.
The serialized request must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.

Disposal of the object will instruct the RDMA subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](write_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.

### `create_readable`

Creates a [`ReadableOperation`](readable_operation.md) for transferring data to a remote worker.

To create the operation, a set of local memory descriptors must be provided that reference memory intended to be transferred to a remote worker.
Once created, the memory referenced by the provided descriptors becomes immediately readable by a remote worker with the necessary metadata.
The metadata required to access the memory referenced by the provided descriptors is accessible via the operation's `.metadata()` method.
Once acquired, the metadata needs to be provided to a remote worker via a secondary channel, most likely HTTP or TCP+NATS.

Disposal of the object will instruct the RDMA subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](readable_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.

### `create_writable`

Creates a [`WritableOperation`](writable_operation.md) for transferring data from a remote worker.

To create the operation, a set of local memory descriptors must be provided which reference memory intended to receive data from a remote worker.
Once created, the memory referenced by the provided descriptors becomes immediately writable by a remote worker with the necessary metadata.
The metadata required to access the memory referenced by the provided descriptors is accessible via the operation's `.metadata()` method.
Once acquired, the metadata needs to be provided to a remote worker via a secondary channel, most likely HTTP or TCP+NATS.

Disposal of the object will instruct the RDMA subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.

Use [`.wait_for_completion()`](writable_operation.md#wait_for_completion) to block the caller until the operation has completed or encountered an error.


## Related Classes

  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation_status.md)
  - [RdmaMetadata](rdma_metadata.md)
  - [ReadOperation](read_operation.md)
  - [ReadableOperation](readable_operation.md)
  - [WritableOperation](writable_operation.md)
  - [WriteOperation](write_operation.md)
