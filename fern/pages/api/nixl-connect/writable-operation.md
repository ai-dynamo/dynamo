---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# dynamo.nixl_connect.WritableOperation

An operation which enables a remote worker to write data to the local worker.

To create the operation, a set of local [`Descriptor`](descriptor.md) objects must be provided which reference memory intended to receive data from a remote worker.
Once created, the memory referenced by the provided descriptors becomes immediately writable by a remote worker with the necessary metadata.
The NIXL metadata ([RdmaMetadata](rdma-metadata.md)) required to access the memory referenced by the provided descriptors is accessible via the operations `.metadata()` method.
Once acquired, the metadata needs to be provided to a remote worker via a secondary channel, most likely HTTP or TCP+NATS.

Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.
Cancellation is handled asynchronously.


## Example Usage

```python
    async def recv_data(
      self,
      local_tensor: torch.Tensor
    ) -> None:
      descriptor = dynamo.nixl_connect.Descriptor(local_tensor)

      with await self.connector.create_writable(descriptor) as write_op:
        op_metadata = write_op.metadata()

        # Send the metadata to the remote worker via sideband communication.
        await self.request_remote_data(op_metadata)
        # Wait the remote worker to complete its write operation to local_tensor.
        # AKA receive data from remote worker.
        await write_op.wait_for_completion()
```


## Methods

### `metadata`

```python
def metadata(self) -> RdmaMetadata:
```

Generates and returns the NIXL metadata ([RdmaMetadata](rdma-metadata.md)) required for a remote worker to write to the operation.
Once acquired, the metadata needs to be provided to a remote worker via a secondary channel, most likely HTTP or TCP+NATS.

### `wait_for_completion`

```python
async def wait_for_completion(self) -> None:
```

Blocks the caller until the operation has received a completion signal from a remote worker.


## Properties

### `status`

```python
@property
def status(self) -> OperationStatus:
```

Returns [`OperationStatus`](operation-status.md) which provides the current state (aka. status) of the operation.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation-status.md)
  - [RdmaMetadata](rdma-metadata.md)
  - [ReadOperation](read-operation.md)
  - [ReadableOperation](readable-operation.md)
  - [WriteOperation](write-operation.md)
