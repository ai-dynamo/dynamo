---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# dynamo.nixl_connect.ReadableOperation

An operation which enables a remote worker to read data from the local worker.

To create the operation, a set of local [`Descriptor`](descriptor.md) objects must be provided that reference memory intended to be transferred to a remote worker.
Once created, the memory referenced by the provided descriptors becomes immediately readable by a remote worker with the necessary metadata.
The NIXL metadata ([RdmaMetadata](rdma-metadata.md)) required to access the memory referenced by the provided descriptors is accessible via the operations `.metadata()` method.
Once acquired, the metadata needs to be provided to a remote worker via a secondary channel, most likely HTTP or TCP+NATS.

Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.


## Example Usage

```python
    async def send_data(
      self,
      local_tensor: torch.Tensor
    ) -> None:
      descriptor = dynamo.nixl_connect.Descriptor(local_tensor)

      with await self.connector.create_readable(descriptor) as read_op:
        op_metadata = read_op.metadata()

        # Send the metadata to the remote worker via sideband communication.
        await self.notify_remote_data(op_metadata)
        # Wait for the remote worker to complete its read operation of local_tensor.
        # AKA send data to remote worker.
        await read_op.wait_for_completion()
```


## Methods

### `metadata`

```python
def metadata(self) -> RdmaMetadata:
```

Generates and returns the NIXL metadata ([RdmaMetadata](rdma-metadata.md)) required for a remote worker to read from the operation.
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
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
