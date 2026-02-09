---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# dynamo.nixl_connect.WriteOperation

An operation which transfers data from the local worker to a remote worker.

To create the operation, NIXL metadata ([RdmaMetadata](rdma-metadata.md)) from a remote worker's [`WritableOperation`](writable-operation.md)
along with a matching set of local [`Descriptor`](descriptor.md) objects which reference memory to be transferred to the remote worker must be provided.
The NIXL metadata must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.
Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.
Cancellation is handled asynchronously.


## Example Usage

```python
    async def write_to_remote(
      self,
      remote_metadata: dynamo.nixl_connect.RdmaMetadata,
      local_tensor: torch.Tensor
    ) -> None:
      descriptor = dynamo.nixl_connect.Descriptor(local_tensor)

      with await self.connector.begin_write(descriptor, remote_metadata) as write_op:
        # Wait for the operation to complete writing local_tensor to the remote worker.
        await write_op.wait_for_completion()
```


## Methods

### `cancel`

```python
def cancel(self) -> None:
```

Instructs the NIXL subsystem to cancel the operation.
Completed operations cannot be cancelled.

### `wait_for_completion`

```python
async def wait_for_completion(self) -> None:
```

Blocks the caller until all provided buffers have been transferred to the remote worker.


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
  - [WritableOperation](writable-operation.md)
