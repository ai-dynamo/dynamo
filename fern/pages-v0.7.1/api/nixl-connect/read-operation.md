---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "dynamo.nixl_connect.ReadOperation"
---

An operation which transfers data from a remote worker to the local worker.

To create the operation, NIXL metadata ([RdmaMetadata](rdma-metadata.md)) from a remote worker's [`ReadableOperation`](readable-operation.md)
along with a matching set of local [`Descriptor`](descriptor.md) objects which reference memory intended to receive data from the remote worker must be provided.
The NIXL metadata must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.
Disposal of the object will instruct the NIXL subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.


## Example Usage

```python
    async def read_from_remote(
      self,
      remote_metadata: dynamo.nixl_connect.RdmaMetadata,
      local_tensor: torch.Tensor
    ) -> None:
      descriptor = dynamo.nixl_connect.Descriptor(local_tensor)

      with await self.connector.begin_read(remote_metadata, descriptor) as read_op:
        # Wait for the operation to complete writing data from the remote worker to local_tensor.
        await read_op.wait_for_completion()
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

Blocks the caller until the memory from the remote worker has been transferred to the provided buffers.


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
  - [ReadableOperation](readable-operation.md)
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
