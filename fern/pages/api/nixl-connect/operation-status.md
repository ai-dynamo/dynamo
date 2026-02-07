---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# dynamo.nixl_connect.OperationStatus(IntEnum)

Represents the current state or status of an operation.


## Values

### `CANCELLED`

The operation has been cancelled by the user or system.

### `COMPLETE`

The operation has been completed successfully.

### `ERRORED`

The operation has encountered an error and cannot be completed.

### `IN_PROGRESS`

The operation has been initialized and is in-progress (not completed, errored, or cancelled).

### `INITIALIZED`

The operation has been initialized and is ready to be processed.

### `UNINITIALIZED`

The operation has not been initialized yet and is not in a valid state.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [RdmaMetadata](rdma-metadata.md)
  - [ReadOperation](read-operation.md)
  - [ReadableOperation](readable-operation.md)
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
