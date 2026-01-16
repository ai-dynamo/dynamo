---
title: "dynamo.nixl_connect.OperationStatus(IntEnum)"
---

{/*
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
*/}

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
