---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Device Mem Type
---

`DeviceMemType` identifies the NIXL memory segment in which a [`Device`](device.md)
allocation resides.

The enum values correspond to canonical NIXL segment names and can be passed
directly to the NIXL API.


## Values

### `DRAM`

System (CPU) memory.

### `VRAM`

GPU memory.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation-status.md)
  - [RdmaMetadata](rdma-metadata.md)
  - [ReadOperation](read-operation.md)
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
