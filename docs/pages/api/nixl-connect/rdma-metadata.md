---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# dynamo.nixl_connect.RdmaMetadata

A Pydantic type intended to provide JSON serialized NIXL metadata about a [`ReadableOperation`](readable-operation.md) or [`WritableOperation`](writable-operation.md) object.
NIXL metadata contains detailed information about a worker process and how to access memory regions registered with the corresponding agent.
This data is required to perform data transfers using the NIXL-based I/O subsystem.

> [!Warning]
> NIXL metadata contains information to connect corresponding backends across agents, as well as identification keys to access specific registered memory regions.
> This data provides direct memory access between workers, and should be considered sensitive and therefore handled accordingly.

Use the respective class's `.metadata()` method to generate an `RdmaMetadata` object for an operation.

> [!Tip]
> Classes using `RdmaMetadata` objects must be paired correctly.
> [`ReadableOperation`](readable-operation.md) with [`ReadOperation`](read-operation.md), and
> [`WritableOperation`](write-operation.md) with [`WriteOperation`](write-operation.md).
> Incorrect pairing will result in an error being raised.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation-status.md)
  - [ReadOperation](read-operation.md)
  - [ReadableOperation](readable-operation.md)
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
