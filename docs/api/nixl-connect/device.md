---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Device
---

`Device` class describes the device a given allocation resides in.
Usually host (`"cpu"`) or GPU (`"cuda"`) memory.

When a system contains multiple GPU devices, specific GPU devices can be identified by including their ordinal index number.
For example, to reference the second GPU in a system `"cuda:1"` can be used.

By default, when `"cuda"` is provided, it is assumed to be `"cuda:0"` or the first GPU enumerated by the system.


## Properties

### `id`

```python
@property
def id(self) -> int:
```

Gets the identity, or ordinal, of the device.

For [`DRAM`](device-mem-type.md#dram) devices, this value is always `0`.

For [`VRAM`](device-mem-type.md#vram) devices, this value identifies a specific accelerator.

### `mem_type`

```python
@property
def mem_type(self) -> DeviceMemType:
```

Gets the [`DeviceMemType`](device-mem-type.md) of the memory segment the instance references.

### `name`

```python
@property
def name(self) -> str:
```

Gets the framework device name (`"cpu"`, `"cuda"`, `"xpu"`, etc.).

This is the string used for display and for serialization via [`SerializedDescriptor`](rdma-metadata.md).


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [OperationStatus](operation-status.md)
  - [ReadOperation](read-operation.md)
  - [ReadableOperation](readable-operation.md)
  - [RdmaMetadata](rdma-metadata.md)
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
