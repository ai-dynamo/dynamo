---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# dynamo.nixl_connect.Device

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

When the device is the [`HOST`](device-kind.md#host), this value is always `0`.

When the device is a [`GPU`](device-kind.md#cuda), this value identifies a specific GPU.

### `kind`

```python
@property
def kind(self) -> DeviceKind:
```

Gets the [`DeviceKind`](device-kind.md) of device the instance references.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [OperationStatus](operation-status.md)
  - [ReadOperation](read-operation.md)
  - [ReadableOperation](readable-operation.md)
  - [RdmaMetadata](rdma-metadata.md)
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
