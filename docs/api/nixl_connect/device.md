<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

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

When the device is the [`HOST`](device_kind.md#host), this value is always `0`.

When the device is a [`GPU`](device_kind.md#cuda), this value identifies a specific GPU.

### `kind`

```python
@property
def kind(self) -> DeviceKind:
```

Gets the [`DeviceKind`](device_kind.md) of device the instance references.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [OperationStatus](operation_status.md)
  - [ReadOperation](read_operation.md)
  - [ReadableOperation](readable_operation.md)
  - [RdmaMetadata](rdma_metadata.md)
  - [WritableOperation](writable_operation.md)
  - [WriteOperation](write_operation.md)
