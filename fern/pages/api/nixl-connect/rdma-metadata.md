---
title: "dynamo.nixl_connect.RdmaMetadata"
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

A Pydantic type intended to provide JSON serialized NIXL metadata about a [`ReadableOperation`](readable-operation.md) or [`WritableOperation`](writable-operation.md) object.
NIXL metadata contains detailed information about a worker process and how to access memory regions registered with the corresponding agent.
This data is required to perform data transfers using the NIXL-based I/O subsystem.

<Warning>
NIXL metadata contains information to connect corresponding backends across agents, as well as identification keys to access specific registered memory regions.
This data provides direct memory access between workers, and should be considered sensitive and therefore handled accordingly.
</Warning>

Use the respective class's `.metadata()` method to generate an `RdmaMetadata` object for an operation.

<Tip>
Classes using `RdmaMetadata` objects must be paired correctly.
[`ReadableOperation`](readable-operation.md) with [`ReadOperation`](read-operation.md), and
[`WritableOperation`](write-operation.md) with [`WriteOperation`](write-operation.md).
Incorrect pairing will result in an error being raised.
</Tip>


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation-status.md)
  - [ReadOperation](read-operation.md)
  - [ReadableOperation](readable-operation.md)
  - [WritableOperation](writable-operation.md)
  - [WriteOperation](write-operation.md)
