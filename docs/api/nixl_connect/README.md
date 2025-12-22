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

# Dynamo NIXL Connect

Dynamo NIXL Connect specializes in moving data between models/workers in a Dynamo Graph, and for the use cases where registration and memory regions need to be dynamic.
Dynamo connect provides utilities for such use cases, using the NIXL-based I/O subsystem via a set of Python classes.
The relaxed registration comes with some performance overheads, but simplifies the integration process.
Especially for larger data transfer operations, such as between models in a multi-model graph, the overhead would be marginal.
The `dynamo.nixl_connect` library can be imported by any Dynamo container hosted application.

> [!Note]
> Dynamo NIXL Connect will pick the best available method of data transfer available to it.
> The available methods depend on the hardware and software configuration of the machines and network running the graph.
> GPU Direct RDMA operations require that both ends of the operation have:
> - NIC and GPU capable of performing RDMA operations
> - Device drivers that support GPU-NIC direct interactions (aka "zero copy") and RDMA operations
> - Network that supports InfiniBand or RoCE
>
> With any of the above not satisfied, GPU Direct RDMA will not be available to the graph's workers, and less-optimal methods will be utilized to ensure basic functionality.
> For additional information, please read this [GPUDirect RDMA](https://docs.nvidia.com/cuda/pdf/GPUDirect_RDMA.pdf) document.

```python
import dynamo.nixl_connect
```

All operations using the NIXL Connect library begin with the [`Connector`](connector.md) class and the type of operation required.
There are four types of supported operations:

 1. **Register local readable memory**:

    Register local memory buffer(s) with the NIXL subsystem to enable a remote worker to read from.

 2. **Register local writable memory**:

    Register local memory buffer(s) with the NIXL subsystem to enable a remote worker to write to.

 3. **Read from registered, remote memory**:

    Read remote memory buffer(s), registered by a remote worker to be readable, into local memory buffer(s).

 4. **Write to registered, remote memory**:

    Write local memory buffer(s) to remote memory buffer(s) registered by a remote worker to writable.

When available, by connecting correctly paired operations, high-throughput GPU Direct RDMA data transfers can be completed.
Given the list above, the correct pairing of operations would be 1 & 3 or 2 & 4.
Where one side is a "(read|write)-able operation" and the other is its correctly paired "(read|write) operation".
Specifically, a read operation must be paired with a readable operation, and a write operation must be paired with a writable operation.

```mermaid
sequenceDiagram
    participant LocalWorker
    participant RemoteWorker
    participant NIXL

    LocalWorker ->> NIXL: Register memory (Descriptor)
    RemoteWorker ->> NIXL: Register memory (Descriptor)
    LocalWorker ->> LocalWorker: Create Readable/WritableOperation
    LocalWorker ->> RemoteWorker: Send NIXL metadata (via HTTP/TCP+NATS)
    RemoteWorker ->> NIXL: Begin Read/WriteOperation with metadata
    NIXL -->> RemoteWorker: Data transfer
    RemoteWorker -->> LocalWorker: Notify completion (unblock awaiter)
```

## Examples

### Generic Example

In the diagram below, Local creates a [`WritableOperation`](writable_operation.md) intended to receive data from Remote.
Local then sends metadata about the requested operation to Remote.
Remote then uses the metadata to create a [`WriteOperation`](write_operation.md) which will perform the GPU Direct RDMA memory transfer, when available, from Remote's GPU memory to Local's GPU memory.

```mermaid
---
title: Write Operation Between Two Workers (RDMA available)
---
flowchart LR
  c1[Remote] --"3: .begin_write()"--- WriteOperation
  WriteOperation e1@=="4: GPU Direct RDMA"==> WritableOperation
  WritableOperation --"1: .create_writable()"--- c2[Local]
  c2 e2@--"2: RDMA Metadata via HTTP"--> c1
  e1@{ animate: true; }
  e2@{ animate: true; }
```

> [!Note]
> When RDMA isn't available, the NIXL data transfer will still complete using non-accelerated methods.

### Multimodal Example

In the case of the [Dynamo Multimodal Disaggregated Example](../../multimodal/vllm.md):

 1. The HTTP frontend accepts a text prompt and a URL to an image.

 2. The prompt and URL are then enqueued with the Processor before being dispatched to the first available Decode Worker.

 3. Decode Worker then requests a Prefill Worker to provide key-value data for the LLM powering the Decode Worker.

 4. Prefill Worker then requests that the image be processed and provided as embeddings by the Encode Worker.

 5. Encode Worker acquires the image, processes it, performs inference on the image using a specialized vision model, and finally provides the embeddings to Prefill Worker.

 6. Prefill Worker receives the embeddings from Encode Worker and generates a key-value cache (KV$) update for Decode Worker's LLM and writes the update directly to the GPU memory reserved for the data.

 7. Finally, Decode Worker performs the requested inference.

```mermaid
---
title: Multimodal Disaggregated Workflow
---
flowchart LR
  p0[HTTP Frontend] i0@--"text prompt"-->p1[Processor]
  p0 i1@--"url"-->p1
  p1 i2@--"prompt"-->dw[Decode Worker]
  p1 i3@--"url"-->dw
  dw i4@--"prompt"-->pw[Prefill Worker]
  dw i5@--"url"-->pw
  pw i6@--"url"-->ew[Encode Worker]
  ew o0@=="image embeddings"==>pw
  pw o1@=="kv_cache updates"==>dw
  dw o2@--"inference results"-->p0

  i0@{ animate: true; }
  i1@{ animate: true; }
  i2@{ animate: true; }
  i3@{ animate: true; }
  i4@{ animate: true; }
  i5@{ animate: true; }
  i6@{ animate: true; }
  o0@{ animate: true; }
  o1@{ animate: true; }
  o2@{ animate: true; }
```

> [!Note]
> In this example, it is the data transfer between the Prefill Worker and the Encode Worker that utilizes the Dynamo NIXL Connect library.
> The KV Cache transfer between Decode Worker and Prefill Worker utilizes a different connector that also uses the NIXL-based I/O subsystem underneath.

#### Code Examples

See [MultimodalPDWorkerHandler](../../../components/src/dynamo/vllm/multimodal_handlers/worker_handler.py) or [MultimodalDecodeWorkerHandler](../../../components/src/dynamo/vllm/multimodal_handlers/worker_handler.py) from our Multimodal example,
for how they coordinate directly with the Encode Worker by creating a [`WritableOperation`](writable_operation.md),
sending the operation's metadata via Dynamo's round-robin dispatcher, and awaiting the operation for completion before making use of the transferred data.

See [MultimodalEncodeWorkerHandler](../../../components/src/dynamo/vllm/multimodal_handlers/encode_worker_handler.py) from our Multimodal example,
for how the resulting embeddings are registered with the NIXL subsystem by creating a [`Descriptor`](descriptor.md),
a [`WriteOperation`](write_operation.md) is created using the metadata provided by the requesting worker,
and the worker awaits for the data transfer to complete for yielding a response.


## Python Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [ReadOperation](read_operation.md)
  - [ReadableOperation](readable_operation.md)
  - [WritableOperation](writable_operation.md)
  - [WriteOperation](write_operation.md)


## References

  - [NVIDIA Dynamo](https://developer.nvidia.com/dynamo) @ [GitHub](https://github.com/ai-dynamo/dynamo)
  - [NVIDIA Inference Transfer Library (NIXL)](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/#nvidia_inference_transfer_library_nixl_low-latency_hardware-agnostic_communication%C2%A0) @ [GitHub](https://github.com/ai-dynamo/nixl)
  - [Dynamo Multimodal Example](../../..//examples/multimodal)
  - [NVIDIA GPU Direct](https://developer.nvidia.com/gpudirect)
