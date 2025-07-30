<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Supporting SGLang's native endpoints via HTTP Server

# Introduction

The SGLang HTTP server provides a REST API interface for managing and monitoring SGLang components running in a dynamo distributed environment. It leverages dynamo's service discovery mechanism to automatically find and communicate with SGLang workers across the cluster.

<details>
<summary>How it works under the hood</summary>

## Architecture Overview

The HTTP server (`sgl_http_server.py`) is built on FastAPI and integrates with dynamo's `DistributedRuntime` to discover and interact with SGLang components. It uses the following discovery flow:

1. **Service Discovery**: Queries dynamo's etcd instance to find components that expose specific endpoints
2. **Dynamic Targeting**: Automatically discovers all matching components across namespaces without requiring manual configuration
3. **Direct Communication**: Establishes direct connections to discovered component instances using dynamo's client infrastructure

## Discovery Mechanism

The server uses dynamo's hierarchical service discovery structure:

- **DistributedRuntime**: Maintains connections to etcd (service discovery) and NATS (messaging)
- **Namespace**: Logical grouping of components (default: "dynamo")
- **Component**: Individual SGLang workers or services
- **Endpoint**: Specific functionality exposed by each component

The discovery process queries etcd with the prefix `instances/` to find all registered components that expose the target endpoint. Components are identified by their namespace, component name, and endpoint, allowing the server to dynamically scale operations across multiple instances.

</details>

## Supported Endpoints

All of these endpoints can be called using

```bash
curl -X POST http://<ip>:9001/<endpoint>
```

#### `/flush_cache`
Flushes the kv cache across all SGLang components. Useful for resetting after a warmup or a benchmarking run.

#### `/start_expert_distribution_record`
Begins recording expert distribution metrics across SGLang components.

#### `/stop_expert_distribution_record`
Stops the expert distribution recording process.

#### `/dump_expert_distribution_record`
Dumps the collected expert distribution data.

## Configuration

The server accepts the following command-line arguments:

- `--port`: HTTP server port (default: 9001)
- `--ns/--namespace`: Target dynamo namespace (default: "dynamo")

## Usage

Start the server:
```bash
python src/dynamo/sglang/utils/sgl_http_server.py --port 9001 --namespace dynamo
```

The server will automatically discover all SGLang components in the specified namespace and provide HTTP endpoints for managing them.
