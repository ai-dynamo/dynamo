<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Supporting SGLang's native endpoints via HTTP Server

# Introduction

The SGLang HTTP server provides a REST API interface for managing and monitoring SGLang components running in a dynamo distributed environment. It leverages dynamo's service discovery mechanism to automatically find and communicate with SGLang workers across the cluster.

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

## Supported Endpoints

### Current Endpoints

#### POST /flush_cache
Flushes the radix cache across all discovered SGLang components.

**Behavior:**
- Discovers all components in the specified namespace that expose the `flush_cache` endpoint
- Sends flush requests to all instances of each discovered component
- Returns success/failure status with details about the operation

**Response:**
```json
{
  "message": "Cache flush initiated",
  "success": true
}
```

### Upcoming Endpoints

The following endpoints will be supported in future releases:

#### POST /start_expert_distribution_record
Begins recording expert distribution metrics across SGLang components.

#### POST /stop_expert_distribution_record
Stops the expert distribution recording process.

#### GET /dump_expert_distribution_record
Retrieves the collected expert distribution data.

## Configuration

The server accepts the following command-line arguments:

- `--port`: HTTP server port (default: 9001)
- `--ns/--namespace`: Target dynamo namespace (default: "dynamo")
- `--comp/--component`: Specific component name to target (default: discover all)
- `--endpoint`: Endpoint name to discover (default: "flush_cache")

## Usage

Start the server:
```bash
python src/dynamo/sglang/utils/sgl_http_server.py --port 9001 --namespace dynamo
```

The server will automatically discover all SGLang components in the specified namespace and provide HTTP endpoints for managing them.
