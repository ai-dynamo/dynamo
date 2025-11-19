<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Service Discovery

## Overview

By default, Dynamo discovers endpoints and model cards through etcd. An experimental Kubernetes backend is available for discovery that uses native Kubernetes EndpointSlices, eliminating the dependency on etcd.

**Using DynamoGraphDeployment (Recommended):**

When deploying with the Dynamo operator, simply add the annotation to your DGD manifest:

```yaml
metadata:
  annotations:
    nvidia.com/dynamo-discovery-backend: kubernetes
```

The operator will automatically configure the required EndpointSlices, labels, and pod environment variables. See [`dgd.yaml`](./dgd.yaml) for a complete example.

## Environment Variables

| **Variable** | **Description** | **Default** |
| ------------ | --------------- | ----------- |
| `DYN_DISCOVERY_BACKEND` | Discovery backend (`kv_store` for etcd or `kubernetes` for experimental EndpointSlice-based discovery) | `kv_store` |

## Metadata Endpoint

The Kubernetes backend exposes a `/metadata` endpoint on each pod that returns registered discovery information. This is used by the system status server to expose the discovery information to the clients on the discovery plane.

### Example Request

```bash
curl -s localhost:9090/metadata | jq
```

### Example Response

```json
{
  "endpoints": {
    "vllm-disagg/backend/generate": {
      "component": "backend",
      "endpoint": "generate",
      "instance_id": 12345678901234567890,
      "namespace": "vllm-disagg",
      "transport": {
        "nats_tcp": "vllm-disagg_backend.generate-abc123"
      }
    }
  },
  "model_cards": {}
}
```
