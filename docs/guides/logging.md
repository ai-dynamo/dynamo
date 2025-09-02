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

# Dynamo Logging 

## Overview

Dynamo provides structured logging in both text as well as JSONL. When
JSONL is enabled logs additionally contain `span` creation and exit
events as well as support for `trace_id` and `span_id` fields for
distributed tracing.

## Environment Variables for configuring Logging

| Environment Variable                | Description                                 | Example Settings                  |
| ----------------------------------- | --------------------------------------------| ---------------------------------------------------- |
| `DYN_LOGGING_JSONL`                | Enable JSONL logging format (default: READABLE)                  | `DYN_LOGGING_JSONL=1`                          |
| `DYN_LOG_USE_LOCAL_TZ`             | Use local timezone for logging timestamps (default: UTC)         | `DYN_LOG_USE_LOCAL_TZ=1`                       |
| `DYN_LOG`                          | Log levels per target (comma-separated key-value pairs)             | `DYN_LOG=info,dynamo_runtime::system_status_server:trace`  |
| `DYN_LOGGING_CONFIG_PATH`          | Path to custom TOML logging configuration file            | `DYN_LOGGING_CONFIG_PATH=/path/to/config.toml`|


## Example Readable Format

Environment Setting:

```

```

## Frontend Liveness Check

The frontend liveness endpoint reports a status of `live` as long as
the service is running. 

#### Example Request

```
curl -s localhost:8080/live -q |  jq
```

#### Example Response

```
{
  "message": "Service is live",
  "status": "live"
}
```

## Frontend Health Check

The frontend liveness endpoint reports a status of `healthy` once a
model has been registered. During initial startup the frontend will
report `unhealthy` until workers have been initialized and registred
with the frontend. Once workers have been registered the `health`
endpoint additionally will list registred endpoints and instances.



#### Example Request

```
curl -s localhost:8080/health -q |  jq
```

#### Example Response

Before workers are registered:

``` 
{
  "instances": [],
  "message": "No endpoints available",
  "status": "unhealthy"
}
```

After workers are registered:

```
{
  "endpoints": [
    "dyn://dynamo.backend.generate"
  ],
  "instances": [
    {
      "component": "backend",
      "endpoint": "clear_kv_blocks",
      "instance_id": 7587888160958628000,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_backend.clear_kv_blocks-694d98147d54be25"
      }
    },
    {
      "component": "backend",
      "endpoint": "generate",
      "instance_id": 7587888160958628000,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_backend.generate-694d98147d54be25"
      }
    },
    {
      "component": "backend",
      "endpoint": "load_metrics",
      "instance_id": 7587888160958628000,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_backend.load_metrics-694d98147d54be25"
      }
    }
  ],
  "status": "healthy"
}
```

## Worker Liveness and Health Check

Health checks for components other than the frontend are enabled
selectively based on environment variables. If a health check for a
component is enabled the starting status can be set along with the set
of endpoints that are required to be served before the component is
declared `ready`.

Once all endpoints declared in `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS`
are served the component transitions to a `ready` state until the
component is shutdown.

> **Note**: Both /live and /ready return the same information

### Environment Variables for Enabling Health Checks

| **Environment Variable** | **Description**     | **Example Settings**                             |
| -------------------------| ------------------- | ------------------------------------------------ |
| `DYN_SYSTEM_ENABLED`     | Enables the system status server.                                            | `true`, `false`                           |
| `DYN_SYSTEM_PORT`        | Specifies the port for the system status server.                              | `9090`                                   |
| `DYN_SYSTEM_STARTING_HEALTH_STATUS`     | Sets the initial health status of the system (ready/not ready).                | `ready`, `notready`      |
| `DYN_SYSTEM_HEALTH_PATH`                | Custom path for the health endpoint.                                         | `/custom/health`           |
| `DYN_SYSTEM_LIVE_PATH`                   | Custom path for the liveness endpoint.                                       | `/custom/live`            |
| `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` | Specifies endpoints to check for determining overall system health status.    | `["generate"]`            |

### Example Environment Setting

```
export DYN_SYSTEM_ENABLED="true"
export DYN_SYSTEM_STARTING_HEALTH_STATUS="notready"
export DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS="[\"generate\"]"
export DYN_SYSTEM_PORT=9090
```

#### Example Request

```
curl -s localhost:9000/health -q |  jq
```

#### Example Response
Before endpoints are being served:

```
{
  "endpoints": {
    "generate": "notready"
  },
  "status": "notready",
  "uptime": {
    "nanos": 775381996,
    "secs": 2
  }
}
```

After endpoints are being served:

```
{
  "endpoints": {
    "clear_kv_blocks": "ready",
    "generate": "ready",
    "load_metrics": "ready"
  },
  "status": "ready",
  "uptime": {
    "nanos": 435707697,
    "secs": 55
  }
}
```

## Related Documentation

- [Distributed Runtime Architecture](../architecture/distributed_runtime.md)
- [Dynamo Architecture Overview](../architecture/architecture.md)
- [Backend Guide](backend.md)
