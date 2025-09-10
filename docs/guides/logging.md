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
| `DYN_LOGGING_JSONL`                | Enable JSONL logging format (default: READABLE)                  | `DYN_LOGGING_JSONL=true`                          |
| `DYN_LOG_USE_LOCAL_TZ`             | Use local timezone for logging timestamps (default: UTC)         | `DYN_LOG_USE_LOCAL_TZ=1`                       |
| `DYN_LOG`                          | Log levels per target `<default_level>,<module_path>=<level>,<module_path>=<level>`             | `DYN_LOG=info,dynamo_runtime::system_status_server:trace`  |
| `DYN_LOGGING_CONFIG_PATH`          | Path to custom TOML logging configuration file            | `DYN_LOGGING_CONFIG_PATH=/path/to/config.toml`|


## Available Logging Levels

| **Logging Levels (Least to Most Verbose)** | **Description**                                                                 |
|-------------------------------------------|---------------------------------------------------------------------------------|
| **ERROR**                                 | Critical errors (e.g., unrecoverable failures, resource exhaustion)              |
| **WARN**                                  | Unexpected or degraded situations (e.g., retries, recoverable errors)           |
| **INFO**                                  | Operational information (e.g., startup/shutdown, major events)                 |
| **DEBUG**                                 | General debugging information (e.g., variable values, flow control)            |
| **TRACE**                                 | Very low-level, detailed information (e.g., internal algorithm steps)           |

## Example Readable Format

Environment Setting:

```
export DYN_LOG="info,dynamo_runtime::system_status_server:trace"
export DYN_LOGGING_JSONL="false"
```

Resulting Log format:

```
2025-09-02T15:50:01.770028Z  INFO main.init: VllmWorker for Qwen/Qwen3-0.6B has been initialized
2025-09-02T15:50:01.770195Z  INFO main.init: Reading Events from tcp://127.0.0.1:21555
2025-09-02T15:50:01.770265Z  INFO main.init: Getting engine runtime configuration metadata from vLLM engine...
2025-09-02T15:50:01.770316Z  INFO main.get_engine_cache_info: Cache config values: {'num_gpu_blocks': 24064}
2025-09-02T15:50:01.770358Z  INFO main.get_engine_cache_info: Scheduler config values: {'max_num_seqs': 256, 'max_num_batched_tokens': 2048}
```

## Example JSONL Format

Environment Setting:

```
export DYN_LOG="info,dynamo_runtime::system_status_server:trace"
export DYN_LOGGING_JSONL="true"
```

Resulting Log format:

```json
{"time":"2025-09-02T15:53:31.943377Z","level":"INFO","target":"log","message":"VllmWorker for Qwen/Qwen3-0.6B has been initialized","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":191,"log.target":"main.init"}
{"time":"2025-09-02T15:53:31.943550Z","level":"INFO","target":"log","message":"Reading Events from tcp://127.0.0.1:26771","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":212,"log.target":"main.init"}
{"time":"2025-09-02T15:53:31.943636Z","level":"INFO","target":"log","message":"Getting engine runtime configuration metadata from vLLM engine...","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":220,"log.target":"main.init"}
{"time":"2025-09-02T15:53:31.943701Z","level":"INFO","target":"log","message":"Cache config values: {'num_gpu_blocks': 24064}","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":267,"log.target":"main.get_engine_cache_info"}
{"time":"2025-09-02T15:53:31.943747Z","level":"INFO","target":"log","message":"Scheduler config values: {'max_num_seqs': 256, 'max_num_batched_tokens': 2048}","log.file":"/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py","log.line":268,"log.target":"main.get_engine_cache_info"}
```

## Trace and Span information

When `DYN_LOGGING_JSONL` is enabled with `DYN_LOG` set to greater than or equal to
`info` level trace information is added to all logged spans along with
`SPAN_CREATED` and `SPAN_CLOSED` events.

### Example Request

```
curl -d '{"model": "Qwen/Qwen3-0.6B", "max_completion_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

### Example Logs

```
# Span Created in HTTP Frontend with trace_id and span_id

{"time":"2025-09-02T16:38:06.656503Z","level":"INFO","file":"/workspace/lib/runtime/src/logging.rs","line":248,"target":"dynamo_runtime::logging","message":"SPAN_CREATED","method":"POST","span_id":"6959a1b2d1ee41a5","span_name":"http-request","trace_id":"425ef761ca5b44c795b4c912f1d84b39","uri":"/v1/chat/completions","version":"HTTP/1.1"}

# Span Created in Worker with trace_id and parent_id from frontend and new span_id

{"time":"2025-09-02T16:38:06.666672Z","level":"INFO","file":"/workspace/lib/runtime/src/pipeline/network/ingress/push_endpoint.rs","line":108,"target":"dynamo_runtime::pipeline::network::ingress::push_endpoint","message":"SPAN_CREATED","component":"backend","endpoint":"generate","instance_id":"7587888160958627596","namespace":"dynamo","parent_id":"6959a1b2d1ee41a5","span_id":"b035f33bdd5c4b50","span_name":"handle_payload","trace_id":"425ef761ca5b44c795b4c912f1d84b39"}
{"time":"2025-09-02T16:38:06.685333Z","level":"WARN","target":"log","message":"cudagraph dispatching keys are not initialized. No cudagraph will be used.","log.file":"/opt/vllm/vllm/v1/cudagraph_dispatcher.py","log.line":101,"log.target":"cudagraph_dispatcher.dispatch"}

# Span Closed in Worker with duration, busy, and idle information

{"time":"2025-09-02T16:38:08.787232Z","level":"INFO","file":"/workspace/lib/runtime/src/pipeline/network/ingress/push_endpoint.rs","line":108,"target":"dynamo_runtime::pipeline::network::ingress::push_endpoint","message":"SPAN_CLOSED","component":"backend","endpoint":"generate","instance_id":"7587888160958627596","namespace":"dynamo","parent_id":"6959a1b2d1ee41a5","span_id":"b035f33bdd5c4b50","span_name":"handle_payload","time.busy_us":1090,"time.duration_us":2121090,"time.idle_us":2120000,"trace_id":"425ef761ca5b44c795b4c912f1d84b39"}

# Span Closed in HTTP Frontend with duration, busy and idle information

{"time":"2025-09-02T16:38:08.788268Z","level":"INFO","file":"/workspace/lib/runtime/src/logging.rs","line":248,"target":"dynamo_runtime::logging","message":"SPAN_CLOSED","method":"POST","span_id":"6959a1b2d1ee41a5","span_name":"http-request","time.busy_us":13000,"time.duration_us":2133000,"time.idle_us":2120000,"trace_id":"425ef761ca5b44c795b4c912f1d84b39","uri":"/v1/chat/completions","version":"HTTP/1.1"}
```

### Example Request with User Supplied `x-request-id`

```
curl -d '{"model": "Qwen/Qwen3-0.6B", "max_completion_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' -H 'x-request-id: 8372eac7-5f43-4d76-beca-0a94cfb311d0' http://localhost:8080/v1/chat/completions
```

### Example Logs

```
# Span Created in HTTP Frontend with x-request-id, trace_id, and span_id

{"time":"2025-09-02T17:01:46.306801Z","level":"INFO","file":"/workspace/lib/runtime/src/logging.rs","line":248,"target":"dynamo_runtime::logging","message":"SPAN_CREATED","method":"POST","span_id":"906902a4e74b4264","span_name":"http-request","trace_id":"3924188ea88d40febdfa173afd246a3a","uri":"/v1/chat/completions","version":"HTTP/1.1","x_request_id":"8372eac7-5f43-4d76-beca-0a94cfb311d0"}

# Span Created in Worker with trace_id, parent_id and x_request_id from frontend and new span_id

{"time":"2025-09-02T17:01:46.307484Z","level":"INFO","file":"/workspace/lib/runtime/src/pipeline/network/ingress/push_endpoint.rs","line":108,"target":"dynamo_runtime::pipeline::network::ingress::push_endpoint","message":"SPAN_CREATED","component":"backend","endpoint":"generate","instance_id":"7587888160958627596","namespace":"dynamo","parent_id":"906902a4e74b4264","span_id":"5a732a3721814f5e","span_name":"handle_payload","trace_id":"3924188ea88d40febdfa173afd246a3a","x_request_id":"8372eac7-5f43-4d76-beca-0a94cfb311d0"}

# Span Closed in Worker with duration, busy, and idle information
{"time":"2025-09-02T17:01:47.975228Z","level":"INFO","file":"/workspace/lib/runtime/src/pipeline/network/ingress/push_endpoint.rs","line":108,"target":"dynamo_runtime::pipeline::network::ingress::push_endpoint","message":"SPAN_CLOSED","component":"backend","endpoint":"generate","instance_id":"7587888160958627596","namespace":"dynamo","parent_id":"906902a4e74b4264","span_id":"5a732a3721814f5e","span_name":"handle_payload","time.busy_us":646,"time.duration_us":1670646,"time.idle_us":1670000,"trace_id":"3924188ea88d40febdfa173afd246a3a","x_request_id":"8372eac7-5f43-4d76-beca-0a94cfb311d0"}

# Span Closed in HTTP Frontend with duration, busy and idle information

{"time":"2025-09-02T17:01:47.975616Z","level":"INFO","file":"/workspace/lib/runtime/src/logging.rs","line":248,"target":"dynamo_runtime::logging","message":"SPAN_CLOSED","method":"POST","span_id":"906902a4e74b4264","span_name":"http-request","time.busy_us":2980,"time.duration_us":1672980,"time.idle_us":1670000,"trace_id":"3924188ea88d40febdfa173afd246a3a","uri":"/v1/chat/completions","version":"HTTP/1.1","x_request_id":"8372eac7-5f43-4d76-beca-0a94cfb311d0"}

```

## Related Documentation

- [Distributed Runtime Architecture](../architecture/distributed_runtime.md)
- [Dynamo Architecture Overview](../architecture/architecture.md)
- [Backend Guide](backend.md)
- [Log Aggregation in Kubernetes](dynamo_deploy/logging.md)
