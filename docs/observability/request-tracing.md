---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Replay Tracing
subtitle: Capture live chat and completion traffic for Mooncake replay
---

Request replay tracing records one compact `request_end` row for each supported
Rust OpenAI chat or completion request. It does not require agent context and
does not record prompts, responses, tool calls, session identifiers, or agent
metadata.

Enable the default rotating gzip sink:

```bash
export DYN_REQUEST_TRACE=1
```

This writes `/tmp/dynamo-request-trace.NNNNNN.jsonl.gz`. To choose another
segment prefix:

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_OUTPUT_PATH=/mnt/captures/run-42/request-trace
```

## Configuration

| Variable | Default when enabled | Description |
| --- | --- | --- |
| `DYN_REQUEST_TRACE` | unset | Truthy master switch. |
| `DYN_REQUEST_TRACE_SINKS` | `jsonl_gz` | Comma-separated `jsonl`, `jsonl_gz`, or `stderr`. |
| `DYN_REQUEST_TRACE_OUTPUT_PATH` | `/tmp/dynamo-request-trace` | Literal JSONL path or gzip segment prefix. |
| `DYN_REQUEST_TRACE_CAPACITY` | `1024` | Best-effort in-process broadcast capacity. |
| `DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES` | `1048576` | JSONL or gzip batching threshold. |
| `DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS` | `1000` | Periodic flush interval. |
| `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES` | `268435456` | Gzip roll threshold in uncompressed bytes. |
| `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES` | unset | Optional gzip roll threshold in records. |

The bus and sinks use the same best-effort delivery behavior as agent tracing.
A slow sink can report lag and drop records. Validate captured row counts before
using a trace as a complete workload.

## Record Shape

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "request": {
    "request_id": "dynamo-request-id",
    "request_received_ms": 1777312800000,
    "output_tokens": 16,
    "replay": {
      "trace_block_size": 64,
      "input_length": 128,
      "input_sequence_hashes": [
        14879255164371896291,
        274632075616497421
      ]
    }
  }
}
```

`input_sequence_hashes` are Dynamo's sequence-aware rolling hashes. The
Mooncake converter maps them to compact `hash_ids`; they are not copied
verbatim into the Mooncake output.

For a canceled response stream, `output_tokens` is the final partial OSL
observed after the inner response stream has been dropped.

## Supported Requests

Initial coverage is the Rust OpenAI chat-completions and completions paths.
Each row must represent one replay request, so tracing skips:

- `n > 1`
- `best_of > 1`
- `prompt_embeds`
- Multimodal inputs
- Requests without a tracker or usable KV cache block size

Skipped requests produce a structured warning and no partial replay row.

## Convert And Replay

The existing converter accepts both `dynamo.agent.trace.v1` and
`dynamo.request.trace.v1`:

```bash
cargo run -p dynamo-bench --bin agent_trace_to_mooncake -- \
  --input-path /tmp/dynamo-request-trace.*.jsonl.gz \
  --output-file /tmp/dynamo-request-trace.mooncake.jsonl
```

Do not pass `--agentic` for request traces; they intentionally contain no agent
context. The converter rejects unknown schema versions.

Use the trace block size printed by the converter for both trace parsing and
the mock engine:

```bash
TRACE_BLOCK_SIZE=64
.venv/bin/python -m dynamo.replay /tmp/dynamo-request-trace.mooncake.jsonl \
  --trace-format mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-request-trace.replay-report.json
```

`DYN_REQUEST_TRACE` is independent of `DYN_AGENT_TRACE`. Enabling both emits
separate files while sharing request completion notification and replay-hash
computation.
