---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Forward Pass Metrics Tracing
subtitle: Persist backend forward pass metrics to rotating gzip JSONL files
---

Forward pass metrics (FPM) tracing is an opt-in, best-effort analysis stream.
It captures the finalized FPM payload in each backend worker before Dynamo
publishes that payload to the event plane. Tracing does not change event-plane
publication, routing, or planner startup behavior.

Enable tracing with the default sampled configuration:

```bash
export DYN_FPM_TRACE=1
```

Each worker writes its own files under `/tmp` by default:

```text
/tmp/dynamo-fpm.<producer-id>.000000.jsonl.gz
/tmp/dynamo-fpm.<producer-id>.000001.jsonl.gz
```

`<producer-id>` is the worker's runtime connection ID, sanitized for use in a
file name. Partitioning files by producer prevents workers on a shared volume
from writing to the same gzip segment.

> [!NOTE]
> `/tmp` is convenient but ephemeral. Mount a host directory or persistent
> volume and set `DYN_FPM_OUTPUT_PATH` if traces must survive pod replacement.

## Configuration

| Variable | Default when enabled | Description |
| --- | --- | --- |
| `DYN_FPM_TRACE` | unset | Master switch. Accepts `1`/`0`, `true`/`false`, `on`/`off`, and `yes`/`no`, case-insensitively. |
| `DYN_FPM_OUTPUT_PATH` | `/tmp/dynamo-fpm` | Output prefix. Files are `<prefix>.<producer-id>.<index>.jsonl.gz`. |
| `DYN_FPM_MODE` | `sampled` | `sampled` keeps the latest changed record per worker and data-parallel rank; `full` captures every valid payload. |
| `DYN_FPM_SAMPLE_INTERVAL_MS` | `5000` | Positive sampling interval. Validated in both modes and used only in `sampled` mode. |
| `DYN_FPM_JSONL_GZ_ROLL_BYTES` | `268435456` | Positive uncompressed-byte threshold. Dynamo rolls before the next JSONL row would exceed it. |
| `DYN_FPM_MAX_SEGMENTS` | `4` | Positive number of segments retained per producer, including the active segment. |

The other variables do not enable tracing by themselves. An invalid value or
an unwritable output path disables tracing only for that worker and produces a
warning. Inference and normal FPM publication continue.

`DYN_FPM_TRACE=1` also enables the existing FPM generation path for vLLM and
SGLang. For vLLM, an explicit `DYN_FORWARDPASS_METRIC_PORT` wins; otherwise
Dynamo uses port `20380`. SGLang uses its existing per-worker IPC endpoint. If
vLLM has an incompatible custom scheduler, Dynamo warns and continues serving
without trace data from that scheduler.

SGLang snapshot mode does not currently produce FPM trace data. Its engine is
created before the Dynamo endpoint identity and IPC path are available, so the
backend disables FPM and warns rather than subscribing to the wrong publisher.

`DYN_FPM_BENCHMARK_OUTPUT_PATH` remains a separate benchmark-only output and
is not used for live tracing.

## Capture Modes

In `sampled` mode, Dynamo retains only the newest pending payload for each
`(namespace, component, worker_id, dp_rank)` key. On each monotonic sampling
interval, it writes keys whose `counter_id` changed. It does not repeat an
unchanged counter. Pending values are flushed during graceful shutdown.

In `full` mode, Dynamo writes every valid payload that reaches the producer,
including idle heartbeats. This mode can generate substantially more data.

Both modes are best effort. Producer enqueueing is nonblocking, and a bounded
in-process queue drops trace records instead of delaying inference. The worker
logs structured dropped-record counts when a trace consumer falls behind.

## Record Shape

Each line uses the shared gzip JSONL envelope. The `event` object is the FPM
trace record; `observed_at_unix_ms` is its absolute observation time. The
outer `timestamp` is milliseconds elapsed since that writer started.

```json
{
  "timestamp": 1250,
  "event": {
    "schema": "dynamo.fpm.trace.v1",
    "source": {
      "namespace": "default",
      "component": "backend",
      "producer_id": "4192"
    },
    "capture_mode": "sampled",
    "observed_at_unix_ms": 1782777601250,
    "fpm": {
      "version": 1,
      "worker_id": "4192",
      "dp_rank": 0,
      "counter_id": 42,
      "wall_time": 0.025,
      "scheduled_requests": {
        "num_prefill_requests": 2,
        "sum_prefill_tokens": 256,
        "var_prefill_length": 100.0,
        "sum_prefill_kv_tokens": 64,
        "num_decode_requests": 3,
        "sum_decode_kv_tokens": 1024,
        "var_decode_kv_tokens": 50.0
      },
      "queued_requests": {
        "num_prefill_requests": 1,
        "sum_prefill_tokens": 128,
        "var_prefill_length": 0.0,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
        "var_decode_kv_tokens": 0.0
      }
    }
  }
}
```

The nested `fpm` object is the complete canonical payload. Use its
`worker_id`, `dp_rank`, and `counter_id` together when checking continuity.
A gap in `counter_id` can result from sampling, local queue pressure, an
upstream vLLM or SGLang ZMQ drop, a crash, or node loss. The trace is not a
durable event plane.

## Read and Size Traces

Decompress all segments for one output prefix in index order:

```bash
gzip -cd /mnt/logs/dynamo-fpm.4192.*.jsonl.gz | jq -c '.event'
```

Replace `4192` with the producer ID in the file name.

The roll threshold counts uncompressed JSONL bytes, while disk usage is the
compressed gzip size. Measure representative records for capacity planning:

```bash
gzip -cd /mnt/logs/dynamo-fpm.4192.*.jsonl.gz | wc -c
gzip -cd /mnt/logs/dynamo-fpm.4192.*.jsonl.gz | wc -l
```

Estimate uncompressed bytes per day as:

```text
average JSONL row bytes * records per second * 86400
```

At the default five-second sampling interval, one continuously changing rank
writes about 17,280 periodic rows per day, plus a possible graceful-shutdown
flush. For example, 600-byte rows are about 10.4 MB per rank per day before
compression. In `full` mode, the rate follows the backend's forward passes:
the same 600-byte row at 10 forward passes per second is about 518 MB per rank
per day before compression.

Rotation starts a new gzip file before adding a row that would cross the
configured threshold. A single oversized row is written intact to an otherwise
empty segment. After a new segment is written, Dynamo removes only the oldest
files that exactly match that producer's prefix. On restart, the next index is
one greater than the highest matching existing index, even if there are gaps.

The segment limit applies independently to each producer. A shared persistent
volume can therefore contain up to `DYN_FPM_MAX_SEGMENTS` files for every
currently or previously used producer ID. Use an external lifecycle policy to
remove stale producer sets.

## Kubernetes Storage

Set the output prefix inside a mounted volume. Existing Dynamo environment and
volume-mount configuration is sufficient; no Dynamo Operator API change is
required.

```yaml
env:
  - name: DYN_FPM_TRACE
    value: "1"
  - name: DYN_FPM_OUTPUT_PATH
    value: /var/log/dynamo/fpm
volumeMounts:
  - name: fpm-traces
    mountPath: /var/log/dynamo
volumes:
  - name: fpm-traces
    persistentVolumeClaim:
      claimName: dynamo-fpm-traces
```

Graceful shutdown flushes records already accepted by the trace pipeline.
`SIGKILL`, node loss, full queues, and upstream transport loss can still lose
records. Treat the files as operational telemetry for analysis, not as input
for planner warm-start or replay.
