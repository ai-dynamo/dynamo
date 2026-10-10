---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Forward Pass Metrics Trace Reference
subtitle: Configuration, topology support, file rotation, and the dynamo.fpm.trace.v1 schema
---

Forward Pass Metrics (FPM) tracing persists finalized FPM payloads immediately before the Rust
publication path sends them to the event plane. Persistence does not replace or reroute event-plane
publication.

## Configuration

| Variable | Default when enabled | Description |
| --- | --- | --- |
| `DYN_FPM_TRACE` | unset | Environment form of `--fpm-trace` and `--no-fpm-trace`. Accepts `1`/`0`, `true`/`false`, `on`/`off`, and `yes`/`no`, case-insensitively. |
| `DYN_FPM_OUTPUT_PATH` | `/tmp/dynamo-fpm` | Output prefix. Files use `<prefix>.<producer-id>.<index>.jsonl.gz`. |
| `DYN_FPM_MODE` | `sampled` | `sampled` keeps the latest changed record per worker and data-parallel rank; `full` captures every valid payload. |
| `DYN_FPM_SAMPLE_INTERVAL_MS` | `5000` | Positive sampling interval. Validated in both modes and used only in `sampled` mode. |
| `DYN_FPM_JSONL_GZ_ROLL_BYTES` | `268435456` | Positive uncompressed-byte threshold. Dynamo rolls before the next JSONL row would exceed it. |
| `DYN_FPM_MAX_SEGMENTS` | `4` | Positive number of segments retained per producer, including the active segment. |

The configuration variables do not enable tracing by themselves. An invalid value or unwritable
output path disables tracing only for that worker and emits a warning. Inference and normal FPM
publication continue.

`DYN_FORWARDPASS_METRIC_PORT` remains a separate legacy backend-generation opt-in and takes
precedence when set. It cannot add a missing Dynamo relay to an unsupported topology.
`DYN_FPM_BENCHMARK_OUTPUT_PATH` is benchmark-only and is not used for live tracing.

## Supported Topologies

| Backend | Worker topology | Trace activation support |
| --- | --- | --- |
| vLLM (`python -m dynamo.vllm`) | Aggregated, prefill, or decode, including native multimodal workers | Supported |
| vLLM (`python -m dynamo.vllm`) | Embedding, multimodal encode, or headless | Not supported; Dynamo warns and does not inject the FPM scheduler |
| SGLang (`python -m dynamo.sglang`) | Aggregated, prefill, decode, or LLM diffusion, including `--enable-multimodal` without a dedicated encoder | Supported |
| SGLang (`python -m dynamo.sglang`) | Embedding or dedicated multimodal encoder | Not supported; Dynamo warns and does not auto-enable FPM |
| SGLang (`python -m dynamo.sglang`) | Image diffusion, video generation, or snapshot mode | Not supported |
| TensorRT-LLM (`python -m dynamo.trtllm`) | Aggregated, prefill, or decode | Supported; also enables the engine's `enable_iter_perf_stats`. `--publish-metrics` enables FPM too, since it turns the same statistics on |
| Mocker | Direct-publisher path | Persistence supported; forward-pass metrics are always published |

For vLLM, an explicit `DYN_FORWARDPASS_METRIC_PORT` wins; otherwise trace activation uses port
`20380`. SGLang uses its existing per-worker IPC endpoint. TensorRT-LLM publishes in-process, so
the port value is unused, but setting the variable still opts the worker in.

## Capture Modes

In `sampled` mode, Dynamo retains the newest pending payload for each
`(namespace, component, worker_id, dp_rank)` key. At each monotonic interval it writes keys whose
`counter_id` changed. It does not repeat an unchanged counter, and it flushes pending values during
graceful shutdown.

In `full` mode, Dynamo writes every valid payload, including idle heartbeats. Each writer flushes a
non-empty batch every second and can flush sooner when its 1 MiB buffer fills. Producer enqueueing is
nonblocking in both modes; a bounded queue drops trace records instead of delaying inference.

## File and Retention Semantics

The default files are:

```text
/tmp/dynamo-fpm.<producer-id>.000000.jsonl.gz
/tmp/dynamo-fpm.<producer-id>.000001.jsonl.gz
```

The producer ID is the sanitized runtime connection ID. Rotation counts uncompressed JSONL bytes.
A single oversized row is written intact to an otherwise empty segment. After rolling, Dynamo removes
only the oldest files that exactly match that producer's prefix. On restart, the next index is one
greater than the highest matching index.

`DYN_FPM_MAX_SEGMENTS` applies independently to every producer, including producer IDs left by old
worker instances.

## Capacity Planning

The roll threshold counts uncompressed JSONL bytes; disk usage is the compressed gzip size.
Estimate uncompressed daily volume with:

```text
average JSONL row bytes * records per second * 86400
```

At the default five-second interval, one continuously changing rank writes about 17,280 periodic
rows per day, plus a possible shutdown flush. A 600-byte row is about 10.4 MB per rank per day before
compression. In `full` mode, the same row at 10 forward passes per second is about 518 MB per rank
per day.

## Record Schema

Each line uses the shared gzip JSONL envelope:

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
        "sum_prefill_kv_tokens": 64,
        "num_decode_requests": 3,
        "sum_decode_kv_tokens": 1024
      },
      "queued_requests": {
        "num_prefill_requests": 1,
        "sum_prefill_tokens": 128,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0
      }
    }
  }
}
```

`observed_at_unix_ms` is the absolute observation time. The outer `timestamp` is milliseconds since
the writer started. The nested `fpm` object is the canonical payload. A `counter_id` gap can result
from sampling, local queue pressure, an upstream ZMQ drop, a crash, or node loss.

## Consuming the Raw FPM Stream

`_FpmPublisherThread` publishes each `ForwardPassMetrics` message on a ZMQ `PUB` socket as a three-frame multipart (`self._pub.send_multipart((topic, seq_bytes, payload), flags=zmq.NOBLOCK)`) with an empty topic frame (`topic = b""`), an 8-byte big-endian sequence number (`seq_bytes = seq.to_bytes(8, "big")`), and the payload last. The payload is MessagePack, not the JSON used by the trace schema above: `msgspec.msgpack.Encoder().encode(...)` on the same `ForwardPassMetrics` struct from `dynamo.common.forward_pass_metrics`. Subscribe with an empty topic filter to receive every message: ZMQ `SUB` topic matching is a byte-prefix match against the first frame, and an empty filter matches unconditionally.

```python
import zmq
from dynamo.common.forward_pass_metrics import decode

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:20380")  # base port + dp_rank
sock.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    topic, seq_bytes, payload = sock.recv_multipart()
    seq = int.from_bytes(seq_bytes, "big")
    metrics = decode(payload)  # ForwardPassMetrics, or None on an unrecognized schema version
```

Before encoding, the publisher stamps `counter_id` with its own send-time sequence number (`metrics = msgspec.structs.replace(metrics, counter_id=seq)`), the same value carried in the sequence frame: it is a per-publisher sequence, not a group step id, stamped from one `itertools.count()` per `_FpmPublisherThread`, and it drifts across data-parallel ranks because each rank's publisher counts from 0 independently. An idle publisher emits a heartbeat once per second, which also advances the sequence. Sends are non-blocking and are dropped rather than queued when the ZMQ send buffer is full, so this raw stream's `counter_id` can show the same kind of gap already described for the trace schema above. The self-benchmark (`--benchmark-mode`) is a separate path: while it runs, this publisher is paused and forward-pass metrics go into the benchmark artifact instead of the socket, with `counter_id` set to the benchmark point id rather than a publisher sequence; see [FPM self-benchmark collection](environment-variables.mdx#fpm-self-benchmark-collection) and Engine Provenance in the Self-Benchmark Artifact below.

Each data-parallel rank binds `DYN_FORWARDPASS_METRIC_PORT + dp_rank`, defaulting to base port `20380`. Under attention data parallelism (`dp_size > 1`), the self-benchmark synchronizer additionally binds `DYN_FORWARDPASS_METRIC_PORT + dp_size`, but only rank 0 (the data-parallel master) binds it, as a ROUTER socket that every other rank's DEALER socket connects to as a client. The occupied block (the base port alone at `dp_size <= 1`, or through `base + dp_size` once the synchronizer also binds) must sit outside the kernel's ephemeral port range, which Dynamo reads from `/proc/sys/net/ipv4/ip_local_port_range` (commonly starting at `32768`). A port inside that range can be claimed by an unrelated outbound connection before the publisher binds it, so the bind then fails with `Address already in use` intermittently and only under load. Dynamo checks this once per engine-core process and logs one warning when the block overlaps the range, staying silent when the proc file cannot be read, such as on a non-Linux host: `FPM ports <first>-<last> overlap this host's ephemeral port range <low>-<high>; an outbound connection can take one of them first and the publisher bind then fails intermittently. Set DYN_FORWARDPASS_METRIC_PORT to a base below <low>.`

## Measurement Protocol in the Self-Benchmark Artifact

Self-benchmark artifacts (`--benchmark-mode`) record a top-level `measurement_protocol` block and a `benchmark_measurement` block inside each retained FPM. The artifact's `schema_version` remains `2`; both blocks have their own `schema_version: 1`. These fields describe benchmark input construction, timing reduction, and observed execution evidence. They do not appear in live-serving traces.

Prompt construction combines `DYN_BENCH_CONTENT_SEED` (default `"0"`) with the point's coordinates, data-parallel rank, and request slot. Selecting a smaller grid or changing benchmark point order does not change that content identity. A decode point recorded with `giant_fake_off_by_batch` in its `sample_reasons` keeps the prompts built for its declared coordinate, whose `total_kv_read_tokens` exceeds the recorded value by `batch_size`; its `benchmark_measurement.point_key` names the recorded coordinate. Different data-parallel ranks intentionally use different content. Compare the same rank's prompt hashes across runs rather than requiring every rank to have the same hash. Pin the model, tokenizer, dataset, content mode, and pool tag when comparing measurements; see [FPM self-benchmark collection](environment-variables.mdx#fpm-self-benchmark-collection).

The shared protocol contains:

| Field | Meaning |
|---|---|
| `content_identity` | `"coordinate_rank_slot_v1"`, the input-construction policy |
| `content_seed`, `synthetic_content`, `synthetic_pool_tag` | Frozen content seed; recorded content mode and pool tag |
| `prompt_hash_encoding` | `"uint32_le"`; prompt token ids are encoded as unsigned 32-bit little-endian integers before SHA-256 hashing |
| `independent_repetitions` | `1`; one engine launch supplies one observation per admitted point |
| `timing_metric` | `"scheduler_wall_time"`; the existing scheduler timing boundary is unchanged, not replaced with CUDA-event timing |
| `execution_evidence` | Versioned contract for per-sample observations and warmup records; identifies their field names, process-local clock and forward indices, graph-statistics source, and validation scope |
| `input_evidence_scope` | `"injected_prompt_token_ids"`; hashes cover the prompt ids available at request injection |
| `unobserved` | Lists inputs or context not established by this evidence: sampled continuation tokens, KV-cache tensors, recurrent-state tensors, decode real-KV chain warmup history, equivalent execution history, and equivalent graph, kernel, or cache warming |
| `preparation` | Warm-up iteration count, prefill real-seed and decode real-KV settings, and giant-KV threshold and repeat count |

Each retained FPM's `benchmark_measurement` contains:

| Field | Meaning |
|---|---|
| `point_key`, `dp_rank` | Coordinate identity and the rank that produced this observation; the point key is independent of the grid digest and benchmark point id |
| `prompts.status` | `"recorded"` when injected prompt ids were captured, otherwise `"unavailable"`; missing evidence is not a match |
| `prompts.sha256`, `prompts.requests` | Combined prompt digest and the request slots, lengths, and individual prompt digests |
| `preparation` | Actual grid digest, number of completed points before this point, KV-seeding regime, and `warmup_records_before`, the number of prior warmup records, including failed attempts |
| `expected_internal_samples` | Number of scheduler steps expected for this point, including an admission step when the path uses one |
| `raw_fpms` | Individual FPMs before reduction, in collection order, including discarded admission measurements and slow samples; each carries its own `benchmark_sample` execution evidence |
| `estimate.method`, `estimate.raw_sample_indices` | Existing reduction (`single_step`, `last_step`, or `adjacent_upper_median`) and the raw samples used for the retained estimate |

The merger requires identical shared protocols in all contributing rank files. It rejects a mixture of legacy artifacts and artifacts with a protocol, and requires point/rank evidence for every member of a synchronized iteration group. Per-rank prompt hashes, raw timings, and execution observations are preserved unchanged; their values need not agree across ranks. Legacy runs without these metadata blocks remain readable, but the merger does not invent evidence for them.

### Per-Sample Timing and Graph Dispatch

Each `benchmark_measurement.raw_fpms` entry identifies one raw scheduler observation. Its `benchmark_sample` block has this shape:

```json
{
  "sample_index": 0,
  "forward_index": 12,
  "timing": {
    "basis": "schedule_to_output",
    "start_monotonic": 150.0,
    "end_monotonic": 150.062
  },
  "cudagraph": {
    "status": "observed",
    "runtime_mode": "PIECEWISE",
    "num_unpadded_tokens": 80,
    "num_padded_tokens": 80,
    "num_paddings": 0
  }
}
```

`sample_index` is the zero-based position in `raw_fpms`. `forward_index` counts nonempty forward outputs during this rank's benchmark, including preparation forwards. Both indices are local to that rank's process; they do not synchronize ranks.

| Timing basis | Interval |
|---|---|
| `schedule_to_output` | Completion of scheduling to arrival of the corresponding model-runner output |
| `inter_output` | Previous model-runner output arrival to current output arrival; used for steady decode samples |

The timestamps use monotonic-clock seconds in the local process, so timestamps from different ranks cannot be subtracted or ordered against each other. An unavailable start timestamp is `null`. `wall_time` retains the native scheduler measurement and existing reduction; these observations do not replace it with GPU execution time. For a reduced estimate, consult `estimate.raw_sample_indices` and the selected raw observations. The retained estimate has no `benchmark_sample`, because a median can combine multiple executions.

In benchmark mode, Dynamo enables vLLM's existing `cudagraph_metrics` option when the runtime exposes it and it is not already on. Before the worker registers for serving, Dynamo turns an option it enabled back off in the model workers and for vLLM's stat loggers, and discards the dispatch statistics the loggers collected during the benchmark, so serving neither records nor logs them. An option you enabled yourself stays on. To reach the model workers, Dynamo sets `--worker-extension-cls` to its own extension class in benchmark mode unless you set one. With your own class, Dynamo skips the two calls that need its extension class, the `cudagraph_metrics` reset and the [worker probe](#worker-probe-sidecar), and each skipped call logs one warning that names your class. The model workers keep recording the statistics, but nothing collects them. The per-sample fields come from `ModelRunnerOutput.cudagraph_stats`, without adding GPU timing or synchronization. `runtime_mode` records `NONE`, `PIECEWISE`, or `FULL`; the token counts describe that dispatch's padding. Runtimes or model runners without complete statistics emit `status: "unavailable"` and `null` values. Missing evidence never means `NONE`.

Observed dispatch is separate from the point's `expected_cudagraph_mode` and `expected_capture_size`, and from the engine's resolved graph configuration. A mode and padded token count do not identify the complete graph descriptor, a capture-versus-replay event, or the attention kernel that ran. Equal values alone do not establish equivalent execution.

### Warmup Completion Evidence

Each rank artifact's `warmup_evidence` contains `status` (`recorded` or `unavailable`) and an ordered `records` list. The list covers global warmup, discarded eager-shape replicas, real-prefix seeding, and real-prefix same-shape preparation. An empty recorded list means none of these attempts was recorded. Decode KV-chain preparation retains its existing `kvwarm` metadata; its individual warmup attempts are not recorded here.

| Record field | Meaning |
|---|---|
| `kind`, `requested_shape` | `global`, `eager_shape`, `real_prefix_seed`, or `real_prefix_shape`, with the requested prompt lengths or benchmark point |
| `status` | `running`, `completed`, or `failed`; completion records that the attempt finished, not that a particular graph or kernel was warmed |
| `forward_index_start`, `forward_index_end` | Start-inclusive, end-exclusive range in this rank's forward indices; an unobserved start or unfinished end can be `null` |
| `completed_points_before` | Number of retained benchmark points before this attempt |
| `observed_forward_count` | Number of nonempty forward outputs observed during the attempt |
| `first_scheduled_requests`, `last_scheduled_requests` | First and last observed scheduler shapes, or `null` when none was observed; intermediate shapes are not retained |
| `validation` | `status` (`not_performed`, `passed`, or `failed`) and optional failure `reason`; only existing eager-replica shape validation produces a pass or failure |
| `validation.scope`, `validation.failed_dp_rank` | For eager-replica shape validation, `attention_dp_group` and the failing DP rank, or `null` when none failed; another rank's shape mismatch can fail the group |

The record count grows with warmup attempts; each attempt retains at most two scheduler shapes. The merged artifact keeps these lists under `rank_warmup_evidence`, keyed by DP rank. A rank whose artifact was not loaded has explicit unavailable evidence. Compare each raw sample's `forward_index` with the same rank's warmup ranges to establish ordering. A completed request without shape validation remains `validation.status: "not_performed"`; neither completion nor a shape-validation pass proves matching cache contents, graph preparation, or kernel history.

<Warning>
Matching prompt hashes establishes only the captured token content. It does not establish matching sampled continuation tokens, initialized cache tensors, memory allocation, graph preparation, or execution history. Full-grid and subset measurements still require a controlled comparison before they can be treated as equivalent repetitions. Adjacent decode steps share preparation and are not independent repetitions; use separate benchmark launches to measure run-to-run variability and retain all attempts.
</Warning>

## Engine Provenance in the Self-Benchmark Artifact

A self-benchmark run (`--benchmark-mode`) writes one JSON artifact per data-parallel rank plus one merged artifact, both separate from the `dynamo.fpm.trace.v1` records above. Each carries an optional top-level `engine` block, captured when the instrumented scheduler starts. Dynamo does not rewrite these artifacts after the benchmark: the post-run worker probe records the resolved worker configuration in a separate [sidecar file](#worker-probe-sidecar), and per-forward dispatch belongs to the raw samples described above. The artifact's `schema_version` stays `2`. Artifacts whose capture never ran, including older artifacts, omit `engine`.

The following example shows selected fields from a rank artifact. The `null` and `"pending_worker_probe"` values are placeholders that stay in the artifact; the worker probe's results are in the sidecar.

```json
{
  "engine": {
    "attention": {
      "backend_requested": null,
      "backend_per_kind": {},
      "mla_prefill_backend_requested": null,
      "flash_attn_version": 3,
      "use_trtllm_attention": false,
      "indexer_kv_dtype": "fp8",
      "indexer_kv_dtype_configured": "auto",
      "backend_resolved": null,
      "mla_prefill_backend_resolved": null,
      "resolution": "pending_worker_probe",
      "indexer": {
        "index_topk": 2048,
        "index_n_heads": 1,
        "index_head_dim": 128
      }
    },
    "kv_cache": {
      "cache_dtype": "auto",
      "block_size": 64,
      "enable_prefix_caching": true,
      "groups": [
        {
          "type": "FullAttentionSpec",
          "dtype": "bfloat16",
          "head_size": 128,
          "num_kv_heads": 8,
          "block_size": 64,
          "kv_quant_mode": "NONE"
        }
      ],
      "kv_cache_layout": "LBHNC"
    },
    "quantization": {
      "method": "fp8",
      "checkpoint_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic"
      },
      "quant_config_class": "Fp8Config"
    },
    "scheduler": {
      "max_num_batched_tokens": 8192,
      "max_num_seqs": 256,
      "async_scheduling": true,
      "enable_chunked_prefill": true,
      "long_prefill_token_threshold": 0
    },
    "model": {
      "dtype": "bfloat16",
      "model_type": "deepseek_v3",
      "architectures": ["DeepseekV3ForCausalLM"],
      "model": "deepseek-ai/DeepSeek-V3"
    },
    "parallel": {
      "data_parallel_size": 4,
      "data_parallel_rank": 0,
      "tensor_parallel_size": 1,
      "pipeline_parallel_size": 1,
      "prefill_context_parallel_size": 1,
      "enable_expert_parallel": true,
      "enable_eplb": false,
      "all2all_backend": "deepep_high_throughput",
      "eplb_config": {
        "window_size": 1000,
        "step_interval": 3000,
        "num_redundant_experts": 0,
        "log_balancedness": false,
        "log_balancedness_interval": 1,
        "use_async": true,
        "policy": "default",
        "communicator": null
      }
    },
    "speculative": null,
    "graph": {
      "cudagraph_mode": "FULL",
      "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
      "resolution": "pre_resolution"
    },
    "versions": {
      "vllm": "0.29.0",
      "vllm_build_commit": "abcdef0",
      "dynamo": "0.9.0",
      "python": "3.12.3"
    },
    "resolved": null,
    "resolution": "pending_worker_probe"
  }
}
```

- `attention` records what the engine-core process asked for. vLLM resolves the real attention backend inside the model workers and never writes it back into the engine-core config, so `backend_requested` and `mla_prefill_backend_requested` stay `null` whenever selection was automatic, and `backend_per_kind` stays `{}` unless a per-attention-kind override was set. `indexer` is present only for sparse-indexer models; `indexer_kv_dtype` is the resolved value and `indexer_kv_dtype_configured` is the raw configured one (`"auto"` resolves to the DeepSeek indexer default, `"fp8"`). `backend_resolved`, `mla_prefill_backend_resolved`, and this sub-block's own `resolution` are `null`, `null`, and `"pending_worker_probe"` in every artifact; the worker probe described below records their resolved values in the sidecar.
- `kv_cache.groups` has one entry per KV-cache group the workers reported (`type` is the spec class name, plus `dtype`, `head_size`, `num_kv_heads`, `block_size`, and `kv_quant_mode` when set). `num_gpu_blocks` is deliberately not here: it is decided per engine-core process after profiling with no cross-rank reduction, so it is rank-varying data rather than engine identity, and it is already recorded per rank at the artifact's `limits.num_gpu_blocks`. `kv_cache_layout` is present starting vLLM 0.29.0 and simply absent on 0.28.0, which keeps it in a worker-process global the engine-core process cannot read. When vLLM reports a group as a `UniformTypeKVCacheSpecs` wrapper (several same-kind specs that differ only in per-layer attributes), the group's own `dtype`, `head_size`, and `num_kv_heads` stay `null`, and the entry additionally carries a `specs` list with one entry of this same shape per wrapped spec.
- `quantization.checkpoint_config` is the normalized quantization dict the checkpoint itself declared, not the user-facing online-quantization spec (`model_config.quantization_config`, deliberately not read here). `method` is the kernel family vLLM resolved, and `quant_config_class` is that resolved config object's class name.
- `scheduler` and `model` are read directly from the engine's own `scheduler_config` and `model_config` at startup and need no further resolution. `model.max_model_len` is deliberately not here: vLLM's `--max-model-len -1` auto-fit can resolve it differently per rank depending on each rank's free GPU memory, so it is rank-varying data rather than engine identity, and it is already recorded per rank at the artifact's `limits.max_model_len`.
- `parallel.data_parallel_rank` differs per rank and is excluded from the startup identity comparison. The post-run probe results described below are recorded after the merge, in the sidecar, so they are not startup identity evidence.
- `speculative` is `null` unless speculative decoding is configured, in which case it is a four-field summary: `method`, `model`, `num_speculative_tokens`, `draft_tensor_parallel_size`.
- `graph` freezes what the engine-core process believed about CUDA graphs before the model worker re-resolved them, tagged `"resolution": "pre_resolution"`. Under a multiprocess executor the engine-core process never sees that rewrite, so this sub-block is never updated after capture. Post-resolution values belong to the individual worker-probe replies in the sidecar, not this startup block.
- `versions` is a best-effort snapshot (`vllm`, `vllm_build_commit`, `dynamo`, `python`); a single lookup failure leaves just that one key `null` rather than failing the whole block.
- `resolved` and `resolution` are `null` and `"pending_worker_probe"` in every artifact whose engine capture completed; the [worker probe sidecar](#worker-probe-sidecar) holds their post-run values.
- `capture_error` appears only when the capture function itself raised partway through; it sits alongside whatever sub-blocks were already built, and a capture failure never costs the run its measurements.
- The merged artifact keeps one copy of this block. Every contributing rank's startup block must be identical except `parallel.data_parallel_rank`. The comparison excludes post-run fields: `resolved`, `resolved_scope`, `resolution`, `worker_probe`, and resolved fields under `attention`. A disagreement in the compared startup fields fails the merge. The merged copy's `parallel.data_parallel_rank` is set to `null`, since the merged document describes every rank at once. A rank whose own capture failed or is missing never blocks the merge; each such rank is instead excluded from the cross-rank identity check and listed, by rank number, in a `capture_errors` map added to the merged block, valued with that rank's own capture error or the literal string `missing engine block`.

### Worker Probe Sidecar

After collection, Dynamo calls the `fpm_engine_probe` method of its worker extension class in every model worker, with one `collective_rpc` under a 30-second timeout. In benchmark mode Dynamo sets `--worker-extension-cls` to that class unless you set one; with your own class Dynamo skips the call and the probe results record `probe_skipped: ...`. Dynamo writes the replies, or the reason there are none, to one JSON file next to the merged artifact, named after it with a `_worker_probe` suffix: `benchmark_results_merged.json` gets `benchmark_results_merged_worker_probe.json`. The merged artifact's top-level `worker_probe_path`, next to `merged_output_path`, names that file. It is written only when the merged artifact has an `engine` block, and rank artifacts do not carry it. The probe neither reads nor rewrites the rank and merged artifacts, which keep their startup `engine` block. The sidecar is written atomically after the merged artifact. When it is missing, no post-run worker evidence was recorded, for example because the probe could not write it. Dynamo deletes a sidecar left by an earlier run before it writes the merged artifact, and the sidecar's `run_id` matches the merged artifact's. The `get_perf_metrics` endpoint serves the merged document with the merged probe result applied to its `engine` block.

The following example shows selected fields from the sidecar of a two-rank run whose RPC returned only DP rank 0's reply. Each probe result's `worker_probe` is described in [Worker Probe Coverage](#worker-probe-coverage).

```json
{
  "schema": "dynamo.fpm.benchmark_worker_probe",
  "schema_version": 1,
  "run_id": "3f6a0c1e9b2d4e7f8a5b6c7d8e9f0a1b",
  "merged_output_path": "/tmp/benchmark_results_merged.json",
  "rank_files": [
    "/tmp/benchmark_results.json",
    "/tmp/benchmark_results_dp1.json"
  ],
  "merged": {
    "resolved": {
      "tp_rank": 0,
      "pp_rank": 0,
      "pcp_rank": 0,
      "worker_rank": 0,
      "dp_rank": 0,
      "attention_backends": {
        "model.layers.0.self_attn": "FLASH_ATTN_MLA"
      },
      "mla_prefill_backend": "FlashAttnPrefillBackend",
      "cudagraph_mode_resolved": "FULL",
      "cudagraph_capture_sizes_resolved": [1, 2, 4, 8, 16, 32],
      "capture_errors": {}
    },
    "resolved_scope": "representative_worker",
    "resolution": "worker_probe_partial",
    "attention": {
      "backend_resolved": "FLASH_ATTN_MLA",
      "mla_prefill_backend_resolved": "FlashAttnPrefillBackend",
      "resolution": "worker_probe_partial"
    }
  },
  "ranks": {
    "0": {
      "rank_file": "/tmp/benchmark_results.json",
      "resolved_scope": "representative_worker",
      "resolution": "worker_probe"
    },
    "1": {
      "rank_file": "/tmp/benchmark_results_dp1.json",
      "resolved": null,
      "resolved_scope": null,
      "resolution": "worker_probe_unobserved"
    }
  }
}
```

| Field | Meaning |
|---|---|
| `schema`, `schema_version` | `dynamo.fpm.benchmark_worker_probe` and `1` |
| `run_id`, `merged_output_path`, `rank_files` | The run and the artifacts the probe describes |
| `merged` | The probe result for every worker of the merged artifact |
| `ranks` | One probe result per contributing rank, keyed by DP rank, each with that rank's `rank_file` |

Each probe result has these fields:

| Field | Meaning |
|---|---|
| `resolved`, `resolved_scope` | One valid, rank-labelled worker reply kept as a representative, with `resolved_scope: "representative_worker"`, or `null` for both when no reply is in scope |
| `resolution` | The probe outcome, described in [Worker Probe Coverage](#worker-probe-coverage) |
| `worker_probe` | Every returned reply and its coverage, described in [Worker Probe Coverage](#worker-probe-coverage) |
| `attention` | `backend_resolved`, `mla_prefill_backend_resolved`, and `resolution` summaries; present when the `engine` block has an `attention` sub-block |

A rank's result uses only replies matching its own DP rank; its `resolved` stays `null` if that rank was not observed. The TP, PP, and PCP sizes a rank's result expects come from the merged `engine` block, which every DP rank covered by the merged artifact shares; a reply whose worker ranks contradict those sizes gets an `issues` entry and is not counted as observed. The merged result can use one observed worker as a representative, but this does not establish that every worker agrees. Use the coverage fields below to interpret the observation.

### Worker Probe Coverage

Each probe result's `worker_probe` preserves every returned reply, including duplicates and malformed replies, rather than selecting only the first result. A worker reply includes DP, tensor-parallel (TP), pipeline-parallel (PP), prefill-context-parallel (PCP), and worker ranks; per-layer `attention_backends`; `mla_prefill_backend`; resolved graph mode and capture sizes; and any per-layer `capture_errors`.

| Field | Meaning |
|---|---|
| `schema_version`, `scope` | `1` and `post_collection_collective_rpc` |
| `responses` | Entries containing the reply's zero-based `rpc_index`, its `response`, and validation `issues`; a non-JSON reply is represented by its type and textual representation |
| `coverage.scope` | `artifact_workers`; coverage is evaluated against the workers of the artifact the probe result describes: every DP rank for `merged`, one DP rank for an entry in `ranks` |
| `coverage.expected_dp_ranks` | Expected DP ranks, or `null` when unknown |
| `coverage.expected_workers_per_dp` | TP size multiplied by PCP size and PP size, or `null` when any size is unknown |
| `coverage.observed_workers` | Valid, unique worker identities in that scope |
| `coverage.missing_dp_ranks` | Expected DP ranks without a usable reply, or `null` when the expected ranks are unknown |
| `coverage.complete` | True only when the expected topology is known and fully observed, without malformed, duplicate, or failed capture evidence |
| `disagreements` | Observed configuration fields that disagree; attention backend comparisons use layers observed by more than one worker, because PP stages own different layers |
| `error` | Probe failure or skip reason, or `null` |

The engine client's returned reply set can cover fewer DP engines than the RPC broadcasts to. Coverage follows the replies actually returned. For example, a four-rank run with TP, PCP, and PP sizes all set to 1 can return only DP rank 0. The merged result then reports missing DP ranks `[1, 2, 3]` and `complete: false`. Rank 0's result can have complete local coverage; the other ranks' results preserve the returned replies but have no local representative. Missing replies do not establish agreement or disagreement.

Each probe result's `resolution` records the outcome; the artifacts keep `pending_worker_probe`:

| Value | Meaning |
|---|---|
| `worker_probe` | Complete coverage with no observed setting disagreements |
| `worker_probe_partial` | Some valid observations, with incomplete or unknown coverage |
| `worker_probe_mixed` | Valid observations disagree on a compared setting; check coverage separately |
| `worker_probe_unobserved` | No valid reply matches the described artifact's workers |
| `probe_failed: ...` | The RPC failed, timed out, or returned no usable replies |
| `probe_skipped: ...` | Dynamo did not send the RPC because the model workers load your own `--worker-extension-cls` class, not one of Dynamo's extension classes |

`attention.backend_resolved` is the single backend name across valid, in-scope layer observations, or `null` when there are none or multiple names. `attention.mla_prefill_backend_resolved` similarly requires agreement among those observations. `attention.resolution` is `worker_probe_mixed` for mixed layer backend names; otherwise it follows the engine's resolution. These summaries do not extend coverage beyond the recorded replies. Probe or sidecar-write failures leave the collected measurements usable.

## Related

- [Observe a Local Deployment](../../cli/operations/observability.mdx#capture-forward-pass-metrics)
- [Observability Architecture](../../developer-guide/knowledge-base/concepts/observability-architecture.md#forward-pass-metrics-persistence)
- [SGLang Observability](../../developer-guide/knowledge-base/modular-components/backends/sglang/observability.md#forward-pass-metrics-fpm)
