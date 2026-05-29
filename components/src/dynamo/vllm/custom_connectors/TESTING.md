# Testing `NixlConnectorWithPendingMetrics`

This document explains what the new connector adds, why it's needed, and the
exact repro used to verify it through Dynamo's runtime.

## The problem in one sentence

In vLLM, when a decode worker dies *after* prefill has completed but *before*
it pulls the KV cache, the prefill request sits in an awkward state where:

- It is **removed** from `self.running` (it's "finished")
- It is **removed** from `self.waiting` (was never there)
- It **keeps** its KV blocks allocated (pinned by the NIXL connector's
  `_reqs_to_send` side-channel map)
- It will stay pinned until either decode pulls (won't happen, decode is dead)
  or the `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` sweep fires (default 480 s)

…and during those up-to-480 seconds, **no standard vLLM metric tracks it**:

| Standard metric | Value during strand |
|---|---|
| `vllm:num_requests_running` | 0 |
| `vllm:num_requests_waiting` | 0 |
| `vllm:num_preemptions_total` | 0 |
| `vllm:kv_cache_usage_perc` | at noise floor (~5e-5 per stranded request) |
| `vllm:nixl_num_failed_transfers_total` | 0 (transfer was never posted) |
| `vllm:nixl_num_kv_expired_reqs_total` | 0 (sweep hasn't fired yet) |
| `vllm:e2e_request_latency_seconds_count` | ➕ goes up — **counts strand as success** |

## What this connector adds

Two Prometheus gauges, P-side only:

```
vllm:nixl_num_pending_sends   = len(NixlConnectorWorker._reqs_to_send)
vllm:nixl_num_in_process_reqs = len(NixlConnectorWorker._reqs_to_process)
```

Both are real-time counts of "how many finished prefill requests are pinned on
this worker right now, waiting for decode to read." The diagnostic gap closes.

## What was changed

Two new files (the connector) plus a small extension to `args.py` so that
Dynamo's auto-detection recognizes the subclass and runs
`ensure_side_channel_host()` for it.

```
components/src/dynamo/vllm/custom_connectors/
├── __init__.py
└── nixl_with_pending_metrics.py
components/src/dynamo/vllm/
└── args.py    (extend _uses_nixl_connector to recognize the subclass)
```

The connector is loaded at vLLM startup via the existing
`kv_connector_module_path` mechanism in vLLM's `KVConnectorFactory`. Activation
is a one-line DGD change (see `nixl_with_pending_metrics.py` docstring).

## How it was tested

### Version stack

```
Dynamo:  v1.1.0        (matches "Dynamo 1.1.0 vllm runtime on 1.0.1 operator")
vLLM:    0.19.0        (exact pin from Dynamo 1.1.0 pyproject.toml)
NIXL:    1.1.0
GPU:     1× RTX 5880 Ada 48 GB (single GPU, single host)
```

Single-host setup; both prefill and decode `python -m dynamo.vllm` processes
share GPU 0 (each sets `--gpu-memory-utilization 0.30`). NIXL uses self-loopback
for the in-host KV transfer. This exercises the full Dynamo +
NixlConnectorScheduler + NixlConnectorWorker + Prometheus path.

### The repro — through `python -m dynamo.vllm`

```bash
# 0. Self-contained Dynamo backplane: file-backed discovery + TCP request plane
#    + ZMQ event plane. Set ONCE in the shell, inherited by all 3 processes.
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq

# 1. Frontend on :8000 (the HTTP entrypoint the client hits)
PYTHONPATH=.../components/src \
DYN_HTTP_PORT=8000 \
python -m dynamo.frontend &

# 2. Decode worker — Dynamo /metrics on :8081
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.../components/src \
DYN_SYSTEM_PORT=8081 \
python -m dynamo.vllm \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnectorWithPendingMetrics",
                           "kv_role":"kv_both",
                           "kv_connector_module_path":
                             "dynamo.vllm.custom_connectors.nixl_with_pending_metrics"}' &
DECODE_PID=$!

# 3. Prefill worker — Dynamo /metrics on :8082, NIXL side-channel on a
#    different port from the decode worker
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.../components/src \
DYN_SYSTEM_PORT=8082 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
VLLM_NIXL_ABORT_REQUEST_TIMEOUT=600 \
python -m dynamo.vllm \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --disaggregation-mode prefill \
    --kv-transfer-config '...same as above...' &

# 4. Wait for both workers' vLLM engines to load (Dynamo /metrics responds
#    with 0 lines before vLLM is up; wait until vllm: lines appear)
until [ $(curl -s localhost:8082/metrics | grep -c '^vllm:') -gt 0 ]; do
    sleep 2; done

# 5. Confirm our connector was loaded (look for the factory log line):
grep "NixlConnectorWithPendingMetrics" prefill.log
# → INFO factory.py: Creating v1 connector with name: NixlConnectorWithPendingMetrics ...

# 6. Send a successful chat completion through the Dynamo frontend:
curl -X POST localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct",
          "messages":[{"role":"user","content":"What is 7+8?"}],
          "max_tokens":32}'
# → {"...","content":"The sum of 7 and 8 is 15.",...}

# 7. Fire another request, then SIGKILL decode mid-transfer to strand it:
curl -X POST localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct",
          "messages":[{"role":"user","content":"Write a 6-line poem."}],
          "max_tokens":128}' &
sleep 3
kill -9 $DECODE_PID

# 8. Query the PREFILL worker's Dynamo /metrics:
curl -s localhost:8082/metrics | grep -E \
    "num_requests_(running|waiting)|nixl_num_pending_sends|num_kv_expired"
```

### Observed result (real run)

```
====================================================================
  T0: cold start — through Dynamo runtime
====================================================================
  vllm:num_requests_running                                  0.0
  vllm:num_requests_waiting                                  0.0
  vllm:nixl_num_pending_sends                                0.0   [NEW]
  vllm:nixl_num_in_process_reqs                              0.0   [NEW]

[T1] response: "The sum of 7 and 8 is 15."

====================================================================
  T1: after successful round-trip via Dynamo frontend
====================================================================
  vllm:nixl_num_pending_sends                                1.0   [NEW]
  vllm:nixl_num_in_process_reqs                              1.0   [NEW]
  vllm:kv_cache_usage_perc                       4.530490199039505e-05

====================================================================
  T2: 3s after decode killed — invisible occupancy, prefill side
====================================================================
  vllm:num_requests_running                                  0.0    ← invisible!
  vllm:num_requests_waiting                                  0.0    ← invisible!
  vllm:nixl_num_pending_sends                                1.0   [NEW]
  vllm:nixl_num_in_process_reqs                              1.0   [NEW]

====================================================================
  T3: 13s after decode killed — strand holds, no traffic
====================================================================
  vllm:nixl_num_pending_sends                                1.0   [NEW] (stable)
```

The strand sits invisibly on the prefill worker — every standard scheduler-
state gauge reports zero — but `vllm:nixl_num_pending_sends` correctly counts
it. 13 seconds later, with no traffic, the strand still holds and the gauge
still reads 1.

### What the actual Prometheus output looks like

On the prefill `/metrics` endpoint, with full Dynamo labels:

```
vllm:nixl_num_pending_sends{
    dynamo_component="backend",
    dynamo_endpoint="generate",
    dynamo_namespace="dynamo",
    engine="0",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    worker_id="4a713c17c835edfd"
} 1.0
```

A scripted version of the above lives at
`verification/dynamo_runtime_test.sh`.

## Production diagnostic the new metric enables

```promql
# Per-worker count of pinned KV transfers right now (leading indicator)
vllm:nixl_num_pending_sends{pod=~".*prefillworker.*"}

# Trailing indicator — non-zero rate = timeout sweep is actively reclaiming
rate(vllm:nixl_num_kv_expired_reqs_total{pod=~".*prefillworker.*"}[5m])

# The "worker is lying about being idle" alert:
(vllm:num_requests_running == 0)
  and (vllm:num_requests_waiting == 0)
  and (vllm:nixl_num_pending_sends > 0)
```

## DGD activation snippet

Add to the prefill worker args in the DGD (decode workers don't need it; their
`_reqs_to_send` is always empty):

```yaml
- --kv-transfer-config
- '{"kv_connector":"NixlConnectorWithPendingMetrics","kv_role":"kv_both","kv_connector_module_path":"dynamo.vllm.custom_connectors.nixl_with_pending_metrics"}'
```

The files under `components/src/dynamo/vllm/custom_connectors/` plus the
extension to `components/src/dynamo/vllm/args.py` need to be present in the
image. Normal Dynamo source-build pipeline will pick them up.
