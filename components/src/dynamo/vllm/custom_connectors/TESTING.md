# Testing `NixlConnectorWithPendingMetrics`

This document explains what the new connector adds, why it's needed, and the
exact methodology used to verify it.

## The problem in one sentence

In vLLM, when a decode worker dies *after*
prefill has completed but *before* it pulls the KV cache, the prefill request
sits in an awkward state where:

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
└── args.py                          (extend _uses_nixl_connector to recognize subclass)
```

The connector is loaded at vLLM startup via the existing
`kv_connector_module_path` mechanism in vLLM's `KVConnectorFactory`. Activation
is a one-line DGD change (see `nixl_with_pending_metrics.py` docstring).

## How it was tested

### 1. Version stack

Tests ran against the same versions Dynamo 1.1.x ships with:

```
Dynamo:  v1.1.0        (matches "Dynamo 1.1.0 vllm runtime on 1.0.1 operator")
vLLM:    0.19.0        (exact pin from Dynamo 1.1.0 pyproject.toml)
NIXL:    1.1.0
GPU:     1× RTX 5880 Ada 48 GB (single GPU, host venv — not in a container)
```

Single-host setup; both prefill and decode vLLM processes share GPU 0 (each
sets `gpu-memory-utilization=0.30` so they fit). NIXL uses self-loopback for
the in-host KV transfer. This is sufficient to exercise the full
NixlConnectorScheduler ↔ NixlConnectorWorker ↔ Prometheus path.

### 2. The "awkward state" repro — kill decode mid-transfer

To force a prefill into the stranded state:

```bash
# 1. Start prefill vLLM (port 8100) with our connector + a long
#    VLLM_NIXL_ABORT_REQUEST_TIMEOUT so the sweep doesn't fire during the test
VLLM_NIXL_ABORT_REQUEST_TIMEOUT=600 \
PYTHONPATH=.../components/src \
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8100 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config '{"kv_connector":"NixlConnectorWithPendingMetrics",
                           "kv_role":"kv_both",
                           "kv_connector_module_path":
                             "dynamo.vllm.custom_connectors.nixl_with_pending_metrics"}'

# 2. Start decode vLLM (port 8200) with the same connector
PYTHONPATH=.../components/src \
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8200 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config '...same...'

# 3. Start the upstream vLLM toy proxy on port 8000 (orchestrates P->D handoff)
python toy_proxy_server.py \
    --prefiller-host localhost --prefiller-port 8100 \
    --decoder-host localhost --decoder-port 8200 \
    --port 8000

# 4. Fire a request, then SIGKILL decode 3s in (mid-transfer)
curl -X POST localhost:8000/v1/chat/completions \
     -d '{"model":"...","messages":[{"role":"user","content":"Write a poem."}],
          "max_tokens":128}' &
sleep 3
kill -9 <DECODE_PID>

# 5. Query prefill /metrics — pending_sends > 0 while standard metrics are at 0
curl -s localhost:8100/metrics | grep -E "num_requests_(running|waiting)|nixl_num_pending"
```

### 3. Observed result (the smoking gun)

```
====================================================================
T2: 2s after decode killed - the awkward state
====================================================================
  vllm:num_requests_running                                  0.0
  vllm:num_requests_waiting                                  0.0
  vllm:kv_cache_usage_perc                            4.57e-05    ← noise floor
  vllm:nixl_num_kv_expired_reqs_total                        0.0    ← timeout hasn't fired
  vllm:num_preemptions_total                                 0.0    ← not preemption
  vllm:nixl_num_pending_sends                                1.0   [NEW]  ← stranded!
  vllm:nixl_num_in_process_reqs                              1.0   [NEW]
```

12 seconds later, with no traffic, the standard metrics didn't move and
`nixl_num_pending_sends` was still 1 — confirming the request is silently
pinned and only our metric reflects it.

### 4. Audit: nothing else covers this

A diff of the full `/metrics` output between "just stranded" and "12 s later,
still stranded" snapshots was **empty** — no standard metric ticked across
that 12-second interval. The strand is completely invisible to every
out-of-the-box vLLM gauge, counter, and histogram.

Notable findings from the audit:

- **`vllm:e2e_request_latency_seconds_count` is actively misleading** — it
  ticks up when the prefill's request reaches `FINISHED_LENGTH_CAPPED`, so it
  reports the stranded request as completed. Operators graphing
  `rate(vllm:e2e_request_latency_seconds_count[5m])` on prefill workers will
  see a falsely high success rate during a strand storm.
- **`vllm:nixl_num_failed_transfers_total` misses this case** — it fires only
  when a transfer is *posted and errors*. When decode dies before initiating
  the pull, no transfer was ever posted, no failure recorded, counter unchanged.
- **`vllm:kv_cache_usage_perc` tracks at noise floor** — mathematically it
  reflects the pinned blocks (1 stranded → ~5e-5, scaling linearly), but
  operationally invisible until thousands of strands accumulate.

### 5. Running through Dynamo's own entrypoint (`python -m dynamo.vllm`)

The repro in section 2 uses `vllm serve` directly. The customer's DGD actually
runs `python -m dynamo.vllm`, which goes through Dynamo's arg parsing
(including `_uses_nixl_connector()` → `ensure_side_channel_host()`) and exposes
metrics on the Dynamo system server (`DYN_SYSTEM_PORT`). Repro through that
launch path, single GPU, no etcd / no NATS:

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
grep "NixlConnectorWithPendingMetrics" /tmp/dynamo_runtime_test/prefill.log
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
# Expected — the "awkward state" through Dynamo:
#   vllm:num_requests_running ...     0.0
#   vllm:num_requests_waiting ...     0.0
#   vllm:nixl_num_pending_sends ...   1.0        ← the new gauge
#   vllm:nixl_num_kv_expired_reqs_total ... 0.0  ← timeout hasn't fired
```

Observed result (real run, file-backed discovery, single RTX 5880 Ada,
Qwen2.5-0.5B):

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
  vllm:nixl_num_pending_sends                                1.0   [NEW]  (stable)
```

The full Prometheus output on Dynamo `/metrics` carries the standard Dynamo
labels in addition to the vLLM ones, e.g.:

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
image. Normal Dynamo source-build pipeline will pick them up; or as a volume
mount overlay if you want to test without a rebuild.

## Verification scripts

For reproducing the tests yourself, the scripts live alongside this branch
under `components/src/dynamo/vllm/custom_connectors/verification/`:

| Script | What it does |
|---|---|
| `e2e_real_vllm.py` | Unit-level checks against real vLLM 0.19.0 (no GPU) |
| `factory_resolution.py` | Confirms `KVConnectorFactory` resolves our class |
| `gpu_smoke_test.py` | Starts ONE vLLM, confirms gauges appear at `/metrics` |
| `single_gpu_pd_test.sh` | Two `vllm serve` processes on one GPU; basic round-trip + decode kill |
| `test_invisible_occupancy.sh` | The side-by-side comparison shown in section 3 above |
| `test_full_metrics_audit.sh` | The full `/metrics` diff audit from section 4 |
| `dynamo_runtime_test.sh` | Same repro but using `python -m dynamo.vllm` instead of `vllm serve` |
| `toy_proxy_server.py` | Upstream-vLLM proxy that orchestrates P→D handoff |
