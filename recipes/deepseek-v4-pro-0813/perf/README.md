<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro-0813 Benchmark Recipe

An [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay against a deployed DGD. The
benchmark is identical for both profiles; only `ENDPOINT` changes. The client waits for the
model on the frontend, runs a short warmup, replays the configured trace at one
`CONCURRENCY` value, and writes raw artifacts to the shared `model-cache` PVC.

## Targeting a variant

| Profile | `ENDPOINT` |
| --- | --- |
| Aggregated (TP8 + DSpark) | `dsv4-pro-0813-agg-gb200-agentic-frontend:8000` |

It serves `deepseek-ai/DeepSeek-V4-Pro-0813`. Deploy from
[`../vllm/agg-gb200-agentic/deploy.yaml`](../vllm/agg-gb200-agentic/deploy.yaml) (see
the recipe [README](../README.md)). A disaggregated profile is not published yet.

## Dataset

The benchmark replays a [Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace via
`aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request
(`input_length`, `output_length`, `hash_ids`, `timestamp`). The
agentic target is median ISL ~64k, OSL ~400, ~90% token-weighted cache hit, block size
**512**, replayed closed-loop at a fixed concurrency. Sweep concurrency to find the SLA
knee: the highest value holding P50 TTFT < 5 s and P50 output >= 50 tok/s/user.

Use the 3,541-request agentic trace in `perf/traces/` (Git LFS). AIPerf builds its replay
schedule from a `timestamp` field; the shipped rows have none, and without it AIPerf
replays a small default instead of the file, so add one:

```bash
git lfs install
# the trace here is a symlink into recipes/kimi-k2.6, where the LFS object lives
git lfs pull --include "recipes/kimi-k2.6/perf/traces/*"
python3 -c "
import json
src='recipes/deepseek-v4-pro-0813/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'
for line in open(src):
    print(json.dumps({'timestamp': 0, **json.loads(line)}))
" > mooncake_agentic.jsonl
```

Confirm a run replayed the whole file: `Request Count` in
`profile_export_aiperf.csv` should be ~3,411, plus ~130 rows that exceed the 262,144-token
context and return errors.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the variant

See the deployment instructions in the recipe [README](../README.md).

### 2. Stage the trace on the PVC

Copy `mooncake_agentic.jsonl` onto the `model-cache` PVC via a helper pod that mounts it:

```bash
kubectl run pvc-helper -n ${NAMESPACE} --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl exec -n ${NAMESPACE} pvc-helper -- mkdir -p /model-cache/traces
kubectl cp mooncake_agentic.jsonl ${NAMESPACE}/pvc-helper:/model-cache/traces/
```

Keep `pvc-helper` around for fetching artifacts later, or
`kubectl delete pod pvc-helper -n ${NAMESPACE}` once you are done staging.

### 3. Run AIPerf

`perf.yaml` in this directory runs the replay as a Job. Edit `CONCURRENCY` to select the
operating point — aggregated is at its SLA knee at c8 — then:

```bash
kubectl apply -f recipes/deepseek-v4-pro-0813/perf/perf.yaml -n ${NAMESPACE}
```

It waits for the model to serve, warms up, replays the whole trace, and writes artifacts to
`/model-cache/perf/<epoch>_<job-name>/`. It aborts if the trace has no `timestamp` field.

To drive AIPerf directly instead (image `nvcr.io/nvidia/ai-dynamo/aiperf:0.11.0`, mounts the
PVC) run against the frontend service:

```bash
aiperf profile deepseek-ai/DeepSeek-V4-Pro-0813 --tokenizer deepseek-ai/DeepSeek-V4-Pro-0813 \
  --url http://${ENDPOINT} --endpoint-type chat \
  --input-file ${TRACE_FILE} \
  --custom-dataset-type mooncake_trace --prompt-input-tokens-block-size 512 \
  --concurrency ${CONCURRENCY} --workers-max ${CONCURRENCY} \
  --extra-inputs ignore_eos:true --streaming --use-server-token-count \
  --artifact-dir /model-cache/perf/<run> --ui none
```

Metrics land in `profile_export_aiperf.{csv,json}`: `Output Token Throughput`,
`Request Throughput`, `Time to First Token`, `Inter Token Latency`,
`Output Token Throughput Per User`. KV-cache hit rate is on the frontend `/metrics`
(`dynamo_component_router_kv_hit_rate_{sum,count}`).

> [!WARNING]
> Do **not** publish throughput measured with the `speculative-config-synthetic` key.
> On this model, forcing AL 2.27 overstates P50 per-user throughput by **~36.5%**
> (75.6 vs 55.9 tok/s/user, same trace, same concurrency, identical error counts),
> because synthetic acceptance skips the draft-model forward passes it claims to have
> paid for. The synthetic key exists for crash-free smoke runs on unpatched images.
> Ship — and measure — with the `speculative-config` key, which uses real DSpark
> rejection sampling and is distribution-preserving.

## Running a concurrency sweep

Run one `CONCURRENCY` at a time; reset vLLM KV and Dynamo router state between independent
runs by restarting the DGD pods:

```bash
DGD=dsv4-pro-0813-agg-gb200-agentic
kubectl delete pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} --timeout=7200s
```

In mooncake mode AIPerf replays the whole trace file provided each row carries a
`timestamp`; `perf.yaml` sets `--num-requests` to the trace line count. Subset the file to
cap request count. Do not compare partial
runs — check `Request Count` and account for successful, errored, and unfinished requests
before reporting aggregate throughput.

## Tunable environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `ENDPOINT` | per profile — see the table above | |
| `CONCURRENCY` | sweep to the SLA knee | Single value per run; restart server pods between values |
| `TRACE_FILE` | `/model-cache/traces/mooncake_agentic.jsonl` | 3,541-request agentic trace with timestamps |
| `TARGET_MODEL` | `deepseek-ai/DeepSeek-V4-Pro-0813` | Must match `--served-model-name` |

## Artifacts

Results are written to `/model-cache/perf/<run>/profile_export_aiperf.{csv,json}`,
`inputs.json`, and warmup/log files.
