<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 Agentic Benchmark

[`perf.yaml`](perf.yaml) runs the NIM Turbo 64k/400/90%-KV agentic trace against
the GLM-5.2 1P1D DGD. It follows the Kimi-K2.6 trace-replay workflow: the client
is co-located with the frontend, waits for `/v1/models`, performs a small
warmup, and stores raw AIPerf artifacts on the model-cache PVC.

## Defaults

| Variable | Default |
| --- | --- |
| `ENDPOINT` | `glm52-b200-tp8-kv-disagg-frontend:8000` |
| `TARGET_MODEL` | `nvidia/GLM-5.2-NVFP4` |
| `TRACE_FILE` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |
| `CONCURRENCY` | `64` |
| `ROOT_ARTIFACT_DIR` | `/model-cache/perf` |

The default 15% trace contains 3,541 requests. Its SHA-256 is
`f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`.

## Stage the trace

The agentic trace assets are shared with the Kimi-K2.6 recipe. Materialize the
Git LFS file, then copy it to the model-cache PVC:

```bash
git lfs pull --include='recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl exec -n ${NAMESPACE} pvc-helper -- mkdir -p /model-cache/traces
kubectl cp \
  recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl \
  ${NAMESPACE}/pvc-helper:/model-cache/traces/
```

## Run

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=glm52-agentic-bench -f
kubectl wait --for=condition=Complete job/glm52-agentic-bench \
  -n ${NAMESPACE} --timeout=10800s
```

The Job creates a tokenizer-only local snapshot before launching AIPerf. This
avoids loading the model architecture config in the client and includes the
minimal AIPerf 0.10.0 local-tokenizer-path fix needed by multiprocessing dataset
workers.

## Fetch artifacts

```bash
kubectl cp \
  ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_glm52-agentic-bench \
  ./results
```

## Clean boundary for another run

Reset both SGLang KV state and Dynamo frontend/router state between independent
trace replays:

```bash
kubectl delete job glm52-agentic-bench -n ${NAMESPACE} --ignore-not-found
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=glm52-b200-tp8-kv-disagg
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=glm52-b200-tp8-kv-disagg \
  --timeout=7200s
```

Do not compare partial runs. A completed run must account for successful,
errored, and unfinished requests before reporting aggregate throughput.
