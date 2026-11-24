<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance benchmark

`perf.yaml` is a Kubernetes Job that replays the public AgentX multi-turn coding-agent trace
(`semianalysis_cc_traces_weka_062126`, aiperf scenario `inferencex-agentx-mvp`) against one deployed recipe at one
concurrency, for a 3600 s measured window after a warmup phase. The dataset is downloaded from Hugging Face at start,
so the Job needs outbound access and no token.

## Variants

The endpoint is `http://<dgd-name>-frontend:8000` in the namespace you deploy to. Set `CONCURRENCY` to the value below
for the variant to reproduce its published result.

| Recipe | DGD name | GPUs | `CONCURRENCY` |
|---|---|---|---|
| `agg-h200-agentx` | `dsv41-flash-h200-agg` | 4 | 111 |
| `disagg-h200-agentx` | `dsv41-flash-h200-disagg` | 16 | 600 |
| `agg-b200-agentx` | `dsv41-flash-b200-agg` | 4 | 800 |
| `disagg-b200-agentx` | `dsv41-flash-b200-disagg` | 28 | 6150 |
| `agg-gb300-agentx` | `dsv41-flash-gb300-agg` | 4 | 1180 |
| `disagg-gb300-agentx` | `dsv41-flash-gb300-disagg` | 4 | 1200 |

tok/s/GPU is the total output tokens per second printed in the aiperf summary, divided by the GPU count. A point passes
when decode p50 is at least 50 tok/s/user, TTFT p50 is under 2 s and errors are under 5%.

## Synthetic acceptance

The trace carries hashed filler blocks instead of real text, so a real DSpark acceptance length measured on it is not
representative. The published points force the acceptance length measured on SPEED-Bench at 5 draft tokens (3.19 on
H200, 3.4 on B200 and GB300; the length includes the bonus token). To reproduce them, replace the `--speculative-config`
value on every worker (both roles for disaggregated targets) with:

```text
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":<3.19 or 3.4>,"enable_adaptive_verification":false,"revision":"1bae0e1088993b9b82de08f59de5be5a34e42170"}
```

Synthetic acceptance fabricates accepted tokens, so the output text is meaningless and the throughput is an upper bound.
Use it for benchmarking only, never for serving or accuracy. With the shipped real-acceptance config the Job measures
a different workload and the result is not comparable to the published numbers.

## Run

```bash
# 1. deploy the recipe and wait until it answers a real completion, not just /v1/models
kubectl apply -f recipes/deepseek-v4.1-flash-1017/vllm/<recipe>/deploy.yaml -n ${NAMESPACE}

# 2. render and apply the Job (CACHE_SALT must be fresh for every run so earlier runs do not warm the prefix cache)
CONCURRENCY=1180 ENDPOINT=http://dsv41-flash-gb300-agg-frontend:8000 CACHE_SALT=$(openssl rand -hex 12) \
  envsubst '$CONCURRENCY $ENDPOINT $CACHE_SALT' < recipes/deepseek-v4.1-flash-1017/perf/perf.yaml \
  | kubectl apply -f - -n ${NAMESPACE}

# 3. follow it; the summary table is the last thing in the log, followed by RC=0 and AGENTX_DONE
kubectl logs -f job/dsv41-flash-agentx-perf -n ${NAMESPACE}
kubectl delete job dsv41-flash-agentx-perf -n ${NAMESPACE}
```

Artifacts are written to an `emptyDir` and are gone when the pod is deleted, so keep the log (the metrics are in the
summary table; `profile_export_aiperf.csv` is in the pod until then). Delete the Job between runs: its name is fixed.

## Notes

- **Warmup is part of the run and grows with concurrency.** The agentic scenario completes about 1.07 requests per
  concurrent session before the measured window starts. Warmup sends requests nearly one at a time and speeds up as it
  goes, so the GPUs look idle at first. At concurrency 1,152 it takes about 110 minutes, so with the 60-minute window the
  Job runs about 3 hours. At 6,150 (`disagg-b200-agentx`) the Job took about 12 hours end to end;
  `activeDeadlineSeconds` is 16 hours.
- **Size the client for the request count.** aiperf keeps every record in memory, about 1 GiB per 1,000 requests; at
  the default 160 GiB limit it was OOM-killed near 157,000 requests. `disagg-b200-agentx` completes about 315,000
  requests in the hour, so raise the Job's memory to `requests: 384Gi` and `limits: 768Gi` for it. The other targets
  stay under 70,000.
- **Keepalive.** At high concurrency hundreds of requests wait for their first token at the end of warmup. The Job
  allows 5 keepalive probes and a 5-minute TCP user timeout (`AIPERF_HTTP_TCP_KEEPCNT`, `AIPERF_HTTP_TCP_USER_TIMEOUT`)
  so aiperf does not drop those connections.
- **Session affinity needs the header.** The Job sets `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1`, which
  makes the Dynamo router pin each trajectory to one worker. Without it, multi-worker results are much worse.
- On arm64 nodes the Job compiles one dependency (`crick`), so the first minutes show a package build.
