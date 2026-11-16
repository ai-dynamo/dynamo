<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mistral Large 4 AgentX benchmark

A single [AIPerf](https://github.com/ai-dynamo/aiperf) Job, [`perf.yaml`](perf.yaml), replays the
AgentX agentic trace (`--scenario inferencex-agentx-mvp`, public dataset
`semianalysis_cc_traces_weka_062126`) against a deployed Mistral Large 4 recipe. It waits for a real
completion from the frontend, runs the AgentX warmup phase (`--warmup-grace-period 1800` allows it up to 30
minutes to finish; profiling starts as soon as it completes), then measures for 60 minutes at one
`CONCURRENCY` value. Raw artifacts go to `/shared-model-cache/perf-artifacts/` on the `shared-model-cache`
PVC. The pod needs internet access once, to fetch the public dataset.

## Targeting a deployment

Edit the `env` block and the `podAffinity` DGD name in [`perf.yaml`](perf.yaml). The shipped defaults are
the Vera Rubin aggregated row; set `ENDPOINT` and the DGD name for the deployment you benchmark:

| Deployment | `ENDPOINT` | `podAffinity` DGD name | `TARGET_MODEL` | `TOKENIZER` |
| --- | --- | --- | --- | --- |
| Vera Rubin aggregated | `http://mistral-large-4-vllm-vr-agg-agentic-frontend:8000` | `mistral-large-4-vllm-vr-agg-agentic` | `mistral-large-4` | `/shared-model-cache/mistral-large-4-nvfp4` |
| Vera Rubin disaggregated | `http://mistral-large-4-vllm-vr-disagg-agentic-frontend:8000` | `mistral-large-4-vllm-vr-disagg-agentic` | `mistral-large-4` | `/shared-model-cache/mistral-large-4-nvfp4` |
| GB300 aggregated | `http://mistral-large-4-vllm-gb300-agg-agentic-frontend:8000` | `mistral-large-4-vllm-gb300-agg-agentic` | `mistral-large-4` | `/shared-model-cache/mistral-large-4-nvfp4` |
| GB300 disaggregated | `http://mistral-large-4-vllm-gb300-disagg-agentic-frontend:8000` | `mistral-large-4-vllm-gb300-disagg-agentic` | `mistral-large-4` | `/shared-model-cache/mistral-large-4-nvfp4` |

Set `ARTIFACT_DIR` to a new path for every run; the Job refuses an existing directory. Leave `CACHE_SALT`
empty so each run gets a fresh salt. The disaggregated deployments pin sessions to a worker, so the Job
exports `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1`.

## Choosing the concurrency

`CONCURRENCY` is per deployment. The gate for this workload is decode-only tok/s/user P50 above 50 and TTFT
P50 below 2 seconds. Run several values and take the highest one that holds both. Judge a point by requests completed as
well as tok/s/GPU, and check the request error rate: AIPerf aborts a run above 10% failed requests (`--failed-request-threshold 0.1`),
and a run with more than a few percent errors is not a valid point.

## Run

```bash
kubectl apply -f recipes/mistral-large-4/perf/perf.yaml -n ${NAMESPACE}
kubectl logs -f job/mistral-large-4-agentx-aiperf -n ${NAMESPACE}
```

Artifacts are in `${ARTIFACT_DIR}`; `profile_export_aiperf.json` holds the summary metrics. Delete the Job
and the ConfigMap before the next run.

Use real speculative-decoding acceptance for any accuracy evaluation. A synthetic-acceptance
deployment (see the recipe's `--speculative-config` comment) is for performance benchmarking only.
