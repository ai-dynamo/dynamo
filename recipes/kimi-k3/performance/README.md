<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi-K3 performance benchmarking

This procedure reproduces the performance workload used to tune these GB300 recipes. First measure
the real DSpark acceptance length on the public SPEED-Bench coding category. Then use that measured
value with vLLM's synthetic rejection sampler to sweep a fixed 64K-input, 400-output workload with
90% prefix reuse. Synthetic rejection is a performance model only; it does not validate output
quality and must not be used for production serving.

Run the commands below from `recipes/kimi-k3` unless a step changes directory explicitly.

## Benchmark definition

| Property | Acceptance-length measurement | Synthetic performance sweep |
| --- | --- | --- |
| Dataset | SPEED-Bench qualitative, `coding` only (80 conversations) | AI Perf synthetic prompts |
| Speculative verification | Real `block` rejection | `synthetic` rejection with the measured coding AL |
| Draft | Inferact DSpark, draft length 7, probabilistic sampling | Identical |
| Sampling | Temperature 0, top-p 1, EOS respected | EOS ignored to hold OSL fixed |
| Load | Concurrency 16, 80 sequential sessions | Concurrency 2, 4, 8, 16, 32, 64, 128, 192, 256; `3*C` requests per point |
| Token shape | Dataset-defined; maximum completion length 4096 | 57,600 shared + 6,400 unique input tokens, exactly 400 output tokens |
| Random seed | Dataset order | 42 |

Keep the image, model and draft checkpoints, draft length, sampling policy, attention backends, and
all vLLM engine settings unchanged between the real-AL and synthetic runs. Re-measure AL whenever
any of those inputs changes.

## 1. Install AI Perf

Use an isolated Python 3.12 environment. The configuration in this directory is validated against
AI Perf 0.12.0 at public commit `c2f5e9d459005d362457716bbd865d247232fa30`.

```bash
python3.12 -m venv .venv-aiperf
. .venv-aiperf/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  https://github.com/ai-dynamo/aiperf/archive/c2f5e9d459005d362457716bbd865d247232fa30.tar.gz
aiperf --version
```

The last command must report `0.12.0`. Pin the same client revision for every point in a comparison.

## 2. Measure coding acceptance length

Use the public [NVIDIA SPEED-Bench dataset](https://huggingface.co/datasets/nvidia/SPEED-Bench)
with the upstream [Model Optimizer speculative-decoding benchmark](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench).
Follow Model Optimizer's SPEED-Bench instructions and dataset license; this recipe intentionally
does not copy or prepare the benchmark data.

Run the qualitative `coding` category only, using all 80 conversations and the acceptance-length
settings in the table above. Deploy the selected GB300 manifest unchanged: it already uses real
`"rejection_sample_method":"block"`. Do not enable synthetic rejection during this step.

Measure vLLM's counters over only the coding profile window. Port-forward the frontend and the vLLM
leader's metrics port in separate terminals; set `DGD` to `kimi-k3-agg` or `kimi-k3-disagg`.

```bash
export NAMESPACE=your-namespace
export DGD=kimi-k3-agg

kubectl port-forward -n ${NAMESPACE} service/${DGD}-frontend 8000:8000
```

```bash
export NAMESPACE=your-namespace
export DGD=kimi-k3-agg
export LEADER=$(kubectl get pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} -o name | \
  sed -n '/vllmdecodeworker.*ldr/{s#pod/##;p;q}')
test -n "${LEADER}"
kubectl port-forward -n ${NAMESPACE} pod/${LEADER} 9090:9090
```

Immediately before sending the 80 coding conversations, and immediately after the last response,
capture the leader metrics:

```bash
curl -fsS http://127.0.0.1:9090/metrics > metrics-before.prom
# Run the SPEED-Bench coding profile here.
curl -fsS http://127.0.0.1:9090/metrics > metrics-after.prom
python performance/calculate_acceptance.py metrics-before.prom metrics-after.prom
```

The script uses counter deltas from the measurement window:

```text
acceptance_length = 1 + accepted_tokens / drafts
acceptance_rate   = accepted_tokens / draft_tokens
```

For reference, the `draft_length=7` run measured `4.25846126489242` on this coding workload.

## 3. Enable synthetic acceptance length

Work on a copy of the selected manifest. In every vLLM worker command that has
`--speculative-config`, replace only the rejection method and add the measured AL:

```json
"rejection_sample_method":"synthetic","synthetic_acceptance_length":4.25846126489242
```

Leave the DSpark model, `num_speculative_tokens: 7`, probabilistic draft sampling, and all other
worker settings unchanged. For the disaggregated manifest, make the same change in both prefill and
decode worker commands. Apply the copied manifest, wait for the DGD to become Ready, verify every
pod has zero restarts, and send one short chat request to complete kernel and request-path warm-up.

```bash
kubectl get pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  -o custom-columns='POD:.metadata.name,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount'

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"Return the number 4."}],"max_tokens":4,"temperature":0}' \
  > /dev/null
```

## 4. Run the synthetic sweep

Keep the frontend and leader-metrics port-forwards from step 2 active. Use a tokenizer directory or
Hugging Face model ID accessible to the AI Perf host:

```bash
export TOKENIZER=moonshotai/Kimi-K3
export MODEL_NAME=moonshotai/Kimi-K3
export INFERENCE_URL=http://127.0.0.1:8000/v1/chat/completions
export SERVER_METRICS_URL=http://127.0.0.1:9090/metrics

for CONCURRENCY in 2 4 8 16 32 64 128 192 256; do
  export CONCURRENCY
  export REQUESTS=$((3 * CONCURRENCY))
  export CACHE_SALT=kimi-k3-64k-c${CONCURRENCY}-$(date +%s%N)
  export AIPERF_ARTIFACT_DIR=artifacts/synthetic-64k/c${CONCURRENCY}
  mkdir -p "${AIPERF_ARTIFACT_DIR}"

  aiperf config validate performance/synthetic-64k.yaml
  aiperf config expand performance/synthetic-64k.yaml --full \
    > "${AIPERF_ARTIFACT_DIR}/resolved-config.txt"
  aiperf profile --config performance/synthetic-64k.yaml

  jq '{requests:.request_count.avg, errors:(.error_summary | length),
       isl:.total_isl.avg, osl:.total_output_tokens.avg,
       output_tps:.output_token_throughput.avg,
       output_tps_per_user:.output_token_throughput_per_user.avg,
       ttft_ms:.time_to_first_token.avg, itl_ms:.inter_token_latency.avg}' \
    "${AIPERF_ARTIFACT_DIR}/profile_export_aiperf.json"
done
```

The configuration calls `POST /reset_prefix_cache` before each point. The shipped GB300 manifests
set `VLLM_SERVER_DEV_MODE=1`, which enables that endpoint. A fresh `cache_salt` additionally prevents
prefix reuse across points; within a point, all requests share the 57,600-token system prefix.

For each point, verify all of the following before retaining its report:

- request count is exactly `3*C`, with no failed or cancelled requests;
- average server-counted input length is approximately 64K and output length is exactly 400;
- the DGD remains Ready and all worker restart counts remain zero;
- the resolved AI Perf configuration, raw report, image digest, manifest, and measured AL are saved.

