<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Solar Open2 250B NVFP4 recipe

This recipe serves [`nota-ai/Solar-Open2-250B-Nota-NVFP4`](https://huggingface.co/nota-ai/Solar-Open2-250B-Nota-NVFP4)
with Dynamo and vLLM on B200 GPUs. It is an aggregated deployment with two
independent TP2 workers behind the Dynamo KV-aware router.

## Configuration

| Setting | Value |
| --- | --- |
| GPU | 4x B200 total; 2x B200 per worker |
| Topology | Aggregated, two worker replicas |
| Parallelism | TP2, DP1, expert parallel disabled |
| Weight precision | NVFP4 |
| KV-cache precision | FP8 |
| Routing | KV-aware, vLLM prefix-cache events |
| Context | 1,048,576 tokens |
| Reasoning parser | `solar_open2` |

Two workers are intentional: a single worker can publish KV events, but the
router needs at least two candidates before KV-aware placement does anything.

Solar Open2 is a hybrid architecture, 12 grouped-query attention layers and 36
linear-attention layers. vLLM accounts for the linear-attention state as a Mamba
page, so it raises the attention block size to 2128 tokens to keep the attention
page at least as large as the Mamba page. The recipe therefore does not set
`--block-size`; any value passed is overridden.

## Deployment

```bash
kubectl apply -f recipes/solar-open2-250b/model-cache/model-cache.yaml
kubectl apply -f recipes/solar-open2-250b/model-cache/model-download.yaml
kubectl wait --for=condition=complete job/model-download --timeout=2h

kubectl apply -f recipes/solar-open2-250b/vllm/agg-b200-chat/deploy.yaml
kubectl wait --for=condition=Ready \
  dynamographdeployment/solar-open2-250b-vllm-b200-agg-chat --timeout=30m
```

## Performance

Measured on a 15% subset of the `nim_turbo` 8k/1k 70kv chat trace: 1805 requests,
mean ISL 37.4k tokens, mean OSL 1008 tokens.

| Workload | Recipe | Framework | SKU | Concurrency | System output tok/s/GPU | User output tok/s (P50) | TTFT P50 (ms) |
|---|---|---|---|---:|---:|---:|---:|
| Chat (15% subset) | Aggregated, 2 replicas, KV-aware routing | vLLM | B200 | 20 | 214.98 | 52.16 | 319 |

The benchmark job is in [`perf/perf.yaml`](perf/perf.yaml).
