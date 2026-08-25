# DeepSeek-V4-Pro-0813

Recipes for the `deepseek-ai/DeepSeek-V4-Pro-0813` checkpoint.

> This is a **different checkpoint** from `deepseek-ai/DeepSeek-V4-Pro` used by the sibling
> `../deepseek-v4-pro` recipes. Do not mix the two model caches or deploy configs.

## Layout

```
deepseek-v4-pro-0813/
├── model-cache/
│   ├── model-cache.yaml       # PVC (1200Gi; checkpoint is ~832 GiB)
│   └── model-download.yaml    # populates the PVC from HF
└── vllm/
    ├── agg-h200-agentic/       # 8x H200,  64K agentic
    ├── disagg-h200-agentic/    # 16x H200, 64K agentic  (best density)
    ├── agg-h200-1m/            # 8x H200,  1M context   (batch)
    └── disagg-h200-1m/         # 16x H200, 1M context   (batch)
```

## Config matrix

|                | Aggregated (8x H200)                    | Disaggregated (16x H200)                  |
|----------------|-----------------------------------------|-------------------------------------------|
| **Agentic 64K**| `agg-h200-agentic` <br> C=2, E2E 56.40 tok/s/user, 11.85 tok/s/GPU | `disagg-h200-agentic` <br> C~7.95, E2E 50, ~24 tok/s/GPU |
| **1M context** | `agg-h200-1m` <br> 998,218 tokens, TTFT ~202 s | `disagg-h200-1m` <br> 1,033,872 tokens, TTFT ~146 s |

Agentic workload is 64K ISL / 400 OSL / 90% prefix reuse; the SLA gate is
E2E >= 50 tok/s/user **and** TTFT p50 < 5 s, jointly, where
`E2E = OSL / (TTFT + OSL x ITL)`. The 1M configs are a batch capability, not
interactive -- time-to-first-token is minutes at that length.

Accuracy on the agentic configs: `gpqa_diamond` 88.26 (agg) / 88.38 (disagg),
`ifeval` 94.48 (agg) / 94.29 (disagg) -- disaggregation does not cost accuracy.

## Quick start

```bash
kubectl apply -f model-cache/model-cache.yaml
kubectl apply -f model-download.yaml   # requires an hf-token-secret in the namespace
kubectl wait --for=condition=complete job/model-download --timeout=6h
```

The download job pulls the default revision. The H200 configs in this directory were validated
against revision `72e1d3230f6c080a530b0a1d46f8eb4602340597`; pin `--revision` in the deploy if
you need to reproduce those numbers exactly.

## Status

| SKU | Topology | Workload | State |
|-----|----------|----------|-------|
| H200 x8  | agg    | agentic 64K | validated |
| H200 x16 | disagg | agentic 64K | validated |
| H200 x8  | agg    | 1M context  | validated |
| H200 x16 | disagg | 1M context  | validated |
| GB200    | -      | -           | separate contribution |
