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
└── vllm/                      # deploy.yaml per SKU x topology (added separately)
```

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
