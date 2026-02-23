<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Qwen3-32B with vLLM — Aggragated
This is a recipe for vLLM + KVBM perf testing.

One use case of this recipe is to test perf regression for vLLM version upgrade in Dynamo.

### Deploy
```bash
cd recipes/qwen3-32b
kubectl apply -f recipes/qwen3-32b/vllm/agg-kvbm/deploy.yaml -n ${NAMESPACE}
```

Test with a sample request
```bash
kubectl port-forward svc/agg-kvbm-frontend 8000:8000 -n ${NAMESPACE} &
```

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dummy' \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role":"user","content":"Say hello!"}],
    "max_tokens": 64
  }'
```

### Perf testing
```bash
cd recipes/qwen3-32b
kubectl apply -f cache.yaml -n ${NAMESPACE}
kubectl apply -f model-downloda.yaml -n ${NAMESPACE}
kubectl apply -f vllm/agg-kvbm/perf.yaml -n ${NAMESPACE}
```
