<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek REAP SGLang

This runbook validates `cerebras/DeepSeek-V3.2-REAP-345B-A37B` on a Dynamo Kubernetes cluster with the SGLang backend. It assumes the production profile is installed and the target node has eight H200 or B200 GPUs available through the NVIDIA device plugin.

The deployment example is opt-in:

```bash
kubectl apply -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
```

## Download the Model

Run the download on every node that can schedule the worker pod. The manifest mounts the model from `/models/cerebras/DeepSeek-V3.2-REAP-345B-A37B` and sets `HF_HUB_OFFLINE=1` in the pod.

```bash
sudo mkdir -p /models/cerebras/DeepSeek-V3.2-REAP-345B-A37B
sudo chown -R "$USER:$USER" /models

python3 -m venv ~/hf-download
source ~/hf-download/bin/activate
python -m pip install --upgrade pip
python -m pip install "huggingface_hub[hf_xet]>=0.36"

export HF_XET_HIGH_PERFORMANCE=1
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "cerebras/DeepSeek-V3.2-REAP-345B-A37B",
    revision="4fd8e8c3e08442c4a6dde6dd3fa3dac481a0205b",
    local_dir="/models/cerebras/DeepSeek-V3.2-REAP-345B-A37B",
)
PY
```

Use a Hugging Face token in the shell environment if the repository requires authentication.

## Deploy

```bash
kubectl apply -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
kubectl get dgd,dcd,pods -n dynamo-system
kubectl logs -n dynamo-system -l app.kubernetes.io/name=deepseek-v32-reap-sglang --all-containers --tail=200
```

The worker launches SGLang with TP + DP on one eight-GPU node:

```bash
python3 -m dynamo.sglang \
  --model-path /models/cerebras/DeepSeek-V3.2-REAP-345B-A37B \
  --served-model-name cerebras/DeepSeek-V3.2-REAP-345B-A37B \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention
```

The full manifest also passes the DeepSeek V3 reasoning/tool parser flags and the REAP benchmark environment used by the `ai-blaise/infrastructure` SGLang REAP harness.

## Smoke Test

Forward the frontend service and send a minimal OpenAI-compatible request:

```bash
kubectl port-forward -n dynamo-system svc/deepseek-v32-reap-sglang-frontend 8000:8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "cerebras/DeepSeek-V3.2-REAP-345B-A37B",
    "messages": [{"role": "user", "content": "Say: dynamo-ready"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Cleanup

```bash
kubectl delete -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
```
