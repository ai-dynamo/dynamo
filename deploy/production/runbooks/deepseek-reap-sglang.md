<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek REAP SGLang

This runbook validates `BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1` on the Dynamo production Kubernetes profile with the SGLang backend from `ai-blaise/optimization-playground`.

The active topology runs on one `a4-us-001-rl9` node with eight allocatable B200 GPUs: four GPUs for prefill and four GPUs for decode. The production profile is HiSparse-first because current SGLang treats HiSparse, HiCache, IndexCache, and dense TurboQuant as separate KV-cache paths:

- target checkpoint: W4A4KV4 NVFP4 via `--quantization modelopt_fp4`
- target KV exception: BF16 target KV storage for HiSparse
- decode-side HiSparse via `--enable-hisparse`
- DSA sparse attention backends via `--nsa-prefill-backend flashmla_sparse` and `--nsa-decode-backend flashmla_sparse`
- no SGLang HiCache, IndexCache, or dense TurboQuant in this profile unless a combined implementation is deliberately added and revalidated
- Dynamo event-backed KV-aware routing via frontend `--router-mode kv --router-kv-events` and worker `--kv-events-config`
- prefill: `--disaggregation-mode prefill`, `--dp 4`, `--tp 4`, DP attention enabled
- decode: `--disaggregation-mode decode`, `--dp 4`, `--tp 4`, DP attention enabled, radix cache disabled as required by HiSparse
- SMC-SD draft on decode only: `BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP`, FP8 draft KV, CUTLASS draft FP8 GEMM

Compatibility note: SGLang documents HiSparse as a decode-side DSA/PD feature
that keeps a hot GPU KV buffer and complete CPU pinned-memory KV, while HiCache
is documented as a RadixAttention/HiRadixTree prefix-KV reuse system. Current
SGLang validation mirrors that split, so this production profile chooses
HiSparse when these paths conflict. See the SGLang HiSparse guide and HiCache
design docs before reintroducing HiCache or dense KV compression here.

## Production Profile

Use the production profile in this repository as the Kubernetes layer. The infrastructure entry point wraps these steps and should be preferred:

```bash
scripts/dynamo-reap/deploy-a4-production.sh
```

The script applies the full `deploy/production` GitOps stack, including baseline add-ons and optional production integrations, then renders and applies the REAP `DynamoGraphDeployment`.

Manual bootstrap is:

```bash
kubectl apply -f deploy/production/gitops/project.yaml
kubectl apply -f deploy/production/gitops/root-app.yaml
kubectl apply -f deploy/production/gitops/optional/keda.yaml
kubectl apply -f deploy/production/gitops/optional/opentelemetry.yaml
kubectl apply -f deploy/production/gitops/optional/actions-runner-controller.yaml
kubectl apply -f deploy/production/gitops/optional/parca.yaml
kubectl apply -f deploy/production/gitops/optional/volcano.yaml
kubectl apply -f deploy/production/gitops/optional/lws.yaml
deploy/pre-deployment/pre-deployment-check.sh --profile production
deploy/pre-deployment/pre-deployment-check.sh --require dynamo-crds,dynamo-webhooks,kai-queue
```

The GitOps manifests read from `https://github.com/ai-blaise/dynamo-prod-k8s.git` on `main`.

## Model Cache

Run the download on every node that can schedule the worker pod. The default manifest mounts the model from:

```text
/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1
```

and the SMC draft from:

```text
/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
```

The target model is private in the `BlaiseAI` organization, so the host environment must have an authenticated Hugging Face token.

```bash
export HF_TOKEN=...
export HF_XET_HIGH_PERFORMANCE=1

sudo mkdir -p /models/BlaiseAI /models/smcsd
sudo chown -R "$USER:$USER" /models

python3 -m venv ~/hf-download
source ~/hf-download/bin/activate
python -m pip install --upgrade pip
python -m pip install "huggingface_hub[hf_xet]>=0.36"

hf download \
  BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1 \
  --local-dir /models/BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1

hf download \
  BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP \
  --local-dir /models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
```

The pod sets `HF_HUB_OFFLINE=1`; model access failures should be fixed during host-side download, not at runtime.

## Engine Image

Build the runtime image from `ai-blaise/optimization-playground` after the remaining custom kernels are added. The manifest defaults to:

```text
ghcr.io/ai-blaise/optimization-playground-sglang-runtime:reap-nvfp4
```

Override this image from the infrastructure script with `DYNAMO_REAP_IMAGE` when testing a candidate image.

## Deploy

```bash
kubectl apply -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
kubectl get dgd,dgdr,dgdsa,dm,pods -n dynamo-system
kubectl logs -n dynamo-system -l app.kubernetes.io/name=deepseek-v32-reap-sglang --all-containers --tail=200
```

The worker launches SGLang through Dynamo with the core stack enabled:

```bash
python3 -m dynamo.sglang \
  --model-path /models/BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1 \
  --served-model-name BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype bfloat16 \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_sparse \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 12345 \
  --disaggregation-mode prefill|decode
```

Omit the HiSparse and SMC-SD flags on the prefill role. The manifest sets them
only on decode:

```bash
--disable-radix-cache \
--enable-hisparse \
--hisparse-config '{"top_k":2048,"device_buffer_size":6144,"host_to_device_ratio":10}' \
--speculative-algorithm SMC
```

SMC-SD draft KV is decode-local in this topology. The prefill/decode transfer
registers target-model KV; it does not register the decode-side draft KV pool
because the prefill worker does not instantiate the draft model.

The `bfloat16` KV dtype is intentional: the checkpoint is W4A4KV4 NVFP4, but
HiSparse currently requires BF16 target KV storage and `flashmla_sparse` NSA
backends.

Do not add `--enable-hierarchical-cache`, `--enable-turboquant-dense-kv-cache`,
or `--nsa-indexer-mode indexcache` to this production profile without first
implementing and validating a combined HiSparse KV path in
`ai-blaise/optimization-playground`.

## Smoke Test

Forward the frontend service and send a minimal OpenAI-compatible request:

```bash
kubectl port-forward -n dynamo-system svc/deepseek-v32-reap-sglang-frontend 8000:8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1",
    "messages": [{"role": "user", "content": "Say: dynamo-ready"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Cleanup

```bash
kubectl delete -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
```
