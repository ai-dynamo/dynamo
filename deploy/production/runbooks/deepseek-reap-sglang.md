<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek REAP SGLang

This runbook validates `BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1` on the Dynamo production Kubernetes profile with the SGLang backend from `ai-blaise/optimization-playground`.

The active topology runs on one `a4-us-001-rl9` node with eight allocatable B200 GPUs: four GPUs for prefill and four GPUs for decode. The production profile uses the combined HiSparse, IndexCache, and dense TurboQuant path from `ai-blaise/optimization-playground`:

- target checkpoint: W4A4KV4 NVFP4 through the checkpoint's `compressed-tensors` quantization metadata
- target KV source dtype: BF16, with dense TurboQuant 2.5-bit compressed MLA storage
- decode-side HiSparse via `--enable-hisparse`
- IndexCache via `--nsa-indexer-mode indexcache`
- dense TurboQuant via `--enable-turboquant-dense-kv-cache --turboquant-dense-kv-preset latent_2p5bit_nc`
- DSA sparse attention backends via `--nsa-prefill-backend flashmla_sparse` and `--nsa-decode-backend flashmla_sparse`
- no SGLang HiCache in this profile because HiSparse requires the decode no-radix path
- no LayerSplit flags in this 4+4 DP=4 profile because the prefill worker does not have effective attention CP size greater than 1
- Dynamo event-backed KV-aware routing via frontend `--router-mode kv --router-kv-events` and worker `--kv-events-config`
- Dynamo-native chat preprocessing via frontend `--dyn-chat-processor dynamo` and worker parser flags `--dyn-tool-call-parser deepseek_v3_2 --dyn-reasoning-parser deepseek_r1`
- prefill: `--disaggregation-mode prefill`, `--dp 4`, `--tp 4`, DP attention enabled
- decode: `--disaggregation-mode decode`, `--dp 4`, `--tp 4`, DP attention enabled, radix cache disabled as required by HiSparse
- SMC-SD draft on decode only: `BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP`, FP8 draft KV, CUTLASS draft FP8 GEMM

Compatibility note: SGLang documents HiSparse as a decode-side DSA/PD feature
that keeps a hot GPU KV buffer and complete CPU pinned-memory KV, while HiCache
is documented as a RadixAttention/HiRadixTree prefix-KV reuse system. This
profile chooses HiSparse when HiSparse and HiCache conflict, but keeps
IndexCache and dense TurboQuant through the combined NSA KV-pool implementation.

Activation artifact note: the current target checkpoint stores packed NVFP4
weights and weight scales, but the observed `compressed-tensors` metadata has
`input_activations: null` and no serialized input activation scale tensors. Do
not treat the W4A4 manifest as fully launchable until the activation
quantization metadata is attached or generated offline. The deployment scripts
preserve the intended W4A4KV4 contract; they do not synthesize activation scales
at runtime.

## Production Profile

Use the production profile in this repository as the Kubernetes layer. The infrastructure entry point wraps these steps and should be preferred:

```bash
scripts/dynamo-reap/deploy-a4-production.sh
```

The script applies the full `deploy/production` GitOps stack, including baseline add-ons and optional production integrations, then renders and applies the REAP `DynamoGraphDeployment`.
The infrastructure wrapper treats `opentelemetry-operator` as deployable when it
is Healthy but OutOfSync, because the chart's webhook certificates and CRDs can
be controller-mutated after sync. Other production applications still need to
satisfy the Synced/Healthy gate.

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
/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1
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
  BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1 \
  --local-dir /models/BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1

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

The production path should use a registry image and let Kubernetes pull it. The
A4 k3s deployment wrapper also supports a validation fallback: if the registry
tag is unavailable, it can build a local overlay image from the
`ai-blaise/optimization-playground` checkout and import that image into k3s
containerd. That fallback is tracked by Docker image ID so unchanged repeat
launches do not pay the `docker save | k3s ctr images import` cost. The fallback
overlay must keep the engine's required `sglang-kernel` version and should not
downgrade `apache-tvm-ffi` below FlashInfer's supported range.

## Deploy

```bash
kubectl apply -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
kubectl get dgd,dgdr,dgdsa,dm,pods -n dynamo-system
kubectl logs -n dynamo-system -l app.kubernetes.io/name=deepseek-v32-reap-sglang --all-containers --tail=200
```

The frontend starts with event-backed KV routing and Dynamo-native chat preprocessing:

```bash
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-kv-events \
  --router-reset-states \
  --dyn-chat-processor dynamo \
  --http-port 8000
```

The workers launch SGLang through Dynamo with the core stack enabled:

```bash
python3 -m dynamo.sglang \
  --model-path /models/BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1 \
  --served-model-name BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1 \
  --quantization compressed-tensors \
  --kv-cache-dtype bfloat16 \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_sparse \
  --nsa-indexer-mode indexcache \
  --nsa-indexcache-pattern FSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSF \
  --enable-turboquant-dense-kv-cache \
  --turboquant-dense-kv-preset latent_2p5bit_nc \
  --turboquant-execution-mode fused_decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 12345 \
  --disaggregation-mode prefill|decode \
  --dyn-tool-call-parser deepseek_v3_2 \
  --dyn-reasoning-parser deepseek_r1
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

The `bfloat16` KV dtype is intentional: dense TurboQuant quantizes BF16 MLA KV
rows into 2.5-bit compressed storage, while HiSparse keeps the decode hot set and
host pool in that compressed row format. Do not add `--enable-hierarchical-cache`
to this production profile without revalidating the HiSparse no-radix contract.

## Smoke Test

Forward the frontend service and send a minimal OpenAI-compatible request:

```bash
kubectl port-forward -n dynamo-system svc/deepseek-v32-reap-sglang-frontend 8000:8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1",
    "messages": [{"role": "user", "content": "Say: dynamo-ready"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Cleanup

```bash
kubectl delete -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
```
