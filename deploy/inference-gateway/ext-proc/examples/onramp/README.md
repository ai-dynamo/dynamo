<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Router on-ramp (raw vLLM workers, no Dynamo control plane)

This directory is the **smallest possible** way to put Dynamo's KV/load-aware
router in front of an existing **vanilla `vllm serve`** fleet that already sits
behind a Gateway-API gateway with the Inference Extension (GAIE).

If you already run GAIE + vLLM, you **replace only your EndpointPicker (EPP)**
with the Dynamo EPP and you are done — no Dynamo operator, no Dynamo runtime,
no Dynamo worker.

| | This on-ramp (router-only mode) | Full Dynamo (full-dynamo-stack mode) |
|---|---|---|
| Workers | stock `vllm serve` | Dynamo workers (vLLM/SGLang/TRT-LLM) |
| Discovery | K8s pod reflector + label selector | Dynamo discovery (etcd/K8s) |
| KV events | per-pod vLLM ZMQ → EPP wrapper | Dynamo event plane |
| Runtime | none, `DYN_EVENT_PLANE=zmq` | etcd + NATS |
| Operator | not required | required (DynamoGraphDeployment) |
| Tokenization | EPP tokenizes for routing; worker re-tokenizes | model-card preprocessor, no double tokenization |

The EPP can be served in 2 modes. The two modes are selected at startup by **`DYN_EPP_MODE`** (`full-dynamo-stack` | `router-only`). The same EPP binary serves both.

## Limitations

| Concern | Router-only mode (gap vs full Dynamo) |
|---|---|
Duplicate store/remove (vLLM "retries") | parity
In-stream ordering |  parity
Transient disconnects | parity-ish. If the EPP's connection to a worker drops for a moment (a brief network blip or a quick pod restart), it reconnects on its own and keeps routing; the only catch is that any KV-cache updates the worker sent during that short gap are missed (ZMQ doesn't replay them), so the router's view is briefly a little out of date until normal traffic refreshes it — almost the same as the full stack, which can replay those missed updates.
Dropped events / gaps | Not handled. (NATS/JetStream is durable + replayable; the standalone indexer does seq-watermark gap detection + replay.)
Initial cache state | Not handled. (The Dynamo path does an initial worker dump).
Backpressure / EPP restart | PUB drops to slow subscribers (HWM) → silent loss; on restart the index is empty and only re-warms from new traffic.
Data Parallelism | Not supported

## Examples

If you are starting from scratch install the [Prerequisites](#1-prerequisites) and apply the provided example file:

- `agg.yaml` — aggregated: one vLLM pool, KV-aware load balancing.
- `disagg.yaml` — disaggregated: prefill + decode pools, KV-aware P/D selection. Note that for the disaggregated serving you need to also build the sidecar to orchestrate the pull of the kv cache from the prefill worker.

```bash
kubectl apply -n <ns> -f agg.yaml        # or disagg.yaml
```

Otherwise follow the steps below.

## Example Progression from vLLM to vLLM + Dynamo + GAIE

### Initial Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-qwen
  template:
    metadata:
      labels:
        app: vllm-qwen
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
          ports:
            - name: http
              containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-qwen
spec:
  selector:
    app: vllm-qwen
  ports:
    - name: http
      port: 8000
      targetPort: 8000
```

### 1. Prerequisites

Set up a Gateway-API gateway + Inference Extension (GAIE) if you don't have one
yet. Follow the instructions for your gateway (e.g. [AgentGateway](https://agentgateway.dev/docs/kubernetes/main/quickstart/install/)) and [Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/guides/).

We provide a convenience script if you are installing from scratch for experimentation.
```bash
# Gateway API + Inference Extension CRDs + a gateway named `inference-gateway`
deploy/inference-gateway/scripts/install_gaie_crd_agentgateway.sh
```

Create the HuggingFace token secret the EPP uses to download the
tokenizer.

```bash
# HF token secret (the EPP downloads the tokenizer to tokenize for routing)
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<your-token>
```

### 2. Create / Wire the InferencePool and HTTPRoute

Now that Gateway and GAIE is installed, we can create an InferencePool and an accompanying Dynamo EPP.

Point the `InferencePool` at your vLLM workers (see the
[`qwen-pool` InferencePool in `agg.yaml`](./agg.yaml#L233) for a complete example):

Specifically, these fields depend on the model you deploy, so make sure the settings below are adjusted to match your workers.

- `spec.selector` — labels matching your vLLM pods.
```yaml
  selector:
    matchLabels:
      app: vllm-qwen
```

- `spec.targetPorts[].number` — the vLLM serving port (usually `8000`).
```yaml
  targetPorts:
    - number: 8000
```

- `spec.endpointPickerRef` (a.k.a. `spec.extensionRef`) — the EPP service + port.
```yaml
  endpointPickerRef:
    kind: Service
    name: qwen-epp
    port:
      number: 9002
```

Attach the `HTTPRoute` to the gateway and target the pool (see the
[`qwen-route` HTTPRoute in `agg.yaml`](./agg.yaml#L253) for a complete example):

- `spec.rules[].backendRefs[]` — targets the `InferencePool`.
```yaml
    - backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: qwen-pool
          port: 8000
          weight: 1
```

- `spec.parentRefs[]` — your gateway (set `namespace` + `sectionName` when the
  gateway lives in another namespace, e.g. `agentgateway-system`).
```yaml
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: inference-gateway
```

### 3. Update vLLM deployment

Add labels to your vLLM pods that match `InferencePool.spec.selector` (and the
EPP's `DYN_EPP_POD_SELECTOR`):

```yaml
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
```

or
```bash
kubectl -n <ns> label deployment <vllm-deployment> app=vllm-qwen --overwrite
```

### 4. Add the EPP to your deployment

Remove the vLLM router (if you have it already installed) and replace it with the EPP (Dynamo Router).
Add the EPP as a `Deployment` + `Service` (see the
[`qwen-epp` Deployment in `agg.yaml`](./agg.yaml#L109) for a complete example) and set:

```yaml
kind: Deployment
metadata:
  name: qwen-epp
  labels:
    app: qwen-epp
```

1. **Set the EPP image.** The Dynamo EPP ships as the "frontend" image with each
   release. (Disaggregated serving also needs the "sidecar" image — use the
   released one or build it, see step 5.)

2. **Point the EPP at the workers' KV-event socket.** The worker publishes KV
   events via `--kv-events-config`; the EPP subscribes on the matching port.

   vLLM worker arg:

   ```text
   --kv-events-config '{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557"}'
   ```

   EPP env:

   ```yaml
   - name: DYN_EPP_KV_EVENT_PORT
     value: "5557"
   ```

3. **Set the model/tokenizer.** The EPP downloads this tokenizer (HF id) to
   tokenize prompts for routing — it MUST match the model the workers serve:

   ```yaml
   - name: DYN_MODEL_NAME
     value: "Qwen/Qwen3-0.6B"
   ```

4. **Match the ZMQ topic.** Must equal the worker's `--kv-events-config` topic
   (default `""`):

   ```yaml
   - name: DYN_EPP_KV_EVENT_TOPIC
     value: ""
   ```

5. **(Optional) Tune KV routing:**

   ```yaml
   # Subscribe to per-pod ZMQ KV events; turn off to fall back to load-only routing.
   - name: DYN_EPP_KV_EVENTS
     value: "true"
   # How heavily prefix-cache (KV) overlap counts vs. load.
   - name: DYN_OVERLAP_SCORE_WEIGHT
     value: "1.0"
   ```

   See the full list in the
   [environment contract](#router-only-mode-environment-contract) below.

In the end your Final vLLM deployment will look like below:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-qwen
  template:
    metadata:
      labels:
        app: vllm-qwen
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
            - "--served-model-name"     # NEW: pin the OpenAI model id
            - "Qwen/Qwen3-0.6B"
            - "--port"                  # NEW: explicit (8000 is vLLM's default)
            - "8000"
            - "--enable-prefix-caching" # NEW: required so vLLM emits prefix KV events
            - "--block-size"            # NEW: MUST equal the EPP's DYN_KV_CACHE_BLOCK_SIZE
            - "16"
            - "--kv-events-config"      # NEW: publish KV events on a ZMQ PUB socket for the EPP
            - '{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557"}'
          ports:
            - name: http
              containerPort: 8000
            - name: kv-events           # NEW: KV-event PUB port the EPP subscribes to
              containerPort: 5557
          env:                          # NEW: HF token lets the worker pull the model (required for gated models)
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
          resources:
            limits:
              nvidia.com/gpu: "1"
      tolerations:
        - effect: NoSchedule
          key: nvidia.com/gpu
          operator: Exists
```

### 5. (Disaggregated only) Build the P/D routing sidecar image

The sidecar is a standalone Rust crate at `deploy/inference-gateway/pd-sidecar/`.
It is a workspace member, so build it with the **repo root as the `dynamo` build
context** (the 2-stage Dockerfile copies the workspace, then compiles only
`dynamo-pd-sidecar`):

```bash
cd deploy/inference-gateway/pd-sidecar
docker buildx build \
  --build-context dynamo=../../.. \
  -t dynamo/pd-sidecar:dev \
  -f Dockerfile --load .

# make it reachable by your cluster, e.g. for kind:
kind load docker-image dynamo/pd-sidecar:dev --name <cluster>
```

Then set the `pd-router-sidecar` container's `image:` in `disagg.yaml` to the tag
you built. The sidecar reads the `x-prefiller-host-port` header the EPP emits and
runs vLLM's `kv_transfer_params` handshake against the selected prefill pod.


## Test

```bash
# terminal 1
kubectl -n agentgateway-system port-forward svc/inference-gateway 8000:80

# terminal 2
GATEWAY_URL=http://localhost:8000

# quick health/smoke first
curl --max-time 20 -sS "$GATEWAY_URL/v1/models" | jq .

# then chat completion
curl --max-time 120 -sS "$GATEWAY_URL/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role":"user","content":"hello"}]
  }' | jq .

```

## router-only-mode environment contract

| Env | Meaning |
|---|---|
| `DYN_EPP_MODE=router-only` | Select router-only (on-ramp) mode; `full-dynamo-stack` (default) keeps Dynamo mode |
| `DYN_EPP_POD_SELECTOR` | Label selector for raw vLLM pods (reflector) |
| `DYN_EPP_TARGET_PORT` | vLLM OpenAI HTTP port (pool targetPort) |
| `DYN_EPP_KV_EVENTS=true` | Subscribe to per-pod vLLM ZMQ KV events |
| `DYN_EPP_KV_EVENT_PORT` | vLLM `--kv-events-config` PUB port (e.g. 5557) |
| `DYN_EPP_KV_EVENT_TOPIC` | vLLM `--kv-events-config` topic (`""` default) |
| `DYN_MODEL_NAME` | Model id (no model card in router-only mode) |
| `DYN_KV_CACHE_BLOCK_SIZE` | MUST equal vLLM `--block-size` |
| `DYN_DISCOVERY_BACKEND=mem` | Inert runtime — no etcd |
| `DYN_EVENT_PLANE=zmq` | Inert runtime — no NATS |
| `DYN_EPP_ROLE_LABEL` | (disagg) pod label key splitting prefill/decode |
| `DYN_ENFORCE_DISAGG` | (disagg) fail instead of agg fallback |
| `DYN_EPP_EMIT_PREFILLER_HOST_PORT` | (disagg) emit `x-prefiller-host-port` |

## How KV events reach the router without NATS

vanilla vLLM publishes native KV-cache events on a ZMQ PUB socket
(`--kv-events-config`). In router-only mode the EPP runs an in-process **ZMQ
wrapper**: it subscribes to each Ready pod's PUB socket, normalizes the events,
and feeds them into the embedded `KvRouter`'s index over the runtime's ZMQ event
plane.
