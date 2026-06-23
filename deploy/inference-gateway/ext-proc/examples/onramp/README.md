<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Router on-ramp (raw vLLM, no Dynamo control plane)

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

| This on-ramp (router-only mode) | Full Dynamo (full-dynamo-stack mode) |
|---|---|
Duplicate store/remove (vLLM "retries") | parity
In-stream ordering |  parity
Transient disconnects | parity-ish. If the EPP's connection to a worker drops for a moment (a brief network blip or a quick pod restart), it reconnects on its own and keeps routing; the only catch is that any KV-cache updates the worker sent during that short gap are missed (ZMQ doesn't replay them), so the router's view is briefly a little out of date until normal traffic refreshes it — almost the same as the full stack, which can replay those missed updates.
Dropped events / gaps | Not handled. (NATS/JetStream is durable + replayable; the standalone indexer does seq-watermark gap detection + replay.)
Initial cache state | Not handled. (The Dynamo path does an initial worker dump).
Backpressure / EPP restart | PUB drops to slow subscribers (HWM) → silent loss; on restart the index is empty and only re-warms from new traffic.
Data Parallelism | Not supported

## Examples

- `agg.yaml` — aggregated: one vLLM pool, KV-aware load balancing.
- `disagg.yaml` — disaggregated: prefill + decode pools, KV-aware P/D selection. Note that for the disaggregated serving you need to also build the sidecar to orchestrate the pull of the kv cache from the prefill worker.

## Prerequisites

```bash
# Gateway API + Inference Extension CRDs + a gateway named `inference-gateway`
deploy/inference-gateway/scripts/install_gaie_crd_agentgateway.sh

# HF token secret (the EPP downloads the tokenizer to tokenize for routing)
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<your-token>
```
The Dynamo EPP image is named "frontend" and provided to you with each release.
For disaggreaged serving you also need to use the "sidecar" image provided with the release or build it with commands below.

### Build the P/D routing sidecar image (disaggregated only)

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


## Run

```bash
kubectl apply -n <ns> -f agg.yaml        # or disagg.yaml, adjust httpRoute there per your setup

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
