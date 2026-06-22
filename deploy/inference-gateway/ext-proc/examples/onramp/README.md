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
no Dynamo worker. For disaggreaged serving you also need to build the disag sidecar.

| | This on-ramp (router-only mode) | Full Dynamo (full-dynamo-stack mode) |
|---|---|---|
| Workers | stock `vllm serve` | Dynamo workers (vLLM/SGLang/TRT-LLM) |
| Discovery | K8s pod reflector + label selector | Dynamo discovery (etcd/K8s) |
| KV events | per-pod vLLM ZMQ → EPP wrapper | Dynamo event plane |
| Runtime | none, `DYN_EVENT_PLANE=zmq` | etcd + NATS |
| Operator | not required | required (DynamoGraphDeployment) |
| Tokenization | EPP tokenizes for routing; worker re-tokenizes | model-card preprocessor, no double tokenization |

The two modes are selected at startup by **`DYN_EPP_MODE`** (`full-dynamo-stack` | `router-only`). The same EPP
binary serves both.

LIMITATIONS:

Concern | Gap vs the Dynamo/NATS path
-- | --
Duplicate store/remove (vLLM "retries") | parity
In-stream ordering |  parity
Transient disconnects | parity-ish
Dropped events / gaps | Not handled. (NATS/JetStream is durable + replayable; the standalone indexer does seq-watermark gap detection + replay.)
Initial cache state | Not handled. (The Dynamo path does an initial worker dump).
Backpressure / EPP restart | PUB drops to slow subscribers (HWM) → silent loss; on restart the index is empty and only re-warms from new traffic.
Data Parallelism | Not supported

## Files

- `agg.yaml` — aggregated: one vLLM pool, KV-aware load balancing.
- `disagg.yaml` — disaggregated: prefill + decode pools, KV-aware P/D selection. Note that for the disaggregated serving you need to also build the sidecar to orchestrate the pull of the kv cache from the prefill worker.

## Prerequisites (install once)

```bash
# Gateway API + Inference Extension CRDs + a gateway named `inference-gateway`
deploy/inference-gateway/scripts/install_gaie_crd_agentgateway.sh

# HF token secret (the EPP downloads the tokenizer to tokenize for routing)
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<your-token>
```

## Run

```bash
kubectl apply -n <ns> -f agg.yaml        # or disagg.yaml
GW=$(kubectl get gateway inference-gateway -o jsonpath='{.status.addresses[0].value}')
curl http://$GW/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role":"user","content":"hello"}]
}'
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
