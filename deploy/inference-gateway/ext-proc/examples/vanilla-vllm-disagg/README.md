<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vanilla-vLLM P/D disaggregation behind the native EPP (no Dynamo runtime)

This example runs **disaggregated prefill/decode (P/D) serving with stock
`vllm serve`** — no Dynamo runtime (no etcd/NATS, no `dynamo.vllm` worker) —
where the native Dynamo EPP makes the routing decision and a decode-side P/D
routing sidecar executes vLLM's KV-transfer handshake.

```
client ─▶ gateway ─(ext_proc)─▶ native EPP
                │  selects a decode worker + a prefill worker (KV-aware),
                │  emits x-prefiller-host-port=<prefill pod ip:8000>
                ▼
        decode pod:  routing sidecar :8000 ─▶ local decode vLLM :8001
                          │ 1) prefill request to x-prefiller-host-port
                          ▼
        prefill pod:  vLLM :8000  ─(NIXL KV transfer)─▶ decode vLLM
```

The EPP only **selects** (which decode worker, which prefill worker) and emits
the prefill endpoint as a request header. The sidecar **executes** vLLM's
mandatory two-step (`kv_transfer_params`) handshake; the KV cache moves
prefill→decode over NIXL between the two GPUs.

> The disaggregation routing sidecar container used in this example is the
> llm-d routing sidecar (`ghcr.io/llm-d/llm-d-routing-sidecar`). Any sidecar
> that reads the `x-prefiller-host-port` prefill-endpoint header works; the EPP
> is sidecar-agnostic.

## Files

* `prefill-decode.yaml` — 1 prefill pod + 1 decode pod (sidecar + decode vLLM).
  Both run stock `vllm serve` with `--kv-transfer-config NixlConnector` on an
  image that ships NIXL/UCX. Labeled `app=epp-vanilla-vllm` (the InferencePool
  selector) and `nvidia.com/dynamo-component-type=prefill|decode` (the EPP's
  role partition).
* `scale-2p2d.yaml` — a second prefill + second decode; the EPP discovers them
  at runtime via its pod reflector (no restart).
* `run-epp-disagg.sh` — launches the EPP in external mode with role-based P/D
  partitioning and enforce-disagg.
* `loadtest.sh` — functional + concurrency test through the gateway.

## Prerequisites

1. An `InferencePool` selecting `app=epp-vanilla-vllm`, `targetPorts: [8000]`,
   `endpointPickerRef` → the EPP service, and an `HTTPRoute` attaching it to
   your gateway (see the `vanilla-vllm/` example for the aggregated wiring).
2. The EPP binary built with the disagg prefill header (`x-prefiller-host-port`).
3. A P/D routing sidecar image (see note above).

## Required environment specifics (and why)

1. **GPU node taint toleration.** GPU nodes are tainted
   `nvidia.com/gpu=true:NoSchedule`; the pods tolerate **only** that key — not
   per-reservation taints — so they land on shared GPU nodes and avoid reserved
   ones.
2. **UCX transport.** vLLM's NixlConnector defaults to the RDMA/RoCE verbs path;
   without RDMA device-plugin resources in the pod, KV registration fails
   (`ibv_create_ah … No such device`). Setting
   `UCX_TLS=tcp,cuda_copy,cuda_ipc,sm,self` + `UCX_NET_DEVICES=eth0` forces the
   TCP+CUDA path (for production, give the pods real RoCE resources instead).
3. **NIXL-capable image.** Stock `vllm/vllm-openai` does not ship NIXL/UCX; this
   example uses `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0` but invokes the
   **vanilla `vllm serve`** entrypoint — no Dynamo runtime is involved.

## Run

```bash
kubectl apply -f prefill-decode.yaml
# optional scale: kubectl apply -f scale-2p2d.yaml
bash run-epp-disagg.sh 8790        # in the EPP pod: tokenizer sidecar + disagg EPP

curl -s -X POST "http://<gateway>/v1/completions" -H "Host: epp-vanilla.local" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16}'
```

Confirm real disaggregation in the decode-pod vLLM log
(`KV Transfer metrics: Num successful transfers=N …`) and a prefill dispatch
(a `POST /v1/completions` from the decode pod's IP) in the prefill-pod log.
