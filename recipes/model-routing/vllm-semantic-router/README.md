<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Semantic Router Mixture of Backends

This experimental recipe serves `nvidia/GLM-5.2-NVFP4` and `Qwen/Qwen3.5-122B-A10B-FP8` behind one endpoint. vLLM Semantic Router selects a model before Gateway API routing. The Dynamo Endpoint Picker Protocol (EPP) then selects a worker inside that model's `InferencePool`.

```text
client -> Envoy ext_proc -> vLLM Semantic Router -> HTTPRoute -> InferencePool -> Dynamo EPP -> backend graph
```

The recipe uses a standalone Envoy bridge because agentgateway v1.0 does not rematch an `HTTPRoute` after a request-body external processor selects a model. The bridge owns model selection only; agentgateway and Dynamo EPP retain Kubernetes endpoint selection, and Dynamo Frontend owns client protocol handling.

`x-selected-model` is internal routing metadata, not authentication. Envoy removes any client-supplied value before semantic routing. Keep the agentgateway service cluster-internal and restrict direct pod access with your cluster's network policy in a multi-tenant namespace.

The models deliberately use different serving topologies:

- Qwen is an aggregated TP8 vLLM deployment on one B200 node.
- GLM is a 1P/1D TP8 SGLang deployment on two B200 nodes. Dynamo EPP selects the prefill and decode endpoints, and SGLang transfers KV state with NIXL over RDMA.

The GLM workers follow SGLang's verified single-node B200 NVFP4 configuration: `modelopt_fp4`, TP8, 8K chunked prefill, and 0.85 static memory. The digest-pinned integration image combines current GLM-5.2 NVFP4 support with the Dynamo SGLang entry point; replace it with the next official Dynamo runtime that includes that support.

This split is intentional: the model router selects model capability, while each model's DGD independently selects the inference backend and serving topology. Replacing either DGD does not change the semantic router, Envoy, Gateway API routes, or the other model.

## Router Contract

A replacement global model router does not implement EPP. EPP is the inner contract between each `InferencePool` and its Dynamo worker graph. The outer router only needs to:

- inspect the request before Gateway API matching;
- select one configured model and set the trusted `x-selected-model` header; and
- replace the unresolved `auto` model with the selected model without changing the client wire protocol.

The HTTPRoutes map that header to model-specific `InferencePool` resources. A future Switchyard integration can therefore replace the semantic-router `ext_proc` service while preserving the Envoy trust boundary, routes, pools, EPP configuration, and worker graphs. If it uses a protocol other than Envoy `ext_proc`, only the bridge must change.

The semantic-router image used here adds `api_format: passthrough`. In that mode it preserves the original OpenAI or Anthropic body and response format and changes only `model`. The EPP uses GAIE's `passthrough-parser`; its Dynamo scorers use a minimal placeholder solely to resolve the required Dynamo worker IDs and do not inject placeholder tokens. The selected Dynamo Frontend parses and tokenizes the original request. This intentionally gives up prompt-aware EPP placement for opaque payloads.

## Relationship to Dynamo Routing and Planning

Model routing is independent from Dynamo's pool routing and capacity planning:

- vLLM Semantic Router selects the concrete model before model-specific tokenization.
- The `HTTPRoute` selects that model's `InferencePool`, and Dynamo EPP selects worker endpoints inside the pool.
- GlobalRouter is optional when one concrete model spans multiple Dynamo pools. This recipe has one pool per model and does not deploy GlobalRouter.
- GlobalPlanner reconciles pool capacity asynchronously. It does not select a model or participate in the request path.

Keep these stages separate: model selection chooses capability, pool routing chooses placement for one model, and planning changes future capacity.

## Prerequisites

- A Kubernetes cluster with three free 8x B200 nodes.
- RDMA devices exposed as `rdma/shared_ib` and UCX devices `mlx5_0:1` through `mlx5_3:1` on the GLM nodes.
- Dynamo operator v1.3.0 and agentgateway v1.0 installed with Gateway API Inference Extension support.
- Access to the pinned GLM runtime image (`Dynamo 1.3.0.dev20260628`, SGLang `aaa31eb0`) with ModelOpt NVFP4 support.
- A ReadWriteMany PVC named `shared-model-cache`.
- Secrets named `hf-token-secret` and `ngc-secret`.
- `git`, Helm 3, `kubectl`, `curl`, and `jq` on the client.

## Cache the Models

The job reuses files already present in `shared-model-cache`; it does not create a PVC.

```bash
export NAMESPACE=model-routing

kubectl apply -n "$NAMESPACE" -f model-cache.yaml
kubectl wait -n "$NAMESPACE" job/model-routing-cache \
  --for=condition=Complete --timeout=7200s
```

## Deploy

Install the v0.3.0 Helm chart. `semantic-router-values.yaml` pins the selection-only semantic-router image:

```bash
git clone --depth 1 --branch v0.3.0 \
  https://github.com/vllm-project/semantic-router.git /tmp/semantic-router-v0.3.0

helm dependency build \
  /tmp/semantic-router-v0.3.0/deploy/helm/semantic-router

helm upgrade --install semantic-router \
  /tmp/semantic-router-v0.3.0/deploy/helm/semantic-router \
  --namespace "$NAMESPACE" \
  --values semantic-router-values.yaml \
  --wait --timeout 30m

kubectl apply -n "$NAMESPACE" -f deploy.yaml
```

Wait for the two serving graphs and the gateway:

```bash
kubectl wait -n "$NAMESPACE" dgd/model-routing-small \
  --for=condition=Ready --timeout=3600s
kubectl wait -n "$NAMESPACE" dgd/model-routing-large \
  --for=condition=Ready --timeout=3600s
kubectl wait -n "$NAMESPACE" gateway/model-routing-gateway \
  --for=condition=Programmed --timeout=300s
kubectl rollout status -n "$NAMESPACE" deployment/model-routing-envoy \
  --timeout=300s
```

## Smoke Test

Forward the recipe endpoint:

```bash
kubectl port-forward -n "$NAMESPACE" service/model-routing-envoy 8000:80
```

Run the functional checks in another terminal:

```bash
./smoke.sh
```

The script checks automatic selection of both model pools, spoofed-header rejection, unknown-model failure, OpenAI streaming, and Anthropic Messages non-streaming and SSE streaming.

Confirm the two routing stages:

```bash
kubectl logs -n "$NAMESPACE" deployment/semantic-router --tail=200
kubectl logs -n "$NAMESPACE" \
  -l nvidia.com/dynamo-component-type=epp --tail=200
```

## Claude Code

Dynamo Frontend terminates the Anthropic Messages API; the semantic router and EPP preserve that protocol. Point Claude Code at the forwarded endpoint:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_MODEL=auto
export ANTHROPIC_SMALL_FAST_MODEL=auto
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export ANTHROPIC_API_KEY=local-dev-token

claude
```

Use `auto` for Anthropic clients so the semantic policy selects the backend. The selected model remains visible in the response and the semantic-router response headers and logs.

## Clean Up

```bash
kubectl delete -n "$NAMESPACE" -f deploy.yaml
kubectl delete -n "$NAMESPACE" -f model-cache.yaml
helm uninstall semantic-router -n "$NAMESPACE"
```
