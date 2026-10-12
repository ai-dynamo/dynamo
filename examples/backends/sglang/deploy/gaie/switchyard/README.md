<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard model routing with SGLang and Dynamo GAIE

Run Switchyard in a separate, single-replica PreProc. It chooses between
`Qwen/Qwen3.5-27B` and `Qwen/Qwen3.5-397B-A17B` served by SGLang; Dynamo's native EPP then selects
a worker within that model's pool. The example adds PreProc and gateway routing to an existing GAIE deployment.
See the [Switchyard PreProc example](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/examples/dynamo-preproc)
for the architecture diagram, service implementation and image build. This directory owns
the Kubernetes deployment and model-pool bindings.

## Prerequisites

Use an existing Kubernetes deployment with the Dynamo operator, Gateway API, GAIE, and the
**agentgateway 1.0.0 controller and CRDs** installed. Follow Dynamo's
[Gateway API installation guide](../../../../../../docs/fern/pages/kubernetes/installation/gateway-api-routing.mdx)
and its [agentgateway setup script](../../../../../../deploy/inference-gateway/scripts/install_gaie_crd_agentgateway.sh).
The script creates `inference-gateway` in `agentgateway-system`, with an `http` listener that
allows routes from workload namespaces. This add-on reuses that Gateway; cluster, operator,
GPU, and worker setup remain outside the example. Use the agentgateway option in the guide:
the PreProc policy below is specific to agentgateway.

The namespace must already contain ready model workers, native Dynamo EPPs, and two
`InferencePool` resources named `qwen-small-pool` and `qwen-large-pool`, serving `Qwen/Qwen3.5-27B`
and `Qwen/Qwen3.5-397B-A17B`. Use the topology in the
[SGLang aggregated GAIE deployment](../agg.yaml) when preparing those pools. That template
serves Qwen3.5-27B as `qwen-small` with one GPU. Size GPU memory and tensor parallelism for
your context length and hardware. For `qwen-large`, replace the model IDs with
`Qwen/Qwen3.5-397B-A17B` and adjust tensor parallelism and GPU resources accordingly. See the model cards for
[27B](https://huggingface.co/Qwen/Qwen3.5-27B) and
[397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) serving guidance.
Name the deployments `qwen-small` and `qwen-large`; the operator creates the corresponding
`-pool` resources. Use matching Dynamo SGLang runtime and frontend image tags, and keep the
worker page size consistent with the EPP cache block size.

Local end-to-end validation used Qwen3-0.6B and Qwen3-1.7B; the larger models have not been
validated with this example.

The SGLang GAIE template follows the native EPP and direct-mode frontend-sidecar topology of
the [vLLM GAIE example](../../../../vllm/deploy/gaie/agg.yaml), using SGLang worker settings.
The gateway and PreProc configuration
is backend-independent and also works with existing vLLM pools that expose the same interface.

The Kustomization deploys PreProc and its policy into the existing Gateway namespace,
`agentgateway-system`. Set that namespace in [kustomization.yaml](kustomization.yaml).
If your Gateway name or namespace differs, update its parent references in
[http-routes.yaml](http-routes.yaml), the policy target in
[preproc-policy.yaml](preproc-policy.yaml), and the gateway URL in [routes.toml](routes.toml).
Apply the HTTPRoutes separately in the namespace containing the model pools.

If model IDs or pool names differ, update the TOML targets, HTTPRoute `X-Gateway-Model-Name`
values, and pool references together. See Dynamo's
[GAIE routing guide](../../../../../../docs/fern/pages/kubernetes/kv-aware-routing/gateway-api.mdx)
for native EPP and pool configuration. Do not downgrade newer controller CRDs for this example.

The PreRouting policy applies to all HTTP requests on the selected Gateway. Use a Gateway
whose traffic is intended for Switchyard; other clients would also pass through PreProc.

You also need Docker, `kubectl` with Kustomize support, and a registry the cluster can pull from.

## Build and deploy

Build and publish the image from the
[Switchyard PreProc example](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/examples/dynamo-preproc#build-the-image).
Then, from the Dynamo repository root:

```bash
export EXAMPLE=examples/backends/sglang/deploy/gaie/switchyard
```

Set the PreProc registry image in [kustomization.yaml](kustomization.yaml). Then apply the add-on
to the existing Gateway and model namespaces:

```bash
export NAMESPACE=switchyard
export AGW_NAMESPACE=agentgateway-system
kubectl get -n "$NAMESPACE" inferencepool qwen-small-pool qwen-large-pool
kubectl wait -n "$AGW_NAMESPACE" gateway/inference-gateway \
  --for=condition=Programmed --timeout=180s
kubectl apply -k "$EXAMPLE"
kubectl rollout status -n "$AGW_NAMESPACE" deployment/switchyard-preproc --timeout=180s
kubectl apply -n "$NAMESPACE" -f "$EXAMPLE/http-routes.yaml"
kubectl get -n "$AGW_NAMESPACE" agentgatewaypolicy switchyard-preproc -o yaml
kubectl get -n "$NAMESPACE" httproute -o yaml
kubectl port-forward -n "$AGW_NAMESPACE" service/inference-gateway 8000:80
```

The policy must report `Accepted=True`. The HTTPRoutes must report `Accepted=True` and
`ResolvedRefs=True`. PreProc and its policy share the Gateway namespace; the HTTPRoutes
share the model pools' namespace.

## Verify both model choices

In a second terminal, send a neutral request. The supplied `efficient_first` policy selects
`Qwen/Qwen3.5-27B`:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

A critical tool failure selects `Qwen/Qwen3.5-397B-A17B`:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Fix the failure."},{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"pytest\"}"}}]},{"role":"tool","tool_call_id":"call-1","content":"MemoryError: out of memory"}],"max_tokens":16}'
```

Check the response's `model` field. Add `"stream":true` and use `curl --no-buffer` to verify SSE.

## Configure routing

Edit [routes.toml](routes.toml) to choose the routing policy, then reapply the
Kustomization. Set the request's `model` to a configured route ID, such as `auto`.

To route to another existing model pool, add its target and policy in the TOML and a matching
rule in [http-routes.yaml](http-routes.yaml), then reapply both the Kustomization and HTTPRoutes. See
[Switchyard routing configuration](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/examples/dynamo-preproc#configure-routing)
for policy restrictions and session headers.

Set `MAX_ACTIVE_STREAMS` in [preproc.yaml](preproc.yaml) and reapply the Kustomization. For the
gateway request timeout, edit and reapply [http-routes.yaml](http-routes.yaml). See
[concurrency and timeouts](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/examples/dynamo-preproc#concurrency-and-timeouts)
for the defaults, overload behavior and the distinction between gateway and PreProc timeouts.

PreProc accepts requests up to 2 MiB and admits up to 4,096 session identities active within
the past hour. This deployment has one replica; restarts and configuration updates interrupt
routing.

## Remove

Remove the model routes and PreProc add-on. The existing Gateway, model deployments and
pools remain:

```bash
kubectl delete -n "$NAMESPACE" -f "$EXAMPLE/http-routes.yaml"
kubectl delete -k "$EXAMPLE"
```
