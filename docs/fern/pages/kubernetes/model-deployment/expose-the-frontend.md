---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Expose the Frontend
subtitle: Route external traffic to a DynamoGraphDeployment's Frontend with a Kubernetes Ingress, a LoadBalancer Service, a Gateway API HTTPRoute to the Frontend Service, or the Inference Gateway (GAIE).
---

The [DGD Guide](deploy-with-dgd.md) reaches the Frontend with `kubectl port-forward`, which is fine for a smoke test but not for production traffic. This page shows how to give the Frontend a stable external address. Whatever the API version, the operator creates a `ClusterIP` Service named `<name>-frontend` (where `<name>` is `metadata.name`) on port 8000 for the Frontend component — the options below route external traffic to that Service. Choose one of four paths depending on what your cluster runs.

| Path | Use when |
|---|---|
| Kubernetes Ingress | You have an ingress controller (NGINX, etc.) and want a host-routed HTTP(S) entry point |
| LoadBalancer Service | You want a cloud load balancer and manage routing yourself |
| Gateway API HTTPRoute to the Frontend Service | Your cluster runs a Kubernetes Gateway, with or without the Gateway API Inference Extension |
| Inference Gateway (GAIE) | The Gateway must pick the worker for each request. GAIE routing does not have full feature parity with Dynamo Frontend routing |

## Option 1: A Kubernetes Ingress

Point a standard Kubernetes [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/) at the operator-created `<name>-frontend` Service. This requires an ingress controller (NGINX, etc.) already installed in the cluster:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dynamo-frontend
  annotations:
    nginx.ingress.kubernetes.io/backend-protocol: HTTP
spec:
  ingressClassName: nginx
  rules:
  - host: dynamo.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-agg-frontend       # <name>-frontend from your DGD
            port:
              number: 8000
  tls:
  - hosts:
    - dynamo.example.com
    secretName: dynamo-tls
```

The operator creates the Ingress object; the controller provisions the actual address.

> [!NOTE]
> The `nvidia.com/v1alpha1` API had a convenience `ingress` field on the Frontend component that generated this Ingress (or a service-mesh VirtualService) for you. That field is **not** part of `nvidia.com/v1beta1` — author the Ingress directly as shown above, or use a LoadBalancer Service (Option 2). See the [API Reference](../../reference/kubernetes-api/full-api-reference.mdx#ingressspec) for the v1alpha1 field.

## Option 2: A LoadBalancer Service

To skip an Ingress and use a cloud load balancer directly, expose the Frontend Service as `type: LoadBalancer`:

```bash
kubectl expose deployment <name>-frontend \
  --type=LoadBalancer --port=8000 --target-port=8000 \
  -n <namespace>
```

```bash
kubectl get svc <name>-frontend -n <namespace> -w   # wait for EXTERNAL-IP
```

Send requests to the external IP on port 8000.

## Option 3: A Gateway API HTTPRoute to the Frontend Service

To put the Frontend behind a Kubernetes [Gateway](https://gateway-api.sigs.k8s.io/), attach an `HTTPRoute` to the Gateway and point it at the operator-created `<name>-frontend` Service. The Frontend still selects the worker for each request, so the deployment keeps every Dynamo routing and request-handling feature. The route needs no Gateway API Inference Extension (GAIE) resources, so it works whether or not GAIE is installed.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: dynamo-frontend
spec:
  parentRefs:
  - name: my-gateway                # your Gateway
  rules:
  - backendRefs:
    - name: vllm-agg-frontend       # <name>-frontend from your DGD
      port: 8000
    timeouts:
      request: 300s
```

Create the route in the DGD namespace, next to the Frontend Service. If the Gateway is in another namespace, add its `namespace` to the `parentRefs` entry and confirm that the Gateway listener allows routes from the DGD namespace. Set `timeouts.request` longer than your slowest expected response; a Gateway implementation's default timeout can cut off long generations.

To serve several deployments behind one Gateway, create an `HTTPRoute` for each Frontend Service and give each route a distinct hostname or header match.

## Option 4: The Inference Gateway (GAIE)

[Using GAIE with Dynamo](../kv-aware-routing/gateway-api.mdx) is a different mechanism: the `HTTPRoute` points at an `InferencePool` instead of the Frontend Service, and the Gateway asks the Dynamo Endpoint Picker Plugin (EPP) to pick a worker for each request and forwards the request to that worker. A Frontend sidecar in each worker pod runs with `--router-mode direct` and respects the EPP's choice.

The Dynamo EPP does not support every Dynamo feature; see [Features Not Supported by the Dynamo EPP](../kv-aware-routing/overview.md#features-not-supported-by-the-dynamo-epp). Prefer Option 3 unless your platform requires the Gateway itself to pick the worker.

> [!NOTE]
> A Kubernetes Ingress and a Gateway are independent. An Ingress (Option 1) exposes the Frontend Service without Gateway API; Options 3 and 4 use a Gateway and do not use an Ingress.

## Related pages

- [KV-Aware Routing on Kubernetes](../kv-aware-routing/overview.md) — choose between Dynamo Frontend routing and GAIE routing.
- [Using GAIE with Dynamo](../kv-aware-routing/gateway-api.mdx) — route requests through the Dynamo EPP.
- [API Reference — IngressSpec](../../reference/kubernetes-api/full-api-reference.mdx#ingressspec) — full field reference.
