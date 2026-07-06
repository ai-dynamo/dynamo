---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Expose the Frontend
subtitle: Route external traffic to a DynamoGraphDeployment's Frontend with a Kubernetes Ingress, a LoadBalancer Service, or the Inference Gateway.
# TODO: either add or drop this page
---

The [DGD Guide](dgd-guide.md) reaches the Frontend with `kubectl port-forward`, which is fine for a smoke test but not for production traffic. This page shows how to give the Frontend a stable external address. Whatever the API version, the operator creates a `ClusterIP` Service named `<name>-frontend` (where `<name>` is `metadata.name`) on port 8000 for the Frontend component — the options below route external traffic to that Service. Choose one of three paths depending on what your cluster runs.

| Path | Use when |
|---|---|
| Kubernetes Ingress | You have an ingress controller (NGINX, etc.) and want a host-routed HTTP(S) entry point |
| LoadBalancer Service | You want a cloud load balancer and manage routing yourself |
| Inference Gateway (GAIE) | You want Gateway API model-aware routing across multiple deployments |

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
> The `nvidia.com/v1alpha1` API had a convenience `ingress` field on the Frontend component that generated this Ingress (or a service-mesh VirtualService) for you. That field is **not** part of `nvidia.com/v1beta1` — author the Ingress directly as shown above, or use a LoadBalancer Service (Option 2). See the [API Reference](api-reference.md#ingressspec) for the v1alpha1 field.

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

## Option 3: The Inference Gateway (GAIE)

The [Gateway API Inference Extension (GAIE)](inference-gateway.md) is a different mechanism: instead of exposing one Frontend, it puts a Gateway in front of one or more deployments and makes model-aware routing decisions in an Endpoint Picker Plugin (EPP). The Frontend runs with `--router-mode direct` and respects the EPP's routing. Use GAIE when you serve multiple models or deployments behind one address. See [Inference Gateway (GAIE)](inference-gateway.md).

> [!NOTE]
> A Kubernetes Ingress and GAIE are independent. An Ingress (Option 1) exposes a single Frontend Service; GAIE routes across deployments through the Gateway API and does not use an Ingress.

## Related pages

- [Inference Gateway (GAIE)](inference-gateway.md) — Gateway API model-aware routing.
- [API Reference — IngressSpec](api-reference.md#ingressspec) — full field reference.
