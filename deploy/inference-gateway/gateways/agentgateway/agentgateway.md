<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Agentgateway

This guide shows how to deploy Dynamo with
[agentgateway](https://agentgateway.dev/) as your inference gateway. By the
end, inference requests will flow from an agentgateway-managed `Gateway` to
your model servers via the Dynamo EPP.

> [!NOTE]
> This guide assumes familiarity with [Gateway API](https://gateway-api.sigs.k8s.io/) and Dynamo.

## Prerequisites

1. The environment variables `${GUIDE_NAME}`, `${MODEL_NAME}` and `${NAMESPACE}` should be set as part of deploying one of the well-lit path guides.
2. A Kubernetes cluster running one of the three most recent [Kubernetes releases](https://kubernetes.io/releases/).
3. [Helm](https://helm.sh/docs/intro/install/).
4. [jq](https://jqlang.org/download/).

## Quickstart

The [`install.sh`](./install.sh) script installs everything needed to run
inference traffic through agentgateway:

- the Gateway API CRDs,
- the Gateway API Inference Extension (GAIE) CRDs,
- the agentgateway control plane (CRDs + controller) into `agentgateway-system`,
- an `AgentgatewayParameters` resource that excludes Istio sidecar injection
  from the `agentgateway-proxy` pods, and
- a `Gateway` named `inference-gateway` in `${NAMESPACE}` using the
  `agentgateway` GatewayClass.

```bash
cd deploy/inference-gateway
export NAMESPACE=my-model
./gateways/agentgateway/install.sh
```

The following environment variables can be overridden:

| Variable | Default | Description |
|----------|---------|-------------|
| `NAMESPACE` | `default` | Namespace where the `Gateway` is created. |
| `AGW_NAMESPACE` | `agentgateway-system` | Namespace where the agentgateway control plane runs. |
| `GATEWAY_API_VERSION` | `v1.5.1` | Gateway API release to install. |
| `IGW_LATEST_RELEASE` | `v1.2.1` | Gateway API Inference Extension release to install. |
| `AGW_VERSION` | `v1.0.0` | agentgateway Helm chart version. |

Verify the `Gateway` is programmed:

```bash
kubectl get gateway inference-gateway -n ${NAMESPACE}
```

Expected output:

```text
NAME                CLASS          ADDRESS         PROGRAMMED   AGE
inference-gateway   agentgateway   10.xx.xx.xx     True         30s
```

Wait until `PROGRAMMED` shows `True` before proceeding.

## Manual installation

If you would rather run the steps by hand, the sections below mirror what
`install.sh` does.

### Step 1: Install Gateway API and GAIE CRDs

```bash
GATEWAY_API_VERSION=v1.5.1
IGW_LATEST_RELEASE=v1.2.1

kubectl apply --server-side --force-conflicts \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

kubectl apply \
  -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml"
```

Verify the APIs are available:

```bash
kubectl api-resources --api-group=gateway.networking.k8s.io
kubectl api-resources --api-group=inference.networking.k8s.io
```

### Step 2: Install agentgateway

Install the agentgateway CRDs and control plane with inference extension support
enabled:

```bash
AGW_VERSION=v1.0.0

helm upgrade --install agentgateway-crds \
  oci://cr.agentgateway.dev/charts/agentgateway-crds \
  --namespace agentgateway-system \
  --create-namespace \
  --version ${AGW_VERSION}

helm upgrade --install agentgateway \
  oci://cr.agentgateway.dev/charts/agentgateway \
  --namespace agentgateway-system \
  --version ${AGW_VERSION} \
  --set inferenceExtension.enabled=true \
  --wait
```

Verify the installation:

```bash
kubectl get pods -n agentgateway-system
kubectl get gatewayclass agentgateway
```

Expected output:

```text
NAME           CONTROLLER                      ACCEPTED   AGE
agentgateway   agentgateway.dev/agentgateway   True       30s
```

### Step 3: Create AgentgatewayParameters (excludes Istio sidecar injection)

When the deployment namespace has `istio-injection=enabled`, the Istio sidecar
intercepts the `ext_proc` gRPC connection from `agentgateway-proxy` to EPP
(port 9002), causing inference requests to return HTTP 500. The
`AgentgatewayParameters` resource below sets `sidecar.istio.io/inject: "false"`
on the proxy pod template so that `ext_proc` traffic reaches EPP directly. This
annotation is a no-op on clusters where Istio is not installed.

`AgentgatewayParameters` must live in the same namespace as the `Gateway`
because Gateway API's `spec.infrastructure.parametersRef` is a
`LocalParametersReference` (no `namespace` field).

```bash
kubectl apply --server-side -n ${NAMESPACE} -f - <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayParameters
metadata:
  name: inference-gateway-params
spec:
  deployment:
    spec:
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
EOF
```

### Step 4: Deploy the Gateway

```bash
kubectl apply -n ${NAMESPACE} -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: agentgateway
  infrastructure:
    parametersRef:
      group: agentgateway.dev
      kind: AgentgatewayParameters
      name: inference-gateway-params
  listeners:
    - name: http
      port: 80
      protocol: HTTP
EOF

kubectl wait gateway/inference-gateway -n ${NAMESPACE} \
  --for=condition=Programmed --timeout=180s
```

## Step 5: Send a Request

> [!IMPORTANT]
> Before sending requests, you must deploy a well-lit path guide. This sets up
> a model server deployment, an `InferencePool`, and an `HTTPRoute` to connect
> the `Gateway` to the pool.

Get the `Gateway` external address:

```bash
export IP=$(kubectl get gateway inference-gateway -n ${NAMESPACE} \
  -o jsonpath='{.status.addresses[0].value}')
```

Send an inference request via the managed `Gateway`:

```bash
curl -X POST http://${IP}/v1/completions \
    -H 'Content-Type: application/json' \
    -H 'X-Gateway-Base-Model-Name: '"$GUIDE_NAME"'' \
    -d '{
        "model": '\"${MODEL_NAME}\"',
        "prompt": "How are you today?"
    }' | jq
```

## Cleanup

```bash
kubectl delete gateway inference-gateway -n ${NAMESPACE}
kubectl delete agentgatewayparameters inference-gateway-params -n ${NAMESPACE}
helm uninstall agentgateway -n agentgateway-system
helm uninstall agentgateway-crds -n agentgateway-system
kubectl delete namespace agentgateway-system
kubectl delete gatewayclass agentgateway
kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml"
kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"
```

## Troubleshooting

### Gateway not showing `PROGRAMMED=True`

```bash
kubectl describe gateway inference-gateway -n ${NAMESPACE}
kubectl get pods -n agentgateway-system
kubectl logs -n agentgateway-system deployment/agentgateway --tail=20
```

Verify the `agentgateway` `GatewayClass` is present and accepted:

```bash
kubectl get gatewayclass agentgateway
```

### HTTPRoute not accepted

```bash
kubectl describe httproute ${GUIDE_NAME} -n ${NAMESPACE}
```

Verify that `parentRefs` matches the `Gateway` name and `backendRefs` matches
the `InferencePool` name.

### No response from Gateway IP

```bash
kubectl get gateway inference-gateway -n ${NAMESPACE} \
  -o jsonpath='{.status.addresses[0].value}'
```

If the address is empty, your `Gateway` may still be waiting for a
LoadBalancer service. Check that your cluster supports external load balancers.

### Inference requests return HTTP 500 with Istio installed

If your `${NAMESPACE}` is labeled `istio-injection=enabled`, confirm that the
`AgentgatewayParameters` resource above is present and that the
`agentgateway-proxy` pods are running without an `istio-proxy` sidecar
container:

```bash
kubectl get agentgatewayparameters -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -l gateway.networking.k8s.io/gateway-name=inference-gateway \
  -o jsonpath='{.items[*].spec.containers[*].name}'
```

The output should not include `istio-proxy`.
