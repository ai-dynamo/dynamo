<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Istio

This guide shows how to deploy Dynamo with [Istio](https://istio.io/) as your
inference gateway. By the end, inference requests will flow from an
Istio-managed `Gateway` to your model servers via the Dynamo EPP.

> [!NOTE]
> This guide assumes familiarity with [Gateway API](https://gateway-api.sigs.k8s.io/) and Dynamo.

## Prerequisites

1. The environment variables `${GUIDE_NAME}`, `${MODEL_NAME}` and `${NAMESPACE}` should be set as part of deploying one of the well-lit path guides.
2. A Kubernetes cluster running one of the three most recent [Kubernetes releases](https://kubernetes.io/releases/).
3. [Helm](https://helm.sh/docs/intro/install/).
4. [jq](https://jqlang.org/download/).

## Quickstart

The [`install.sh`](./install.sh) script installs everything needed to run
inference traffic through Istio:

- the Gateway API CRDs,
- the Gateway API Inference Extension (GAIE) CRDs,
- Istio (via `istioctl`) with the
  `ENABLE_GATEWAY_API_INFERENCE_EXTENSION` feature flag enabled, and
- a `Gateway` named `inference-gateway` in `${NAMESPACE}` using the `istio`
  GatewayClass.

```bash
cd deploy/inference-gateway
export NAMESPACE=my-model
./gateways/istio/install.sh
```

The following environment variables can be overridden:

| Variable | Default | Description |
|----------|---------|-------------|
| `NAMESPACE` | `default` | Namespace where the `Gateway` is created. |
| `ISTIO_NAMESPACE` | `istio-system` | Namespace where the Istio control plane runs. |
| `GATEWAY_API_VERSION` | `v1.5.1` | Gateway API release to install. |
| `IGW_LATEST_RELEASE` | `v1.2.1` | Gateway API Inference Extension release to install. |
| `ISTIO_VERSION` | `1.29.2` | Istio version to install (used when `istioctl` is not already on `PATH`). |
| `ISTIO_DOWNLOAD_DIR` | `$(pwd)` | Directory where Istio is downloaded if `istioctl` is not already installed. |

Verify the `Gateway` is programmed:

```bash
kubectl get gateway inference-gateway -n ${NAMESPACE}
```

Expected output:

```text
NAME                CLASS   ADDRESS         PROGRAMMED   AGE
inference-gateway   istio   10.xx.xx.xx     True         30s
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

### Step 2: Install Istio

Install Istio with the Gateway API Inference Extension feature flag enabled:

```bash
ISTIO_VERSION=1.29.2
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} sh -
export PATH="$PWD/istio-${ISTIO_VERSION}/bin:$PATH"

istioctl install -y \
  --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true
```

Verify the installation:

```bash
kubectl get pods -n istio-system
```

Expected output:

```text
NAME                      READY   STATUS    RESTARTS   AGE
istiod-xxxxxxxxxx-xxxxx   1/1     Running   0          30s
```

### Step 3: Deploy the Gateway

Create a `Gateway` resource. Istio watches this resource and creates an
Envoy-based proxy that accepts incoming traffic.

```bash
kubectl apply -n ${NAMESPACE} -f - <<'EOF'
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: istio
  listeners:
    - name: http
      port: 80
      protocol: HTTP
EOF

kubectl wait gateway/inference-gateway -n ${NAMESPACE} \
  --for=condition=Programmed --timeout=180s
```

## Step 4: Send a Request

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
    -d '{
        "model": '\"${MODEL_NAME}\"',
        "prompt": "How are you today?"
    }' | jq
```

## Cleanup

```bash
kubectl delete gateway inference-gateway -n ${NAMESPACE}
istioctl uninstall --purge -y
kubectl delete namespace istio-system
kubectl delete gatewayclass istio istio-remote
kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml"
kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"
```

## Troubleshooting

### Gateway not showing `PROGRAMMED=True`

```bash
kubectl describe gateway inference-gateway -n ${NAMESPACE}
kubectl get pods -n istio-system
kubectl logs -n istio-system deployment/istiod --tail=20
```

Verify Istio was installed with the inference extension flag enabled.

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
