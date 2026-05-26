# Gateway Integration

Per-gateway recipes for fronting a Dynamo Frontend service with smart
routing, TLS, and auth. The Dynamo source provides docs at
`docs/kubernetes/inference-gateway.md` on the target release branch.

---

## Decision Matrix

| Gateway | Best for | Watch out |
|---|---|---|
| **GAIE** (Gateway API Inference Extension) | Multi-deployment fleets; want SIG-AI routing semantics | Requires the EPP image; the spec is still evolving (`v1.5.0-rc.2` per `container/context.yaml`) |
| **kgateway** | Already running kgateway | Istio sidecar injection breaks inference traffic — see DYN-3077 |
| **Istio** | Already running Istio | Same DYN-3077 issue; explicit sidecar opt-out required on the Frontend pod |
| **Plain Service + Ingress** | Single deployment; no routing intelligence | Loses cross-deployment / SLO-aware routing |

---

## GAIE (Gateway API Inference Extension)

Reference: NVIDIA Inference Gateway docs and the SIG-Network Gateway
API Inference Extension upstream project.

EPP image (per `container/context.yaml`):

```
us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:v1.5.0-rc.2
```

Minimal install (after a `gateway.networking.k8s.io` GatewayClass is
already provisioned):

```yaml
---
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: dynamo-inference-gw
  namespace: dynamo-system
spec:
  gatewayClassName: inference-gateway
  listeners:
    - name: http
      port: 80
      protocol: HTTP
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: dynamo-pool
  namespace: dynamo-system
spec:
  selector:
    app.kubernetes.io/component: frontend
  targetPortNumber: 8000
  extensionRef:
    name: dynamo-epp
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: dynamo-route
  namespace: dynamo-system
spec:
  parentRefs:
    - name: dynamo-inference-gw
  rules:
    - backendRefs:
        - group: inference.networking.x-k8s.io
          kind: InferencePool
          name: dynamo-pool
```

The `InferencePool` is the GAIE primitive that adds inference-aware
routing on top of standard Gateway API.

---

## kgateway

Reference: kgateway upstream documentation. Dynamo-specific
integration is partial — the platform chart does not pre-install
kgateway; you bring your own.

Pattern:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: dynamo-gw
  namespace: dynamo-system
spec:
  gatewayClassName: kgateway
  listeners:
    - name: http
      port: 80
      protocol: HTTP
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: dynamo-route
  namespace: dynamo-system
spec:
  parentRefs:
    - name: dynamo-gw
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /v1
      backendRefs:
        - name: <frontend-svc>
          port: 8000
```

**Known issue:** kgateway + Istio sidecar injection causes `500` on
inference requests — see [references/known-issues.md](known-issues.md)
and DYN-3077 / NVBug 6194957 (P0 open as of 1.2.0 RC5).

---

## Istio

For deployments already running Istio:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: dynamo-gw
  namespace: dynamo-system
spec:
  gatewayClassName: istio
  listeners:
    - name: http
      port: 80
      protocol: HTTP
```

**Required:** disable sidecar injection on the Frontend pod's
namespace OR explicitly opt the Frontend pod out:

```bash
# Option A: namespace-level disable (affects everything in the ns).
kubectl label namespace dynamo-system istio-injection=disabled --overwrite

# Option B: per-pod opt-out (preferred when other workloads in the ns need the sidecar).
kubectl patch deploy <frontend-deploy> -n dynamo-system --type=merge -p '{
  "spec": {"template": {"metadata": {"annotations": {"sidecar.istio.io/inject": "false"}}}}
}'
```

The DYN-3077 issue documented in
[references/known-issues.md](known-issues.md) is what makes this
opt-out load-bearing.

---

## Plain Service + Ingress

The simplest path: standard `Service` of type `LoadBalancer` or
`ClusterIP` + `Ingress`. No routing intelligence.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: dynamo-public
  namespace: dynamo-system
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8000
  selector:
    app.kubernetes.io/component: frontend
    nvidia.com/dgd-name: <dgd-name>
```

This loses cross-deployment routing but is the lowest-overhead choice
for single-deployment clusters.

---

## TLS Termination

All three gateway options support TLS termination. Recommended path:
use cert-manager to issue certs and reference them in the Gateway
spec:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: dynamo-gw
spec:
  listeners:
    - name: https
      port: 443
      protocol: HTTPS
      tls:
        certificateRefs:
          - name: inference-tls
            kind: Secret
```

The `inference-tls` Secret is provisioned by cert-manager from a
ClusterIssuer or Issuer. Dynamo does not own this layer; treat it as
standard Kubernetes TLS provisioning.

---

## Auth

Dynamo's Frontend does not implement auth natively. Common patterns:

| Pattern | Where to configure |
|---|---|
| API key in `Authorization: Bearer ...` validated by the gateway | Gateway-level (GAIE / kgateway / Istio) |
| mTLS client certs | Gateway-level TLS config with `mode: Terminate` and client cert requirements |
| OIDC / JWT | Gateway-level auth filter (kgateway has `kgateway-policies`; Istio has `RequestAuthentication`) |
| Per-route ACL | HTTPRoute filters or per-Gateway policy |

The Dynamo Frontend logs the resolved identity if the gateway forwards
it (typically as `X-Forwarded-User` or similar). Useful for per-
tenant metrics tagging via the DynamoModel CR's `metadata.annotations`.
