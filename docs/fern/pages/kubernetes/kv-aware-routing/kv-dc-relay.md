---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy the DC KV Relay
subtitle: Discover existing workers and publish endpoint-local KV pool facts from Kubernetes
---

**Experimental.** Deploy NVIDIA Dynamo's DC KV Relay alongside existing inference workers.
The Relay uses the shared Dynamo runtime and universal publisher; it does not serve inference
requests or choose a destination data center. For the producer model, see
[DC KV Relay Concepts](../../developer-guide/knowledge-base/modular-components/router/multi-dc-kv-routing.md).

## Prerequisites

- A Dynamo operator installation with the `DynamoWorkerMetadata` CRD.
- Ready inference workers using Kubernetes discovery in one Kubernetes namespace. They must
  advertise model cards, KV event sources, and a recoverable KV-state endpoint. Enabling a
  listener on Relay does not enable worker KV events.
- A container image built from a revision that includes `dynamo.kv_dc_relay`, its Rust bindings,
  and the WAN protocol. Older released images may not contain this module; use the repository's
  [container build instructions](https://github.com/ai-dynamo/dynamo/blob/main/container/README.md)
  from the same revision.
- The workers' NATS connection settings. This example uses Kubernetes discovery with the NATS
  event plane and TCP request plane; it does not deploy a second NATS server.
- Network access from Relay to the Kubernetes API, NATS, and advertised worker recovery endpoints.
- `kubectl`, and `grpcurl` on the machine used for verification.
- Kubernetes support for native gRPC startup and readiness probes.

See [Using the Dynamo Frontend](dynamo-frontend.md) for worker KV-event configuration and
[Runtime Configuration](../../reference/components/runtime-configuration.mdx) for shared runtime settings.

## Discovery Scope

The example assumes an existing Kubernetes namespace named `dynamo` and watches every Dynamo
namespace visible within it. Change the namespace consistently in the commands and RoleBinding
if your workers run elsewhere.

> [!IMPORTANT]
> `--namespaces` selects logical Dynamo namespaces, not Kubernetes namespaces. The current
> Kubernetes backend watches only the Relay pod's Kubernetes namespace. Workers in other
> Kubernetes namespaces are invisible even with `--watch-all` and cluster-wide RBAC.

`DYN_NAMESPACE` names Relay's own runtime endpoints and does not select the watched workers.
To narrow the visible logical scope, replace `--watch-all` with `--namespaces <dynamo-namespace>`.
Use the namespace in the workers' advertised endpoint identities, not an assumed Kubernetes name.

Before deployment, confirm discovery resources exist:

```bash
kubectl get crd dynamoworkermetadatas.nvidia.com
kubectl -n dynamo get dynamoworkermetadatas
kubectl -n dynamo get endpointslices \
  -l nvidia.com/dynamo-discovery-backend=kubernetes,nvidia.com/dynamo-discovery-enabled=true
```

This example uses pod-mode discovery, which joins ready EndpointSlices with worker metadata.
Worker Services must carry the discovery labels so their EndpointSlices are watched. Container-mode
discovery instead watches labeled Pods and needs Pod `get/list/watch` permissions; do not mix
discovery modes without checking the workers' registration mode.

## Deploy the Relay

Save the following manifest as `kv-dc-relay.yaml`. Replace `REPLACE_WITH_RELAY_IMAGE` with your
image and `nats://nats.dynamo-system.svc.cluster.local:4222` with the workers' NATS address.
If NATS requires authentication, supply the same runtime connection credentials through Secrets,
not literal credentials in the manifest. Set `--dc-id` to your stable logical data-center name.
Add `imagePullSecrets` if your registry requires them.

The manifest runs one replica with `Recreate` updates. Restarting Relay changes its incarnation
and requires consumers to reconnect; this is not an HA deployment. CPU and memory values are
starting allocations, not sizing guarantees: pool count and expected unique blocks affect memory.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kv-dc-relay
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: kv-dc-relay
rules:
  - apiGroups: ["nvidia.com"]
    resources: ["dynamoworkermetadatas"]
    verbs: ["get", "list", "watch", "create", "patch", "delete"]
  - apiGroups: ["discovery.k8s.io"]
    resources: ["endpointslices"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kv-dc-relay
subjects:
  - kind: ServiceAccount
    name: kv-dc-relay
    namespace: dynamo
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: kv-dc-relay
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kv-dc-relay
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: kv-dc-relay
  template:
    metadata:
      labels:
        app.kubernetes.io/name: kv-dc-relay
        nvidia.com/dynamo-discovery-backend: kubernetes
        nvidia.com/dynamo-discovery-enabled: "true"
    spec:
      serviceAccountName: kv-dc-relay
      terminationGracePeriodSeconds: 60
      containers:
        - name: relay
          image: REPLACE_WITH_RELAY_IMAGE
          command: ["python3", "-m", "dynamo.kv_dc_relay"]
          args: ["--dc-id", "dc-a", "--watch-all", "--bind", "0.0.0.0:5561"]
          env:
            - name: DYN_NAMESPACE
              value: relay
            - name: DYN_DISCOVERY_BACKEND
              value: kubernetes
            - name: DYN_KUBE_DISCOVERY_MODE
              value: pod
            - name: DYN_REQUEST_PLANE
              value: tcp
            - name: DYN_EVENT_PLANE
              value: nats
            - name: NATS_SERVER
              value: nats://nats.dynamo-system.svc.cluster.local:4222
            - name: DYN_SYSTEM_PORT
              value: "8081"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          ports:
            - name: grpc
              containerPort: 5561
            - name: system
              containerPort: 8081
          startupProbe:
            grpc:
              port: 5561
              service: dynamo.kvrelay.v1.KvEventRelay
            periodSeconds: 5
            failureThreshold: 60
          readinessProbe:
            grpc:
              port: 5561
              service: dynamo.kvrelay.v1.KvEventRelay
            periodSeconds: 10
          resources:
            requests:
              cpu: "1"
              memory: 1Gi
            limits:
              memory: 4Gi
---
apiVersion: v1
kind: Service
metadata:
  name: kv-dc-relay
  labels:
    nvidia.com/dynamo-discovery-backend: kubernetes
    nvidia.com/dynamo-discovery-enabled: "true"
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: kv-dc-relay
  ports:
    - name: grpc
      port: 5561
      targetPort: grpc
```

Relay needs write access to worker metadata because its runtime registers its own endpoints.
The Role is namespaced; it does not grant cross-namespace discovery.
This example assumes a trusted cluster network. The Service exposes plaintext gRPC without
authentication; `ClusterIP` does not itself restrict which pods can connect. Use local
port-forwarding to inspect runtime diagnostic ports, which are not included in the Service.

```bash
kubectl -n dynamo apply -f kv-dc-relay.yaml
kubectl -n dynamo rollout status deployment/kv-dc-relay --timeout=300s
kubectl -n dynamo logs deployment/kv-dc-relay --tail=100
```

## Verify Discovery and Published Metadata

Forward the listener to your machine:

```bash
kubectl -n dynamo port-forward service/kv-dc-relay 5561:5561
```

In another terminal, query the protocol identity. The decimal marker below is `KVR1`
(`0x4B565231`); it is required by every Relay request.

```bash
grpcurl -plaintext -d '{"contractMarker":1263948337}' \
  localhost:5561 dynamo.kvrelay.v1.KvEventRelay/GetRelayInfo
```

Expect this shape; identity values vary per deployment and restart:

```json
{
  "protocolVersion": 1,
  "relay": {"drtInstanceId": "123", "relayIncarnation": "456"},
  "contractMarker": 1263948337
}
```

Then inspect catalog and readiness. Each command opens a stream; stop it with Ctrl-C after
the first update.

```bash
grpcurl -plaintext -max-msg-sz 8388608 \
  -d '{"contractMarker":1263948337,"subscriberId":"deployment-check-catalog"}' \
  localhost:5561 dynamo.kvrelay.v1.KvEventRelay/WatchKvPoolCatalog
```

```bash
grpcurl -plaintext -max-msg-sz 8388608 \
  -d '{"contractMarker":1263948337,"subscriberId":"deployment-check-readiness"}' \
  localhost:5561 dynamo.kvrelay.v1.KvEventRelay/SubscribeServingReadiness
```

Check these fields in the responses:

| Response | Expected fields |
| --- | --- |
| Catalog | `snapshot.pools[]`: a `producer`, the expected `servingEndpoint`, model `registrations`, and `querySemantics`. |
| Readiness | `entries[]`: expected `namespace` and `canonicalModelId`, `state`, and `members` with optional `poolId` links. |

For a ready disaggregated model, expect separate Prefill and Decode pools but one readiness
entry with both roles. LoRA readiness appears under the base entry's `adapters`, not as another
top-level entry. Catalog and readiness revisions are independent.

An empty catalog does not verify discovery; a passing pod probe does not prove model readiness.
These checks expose metadata, not CKF contents. CKF validation requires subscribing to an
advertised producer and validating its complete CBI1 snapshot and subsequent deltas.

## Expose the WAN Listener

The example Service is cluster-internal. A trusted in-cluster consumer can use
`kv-dc-relay.dynamo.svc.cluster.local:5561`. Do not turn this plaintext Service into an
unrestricted LoadBalancer or expose the pod port to another data center directly.

For access across a trust boundary, terminate TLS in an external proxy and route only its
protected listener through your network ingress. Keep its upstream connection HTTP/2
and allow long-lived server streams. See the
[gRPC contract](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_dc_relay/docs/grpc-contract.md)
for message sizes, reconnect behavior, and error reasons.

## Optional mTLS Sidecar

Mutual TLS (mTLS) is optional and implemented outside Relay. Relay has no built-in TLS configuration,
certificate loading, or authentication. For a protected deployment:

1. Change Relay's bind address to `127.0.0.1:5561`.
2. Add a gRPC-capable sidecar that accepts authenticated TLS connections on a separate pod port
   and forwards HTTP/2 to `127.0.0.1:5561`, without retries or buffering.
3. Mount the sidecar's certificate, key, and trust bundle from Secrets; configure client
   authorization, certificate rotation, and expiry monitoring in the sidecar.
4. Point the Service at the proxy port only; keep Relay's port `5561` bound to loopback.
5. Replace the pod-IP gRPC probes from the example: they cannot reach a loopback-only listener.
   Use probes suitable for your proxy and a local Relay check; native Kubernetes gRPC probes do
   not authenticate through mTLS.

The sidecar's image and configuration depend on your organization's proxy and PKI.

For local inspection after this change, forward directly to Relay's loopback listener:

```bash
kubectl -n dynamo port-forward deployment/kv-dc-relay 5561:5561
```

The plaintext `grpcurl` commands above still apply. This checks Relay through the Kubernetes
tunnel, not the externally exposed mTLS path.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Module or binding import fails | The image must include this Relay revision and matching Python/Rust bindings. |
| Kubernetes discovery reports forbidden | Check the ServiceAccount, RoleBinding namespace, and metadata/EndpointSlice permissions. |
| Missing pod identity | Supply `POD_NAME`, `POD_UID`, and `POD_NAMESPACE` through the Downward API. |
| Running pod, empty catalog | Check the Kubernetes namespace, logical filters, ready discovery-labeled EndpointSlices, metadata, KV event advertisements, and recovery endpoints. |
| NATS connection failure or absent load | Match workers' NATS address and credentials; allow DNS and event-plane egress. |
| Catalog present, model not ready | Inspect the readiness stream's missing roles and member availability. Pool presence alone is not readiness. |
| Listener unavailable | Check bind errors, pod logs, Service selectors, network connectivity, and whether a sidecar requires TLS. |
| Client reports an oversized message | Raise client/proxy receive limits to match the Relay message limit. |
| Resource exhaustion | Inspect the machine-readable error reason; distinguish admission limits from lag or snapshot progress timeout. |

## Clean Up

Remove only the resources created by this guide; retain the existing workers and namespace:

```bash
kubectl -n dynamo delete -f kv-dc-relay.yaml
```

For all CLI and tuning options, see [DC KV Relay Configuration](../../reference/components/kv-dc-relay-configuration.md).
