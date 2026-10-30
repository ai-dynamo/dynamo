---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy the Pool Relay
subtitle: Discover existing workers and publish endpoint-local KV pool facts from Kubernetes
---

**Experimental.** Deploy NVIDIA Dynamo's Pool Relay (formerly KV DC Relay) alongside existing inference workers.
The Relay uses the shared Dynamo runtime and universal publisher; it does not serve inference
requests or choose a destination data center. For the producer model, see
[Pool Relay Concepts](../../developer-guide/knowledge-base/modular-components/router/multi-dc-kv-routing.md).

## Prerequisites

- A Dynamo operator installation with the `DynamoWorkerMetadata` CRD.
- Ready inference workers using Kubernetes discovery in one Kubernetes namespace. They must
  advertise model cards, KV event sources, and a recoverable KV-state endpoint. Enabling a
  listener on Relay does not enable worker KV events.
- A container image built from a revision that includes `dynamo.pool_relay`, its Rust bindings,
  and the WAN protocol. Older released images may not contain this module; use the repository's
  [container build instructions](https://github.com/ai-dynamo/dynamo/blob/main/container/README.md)
  from the same revision.
- The workers' event-plane settings: direct ZeroMQ (ZMQ) over TCP, or NATS. Only the NATS variant
  requires a NATS server and its connection credentials.
- Network access to the Kubernetes API and advertised worker event/recovery endpoints. TCP
  recovery also needs a return path from workers to Relay's advertised response-stream address.
  The NATS variant additionally requires access to the workers' NATS server.
- `kubectl`, and `grpcurl` on the machine used for verification.
- Kubernetes support for native gRPC startup and readiness probes.

See [Using the Dynamo Frontend](dynamo-frontend.md) for worker KV-event configuration and
[Runtime Configuration](../../reference/components/runtime-configuration.mdx) for shared runtime settings.

## Deploy with the Operator

Use `DynamoPoolRelay` for active Relay replicas managed by the operator. Install an operator that
includes this CRD and use a Relay image built from the same revision. The image must run
`dynamo.pool_relay` with the `--sources-file` option; an older image with only
`dynamo.kv_dc_relay` will not start this workload.

Create the resource in the same Kubernetes namespace and with the same name as an existing DGD.
The DGD must use Kubernetes pod-mode discovery, have at least one worker, and avoid
`globalDynamoNamespace` on its workers and frontends. Its name must start with a lowercase letter,
contain only lowercase letters, digits, and hyphens, and be at most 52 characters so the operator
can create a `<dgd>-pool-relay` Service.

Save the following as `pool-relay-operator.yaml`. Replace the image and cluster ID, then match the
transport settings to the DGD's workers. The [operator sample](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/samples/dynamo-pool-relay.yaml)
has the same fields.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoPoolRelay
metadata:
  name: green
  namespace: inference
spec:
  clusterId: example-cluster
  replicas: 2
  podTemplate:
    spec:
      containers:
        - name: main
          image: REPLACE_WITH_RELAY_IMAGE
          env:
            - name: DYN_REQUEST_PLANE
              value: tcp
            - name: DYN_EVENT_PLANE
              value: zmq
```

```bash
kubectl -n inference get dgd green
kubectl apply -f pool-relay-operator.yaml
kubectl -n inference wait dynamopoolrelay/green --for=condition=Ready --timeout=300s
kubectl -n inference get dynamopoolrelay green -o jsonpath='{.status.endpoint}{"\n"}'
```

`Ready` means all requested Relay listeners are available. It does not mean that the DGD's models
or KV pools are ready. The endpoint is cluster-local and is empty after withdrawal or a failed
reconciliation. If `Ready` is False, inspect the condition reason with
`kubectl -n inference describe dynamopoolrelay green` and check the Relay Pod logs. To publish
serving load, set `DYN_SERVING_LOAD_PUBLISH=1` on the DGD's frontend Pods.

The `podTemplate` follows DGD's Pod configuration shape. Put the image and transport environment
on its `main` container. You can also set image pull secrets, resources, and sidecars there. The
operator supplies the main command, listener ports and probes, source volume, service account,
and Relay identity environment variables; leave those out of the template. Reserved environment
names are `DYN_NAMESPACE`, `DYN_DISCOVERY_BACKEND`, `DYN_KUBE_DISCOVERY_MODE`, `DYN_SYSTEM_PORT`,
`DYN_TCP_RESPONSE_STREAM_HOST`, `DYN_TCP_RESPONSE_STREAM_PORT`, `DYN_CLUSTER_ID`, `DYN_RELAY_BIND`,
`DYN_RELAY_CONNECTION_REVISION`, `DYN_RELAY_NAMESPACES`, `DYN_RELAY_WATCH_ALL`,
`DYN_RELAY_ENDPOINT_PREFIXES`, `POD_NAME`, `POD_NAMESPACE`, and `POD_UID`. The main container
cannot use `envFrom` because it could override those values.

The manual manifest below runs one Relay replica and is not an HA deployment. Use it when the
operator does not manage your Relay.

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

Save the following manifest as `pool-relay.yaml`. Replace `REPLACE_WITH_RELAY_IMAGE` with your
image and set `--cluster-id` to your stable logical data-center name. Add `imagePullSecrets` if needed.
The manifest uses TCP requests and NATS events. For a deployment without NATS, apply the
[TCP-only settings](#tcp-only-local-planes-no-nats) below before deploying. Otherwise, replace
the sample NATS address with the workers' address and supply any credentials through Secrets.

The manifest runs one replica with `Recreate` updates. Restarting Relay changes its incarnation
and requires consumers to reconnect; this is not an HA deployment. CPU and memory values are
starting allocations, not sizing guarantees: pool count and expected unique blocks affect memory.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pool-relay
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pool-relay
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
  name: pool-relay
subjects:
  - kind: ServiceAccount
    name: pool-relay
    namespace: dynamo
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: pool-relay
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pool-relay
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: pool-relay
  template:
    metadata:
      labels:
        app.kubernetes.io/name: pool-relay
        nvidia.com/dynamo-discovery-backend: kubernetes
        nvidia.com/dynamo-discovery-enabled: "true"
    spec:
      serviceAccountName: pool-relay
      terminationGracePeriodSeconds: 60
      containers:
        - name: relay
          image: REPLACE_WITH_RELAY_IMAGE
          command: ["python3", "-m", "dynamo.pool_relay"]
          args: ["--cluster-id", "cluster-a", "--watch-all", "--bind", "0.0.0.0:5561"]
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
  name: pool-relay
  labels:
    nvidia.com/dynamo-discovery-backend: kubernetes
    nvidia.com/dynamo-discovery-enabled: "true"
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: pool-relay
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

### TCP-Only Local Planes (No NATS)

Use Kubernetes discovery, TCP requests, and direct ZMQ events for a deployment without a
messaging server. `DYN_EVENT_PLANE=zmq` selects ZeroMQ over TCP; `tcp` is not an event-plane value.
This is independent of the WAN Protobuf/gRPC listener on `5561`.

In the Relay container's `env` list, replace the request/event-plane entries with the following
and remove `NATS_SERVER`. Keep the other environment entries from the manifest:

```yaml
- name: DYN_REQUEST_PLANE
  value: tcp
- name: DYN_EVENT_PLANE
  value: zmq
- name: DYN_TCP_RESPONSE_STREAM_HOST
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: DYN_TCP_RESPONSE_STREAM_PORT
  value: "5562"
```

The fixed response-stream port is optional; without it, the runtime allocates a free port.
Port `5562` is used for worker responses to Relay's runtime requests, not for WAN subscriptions.
It does not need to be added to the WAN Service: workers connect to the advertised pod address.

Workers and other local event consumers must also use `DYN_EVENT_PLANE=zmq`. For an entirely
TCP-based request path, configure workers with `DYN_REQUEST_PLANE=tcp`. Changing Relay alone
does not migrate worker publishers. Keep worker KV-event publication enabled, and do not
register a ZMQ broker for these scopes when using direct mode.

Relay discovers direct ZMQ publishers through Kubernetes metadata and connects to their
advertised TCP addresses. Allow those pod-to-pod connections, worker recovery requests, and
the return path to Relay's response port. Publisher ports may be dynamically allocated;
opening only `5561` is insufficient. On multi-interface workers, set `DYN_EVENT_PLANE_HOST`
to a reachable IPv4 or IPv6 pod address if automatic selection advertises the wrong address.
Direct ZMQ publishers bind `0.0.0.0` for an advertised IPv4 address or `[::]` for an advertised
IPv6 address. Relay connects to the advertised address and the port assigned to that listener.

Discovery RBAC, the WAN Service, optional mTLS sidecar, and the gRPC checks below are unchanged.

### Apply the Manifest

```bash
kubectl -n dynamo apply -f pool-relay.yaml
kubectl -n dynamo rollout status deployment/pool-relay --timeout=300s
kubectl -n dynamo logs deployment/pool-relay --tail=100
```

## Verify Discovery and Published Metadata

Forward the listener to your machine:

```bash
kubectl -n dynamo port-forward service/pool-relay 5561:5561
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
`pool-relay.dynamo.svc.cluster.local:5561`. Do not turn this plaintext Service into an
unrestricted LoadBalancer or expose the pod port to another data center directly.

For access across a trust boundary, terminate TLS in an external proxy and route only its
protected listener through your network ingress. Keep its upstream connection HTTP/2
and allow long-lived server streams. See the
[gRPC contract](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/pool_relay/docs/grpc-contract.md)
for message sizes, reconnect behavior, and error reasons.
Within one trusted network whose clusters route each other's pod addresses, the Kubernetes
Multi-Cluster Services API can make a Relay Service visible in the other clusters instead; see
[Deploy Multi-Cluster Services](multi-cluster-services.md).

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
kubectl -n dynamo port-forward deployment/pool-relay 5561:5561
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
| NATS connection failure | Match workers' NATS address and credentials; allow DNS and event-plane egress. |
| Missing events or load with ZMQ | Check worker/Relay event-plane settings, publisher discovery metadata, and reachability of advertised pod addresses and ports. |
| TCP recovery timeout | Check Relay-to-worker recovery connectivity and the worker-to-Relay response-stream address/port. |
| Catalog present, model not ready | Inspect the readiness stream's missing roles and member availability. Pool presence alone is not readiness. |
| Listener unavailable | Check bind errors, pod logs, Service selectors, network connectivity, and whether a sidecar requires TLS. |
| Client reports an oversized message | Raise client/proxy receive limits to match the Relay message limit. |
| Resource exhaustion | Inspect the machine-readable error reason; distinguish admission limits from lag or snapshot progress timeout. |

## Clean Up

Remove only the resources created by this guide; retain the existing workers and namespace:

```bash
kubectl -n dynamo delete -f pool-relay.yaml
```

For all CLI and tuning options, see [Pool Relay Configuration](../../reference/components/pool-relay-configuration.md).
