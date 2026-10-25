<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Discovery-backed global round robin

This example deploys one Global Discovery Service (GDS) and one Global Ingress
(Global Frontend plus embedded round-robin router) per cluster. Services remain
private ClusterIP services. It uses existing Karmada members and private pod
connectivity; it does not create clusters, node groups, or GPU capacity.

The ingress polls its local GDS `/pools` every two seconds. GDS joins ready MCS
EndpointSlices for `<dgd>-pool-relay` and `<dgd>-frontend` using source cluster,
namespace, and DGD name. Model identity comes from an operator-supplied JSON file;
frontend addresses and readiness come from MCS. A pool needs a ready Relay to be
listed and a ready frontend to receive requests. Replacing a frontend updates its
address without restarting GDS or the ingress.

GDS `/routers` lists all imported ready ingress replicas. This example advertises
HTTP port `8000` for inspection. Global Ingress has no Relay state receiver; do not
configure Relay state-stream pushes to this HTTP port. Load-aware and KV-aware
routing require additional integration. Round robin is per ingress process and
per model, with no cross-replica cursor coordination or automatic request retry.
Membership converges through MCS and the ingress polling interval; exact balance
checks must wait for a stable pool set.

## Build and run

Use the workspace-pinned Rust toolchain and its normal Linux build dependencies
on Ubuntu 24.04, matching the runtime image libc.
Build the runtime image from the two binaries:

```bash
cargo build --locked -p dynamo-global-ingress -p dynamo-global-discovery-service
cp target/debug/global-ingress target/debug/global-discovery-service examples/global-router/rr-control-plane/
docker build -t global-router-rr:local examples/global-router/rr-control-plane
```

Run local container health and request checks before publishing to your private
registry. Remove the copied binaries afterward. Use the registry image digest when
rendering manifests, and provide a private routing file modeled on
`routing.example.json`:

```bash
python3 examples/global-router/rr-control-plane/manifests.py   --namespace global-router-rr   --image "$RUNTIME_IMAGE_DIGEST"   --routing-config /path/to/routing.local.json > /tmp/global-router-rr.json
kubectl apply -f /tmp/global-router-rr.json
```

For private clusters reached through an existing CDK kubectl provider, apply the
same manifest with `rr-backend/cluster.py` and operator-local fleet configuration.

GDS receives only get/list/watch permissions for EndpointSlices. The ingress does
not mount a Kubernetes token. Both processes run as a non-root user, with bounded
memory, read-only root filesystems, and no added capabilities. Model configuration
changes require a GDS rollout because the file is loaded at startup.

## Export and import through Karmada

1. Create the workload Services directly in each source member cluster. Export the
   same service names from each cluster: `global-router` (`http`, 8000),
   `<dgd>-frontend` (`http`, 8000), and `<dgd>-pool-relay` (the Relay's actual gRPC port).
2. Create the workload namespace in the Karmada API with label
   `namespace.karmada.io/skip-auto-propagation=true` when namespace lifecycle is
   managed independently in the members.
3. Create a ServiceExport for each name in Karmada and propagate those exports to
   source members. Do not propagate the Services: each source Service selects its
   own local pods. Applying the exports through Karmada also initializes fresh
   agents' ServiceExport informers.
4. Create a ServiceImport for each name in Karmada. Propagate imports to every GDS
   member with `propagateDeps: true`, so imported EndpointSlices reach every GDS.
5. Verify local GDS `/healthz`, `/routers`, and `/pools`. Each pool must have the
   expected model and ready frontend URL. Verify `/v1/models` on every ingress.

Cross-cluster forwarding uses private pod addresses from imported EndpointSlices.
Overlapping Kubernetes Service CIDRs are supported because remote ClusterIPs are
not used. Private pod routes, security groups, and CNI SNAT exclusions must allow
all directed cluster-to-cluster paths. ClusterSet DNS is not required.

## Validate

`validate.py` runs in a short-lived Python pod with read-only access to pods and
pod logs in the workload namespace. Set `POD_NAMESPACE`, `MODEL`, `EXPECTED_POOLS`,
and optionally `REQUEST_COUNT` (default 40). Enable JSON logs with
`DYN_LOGGING_CONSOLE_FORMAT=jsonl` and forwarding debug logs, as the renderer does.
The script verifies the discovery inventory, contacts every advertised ingress,
waits for stable discovery membership, sends streaming and non-streaming
chat/completion requests, and checks actual
forwarding log counts for equal per-pool distribution. It writes a compact result
to `/dev/termination-log`. Remove the validation pod and its temporary RBAC after
collecting results.

To inspect from a host with access to a member cluster:

```bash
kubectl -n global-router-rr port-forward svc/global-router 8000:8000
kubectl -n global-router-rr port-forward svc/global-discovery 8080:8080
```

These commands target the active kubeconfig context. Use explicit contexts when
working with multiple clusters in the same region.
