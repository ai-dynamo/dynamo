<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-end test on kind

This directory holds the end-to-end test of `mcs-registry` and
`mcs-controller` from the Helm chart `deploy/helm/charts/global-discovery-mcs`
on three local kind clusters, and, in phase 3, of the Global Discovery Service
(`lib/global-discovery-service`) reading what the controller imports.

```bash
make -C deploy/global-discovery-mcs e2e E2E_PHASES="1 2 3"
make -C deploy/global-discovery-mcs e2e-down   # delete the clusters
```

The run prints a PASS, FAIL, GAP or SKIP line per assertion and a summary with
the counts of every phase. It exits non-zero when an assertion failed.

## What it sets up

| Cluster | Pod subnet | Service subnet | Region | Runs |
| :---- | :---- | :---- | :---- | :---- |
| `mcs-e2e-hub` | `10.241.0.0/16` | `10.251.0.0/16` | `region-hub` | `mcs-registry` (memory store, 1 replica, Service type NodePort) and `mcs-controller` (cluster id `hub`); in phase 3 the Global Discovery Service |
| `mcs-e2e-a` | `10.242.0.0/16` | `10.252.0.0/16` | `region-a` | `mcs-controller` (cluster id `a`) |
| `mcs-e2e-b` | `10.243.0.0/16` | `10.253.0.0/16` | `region-b` | `mcs-controller` (cluster id `b`) |

1. `docker build` of the module `Dockerfile` into `global-discovery-mcs` for
   the host platform (linux/arm64 on Apple Silicon), tagged
   `e2e-<image id>`. With phase 3 also `container/Dockerfile.global-discovery-service`
   from the repository root into `global-discovery-service`; a first Rust build
   takes 10 to 20 minutes.
2. Three kind clusters (`E2E_PREFIX-hub`, `-a`, `-b`) with the pinned node
   image, distinct pod and service subnets, on the Docker network `kind`.
3. A static route on every node to the pod subnet of each other cluster,
   through that cluster's node address on the Docker network, as VPC peering
   does on EKS. kindnet masquerades pod traffic that leaves the cluster, so
   the replies come back to the node. Service subnets are not routed: a derived
   Service ClusterIP is served by the kube-proxy of the importing cluster, which
   DNATs it to a remote pod IP, so only the pod subnets must be reachable.
4. The images loaded into every node (`kind load`, with a `docker save`
   archive as the fallback).
5. The chart in each cluster, release `mcs` in namespace `mcs-system`:
   the hub with `registry.enabled=true`, `registry.store=memory`,
   `registry.replicas=1`, `registry.service.type=NodePort` and the controller
   on the in-release registry URL; a and b with
   `controller.registryUrls={http://<hub node address>:<registry gRPC NodePort>}`.
   Every controller has `controller.registryAllowInsecureHttp=true`,
   `controller.clusterIdentitySource=flags`, `controller.clustersetId=e2e`,
   `controller.clusterId`, `controller.region`, `controller.replicas=1` and
   `controller.leaseTtl=10s` (the minimum, so the lease expiry test is quick).

The test reaches the registry REST API and the Global Discovery Service through
the API server's Service proxy of the hub (`kubectl get --raw
/api/v1/namespaces/<ns>/services/<name>:<port>/proxy/...`), so it needs no
port-forward and no route from the host into the Docker network (colima's VM
does not offer one). Every `kubectl` and `helm` call uses its own kubeconfig
file in `bin/e2e/kubeconfig`, never the default one.

## Phases

| Phase | Namespace | Assertions |
| :---- | :---- | :---- |
| 1 | `e2e-p1` | The registry is ready; it lists the clusters of the clusterset with ACTIVE leases and their regions; every controller holds its leader election Lease. Labelling Service `echo` (2 pods) in a creates a ServiceExport that is Valid and Ready, and the registry import shows cluster a with 2 ready endpoints. In a, b and the hub: a ClusterSetIP ServiceImport whose `spec.ips` is the ClusterIP of its `derived-<hash>` Service, and EndpointSlices labelled `multicluster.kubernetes.io/service-name`, `multicluster.kubernetes.io/source-cluster=a`, `topology.kubernetes.io/region=region-a` and the controller's managed-by labels, holding the ready pod IPs of a with `targetRef` kind Pod and the pod names. A pod in b and one in the hub reach a pod of a through the derived Service ClusterIP. |
| 2 | `e2e-p2` | Scaling the Deployment to 1, 3 and 2 replicas changes the endpoints in every cluster and the registry import. A ServiceExport created as `multicluster.x-k8s.io/v1alpha1` works like a labelled Service. A conflicting export from b (same port name, port 9090 against 8080) gives Conflict=True, reason PortConflict, on b's ServiceExport (and, as KEP-1645 asks, on a's); the registry import keeps the oldest export (a), with its port only; deleting b's Service clears the conflict. Stopping the controller of b expires its lease (10 s TTL plus skew): its export stops participating and its slices leave the imports of a and the hub; starting it again brings all back. Removing the export label (or deleting a ServiceExport) removes the ServiceImport, the derived Service and the slices everywhere, and the registry import. |
| 3 | `demo`, `gr` | The Global Discovery Service runs in the hub with `--pool-discovery mcs --router-namespace gr --router-service global-router --router-port-name grpc` and read access to EndpointSlices. In a, Deployment `demo` (2 pods) with Services `demo-pool-relay` (exported as `v1alpha1`) and `demo-frontend`; in the hub, Deployment `global-router` (2 pods) with Service `global-router`, port `grpc`, exported. `GET /pools` lists one complete pool with key `{cluster_id: a, namespace: demo, dgd_name: demo}` (`demo-frontend` is not a pool); its `location.region` and `relay_identities` are checked as below; `GET /routers` lists `hub/<pod>` for both pods with `http://<pod IP>:8080` URLs. Scaling `demo` to 0 removes the pool, scaling it to 2 brings it back, and deleting the ServiceExport removes it again; the routers stay. |

Phases are independent: each resets its own namespaces and creates its own
workloads. Phases 1 and 2 use all three clusters; phase 3 needs only the hub and
a, so `E2E_PHASES=3` creates two clusters.

All waits are polls (every 2 s, `E2E_TIMEOUT` seconds each) on the state that
is asserted; there are no fixed sleeps.

### Known gaps

A GAP line is a behaviour the module documents but the Global Discovery
Service does not have yet. It does not fail the run, and `E2E_STRICT=1` turns
it into a failure. Phase 3 has one:

- `location.region` of a pool is the source cluster id (`a`), not the
  `topology.kubernetes.io/region` label of the slices (`region-a`), although
  `README.md` and `docs/mcs-controller.md` say the service reads the label.
  The test passes when the region is `region-a` and records a GAP when it is
  the cluster id.

`relay_identities` is `["ip:<address>"]` of the one relay pod with the lowest
address (the service binds a single relay), not the pod names; the test accepts
either the pod names or that address form. The slices carry the pod names in
`targetRef`, which the router replicas (`hub/<pod>`) use.

## Prerequisites

- Docker (on macOS, colima) and `kubectl`, `helm` 3, `jq`, `curl`, `go`
  (for `make kind`), and `shellcheck` for `make` linting of the scripts.
- `kind` is installed into `bin/` at the version `KIND_VERSION` of the
  Makefile, by `make e2e` itself; nothing is installed globally.
- The Docker host's `fs.inotify.max_user_instances` must be at least 256.
  Every kind node is a systemd container holding dozens of inotify instances
  of the host's per-user pool, 128 by default in colima: with three clusters
  the node that starts last fails with `too many open files` (kube-proxy
  crash-loops) and the run stops. The script checks the limit after creating
  the hub and prints the fix: `colima ssh -- sudo sysctl -w
  fs.inotify.max_user_instances=512`. The setting is lost when the VM
  restarts; `E2E_SKIP_INOTIFY_CHECK=1` skips the check.
- Network access to Docker Hub (`busybox`, `golang`, `rust`), `nvcr.io`
  (the distroless base) and the `kindest/node` image.

## Settings

| Variable | Default | Meaning |
| :---- | :---- | :---- |
| `E2E_PHASES` | `1 2 3` | Phases to run, separated by spaces or commas |
| `E2E_KEEP` | `0` | `1` keeps the clusters when the run fails, `always` also after a pass; `make e2e-down` deletes them |
| `E2E_SKIP_BUILD` | `0` | `1` reuses the images of an earlier run (`<repo>:e2e-latest` in Docker) |
| `E2E_STRICT` | `0` | `1` fails on a known gap |
| `E2E_TIMEOUT` | `90` | Seconds each assertion polls |
| `E2E_HELM_TIMEOUT` | `5m` | `helm --wait` timeout |
| `E2E_PREFIX` | `mcs-e2e` | Cluster name prefix (`<prefix>-hub`, `-a`, `-b`); the test only touches these |
| `E2E_CLUSTER_KEYS` | by phase | Clusters to create, for example `hub a`; steps that need b are skipped |
| `E2E_PLATFORM` | host | Image platform, `linux/arm64` or `linux/amd64` |
| `E2E_WORK_DIR` | `bin/e2e` | Kubeconfigs, kind configs and logs |
| `KIND_NODE_IMAGE` | `kindest/node:v1.35.8@sha256:...` | Node image, pinned by digest |
| `POD_SUBNET_*`, `SERVICE_SUBNET_*` | see above | Subnets of `HUB`, `A` and `B`; keep them distinct and clear of the Docker network |
| `MCS_IMAGE_REPO`, `GDS_IMAGE_REPO`, `BUSYBOX_IMAGE` | `global-discovery-mcs`, `global-discovery-service`, `busybox:1.37` | Images |

A run reuses clusters that exist (for example kept ones) and re-applies the
routes and the chart, so it is idempotent. Unless kept, the clusters are
deleted at the end, passed or failed. On a failure the script prints the pods,
the controller and registry logs and the MCS objects of every cluster first.

## Files

| File | Contents |
| :---- | :---- |
| `run.sh` | Entry point: setup, phases, summary, teardown; `run.sh down` deletes the clusters |
| `lib.sh` | Settings, output, polling and result counters, kubectl, helm and registry REST wrappers |
| `setup.sh` | Image builds and loading, kind clusters, routes, Helm releases |
| `checks.sh` | Check functions on MCS objects, shared by the phases |
| `phase1.sh`, `phase2.sh`, `phase3.sh` | The phases |
| `manifests/` | The workloads, the Global Discovery Service and the ServiceExports the phases apply |
