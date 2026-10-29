<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# mcs-controller

`mcs-controller` implements the Kubernetes Multi-Cluster Services API
(KEP-1645, `multicluster.x-k8s.io/v1beta1`) on top of the registry
(`mcs-registry`). One controller runs in every member cluster. It exports
the cluster's ServiceExports to the registry and materialises the
ServiceImports of the whole clusterset, with their derived Services and
EndpointSlices, from the registry.

Code: `cmd/mcs-controller` and `internal/mcs/*`. Image: the module's
`Dockerfile` (`/mcs-controller`).

## Deployment

One installation per member cluster needs:

- the CRDs in `config/crd/bases` (ServiceExport, ServiceImport and the
  KEP-2149 ClusterProperty), also shipped in the image under
  `/opt/global-discovery-mcs/crds`;
- the ClusterRole `config/rbac/role.yaml` (see [RBAC](#rbac)) and a Role in
  the controller's namespace for `coordination.k8s.io` leases (get, list,
  watch, create, update, patch, delete) for leader election;
- two replicas with `--leader-elect` and `POD_NAMESPACE` from the downward
  API, and `--registry-url` (or `MCS_REGISTRY_URLS`) naming the registry.

`make manifests` also copies the ClusterRole to
`deploy/helm/charts/global-discovery-mcs/files/controller-role.yaml`.

## Components

All run on one controller-runtime manager. Everything that writes to the
registry or applies imports is leader-only (`--leader-elect`, Lease
`mcs-controller.dynamo.nvidia.com` in `--operator-namespace`).

| Component | Package | Role |
| :---- | :---- | :---- |
| Identity | `internal/mcs/identity` | Resolves the clusterset and cluster ids, region, zones and provider at startup; optionally writes ClusterProperty objects |
| Lease | `internal/mcs/lease` | `RegisterCluster`, then `HeartbeatCluster` every TTL/3 with 10 percent jitter |
| ServiceExport reconciler | `internal/mcs/export` | Writes exports and endpoint sets, writes ServiceExport conditions |
| Resyncer | `internal/mcs/export` | Garbage-collects registry records and resends every export |
| Auto-export | `internal/mcs/export` | Label-driven ServiceExports |
| Importer | `internal/mcs/importer` | `WatchServiceImports` stream and the import controller |

## Identity

- `--cluster-identity-source=auto` (default): `--clusterset-id` and
  `--cluster-id` win; an id without its flag comes from the ClusterProperty
  `clusterset.k8s.io` or `cluster.clusterset.k8s.io` (`about.k8s.io/v1alpha1`,
  KEP-2149). A property is read only for an id without its flag; a
  missing ClusterProperty CRD counts as no property.
- `flags`: both flags are required; ClusterProperty objects are not read.
- `clusterproperty`: both properties are required; flags, when set, must
  agree with them.
- Both ids are DNS labels. The cluster id is the `source-cluster` the
  Global Discovery Service sees for this cluster's endpoints.
- Region: `--region`, else the most common `topology.kubernetes.io/region`
  node label. The region becomes the `topology.kubernetes.io/region` label
  of the EndpointSlices every importing cluster writes, so `--region` must
  be a valid label value (the registry checks it too; an importer skips the
  label for a value that is not). Zones: every `topology.kubernetes.io/zone`
  node label. Provider: the scheme of the nodes' `spec.providerID`
  (`aws`, ...).
- `--write-cluster-properties` creates the two ClusterProperty objects for
  other MCS tooling. An existing property with another value stops the
  controller instead of being overwritten.

## Cluster lease

The lease registers the cluster with `--lease-ttl` (default 30 s, accepted
range 10 s to 5 m) and heartbeats every TTL/3. A heartbeat refused with
`ABORTED LEASE_EPOCH_MISMATCH` (another registration took the lease) or
`NOT_FOUND CLUSTER_NOT_REGISTERED` (the registry lost or deleted the
cluster) registers again and triggers a full resync of the exports. Other
failures are retried with backoff; while they last, the registry lets the
lease expire after TTL plus its skew allowance, which hides the cluster's
endpoints from every import until the heartbeats resume.

`--deregister-on-shutdown` (default false) deregisters the cluster when the
controller stops, which deletes its exports in the registry. Leave it off
for normal restarts and upgrades; the lease covers a short absence.

## Exports

### ServiceExport reconciler

For every ServiceExport in a watched namespace, created by hand or a chart
(KEP behaviour) or by auto-export:

1. Adds the finalizer `dynamo.nvidia.com/mcs-service-export`.
2. Reads the Service of the same name. No Service: `Valid=False`, reason
   `NoService`. Type ExternalName: `Valid=False`, reason
   `InvalidServiceType`. In both cases the registry record is deleted once
   and an `InvalidExport` Event is emitted.
3. Builds the export spec: type `HEADLESS` for `clusterIP: None`,
   `CLUSTER_SET_IP` otherwise (NodePort and LoadBalancer Services export
   their pod endpoints); ports; session affinity and its ClientIP timeout;
   IP families; internal traffic policy; traffic distribution;
   `export_creation_timestamp` = the ServiceExport's creationTimestamp. The
   Service's `dynamo.nvidia.com/mcs-*` labels and annotations are forwarded
   as exported labels and annotations together with `spec.exportedLabels`
   and `spec.exportedAnnotations`, which win on the same key. The
   controller's own markers (`dynamo.nvidia.com/mcs-managed-by`,
   `dynamo.nvidia.com/mcs-clusterset`) and the export trigger label
   (`dynamo.nvidia.com/mcs-export`) are not forwarded.
4. `PutServiceExport` when the hash of the spec changed since the last
   successful write.
5. Builds the endpoint set from the Service's EndpointSlices
   (`kubernetes.io/service-name=<name>`), skipping slices labelled
   `endpointslice.kubernetes.io/managed-by=mcs-controller.dynamo.nvidia.com`
   (imports): per endpoint the addresses, the slice address type, `ready`
   (unset means ready), `serving` (unset means serving), `terminating`,
   hostname, node name, zone, the slice ports, and the target reference as
   attributes `targetRef.kind`, `targetRef.name`, `targetRef.namespace`,
   `targetRef.uid`. The set is
   sorted and de-duplicated, so its hash is stable. Above 1000 endpoints the
   ready ones are kept first and an `EndpointsTruncated` Event is emitted.
6. `ReplaceEndpoints` when the hash changed.
7. Writes the status: `Valid` locally; `Ready` and `Conflict` copied from the
   cluster status the registry reports for this cluster in the import
   (delivered by the importer). Before the registry reports,
   `Ready=Unknown`, reason `Pending`. A write the registry rejects as invalid
   sets `Ready=False`, reason `Failed`, and an `ExportFailed` Event.

The registry's report of this cluster also says what the registry still
holds. When the import no longer lists this cluster although the
ServiceExport exists, or lists it as participating without an endpoint
set, the record or the endpoint set was lost: the reconciler forgets the
hashes it wrote and writes again on the next reconcile, whatever the
heartbeats say.

Events go through the `events.k8s.io/v1` API with reporting controller
`mcs-controller` and action `Export`.

Generations: each export has a spec and an endpoint generation,
`max(previous + 1, unix milliseconds)` (`client.Generation`), monotonic
across restarts and registry replicas. A write refused with
`FAILED_PRECONDITION STALE_GENERATION` is retried once with a generation
above the `current_generation` the registry returned.

Writes need a registered cluster: while the lease is not registered the
export waits (`Ready=Unknown`, requeued after 2 s), and a write refused with
`CLUSTER_NOT_REGISTERED` wakes the lease.

Deleting a ServiceExport deletes the registry record
(`DeleteServiceExport`) before the finalizer is removed, so while the
registry is unreachable a deletion waits. Removing the finalizer by hand
skips that; the next resync deletes the record.

### Resync and garbage collection

At startup (once the cluster is registered and the caches are synced),
after every reconnect of the registry client, after every
re-registration, and every `--export-resync-period` (default 10 m, `0`
disables), the leader:

1. forgets which hashes it wrote (generations are kept);
2. lists this cluster's exports in the registry (`ListServiceExports`);
3. lists the ServiceExports of the watched namespaces and deletes every
   registry record without one (the registry is listed first, so an export
   written while the resync runs is never taken for an orphan);
4. reconciles every ServiceExport, so each spec and endpoint set is written
   again.

Triggers that arrive while a resync is pending coalesce into it; a
reconnect seen by a replica before it became the leader is covered by the
leader's startup resync. `resync_attempts_total` counts resyncs started,
`resyncs_total` those completed; a failed resync is retried with backoff.

The controller owns every registry export record of its cluster id. Run
exactly one controller per cluster id; records in namespaces that are no
longer watched are deleted.

### Narrowing `--watch-namespaces` and uninstalling

The caches cover the watched namespaces only, so objects in a namespace
that is no longer watched get no reconcile. At leader start the controller
therefore lists the API server directly and, in every namespace outside the
watch set, removes its finalizer from ServiceExports and deletes the
managed ServiceImports, derived Services and EndpointSlices; the startup
resync then deletes the registry records of those ServiceExports. Narrow
the flag and restart; nothing else is needed.

To uninstall: delete the ServiceExports first and wait until the
controller has removed their finalizers (it deletes the registry records
on the way), delete the managed ServiceImports (the derived Services and
slices go with them through owner references), and only then stop the
controller. A controller stopped first leaves ServiceExports that cannot
be deleted until the finalizer `dynamo.nvidia.com/mcs-service-export` is
removed by hand, and records in the registry until its cluster lease
expires and compaction removes them (`--deregister-on-shutdown` removes
them at once).

### Label-driven export

A Service labelled `dynamo.nvidia.com/mcs-export: "true"` gets a
ServiceExport of the same name with an owner reference to the Service and
the labels `dynamo.nvidia.com/mcs-managed-by: mcs-controller` and
`dynamo.nvidia.com/mcs-clusterset: <clusterset>`. Removing the label
deletes that ServiceExport; deleting the Service deletes it through the
owner reference. A ServiceExport without the managed-by label is never
created, changed or deleted by the controller.

The controller does not interpret other labels: every
`dynamo.nvidia.com/mcs-*` label and annotation of the Service is forwarded
as it is (step 3 above), so new keys need no controller change.

## Imports

The leader holds one `WatchServiceImports` stream with endpoints (with a
namespace filter when exactly one namespace is watched). `SNAPSHOT_START`
clears the local map of imports, `UPSERT` and `DELETE` update it, and
`SNAPSHOT_END` garbage-collects. Every change is applied by the import
controller with server-side apply, field manager `mcs-controller`:

| Object | Name | Content |
| :---- | :---- | :---- |
| ServiceImport | `<name>` | `spec.type` (ClusterSetIP or Headless), merged ports, session affinity and config, IP families (the exporters'), `spec.ips` = the derived Service's ClusterIPs (at most two, none for Headless); exported labels and annotations; labels `dynamo.nvidia.com/mcs-managed-by`, `dynamo.nvidia.com/mcs-clusterset`; annotation `multicluster.kubernetes.io/derived-service`; `status.clusters` = the participating clusters; conditions `Ready` and `Conflict` (the import-level conflict) |
| Derived Service | `derived-<hash>` (`api.DerivedServiceName`) | No selector; type ClusterIP; `clusterIP: None` for Headless; merged ports; session affinity; `ipFamilyPolicy` SingleStack for one imported family and PreferDualStack for two, listing only the families the importing cluster's Service network has (read from the `kubernetes` Service at startup), so the API server picks when the import's family is not available; labels managed-by, clusterset and `multicluster.kubernetes.io/service-name`; owner reference to the ServiceImport |
| EndpointSlices | `<derived>-<cluster>-<v4\|v6\|fqdn>-<i>` | One set per source cluster, the local one included, at most `--endpoints-per-slice` endpoints each, grouped by address type and port set and indexed per address type (`addressType` is immutable, so a slice never changes its family); ports filtered to the port names of the merged import; the `ready`, `serving` and `terminating` conditions as exported; node names kept for local endpoints only; `targetRef` from the exported target attributes (below) unless `--import-target-ref=false`; labels `kubernetes.io/service-name=<derived>`, `multicluster.kubernetes.io/service-name=<name>`, `multicluster.kubernetes.io/source-cluster=<cluster>`, `topology.kubernetes.io/region=<region>` when the source cluster registered a region that is a valid label value, `endpointslice.kubernetes.io/managed-by=mcs-controller.dynamo.nvidia.com`, managed-by and clusterset; owner reference to the ServiceImport |

Details:

- An imported endpoint whose export carries the attribute `targetRef.name`
  gets `targetRef` with that name, the kind from `targetRef.kind` (Pod when
  absent) and the namespace from `targetRef.namespace` when present. The
  reference names an object of the source cluster, so it is informational
  in the importing cluster; the Global Discovery Service uses it to name
  replicas after their pods and to group a dual-stack pod's addresses, so
  it needs `--import-target-ref` on (the default). A service mesh that
  resolves endpoint target references locally (Linkerd looks the pod up
  and skips endpoints whose pod it cannot find) needs
  `--import-target-ref=false`, which leaves `targetRef` off.
- The merged ports of an import are the registry's; endpoint ports of a
  cluster whose Service named a port differently from the winning export
  arrive under the merged name with their own target port, so every
  participating cluster is reachable through the merged ports.
- The status is written with a merge patch of the whole status; the spec
  and metadata by server-side apply with forced ownership.
- A ServiceImport without the managed-by label belongs to another MCS
  implementation and is skipped, unless `--adopt-unmanaged-imports`.
- A Service named like the derived Service but not managed by the
  controller blocks the import (the reconcile fails and retries).
- A change between ClusterSetIP and Headless, or of the primary IP family,
  deletes and re-creates the derived Service, since `clusterIP` and the
  primary family are immutable. The ClusterIPs of a created Service come
  from the answer of the apply; `spec.ips` follows in the same reconcile.
- Slices of a source cluster that left the import, or beyond the needed
  count, are deleted: first by the names the previous apply wrote, whatever
  the informer cache says, then those the cache (or, when the count
  shrinks, the API server) still lists.
- An apply that fails after the ServiceImport exists (for example a port
  set that no Service can carry, from an older registry) sets
  `Ready=False`, reason `ApplyFailed`, naming the step, and emits an
  `ApplyFailed` Warning Event; the next successful apply clears it. A
  ServiceImport of another implementation that blocks an import is checked
  again when it is deleted or its labels change, and every 5 minutes.
- A `DELETE` event deletes the ServiceImport, the derived Service and the
  slices explicitly; owner references would cascade too, but the controller
  does not depend on the garbage collector.
- After `SNAPSHOT_END` every import that has managed objects in the cluster
  (ServiceImports, derived Services or slices) but none in the registry is
  deleted the same way. This covers imports deleted while the controller
  was down or disconnected. Before `SNAPSHOT_END` nothing is deleted for a
  missing import, so a reconnect never flaps imports.
- Reconciliation is level-based. Deleting a managed object by hand
  re-creates it; editing one (the spec, labels or owner references of the
  ServiceImport or the derived Service, the address type, endpoints, ports,
  labels or owner references of a slice) restores it: the watches pass the
  changes the controller did not write itself (it remembers the
  `resourceVersion` of each write), and every `--import-resync-period`
  (default 10 m, `0` disables) every import is applied again, so drift the
  watches missed is repaired within one period. An edited port of the
  derived Service would survive a server-side apply as an extra entry
  (`spec.ports` is a map list); the controller replaces the list when that
  happens.
- The controller remembers a hash of the import it applied last per name,
  covering what it renders (not the registry's timestamps and generations,
  which every resync of an exporting cluster bumps); an unchanged import
  (for example in the snapshot after a reconnect) costs no API call. Drift,
  a deleted managed object, a re-created namespace, a failed apply and the
  periodic resync forget the hash; an apply that started before the forget
  does not record it.
- A pending import that disappears from the registry while the watch is
  down is dropped by the next snapshot.
- This cluster's status of every import (`clusters[]` entry of the cluster
  id) goes to the ServiceExport reconciler, which copies `Ready` and
  `Conflict` onto the ServiceExport.

### Pending namespaces

Namespace sameness means an import lands in the namespace of the exported
Service. When that namespace does not exist (or is terminating) the import
is held pending, counted by
`dynamo_mcs_controller_imports_pending_namespace`, and applied when the
Namespace informer sees the namespace created; while it is terminating the
import is checked again every 5 s, and the namespace's deletion event
reconciles its imports too, so `--create-namespaces` re-creates it without
waiting for an unrelated change. A re-created namespace gets all its
imports again. `--create-namespaces` creates a missing namespace instead
(labelled managed-by and clusterset); the controller never deletes
namespaces.

## Labels, annotations and finalizer

| Key | On | Value |
| :---- | :---- | :---- |
| `dynamo.nvidia.com/mcs-export` | Service | `"true"` enables label-driven export |
| Other `dynamo.nvidia.com/mcs-*` labels and annotations | Service | Forwarded to the registry as exported labels and annotations |
| `dynamo.nvidia.com/mcs-managed-by` | ServiceExports, ServiceImports, derived Services, slices, namespaces the controller creates | `mcs-controller` |
| `dynamo.nvidia.com/mcs-clusterset` | same | the clusterset |
| `dynamo.nvidia.com/mcs-service-export` | ServiceExport finalizer | |
| `endpointslice.kubernetes.io/managed-by` | imported slices | `mcs-controller.dynamo.nvidia.com` |
| `multicluster.kubernetes.io/service-name`, `multicluster.kubernetes.io/source-cluster` | imported slices | import name, source cluster |
| `topology.kubernetes.io/region` | imported slices | the source cluster's region, absent when the cluster registered none |
| `multicluster.kubernetes.io/derived-service` | ServiceImport annotation | derived Service name |

## Conditions

| Object | Type | Status and reason |
| :---- | :---- | :---- |
| ServiceExport | `Valid` | True/`Valid`; False/`NoService`; False/`InvalidServiceType` |
| ServiceExport | `Ready` | Copied from the registry (True/`Ready`; False/`ClusterLeaseExpired`); Unknown/`Pending` before the registry reports or while the cluster is not registered; False/`Failed` when the registry rejects the export or it is invalid |
| ServiceExport | `Conflict` | Copied from the registry: False/`NoConflicts`, True with exactly one reason, the first found in the order ports (`PortConflict`), IP families (`IPFamilyConflict`), type, sessionAffinity, sessionAffinityConfig, annotations, labels, internalTrafficPolicy, trafficDistribution (each with its mcs-api reason, such as `TypeConflict`); Unknown/`ClusterLeaseExpired` |
| ServiceImport | `Ready` | True/`Ready` with the count of participating clusters; False/`Pending` when no exporting cluster participates (all leases expired); False/`ApplyFailed` naming the object that could not be applied |
| ServiceImport | `Conflict` | The import-level conflict: False/`NoConflicts` or True with its reason |

## Reachability prerequisite

A ClusterSetIP import works only when the pod IPs of every source cluster
are routable from the importing cluster: on EKS, VPC peering (or a transit
gateway) between the cluster VPCs, non-overlapping pod CIDRs (the VPC CNI
uses VPC addresses), and security groups that allow the importing
cluster's pod CIDR to the exported ports. kube-proxy DNATs the derived
ClusterIP to remote pod IPs on the client's node; there is no extra hop.
The same holds for headless imports and for consumers, such as the Global
Discovery Service, that dial imported endpoint addresses directly. The
controller does not detect routability.

Health propagation across clusters takes roughly 3 to 6 s (EndpointSlice
change, `ReplaceEndpoints`, replication, refresh, watch, apply), so
clients of a derived VIP keep retrying another endpoint on failure.

## DNS: CoreDNS multicluster plugin (optional)

Clients compute the derived Service name and dial
`derived-<hash>.<namespace>.svc.cluster.local`; no extra DNS is needed. For
`<service>.<namespace>.svc.clusterset.local` names, add the CoreDNS
[multicluster](https://github.com/coredns/multicluster) plugin. It answers
from ServiceImports (`spec.ips` for ClusterSetIP) and from the slices
labelled `multicluster.kubernetes.io/service-name` (Headless), both of
which the controller writes. Stock CoreDNS images (EKS, k3s) do not
include the plugin: build CoreDNS with `multicluster:github.com/coredns/multicluster`
added to `plugin.cfg`, before `kubernetes`.

Corefile stanza, inside the existing `.:53` server block:

```text
multicluster clusterset.local {
    fallthrough
}
```

CoreDNS also needs to list and watch the MCS objects:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: coredns-multicluster
rules:
  - apiGroups: ["multicluster.x-k8s.io"]
    resources: ["serviceimports"]
    verbs: ["list", "watch"]
  - apiGroups: ["discovery.k8s.io"]
    resources: ["endpointslices"]
    verbs: ["list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: coredns-multicluster
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: coredns-multicluster
subjects:
  - kind: ServiceAccount
    name: coredns
    namespace: kube-system
```

## Flags

The flags are listed in the module `README.md`. Readiness (`/readyz`): the
registry client connected at least once and, on the leader, the cluster is
registered. Liveness (`/healthz`) is a ping.

## RBAC

`config/rbac/role.yaml` is the ClusterRole `mcs-controller` generated from
the `+kubebuilder:rbac` markers by `make manifests`:

| Resource | Verbs | Why |
| :---- | :---- | :---- |
| `serviceexports`, `serviceimports` | get, list, watch, create, update, patch, delete | Reconciler, auto-export and imports |
| `serviceexports/status`, `serviceimports/status` | get, update, patch | Conditions and import status |
| `serviceexports/finalizers` | update | Finalizer |
| `services`, `endpointslices` | get, list, watch, create, update, patch, delete | Export sources, derived Services, imported slices |
| `namespaces` | get, list, watch, create | Pending imports; `--create-namespaces` |
| `nodes` | get, list, watch | Region and zones |
| `clusterproperties` (`about.k8s.io`) | get, list, watch, create, update, patch | Identity; `--write-cluster-properties` |
| `events` (core and `events.k8s.io`) | create, patch | Events on ServiceExports and ServiceImports |

A deployment may drop `namespaces` create and the ClusterProperty writes
when their flags are off, and adds a Role for `coordination.k8s.io` leases
(get, list, watch, create, update, patch, delete) in the controller's
namespace for leader election.

## Metrics

`dynamo_mcs_controller_*`, counters pre-initialised:

| Metric | Type | Labels | Meaning |
| :---- | :---- | :---- | :---- |
| `exports` | gauge | `state` (exported, pending, invalid, failed) | ServiceExports by state |
| `reconcile_duration_seconds` | histogram | `controller` (serviceexport, autoexport, serviceimport) | Reconcile duration |
| `reconciles_total` | counter | `controller`, `result` (success, error) | Reconciles |
| `export_writes_total` | counter | `op` (put_export, delete_export), `result` (success, stale_generation, not_registered, error) | Export writes |
| `endpoint_replacements_total` | counter | `result` | `ReplaceEndpoints` calls |
| `exported_endpoints` | gauge | | Endpoints of all exports as last written |
| `imports` | gauge | | Imports in the controller's view |
| `import_events_total` | counter | `type` (snapshot_start, upsert, delete, snapshot_end, bookmark) | Watch events |
| `imports_pending_namespace` | gauge | | Imports waiting for their namespace |
| `stream_reconnects_total` | counter | | Registry client reconnects |
| `resyncs_total` | counter | `reason` (startup, reconnect, reregister, periodic, snapshot, import_periodic) | Completed full resyncs: export resyncs by trigger, the import watch's new snapshots and the periodic import resync |
| `resync_attempts_total` | counter | `reason` | Full resyncs started, failed ones included |
| `gc_deletions_total` | counter | `kind` (registry_export, service_import, service, endpoint_slice) | Objects deleted because their source is gone |
| `heartbeat_failures_total` | counter | `reason` (epoch_mismatch, not_registered, error) | Failed heartbeats |
| `registrations_total` | counter | `result` | `RegisterCluster` calls |
| `cluster_registered` | gauge | | 1 while this replica holds the lease |

The registry client adds `dynamo_mcs_client_*` (`pkg/client/README.md`) and
controller-runtime its `controller_runtime_*` and `workqueue_*` metrics.

## Logs and traces

Logs are controller-runtime zap, JSON by default (`--zap-devel` switches
to the console encoder), with `clusterset` and `cluster` on every component
logger and the controller name, namespace and name on reconcile logs. With
`--otel-enabled` every registry call gets an otelgrpc client span with W3C
propagation, inside the controller spans `mcs-controller.register_cluster`,
`heartbeat_cluster`, `put_service_export`, `replace_endpoints`,
`delete_service_export` and `apply_service_import`, which carry
`dynamo.clusterset`, `dynamo.cluster.id`, `k8s.namespace.name` and
`k8s.service.name`, and `error=true` on failure.

## Run and test

```bash
# Unit tests (fake Kubernetes client and pkg/fakeregistry) and the envtest
# suite, part of make test:
make test
# The envtest suite alone, against kube-apiserver and etcd from setup-envtest:
make envtest
# Out of cluster against the current kubeconfig and a local mcs-registry:
kubectl apply -f config/crd/bases
go run ./cmd/mcs-controller --registry-url=http://localhost:9090 \
  --registry-allow-insecure-http --clusterset-id=dev --cluster-id=a --region=local \
  --leader-elect=false --metrics-bind-address=:18080 --health-probe-bind-address=:18081
```

`config/crd/bases` holds the three CRDs: `multicluster.x-k8s.io_serviceexports.yaml`
and `multicluster.x-k8s.io_serviceimports.yaml`, copied unchanged from
`sigs.k8s.io/mcs-api` at the version in `go.mod` by `make sync-crds`, and
`about.k8s.io_clusterproperties.yaml`. Package `config/crd` embeds them and
the envtest suite installs them from there; a unit test fails when the
copies differ from the module's mcs-api version. The ClusterProperty CRD
carries the `api-approved.kubernetes.io` annotation that the protected
`about.k8s.io` group requires.

The envtest suite (`internal/mcs/integration`) covers: an export creating
`PutServiceExport` and `ReplaceEndpoints`; scaling replacing the endpoint
set with a higher generation; this cluster's conditions on the
ServiceExport; deleting a ServiceExport calling `DeleteServiceExport` and
removing the finalizer; an import creating the ServiceImport, the derived
Service with `spec.ips`, and slices with their labels, owner references and
target references; a `DELETE` cascading; a snapshot after reconnecting to a
new replica collecting an orphan import; a namespace created later
receiving its pending import; label-driven export, a hand-made
ServiceExport giving way to a managed one, and the removal of the label; a
deleted or edited managed object repaired; dual-stack slice names across
the slice boundary and a family transition; IP families on an IPv4-only
importer; a foreign ServiceImport releasing an import; an apply failure
reported on the ServiceImport; narrowing `--watch-namespaces`; and the
controller against an in-process `mcs-registry` (memory store) with a
second cluster exporting the same Service under another port name, and
the registry losing this cluster's endpoint set and record.
