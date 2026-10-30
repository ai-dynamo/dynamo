---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multi-Cluster Services Configuration
subtitle: Flags, labels, conditions, metrics, ports and RBAC of mcs-registry, mcs-controller and mcs-admin
---

**Experimental.** This page is the reference for the Kubernetes Multi-Cluster Services (MCS)
implementation in
[`deploy/global-discovery-mcs`](https://github.com/ai-dynamo/dynamo/tree/main/deploy/global-discovery-mcs):
the registry server `mcs-registry`, the per-cluster `mcs-controller`, the `mcs-admin` tool, and the
Global Discovery Service flags that read its imports. The `global-discovery-mcs` image holds the three
binaries (`/mcs-registry`, the entrypoint, `/mcs-controller` and `/mcs-admin`), runs as user
`65532`, and ships the CRDs in `/opt/global-discovery-mcs/crds`.

For deployment, see [Deploy Multi-Cluster Services](../../kubernetes/kv-aware-routing/multi-cluster-services.md).
For the design, see
[Multi-Cluster Services Registry](../../developer-guide/knowledge-base/modular-components/router/multi-cluster-services-registry.md).
The Helm chart's values are listed in the
[chart README](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/global-discovery-mcs/README.md).

## Ports

| Component | Port | Serves |
|---|---|---|
| `mcs-registry` | `9090` | gRPC API, package `dynamo.mcs.v1` (h2c, or TLS with `--tls-cert-file`), and the gRPC health service |
| `mcs-registry` | `8080` | REST/JSON gateway of the same API, under `/v1/` |
| `mcs-registry` | `9091` | `/metrics`, `/healthz`, `/readyz`, `/debug/pprof/` |
| `mcs-controller` | `8080` | `/metrics` |
| `mcs-controller` | `8081` | `/healthz`, `/readyz` |

## mcs-registry Flags

Every flag has an environment fallback named `MCS_` plus the flag name in upper case with dashes
replaced by underscores. A flag on the command line wins over its environment variable. An invalid
configuration exits with status 2 and lists every invalid setting.

| Flag | Environment variable | Default | Description |
|---|---|---|---|
| `--grpc-listen-address` | `MCS_GRPC_LISTEN_ADDRESS` | `:9090` | gRPC listener. `host:port`, different from the other two listeners. |
| `--http-listen-address` | `MCS_HTTP_LISTEN_ADDRESS` | `:8080` | REST gateway listener. The gateway dials the gRPC listener over loopback. `host:port`. |
| `--admin-listen-address` | `MCS_ADMIN_LISTEN_ADDRESS` | `:9091` | Admin listener for metrics, health and pprof. `host:port`. |
| `--region` | `MCS_REGION`, then `AWS_REGION` | none | Region returned by `GetServerInfo` and used by the DynamoDB store. |
| `--store` | `MCS_STORE` | `memory` | Store backend: `memory` or `dynamodb`. |
| `--dynamodb-table` | `MCS_DYNAMODB_TABLE` | `global-discovery-mcs` | DynamoDB table. Must not be empty with `--store=dynamodb`. |
| `--dynamodb-endpoint` | `MCS_DYNAMODB_ENDPOINT` | none | DynamoDB endpoint override, for dynamodb-local. An `http` or `https` URL with a host. |
| `--refresh-interval` | `MCS_REFRESH_INTERVAL` | `2s` | Interval of the full reload of each clusterset from the store. At least `100ms`. |
| `--lease-eval-interval` | `MCS_LEASE_EVAL_INTERVAL` | `1s` | Interval of the lease evaluation between refreshes, without a store read. At least `100ms`. |
| `--clusterset-discovery-interval` | `MCS_CLUSTERSET_DISCOVERY_INTERVAL` | `30s` | Interval of the listing of the store's clustersets. A request that names an unknown clusterset loads it at once. At least `1s`. |
| `--lease-skew` | `MCS_LEASE_SKEW` | `2s` | Clock skew allowance before a lease expires. `0s` to `1m`. |
| `--retention` | `MCS_RETENTION` | `24h` | How long the records of an expired cluster are kept before compaction removes them. At least `5m`. |
| `--compaction-interval` | `MCS_COMPACTION_INTERVAL` | `10m` | Interval of the retention cleanup. At least `1s`. |
| `--bookmark-interval` | `MCS_BOOKMARK_INTERVAL` | `5s` | Interval of watch `BOOKMARK` events. `1s` to `5m`. |
| `--watch-history` | `MCS_WATCH_HISTORY` | `10000` | Watch events kept per clusterset for resume. At least `1`. |
| `--watch-queue-size` | `MCS_WATCH_QUEUE_SIZE` | `1024` | Per-watcher event queue. A watcher that overflows it is sent a new snapshot. At least `1`. |
| `--max-watchers` | `MCS_MAX_WATCHERS` | `5000` | Concurrent watch streams per replica. Above it a watch fails with `TOO_MANY_WATCHERS`. At least `1`. |
| `--max-message-bytes` | `MCS_MAX_MESSAGE_BYTES` | `4194304` | gRPC send and receive message size limit. At least `1048576`. |
| `--advertise-endpoints` | `MCS_ADVERTISE_ENDPOINTS` | none | JSON array of `{"region","grpcUrl","httpUrl","primary"}` returned by `GetServerInfo`; clients append the `grpcUrl` values to their URL list as fallbacks. Each entry needs `region` and `grpcUrl` or `httpUrl` (`http` or `https` URLs); at most one entry is `primary`. |
| `--tls-cert-file` | `MCS_TLS_CERT_FILE` | none | TLS certificate of the gRPC and REST listeners. Set together with `--tls-key-file`. |
| `--tls-key-file` | `MCS_TLS_KEY_FILE` | none | TLS private key of the gRPC and REST listeners. |
| `--shutdown-drain` | `MCS_SHUTDOWN_DRAIN` | `5s` | Time the replica reports not ready on SIGTERM before it stops serving. At least `0s`. |
| `--shutdown-timeout` | `MCS_SHUTDOWN_TIMEOUT` | `25s` | Graceful stop of the gRPC and REST servers. At least `1s`. |
| `--otel-enabled` | `MCS_OTEL_ENABLED` | `false` | OpenTelemetry tracing over OTLP, configured by the standard `OTEL_*` environment variables. |
| `--log-level` | `MCS_LOG_LEVEL` | `info` | `debug`, `info`, `warn` or `error`. Logs are JSON. |

The DynamoDB store takes its credentials from the default AWS credential chain. The gRPC server
pings idle connections after 60 s (timeout 20 s) and accepts client pings every 10 s or more, also
without active streams.

### Registry Health

| Check | Passes when |
|---|---|
| `/healthz` | The process serves the admin port. |
| `/readyz` and the gRPC health service | The store answers a ping, every clusterset the store knows was loaded at least once, and no clusterset view is older than three refresh intervals. Fails from the start of shutdown. |

On SIGTERM a replica reports not ready, waits `--shutdown-drain`, ends open watches with
`UNAVAILABLE` and reason `SHUTTING_DOWN`, stops the REST and gRPC servers within
`--shutdown-timeout`, and flushes traces. Set the pod's `terminationGracePeriodSeconds` above the
sum of the two.

## mcs-controller Flags

The controller reads two environment variables: `MCS_REGISTRY_URLS` and `POD_NAMESPACE`, as the
defaults of `--registry-url` and `--operator-namespace`. It has no other environment fallbacks. An
invalid configuration exits with status 2 and lists every invalid setting.

| Flag | Default | Description |
|---|---|---|
| `--registry-url` | `$MCS_REGISTRY_URLS` | Registry gRPC URLs in failover order, repeatable or comma-separated. Required. Every URL uses the same scheme, `https` or `http`; `http` needs `--registry-allow-insecure-http`. A URL without a port uses `443` (`https`) or `80` (`http`). Any `--registry-url` on the command line replaces the environment value. |
| `--registry-tls-ca-file` | system roots | PEM bundle of the CAs that sign the registry certificates. The file must exist. |
| `--registry-allow-insecure-http` | `false` | Permit plaintext `http://` registry URLs. |
| `--clusterset-id` | none | Clusterset name, a DNS label. Without it, the ClusterProperty `clusterset.k8s.io` provides it. |
| `--cluster-id` | none | Cluster id, a DNS label unique in the clusterset. It is the `multicluster.kubernetes.io/source-cluster` value of this cluster's endpoints in every importing cluster. Without it, the ClusterProperty `cluster.clusterset.k8s.io` provides it. |
| `--cluster-identity-source` | `auto` | Where the two ids come from. `auto`: a flag wins, an id without its flag comes from its ClusterProperty, and a missing ClusterProperty CRD counts as no property. `flags`: both flags are required and ClusterProperty objects are not read. `clusterproperty`: both ClusterProperty objects are required, and flags, when set, must agree with them. |
| `--region` | none | Region of the registration. Empty uses the most common `topology.kubernetes.io/region` node label. |
| `--cluster-labels` | none | Comma-separated `key=value` labels of the registration. Keys are qualified names and values valid label values; at most 64. |
| `--lease-ttl` | `30s` | Cluster lease TTL requested from the registry, `10s` to `5m`. The leader heartbeats every TTL/3, with 10 percent jitter. |
| `--watch-namespaces` | all namespaces | Comma-separated namespaces to export from and import into. Restricts the caches of Services, EndpointSlices, ServiceExports and ServiceImports. |
| `--create-namespaces` | `false` | Create a missing namespace of an import instead of holding the import pending. The controller never deletes namespaces. |
| `--adopt-unmanaged-imports` | `false` | Take over ServiceImports that the controller did not create. |
| `--write-cluster-properties` | `false` | Create the ClusterProperty objects `clusterset.k8s.io` and `cluster.clusterset.k8s.io` from the resolved ids. An existing property with another value stops the controller. |
| `--deregister-on-shutdown` | `false` | Deregister the cluster when the controller stops, which deletes its exports in the registry. |
| `--endpoints-per-slice` | `100` | Endpoints per imported EndpointSlice, `1` to `1000`. |
| `--max-concurrent-reconciles` | `4` | Workers of each controller. At least `1`. |
| `--metrics-bind-address` | `:8080` | Address of `/metrics`. `0` disables it. |
| `--health-probe-bind-address` | `:8081` | Address of `/healthz` and `/readyz`. Required. |
| `--leader-elect` | `true` | Leader election. Only the leader writes to the registry and applies imports; required for more than one replica. |
| `--operator-namespace` | `$POD_NAMESPACE` | Namespace of the leader election Lease, a namespace name. Required with `--leader-elect`. |
| `--otel-enabled` | `false` | OpenTelemetry tracing over OTLP, configured by the standard `OTEL_*` environment variables. |
| `--kubeconfig` | in-cluster configuration | Kubeconfig path, for running out of cluster. |
| `--zap-devel` | `false` | Development logging: console encoder, debug level, stack traces from warn. |
| `--zap-encoder` | `json` | Log encoding, `json` or `console`. |
| `--zap-log-level` | `info` | `debug`, `info`, `error`, `panic`, or an integer for more verbose debug levels. |
| `--zap-stacktrace-level` | `error` | Level at and above which stack traces are logged: `info`, `error` or `panic`. |
| `--zap-time-encoding` | `epoch` | `epoch`, `millis`, `nano`, `iso8601`, `rfc3339` or `rfc3339nano`. |

### Controller Health

| Check | Passes when |
|---|---|
| `/healthz` | The process answers. |
| `/readyz` | The registry client connected at least once and, on the leader, the cluster lease is registered. |

## mcs-admin

`mcs-admin` reads and repairs a store directly, without a running registry:

```text
mcs-admin [store flags] describe [--clusterset NAME] [--output text|json]
mcs-admin [store flags] delete-cluster --clusterset NAME --cluster NAME --yes
mcs-admin [store flags] create-table [create-table flags]
```

`describe` and `delete-cluster` act on the store the store flags open. The memory store lives in
the registry process, so use them with `--store=dynamodb`; with the default `--store=memory` they
print a warning to stderr and see an empty store. Exit status: `0` on success, `1` on a
failure, `2` on a usage error.

### Store Flags

Given before the command. Every store flag has the same `MCS_` environment fallback as the
`mcs-registry` flag of the same name.

| Flag | Environment variable | Default | Description |
|---|---|---|---|
| `--store` | `MCS_STORE` | `memory` | `memory` or `dynamodb`. |
| `--dynamodb-table` | `MCS_DYNAMODB_TABLE` | `global-discovery-mcs` | DynamoDB table. |
| `--dynamodb-endpoint` | `MCS_DYNAMODB_ENDPOINT` | none | DynamoDB endpoint override, for dynamodb-local. |
| `--region` | `MCS_REGION`, then `AWS_REGION` | none | AWS region. |
| `--lease-skew` | `MCS_LEASE_SKEW` | `2s` | Clock skew allowance when `describe` evaluates lease states. |

### Commands

| Command | Flags | Description |
|---|---|---|
| `describe` | `--clusterset` (default: every clusterset), `--output` (`text` or `json`, default `text`) | Lists each clusterset's clusters with region, lease state, epoch, renewal, expiry and generation; its exports with type, generation and export creation time; and its endpoint sets with endpoint and ready counts. |
| `delete-cluster` | `--clusterset`, `--cluster` (DNS labels, required), `--yes` | Deletes the cluster with its exports and endpoint sets. Without `--yes` it prints what it would delete and exits with status 2. |
| `create-table` | See below | Creates the DynamoDB table, waits until it is `ACTIVE`, then enables TTL on `expires_at` and point-in-time recovery. On an existing table it checks the key schema and enables only what is missing. |

`create-table` takes its own flags after the command. `--table`, `--region` and `--endpoint` default
to the store flags `--dynamodb-table`, `--region` and `--dynamodb-endpoint`, so the order of
precedence is the `create-table` flag, then the store flag, then its `MCS_` environment variable,
then the built-in default:

| Flag | Default | Description |
|---|---|---|
| `--table` | `--dynamodb-table`, else `global-discovery-mcs` | Table name. |
| `--region` | `--region` store flag, else `AWS_REGION` or the shared AWS configuration | AWS region. |
| `--endpoint` | `--dynamodb-endpoint` store flag, else none | DynamoDB endpoint override, for dynamodb-local. |
| `--deletion-protection` | `true` | Create the table with deletion protection. |
| `--pitr` | `true` | Enable point-in-time recovery. dynamodb-local does not support it; pass `--pitr=false`. |
| `--ttl` | `true` | Enable TTL on `expires_at`. |
| `--wait-timeout` | `5m` | How long to wait for the table to become `ACTIVE`. |

## Global Discovery Service Flags for MCS

The Global Discovery Service reads the EndpointSlices that `mcs-controller` imports with these
flags. It does not connect to the registry.

| Flag | Default | Description |
|---|---|---|
| `--pool-discovery` | `registrations` | `mcs` reads imported EndpointSlices labelled `multicluster.kubernetes.io/source-cluster` and `multicluster.kubernetes.io/service-name`. `karmada` reads Karmada's imports. |
| `--pool-service-suffix` | `-pool-relay` | A pool's PoolRelay Service is named `<DGD name><suffix>`. Must not be empty. |
| `--router-namespace` | none | Namespace of the exported Global Router Service. Required with `mcs`. |
| `--router-service` | none | The exported Global Router Service. Required with `mcs`. |
| `--router-port-name` | none | Name of the router Service's TCP port that PoolRelays connect to. Required with `mcs`. |

With `--pool-discovery mcs`, the flags of the `registrations` mode (`--registration-storage`,
`--heartbeat-interval-secs`, `--lease-duration-secs` and `--router`) fail at startup, and
`POST /pools` answers `501`. The service needs `get`, `list` and `watch` on
`discovery.k8s.io` `endpointslices` in every namespace.

## Labels, Annotations and Names

| Key or name | On | Value |
|---|---|---|
| `dynamo.nvidia.com/mcs-export` | Service | `"true"` makes the controller create and own a ServiceExport of the same name. |
| Other `dynamo.nvidia.com/mcs-*` labels and annotations | Service | Forwarded to the registry as exported labels and annotations, with `spec.exportedLabels` and `spec.exportedAnnotations` of the ServiceExport, which win on the same key. |
| `dynamo.nvidia.com/mcs-managed-by` | ServiceExports, ServiceImports, derived Services, EndpointSlices and namespaces the controller creates | `mcs-controller`. Not forwarded. |
| `dynamo.nvidia.com/mcs-clusterset` | The same objects | The clusterset. Not forwarded. |
| `dynamo.nvidia.com/mcs-service-export` | ServiceExport finalizer | The controller deletes the registry record before removing it. |
| `multicluster.kubernetes.io/service-name` | Derived Services and imported EndpointSlices | Name of the exported Service. |
| `multicluster.kubernetes.io/source-cluster` | Imported EndpointSlices | Cluster id of the exporting cluster. |
| `kubernetes.io/service-name` | Imported EndpointSlices | Name of the derived Service. |
| `endpointslice.kubernetes.io/managed-by` | Imported EndpointSlices | `mcs-controller.dynamo.nvidia.com`. The controller never exports slices with this value. |
| `multicluster.kubernetes.io/derived-service` | ServiceImport annotation | Name of the derived Service. |
| Derived Service name | Importing clusters | `derived-` plus the first ten characters of the lowercase, unpadded base32hex encoding of the SHA-256 of `<namespace>/<name>`, as in `sigs.k8s.io/mcs-api`. |
| Imported EndpointSlice name | Importing clusters | `<derived Service>-<source cluster>-<index>`. |
| `targetRef.kind`, `targetRef.name`, `targetRef.uid` | Endpoint attributes in the registry | The exported endpoint's target reference. The importer sets the imported endpoint's `targetRef` from the name, the kind (`Pod` when absent) and `targetRef.namespace` when present. |
| `clusterset.k8s.io`, `cluster.clusterset.k8s.io` | ClusterProperty objects (`about.k8s.io/v1alpha1`) | Clusterset name and cluster id. |
| `mcs-controller` | Server-side apply field manager | |
| `mcs-controller.dynamo.nvidia.com` | Leader election Lease, in `--operator-namespace` | |
| `mcs.dynamo.nvidia.com` | Domain of every registry error's `ErrorInfo` | |

## Conditions

### ServiceExport

The controller sets `Valid` itself and copies `Ready` and `Conflict` from the status the registry
reports for its cluster.

| Type | Status | Reason | Meaning |
|---|---|---|---|
| `Valid` | `True` | `Valid` | The Service exists and is exportable. |
| `Valid` | `False` | `NoService` | No Service of the same name exists. The registry record is deleted. |
| `Valid` | `False` | `InvalidServiceType` | The Service is of type ExternalName. The registry record is deleted. |
| `Ready` | `True` | `Ready` | The registry holds the export and the cluster's lease is active. |
| `Ready` | `Unknown` | `Pending` | The registry has not reported the export yet, or the cluster is not registered. |
| `Ready` | `False` | `Failed` | The registry rejected the export, or the export is not valid. |
| `Ready` | `False` | `ClusterLeaseExpired` | The cluster's lease expired; its endpoints left every import. |
| `Conflict` | `False` | `NoConflicts` | The exports of the Service agree. |
| `Conflict` | `True` | A conflict reason | The exports of the Service disagree. Set on every participating cluster, whichever cluster's export differs. |
| `Conflict` | `Unknown` | `ClusterLeaseExpired` | The cluster's lease expired. |

An invalid ServiceExport has no `Conflict` condition. Conflict reasons, in the order the registry
checks them; it reports the first that applies:

| Reason | The exports differ in |
|---|---|
| `PortConflict` | Ports: a name or app protocol of the same (port, protocol), or a name used by another (port, protocol) |
| `IPFamilyConflict` | IP families, with no family in common |
| `TypeConflict` | Type: ClusterSetIP or Headless |
| `SessionAffinityConflict` | Session affinity |
| `SessionAffinityConfigConflict` | Session affinity client IP timeout |
| `AnnotationsConflict` | Exported annotations |
| `LabelsConflict` | Exported labels |
| `InternalTrafficPolicyConflict` | Internal traffic policy |
| `TrafficDistributionConflict` | Traffic distribution |

`IPFamilyConflict` and `ClusterLeaseExpired` are registry additions; the other reasons are the
`sigs.k8s.io/mcs-api` constants.

### ServiceImport

| Type | Status | Reason | Meaning |
|---|---|---|---|
| `Ready` | `True` | `Ready` | At least one exporting cluster participates. The message counts them. |
| `Ready` | `False` | `Pending` | No exporting cluster participates: their leases expired. |
| `Conflict` | `False` | `NoConflicts` | The exports agree. |
| `Conflict` | `True` | A conflict reason | The exports disagree. |

`status.clusters` lists the participating clusters.

### Events

The controller writes Events on ServiceExports through the `events.k8s.io` API, with reporting
controller `mcs-controller` and action `Export`.

| Reason | Type | When |
|---|---|---|
| `InvalidExport` | Warning | The ServiceExport became invalid (`NoService` or `InvalidServiceType`). |
| `EndpointsTruncated` | Warning | The Service has more than 1000 endpoints. Ready endpoints are kept first. |
| `ExportFailed` | Warning | The registry rejected a write as invalid. |

## Registry API

The gRPC services are `ClusterService`, `ServiceExportService`, `ServiceImportService` and
`ServerInfoService` in package `dynamo.mcs.v1`. The REST gateway maps them as follows, where `{cs}`
is the clusterset; every path segment is a DNS-1123 label:

| RPC | Method and path |
|---|---|
| `RegisterCluster` | `PUT /v1/clustersets/{cs}/clusters/{cluster}` |
| `HeartbeatCluster` | `POST /v1/clustersets/{cs}/clusters/{cluster}:heartbeat` |
| `DeregisterCluster` | `DELETE /v1/clustersets/{cs}/clusters/{cluster}` |
| `GetCluster` | `GET /v1/clustersets/{cs}/clusters/{cluster}` |
| `ListClusters` | `GET /v1/clustersets/{cs}/clusters` |
| `WatchClusters` | `GET /v1/clustersets/{cs}/clusters:watch` |
| `ListClusterSets` | `GET /v1/clustersets` |
| `PutServiceExport` | `PUT /v1/clustersets/{cs}/clusters/{cluster}/exports/{namespace}/{name}` |
| `DeleteServiceExport` | `DELETE /v1/clustersets/{cs}/clusters/{cluster}/exports/{namespace}/{name}` |
| `ListServiceExports` | `GET /v1/clustersets/{cs}/clusters/{cluster}/exports` |
| `ReplaceEndpoints` | `PUT /v1/clustersets/{cs}/clusters/{cluster}/exports/{namespace}/{name}/endpoints` |
| `GetServiceImport` | `GET /v1/clustersets/{cs}/imports/{namespace}/{name}` |
| `ListServiceImports` | `GET /v1/clustersets/{cs}/imports` |
| `WatchServiceImports` | `GET /v1/clustersets/{cs}/imports:watch` |
| `GetServerInfo` | `GET /v1/serverinfo` |

JSON follows the proto3 JSON mapping: lowerCamelCase field names, enums as their full names (for
example `SERVICE_TYPE_CLUSTER_SET_IP`), 64-bit integers as strings. GET filters are query parameters,
for example `?namespace=dynamo&includeEndpoints=true`. A REST watch is a chunked response of
newline-delimited JSON objects, each wrapped in `result`.

### Errors

Every error is a `google.rpc.Status` with a `google.rpc.ErrorInfo` detail in domain
`mcs.dynamo.nvidia.com`:

| Reason | gRPC code | HTTP | Meaning |
|---|---|---|---|
| `STALE_GENERATION` | `FAILED_PRECONDITION` | 400 | The write's generation does not exceed the stored one; `metadata.current_generation` holds it. |
| `LEASE_EPOCH_MISMATCH` | `ABORTED` | 409 | The heartbeat's lease epoch is not current. Register again. |
| `CLUSTER_NOT_REGISTERED` | `FAILED_PRECONDITION` or `NOT_FOUND` | 400 or 404 | The cluster has no registration. |
| `TOO_MANY_ENDPOINTS` | `INVALID_ARGUMENT` | 400 | The endpoint set exceeds 1000 endpoints. |
| `TOO_MANY_WATCHERS` | `RESOURCE_EXHAUSTED` | 429 | The replica is at `--max-watchers`. |
| `STORE_UNAVAILABLE` | `UNAVAILABLE` | 503 | The store did not answer. |
| `SHUTTING_DOWN` | `UNAVAILABLE` | 503 | The replica is draining. |
| `INVALID_ARGUMENT` | `INVALID_ARGUMENT` | 400 | Invalid input; a `google.rpc.BadRequest` detail names every invalid field. |
| `NOT_FOUND` | `NOT_FOUND` | 404 | No such ServiceImport. |
| `INTERNAL` | `INTERNAL` | 500 | An unexpected error. |

### Limits

| Item | Limit |
|---|---|
| Clusterset, cluster, namespace and name | DNS-1123 label |
| Labels per object | 64 |
| Annotations per object | 32, at most 16 KiB in total |
| Ports per export | 64 |
| Endpoints per export per cluster | 1000 |
| Attributes per endpoint | 16 |
| Message size | 4 MiB (`--max-message-bytes`) |
| Cluster lease TTL | 10 s to 5 m, default 30 s |

## Metrics

Counters and labelled gauges are created with every known label value, so they appear on the first
scrape.

### mcs-registry

Served on the admin port, `9091`.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `dynamo_mcs_registry_clusters` | gauge | `clusterset`, `state` (`active`, `expired`) | Registered clusters by lease state |
| `dynamo_mcs_registry_exports` | gauge | `clusterset` | ServiceExport records |
| `dynamo_mcs_registry_endpoints` | gauge | `clusterset`, `condition` (`ready`, `not_ready`) | Endpoints in stored endpoint sets |
| `dynamo_mcs_registry_service_imports` | gauge | `clusterset`, `conflict` (`true`, `false`) | Merged ServiceImports |
| `dynamo_mcs_registry_lease_expirations_total` | counter | `kind` (`cluster`) | Leases this replica saw expire |
| `dynamo_mcs_registry_watchers` | gauge | `kind` (`cluster`, `service_import`) | Open watch streams |
| `dynamo_mcs_registry_watch_events_sent_total` | counter | `kind`, `type` (`snapshot_start`, `upsert`, `delete`, `snapshot_end`, `bookmark`) | Watch events sent |
| `dynamo_mcs_registry_watch_resyncs_total` | counter | `kind`, `reason` (`lagged`, `revision_unknown`, `revision_expired`) | Snapshots sent instead of deltas |
| `dynamo_mcs_registry_watch_rejected_total` | counter | `reason` (`too_many_watchers`, `shutting_down`) | Watch streams refused |
| `dynamo_mcs_registry_view_refresh_duration_seconds` | histogram | | Duration of one clusterset refresh from the store |
| `dynamo_mcs_registry_view_refresh_failures_total` | counter | | Failed clusterset refreshes and discoveries |
| `dynamo_mcs_registry_view_age_seconds` | gauge | | Age of the oldest clusterset view since its last successful refresh |
| `dynamo_mcs_registry_store_request_duration_seconds` | histogram | `operation`, `result` (`ok`, `conditional`, `error`) | Store request duration |
| `dynamo_mcs_registry_store_conditional_failures_total` | counter | `operation`, `reason` | Store writes refused by their condition: `renew_cluster_lease` (`lease_epoch_mismatch`, `not_found`), `put_export` (`stale_generation`), `put_endpoint_set` (`stale_generation`, `too_large`) |
| `dynamo_mcs_registry_compaction_deleted_total` | counter | `kind` (`cluster`, `export`, `endpoint_set`) | Records removed by compaction |
| `dynamo_mcs_registry_build_info` | gauge | `version`, `git_hash` | Build metadata; always 1 |

Store operations: `put_cluster`, `renew_cluster_lease`, `get_cluster`, `delete_cluster`,
`put_export`, `delete_export`, `put_endpoint_set`, `delete_endpoint_set`, `list_cluster_sets`,
`load_cluster_set` and `ping`.

The gRPC server adds `dynamo_mcs_registry_grpc_server_started_total`,
`dynamo_mcs_registry_grpc_server_handled_total` (with `grpc_code`),
`dynamo_mcs_registry_grpc_server_msg_received_total`,
`dynamo_mcs_registry_grpc_server_msg_sent_total` and the histogram
`dynamo_mcs_registry_grpc_server_handling_seconds`, labelled `grpc_type`, `grpc_service` and
`grpc_method`.

### mcs-controller

Served on `--metrics-bind-address`, `:8080`.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `dynamo_mcs_controller_exports` | gauge | `state` (`exported`, `pending`, `invalid`, `failed`) | ServiceExports by state |
| `dynamo_mcs_controller_reconcile_duration_seconds` | histogram | `controller` (`serviceexport`, `autoexport`, `serviceimport`) | Reconcile duration |
| `dynamo_mcs_controller_reconciles_total` | counter | `controller`, `result` (`success`, `error`) | Reconciles |
| `dynamo_mcs_controller_export_writes_total` | counter | `op` (`put_export`, `delete_export`), `result` (`success`, `stale_generation`, `not_registered`, `error`) | `PutServiceExport` and `DeleteServiceExport` calls |
| `dynamo_mcs_controller_endpoint_replacements_total` | counter | `result` (`success`, `stale_generation`, `not_registered`, `error`) | `ReplaceEndpoints` calls |
| `dynamo_mcs_controller_exported_endpoints` | gauge | | Endpoints of all exports as last written |
| `dynamo_mcs_controller_imports` | gauge | | ServiceImports in the controller's view of the registry |
| `dynamo_mcs_controller_import_events_total` | counter | `type` (`snapshot_start`, `upsert`, `delete`, `snapshot_end`, `bookmark`) | `WatchServiceImports` events received |
| `dynamo_mcs_controller_imports_pending_namespace` | gauge | | Imports waiting for their namespace |
| `dynamo_mcs_controller_stream_reconnects_total` | counter | | Registry client reconnects after the first connection |
| `dynamo_mcs_controller_resyncs_total` | counter | `reason` (`startup`, `reconnect`, `reregister`, `snapshot`) | Full resyncs of the exports for `startup`, `reconnect` and `reregister`; restarts of the import watch with a new snapshot for `snapshot` |
| `dynamo_mcs_controller_gc_deletions_total` | counter | `kind` (`registry_export`, `service_import`, `service`, `endpoint_slice`) | Objects deleted because their source is gone |
| `dynamo_mcs_controller_heartbeat_failures_total` | counter | `reason` (`epoch_mismatch`, `not_registered`, `error`) | Failed cluster heartbeats |
| `dynamo_mcs_controller_registrations_total` | counter | `result` (`success`, `error`) | `RegisterCluster` calls |
| `dynamo_mcs_controller_cluster_registered` | gauge | | 1 while this replica holds a registered cluster lease |

controller-runtime adds its `controller_runtime_*` and `workqueue_*` metrics on the same endpoint.

### Registry Client

The controller's registry client adds these metrics on the controller's endpoint:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `dynamo_mcs_client_requests_total` | counter | `rpc`, `code` | Calls by RPC and gRPC status code; a watch counts once, when it ends |
| `dynamo_mcs_client_request_duration_seconds` | histogram | `rpc` | Unary call duration, retries included |
| `dynamo_mcs_client_endpoint_failovers_total` | counter | | Connections to a different registry URL than the previous one, the return to the primary included |
| `dynamo_mcs_client_connected` | gauge | `url` | 1 for the registry URL in use, 0 for the other configured and advertised URLs |
| `dynamo_mcs_client_watch_reconnects_total` | counter | `watch` (`clusters`, `service_imports`) | Watch streams reopened |

## RBAC

`mcs-registry` does not use the Kubernetes API. With the DynamoDB store, its IAM role needs
`dynamodb:GetItem`, `Query`, `PutItem`, `UpdateItem`, `DeleteItem` and `BatchWriteItem` on the
table in each region; the module's
[IAM document](https://github.com/ai-dynamo/dynamo/blob/main/deploy/global-discovery-mcs/docs/iam.md)
has the policies for the registry and for `mcs-admin create-table`.

`mcs-controller` needs the ClusterRole below, generated from the code into
`deploy/global-discovery-mcs/config/rbac/role.yaml`, and a Role in its own namespace for leader
election on `coordination.k8s.io` `leases` (`get`, `list`, `watch`, `create`, `update`, `patch`,
`delete`).

| API group | Resources | Verbs | Used for |
|---|---|---|---|
| `multicluster.x-k8s.io` | `serviceexports`, `serviceimports` | `get`, `list`, `watch`, `create`, `update`, `patch`, `delete` | Exports, label-driven exports and imports |
| `multicluster.x-k8s.io` | `serviceexports/status`, `serviceimports/status` | `get`, `update`, `patch` | Conditions and import status |
| `multicluster.x-k8s.io` | `serviceexports/finalizers` | `update` | The ServiceExport finalizer |
| core | `services` | `get`, `list`, `watch`, `create`, `update`, `patch`, `delete` | Exported Services and derived Services |
| `discovery.k8s.io` | `endpointslices` | `get`, `list`, `watch`, `create`, `update`, `patch`, `delete` | Exported endpoints and imported EndpointSlices |
| core | `namespaces` | `get`, `list`, `watch`, `create` | Pending imports; `create` only for `--create-namespaces` |
| core | `nodes` | `get`, `list`, `watch` | Region and zones |
| `about.k8s.io` | `clusterproperties` | `get`, `list`, `watch`, `create`, `update`, `patch` | Identity; writes only for `--write-cluster-properties` |
| core, `events.k8s.io` | `events` | `create`, `patch` | Events on ServiceExports |

The Global Discovery Service needs `get`, `list` and `watch` on `discovery.k8s.io` `endpointslices`
in every namespace.

## CRDs

The controller needs three CRDs, which the image ships in `/opt/global-discovery-mcs/crds`:

| CRD | Versions | Source |
|---|---|---|
| `serviceexports.multicluster.x-k8s.io` | `v1alpha1` and `v1beta1` served, `v1beta1` stored | `sigs.k8s.io/mcs-api` v0.5.2 |
| `serviceimports.multicluster.x-k8s.io` | `v1alpha1` and `v1beta1` served, `v1beta1` stored | `sigs.k8s.io/mcs-api` v0.5.2 |
| `clusterproperties.about.k8s.io` | `v1alpha1` | KEP-2149 |

The controller reads and writes `multicluster.x-k8s.io/v1beta1`, so a cluster where another
installation owns these CRDs must serve that version.
