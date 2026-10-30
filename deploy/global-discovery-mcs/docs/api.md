<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Registry API

The registry API is the protobuf package `dynamo.mcs.v1` in
`api/proto/dynamo/mcs/v1`. Every RPC is served over gRPC and mirrored as
REST/JSON by grpc-gateway. The OpenAPI description is
`api/gen/openapi/mcs.swagger.json`.

The API is additive only. A breaking change means a new `v2` package.

## Services

| Service | File | Callers |
| :---- | :---- | :---- |
| `ClusterService` | `cluster.proto` | `mcs-controller` (writes), anyone (reads) |
| `ServiceExportService` | `mcs.proto` | `mcs-controller` of the exporting cluster |
| `ServiceImportService` | `mcs.proto` | `mcs-controller` of every cluster |
| `ServerInfoService` | `server.proto` | every client, for fallback endpoints |

## REST paths

`{cs}` is the clusterset. Path segments are DNS-1123 labels.

| RPC | Method and path |
| :---- | :---- |
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

Writes with a body take the request message as JSON (`body: "*"`); path
parameters override body fields. GET filters are query parameters named
after the request fields, for example
`GET /v1/clustersets/prod/imports?namespace=llm&selector.matchLabels[app]=frontend&includeEndpoints=true`.
`api/gateway_test.go` pins every path to its RPC.

## JSON

JSON follows the proto3 JSON mapping:

- Field names are lowerCamelCase on output (`readyEndpointCount`). Input
  accepts lowerCamelCase and the proto names (`ready_endpoint_count`).
- Enums are strings with their full names, for example
  `"SERVICE_TYPE_CLUSTER_SET_IP"` and `"LEASE_STATE_ACTIVE"`.
- Responses omit fields with default values (empty strings, zero counts,
  empty lists); read a missing field as its default. Unknown input fields
  are ignored.
- `uint64` fields (`generation`, `epoch`) are strings: `"generation": "1761480000000"`.
- Durations are strings with an `s` suffix (`"leaseTtl": "30s"`); timestamps
  are RFC 3339 strings.

Register a cluster, then heartbeat it with the returned epoch:

```bash
curl -sX PUT http://mcs-registry:8080/v1/clustersets/prod/clusters/use1-a \
  -d '{"region":"us-east-1","zones":["us-east-1a"],"leaseTtl":"30s"}'
curl -sX POST http://mcs-registry:8080/v1/clustersets/prod/clusters/use1-a:heartbeat \
  -d '{"epoch":"1761480000000"}'
```

## Writers, generations and leases

- Every record has one logical writer: the `mcs-controller` of its
  cluster.
- Writes carry a `generation` that must exceed the stored one. Writers use
  `max(previous + 1, unix milliseconds)`, which stays monotonic across
  restarts and region failover. A stale write fails with
  `STALE_GENERATION` and the stored generation.
- `RegisterCluster` mints a new lease epoch, `max(previous + 1, unix
  milliseconds)`. Heartbeats must carry it; any other epoch fails with
  `LEASE_EPOCH_MISMATCH` and the writer registers again.
- Lease TTLs are clamped to [10 s, 5 m], default 30 s. Writers heartbeat
  every TTL/3.
- A lease expires at read time once `renewed_at + ttl + skew` (skew 2 s) is
  in the past. Nothing writes expiry.
- `ReplaceEndpoints` replaces the whole endpoint set; there is no
  per-endpoint write. An empty list clears it.
- A write with `generation` 0 (or none) asks the registry to assign
  `max(stored + 1, unix milliseconds)`. That suits a single writer such as a
  curl session; controllers and clients send their own.
- `PutServiceExport` and `ReplaceEndpoints` need a registered cluster
  (`FAILED_PRECONDITION` `CLUSTER_NOT_REGISTERED`). A heartbeat for an
  unknown cluster is `NOT_FOUND` `CLUSTER_NOT_REGISTERED`.
- Defaults: port protocol TCP, session affinity None, and an endpoint
  address type inferred from its first address (IPv4, IPv6, else FQDN).
  `spec.type` is required. Without `export_creation_timestamp` the registry
  keeps the stored one, or uses the time of the first write; controllers
  send the ServiceExport's creationTimestamp.
- Endpoint `attributes` are opaque to the registry and copied into the
  import. `mcs-controller` uses `targetRef.kind`, `targetRef.name` and
  `targetRef.uid` (`api/labels.go`).

## ServiceImports

The registry merges the exports of one `(namespace, name)` from every
cluster with the conflict resolution of KEP-1645
(`internal/registry/merge`):

- Only exports of clusters with an active lease take part, oldest
  `export_creation_timestamp` first, ties broken by cluster name. The first
  one wins.
- Ports are the union by (port, protocol); a name or app protocol
  disagreement resolves to the oldest export, and a port whose name is
  already used by another (port, protocol) is skipped.
- IP families are intersected; an export without a common family is
  skipped. An export that lists no family does not constrain the result.
- Type, session affinity and its config, internal traffic policy, traffic
  distribution, annotations and labels come from the winner.

`conflict` is set when any participating export disagrees with the winner
(on ports, IP families, type, session affinity, its client IP timeout,
annotations, labels, internal traffic policy or traffic distribution), with
the mcs-api reason (`PortConflict`, `TypeConflict`, ...). Each
`clusters[]` entry carries:

- `participating`: false when the cluster's lease is not active.
- `conflict`: set when this cluster's own export disagrees with the winner.
- `conditions`: `Valid` (`True/Valid`), `Ready` (`True/Ready`, or
  `False/ClusterLeaseExpired`) and `Conflict` (`True` with the import's
  conflict reason on every participating cluster, as KEP-1645 asks,
  `False/NoConflicts`, or `Unknown/ClusterLeaseExpired`). Controllers copy
  them onto their ServiceExports.

`endpoints[]` has one entry per participating cluster, with the cluster's
region and counts always set and the endpoint list only with
`include_endpoints`. An import whose exporting clusters all expired keeps
its shape with no participants and no endpoints until compaction removes
the records.

### Traffic policies

The read-side `ServiceImport` carries `internal_traffic_policy` (field 15)
and `traffic_distribution` (field 16) from the winning export. These are
computed fields, not separately writable import settings; the existing
export validation and oldest-active-export ownership rules apply.
`include_endpoints=false` omits endpoint lists only, not either policy.

For example, the source Kubernetes Service can set:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference
  namespace: llm
spec:
  selector:
    app: inference
  internalTrafficPolicy: Local
  trafficDistribution: PreferClose
  ports:
    - name: http
      port: 80
      targetPort: 8080
```

Its winning import includes the following REST/JSON fields:

```json
{
  "internalTrafficPolicy": "INTERNAL_TRAFFIC_POLICY_LOCAL",
  "trafficDistribution": "PreferClose"
}
```

An absent source policy remains `INTERNAL_TRAFFIC_POLICY_UNSPECIFIED` on
the wire; absent traffic distribution remains empty. JSON omits these
default-valued fields. Kubernetes consumers interpret an unspecified
internal traffic policy as the Kubernetes default, `Cluster`, and omit an
empty traffic distribution rather than inventing a preference.

The two fields are additive: older protobuf clients ignore them, while a
newer client reading an older registry receives their unspecified/empty
values.
Upgrade the registry and controller to propagate non-default policies
end to end. No stored record or table schema changes are required.

## Watches

`WatchClusters` and `WatchServiceImports` stream events. Each response has a
`type` (`EventType`), the object for `UPSERT` and `DELETE`, and a
`revision`.

1. `EVENT_TYPE_SNAPSHOT_START`: discard local state.
2. One `EVENT_TYPE_UPSERT` per object.
3. `EVENT_TYPE_SNAPSHOT_END` with the revision.
4. Deltas: `EVENT_TYPE_UPSERT` with the full object, `EVENT_TYPE_DELETE` with
   the last known object.
5. `EVENT_TYPE_BOOKMARK` with the current revision every 5 s. It keeps
   load balancer idle timeouts (350 s on an AWS NLB) and chunked REST
   streams alive.

The revision is an opaque `view_epoch:seq` token of the serving replica.
Pass the last revision seen as `resume_revision` to resume: the same replica
replays the missed deltas; any other replica, or a revision too old for its
history (10000 events per clusterset), starts with a new snapshot. A resume
never fails and never skips events.

A watcher that falls 1024 events behind is re-snapshotted, not
disconnected. A replica serves at most `--max-watchers` (5000) streams;
above that a watch fails with `TOO_MANY_WATCHERS`. Filters (`namespace`,
`selector` and `include_endpoints` for imports, `include_expired` for
clusters) apply per subscriber: an object that stops matching arrives as
`DELETE` with its new state. A bookmark's revision covers changes the filter
skipped. A draining replica ends watches with `SHUTTING_DOWN`; reconnect,
usually to another replica, which sends a new snapshot. See
`consistency.md` for revisions and resume.

Over REST a watch is a chunked response of newline-delimited JSON objects,
one per event, each wrapped in `result`:

```text
{"result":{"type":"EVENT_TYPE_SNAPSHOT_START"}}
{"result":{"type":"EVENT_TYPE_UPSERT","serviceImport":{"clusterset":"prod","namespace":"llm","name":"frontend","type":"SERVICE_TYPE_CLUSTER_SET_IP"}}}
{"result":{"type":"EVENT_TYPE_SNAPSHOT_END","revision":"1761480000000-1a2b3c4d:42"}}
{"result":{"type":"EVENT_TYPE_BOOKMARK","revision":"1761480000000-1a2b3c4d:42"}}
```

A stream that ends with an error sends one last line `{"error":{...}}` with
the status described below.

```bash
curl -sN 'http://mcs-registry:8080/v1/clustersets/prod/imports:watch?namespace=llm'
```

## Errors

Errors are `google.rpc.Status` values. Over REST the body is
`{"code": <gRPC code>, "message": "...", "details": [...]}`. Every registry
error carries a `google.rpc.ErrorInfo` detail with domain
`mcs.dynamo.nvidia.com` and a reason:

| Reason | gRPC code | HTTP | Meaning and client action |
| :---- | :---- | :---- | :---- |
| `STALE_GENERATION` | `FAILED_PRECONDITION` | 400 | The generation does not exceed the stored one. `metadata.current_generation` and a `PreconditionFailure` detail hold the stored generation; retry with a larger one. |
| `LEASE_EPOCH_MISMATCH` | `ABORTED` | 409 | The heartbeat epoch is not current. Register again and resend state. |
| `CLUSTER_NOT_REGISTERED` | `FAILED_PRECONDITION` or `NOT_FOUND` | 400 or 404 | The cluster has no registration. Register first. |
| `TOO_MANY_ENDPOINTS` | `INVALID_ARGUMENT` | 400 | The endpoint set exceeds 1000 endpoints. |
| `TOO_MANY_WATCHERS` | `RESOURCE_EXHAUSTED` | 429 | The replica is at `--max-watchers`. Back off or try another replica. |
| `STORE_UNAVAILABLE` | `UNAVAILABLE` | 503 | The store did not answer. Retry with backoff. |
| `SHUTTING_DOWN` | `UNAVAILABLE` | 503 | The replica is draining. Reconnect; another replica serves. |
| `INVALID_ARGUMENT` | `INVALID_ARGUMENT` | 400 | Invalid input. A `google.rpc.BadRequest` detail names every invalid field. |
| `NOT_FOUND` | `NOT_FOUND` | 404 | No such ServiceImport. |
| `INTERNAL` | `INTERNAL` | 500 | An unexpected error, logged by the registry. |

The Go constants are in `api/errors.go`.

Example body:

```json
{
  "code": 9,
  "message": "export generation 5 does not exceed stored generation 7",
  "details": [
    {
      "@type": "type.googleapis.com/google.rpc.ErrorInfo",
      "reason": "STALE_GENERATION",
      "domain": "mcs.dynamo.nvidia.com",
      "metadata": {"current_generation": "7"}
    }
  ]
}
```

## Limits

| Item | Limit |
| :---- | :---- |
| Identifiers (clusterset, cluster, namespace, name) | DNS-1123 label |
| Labels per object | 64 |
| Annotations per object | 32, at most 16 KiB in total |
| Ports per export | 64 |
| Endpoints per export per cluster | 1000 |
| Attributes per endpoint | 16 |
| Message size | 4 MiB |

The constants are in `api/errors.go`.

## Labels and derived Services

`mcs-controller` exports a Service labelled `dynamo.nvidia.com/mcs-export:
"true"` and forwards the Service's other `dynamo.nvidia.com/mcs-*` labels
and annotations as exported labels and annotations; the registry stores and
merges them without interpreting them. The keys are in `api/labels.go`.

Clients dial an import through its derived Service,
`derived-<hash>.<namespace>.svc.cluster.local`, where the name is
`api.DerivedServiceName(namespace, name)`, the same algorithm as
`sigs.k8s.io/mcs-api`. Consumers that need every cluster's endpoints of
one Service, such as the Global Discovery Service, read the imported
EndpointSlices instead, which carry the source cluster.
