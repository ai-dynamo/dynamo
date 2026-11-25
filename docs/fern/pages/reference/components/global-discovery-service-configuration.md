---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Global Discovery Service Configuration
subtitle: Flags, HTTP endpoints, and the DynamoDB item layout
---

**Experimental.** The Global Discovery Service serves the pool and router lists that Global Routers and
[Pool Relays](../../kubernetes/kv-aware-routing/pool-relay.md) read over HTTP. This reference describes
the `global-discovery-service` binary. For choosing and setting up a registration store, see
[Global Discovery Service Registration Storage](../../kubernetes/kv-aware-routing/global-discovery-service-storage.md).

## Flags

`--help` groups the flags under four headings. A flag under the heading of another
`--pool-discovery` value fails at startup when you set it, and so does a DynamoDB flag without
`--registration-storage dynamodb`, which itself needs `--pool-discovery registrations`. A flag left at
its default is never refused.

### All modes

| Flag | Default | Meaning |
| --- | --- | --- |
| `--listen <addr>` | `0.0.0.0:8080` | Address to serve on. Use `[::]:8080` on clusters with IPv6 Pods. |
| `--tls-cert-path <path>` | Unset | Serve HTTPS with this certificate. Requires `--tls-key-path`. |
| `--tls-key-path <path>` | Unset | Private key for `--tls-cert-path`. |
| `--pool-discovery <mode>` | `registrations` | Where the pool and router lists come from: `registrations`, `karmada`, or `mcs`. |

### With `--pool-discovery registrations`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--registration-storage <store>` | `memory` | Where registrations are kept: `memory` or `dynamodb`. |
| `--heartbeat-interval-secs <n>` | `5` | How often a Pool Relay registers again. Must be above zero and below the lease. |
| `--lease-duration-secs <n>` | `15` | How long a registration lasts. Longer than the heartbeat interval and at most `86400` (one day). |
| `--router <id>=<url>` | None | A static Global Router replica. Repeat for each replica. IDs must be unique. Conflicts with `--router-discovery registrations`. |
| `--router-discovery <mode>` | `static` | `static` reads `--router` entries; `registrations` accepts trusted-network router registration using the selected registration store. |

### With `--pool-discovery karmada` or `mcs`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--pool-service-suffix <suffix>` | `-pool-relay` | Each Pool Relay Service is named `<DGD name><suffix>`. Must not be empty. |
| `--frontend-service-suffix <suffix>` | `-frontend` | Each frontend Service is named `<DGD name><suffix>`. Must not be empty or overlap the Pool Relay suffix. |
| `--frontend-port-name <name>` | `http` | The frontend Service's named port for HTTP. Must not be empty. |
| `--router-namespace <ns>` | Required | Namespace of the Global Router Service. |
| `--router-service <name>` | Required | The Global Router Service, exported through MCS. |
| `--router-port-name <name>` | Required | The router Service's named port for state streams. |
| `--cluster-region <cluster>=<region>` | None | A pool cluster's region, by the cluster's name in MCS, for a cluster whose region neither its imported slices nor Karmada give. Repeat for each cluster; each cluster once. |

These modes list the pools whose Pool Relay Services MCS imports, and take no registrations:
`POST /pools` answers 501, and `GET /routers` tells Pool Relays not to call it. A pool is listed
while MCS imports a Ready Pool Relay endpoint for it, so `complete` needs no lease after a restart, and
any number of replicas can run without shared storage. MCS supplies the pool's key, relay identities
and frontend addresses; export each pool's frontend Service to supply its addresses. Global Routers
learn what each pool serves from its Pool Relays' state streams, so a Pool Relay's `--cluster-id` must
be the MCS cluster ID or Karmada member cluster name: a stream names its pool by that key. These modes
assume a trusted network on which pod addresses are reachable between clusters.

A pool's region is the `topology.kubernetes.io/region` label of its imported EndpointSlices, when the
MCS implementation sets it. Otherwise, with `karmada`, it is the `spec.region` of its member `Cluster`
on the Karmada control plane. A cluster without either takes its `--cluster-region`. A cluster with no
region is listed with its name as the region, and the service logs a warning once per cluster.

The service needs read access to `endpointslices`, and with `karmada`, to `clusters.cluster.karmada.io`
for regions; without that access it logs a warning and uses `--cluster-region` or the cluster's name.
When it reads a member cluster rather than the Karmada control plane, it finds no `Cluster` objects,
logs that once, and takes every region from `--cluster-region`.
`karmada` reads slices named the Karmada way, and `mcs` reads slices with the standard
`multicluster.kubernetes.io` labels.

The service sees only what MCS imports into the cluster it reads. With `karmada`, reading a member
cluster, each pool's Pool Relay and frontend Services and the router Service need a `ServiceImport`
that Karmada propagates to that cluster, for pools in that same cluster too; the Karmada control plane
holds every exported Service's slices, so reading it needs none. Karmada also watches a member's
`ServiceExport`s only while it has propagated at least one `ServiceExport` there. With `mcs`, check
where the implementation imports: some import only into namespaces that exist in the cluster, or that
they are set to watch.

An MCS implementation can drop a healthy cluster's imports for a while, as when its registry restarts or
the cluster's lease lapses, and an empty list looks like one whose imports all went. So a list is
partial for a minute after a pool or router leaves it and while it is empty, and readers keep what it
leaves out. A list cannot show that MCS stopped updating it: while Karmada cannot reach a member, or its
control plane is down, the imports keep their last pools, frontend addresses and router replicas, and the
list stays complete. Routers send requests only to pools whose state streams are live, but an address
that changed meanwhile, as in a rollout, is not updated until MCS is back, and requests can reach
whichever pod has the old address by then.

These modes trust what MCS imports: whoever can export a Service named like a Pool Relay, frontend or
router Service, or write to an MCS implementation that does not authenticate its clients, can add
frontends or router replicas.

### With `--registration-storage dynamodb`

These flags work only with `--pool-discovery registrations`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--lease-skew-secs <n>` | `2` | Clock skew allowed when evaluating leases. At most `86400`. |
| `--dynamodb-replication-grace-secs <n>` | `60` | With a Global Table, how long past its lease a pool stays listed when its last registration was written through another region's replica, since that region's renewals may arrive late. `complete` is `false` meanwhile. `0` drops such a pool when its lease lapses. At most `86400`. |
| `--dynamodb-table <name>` | None | Table name: 3 to 255 letters, digits, `_`, `-`, or `.`. Required with `--registration-storage dynamodb`. |
| `--dynamodb-region <region>` | SDK default chain (`AWS_REGION`) | Region of the table replica this process uses. |
| `--dynamodb-endpoint <url>` | Unset | Endpoint override, for DynamoDB Local. Must be an `http` or `https` URL. |
| `--dynamodb-poll-interval-secs <n>` | `2` | How often each replica reads the table again. `GET /pools` answers from these reads. This plus `--heartbeat-interval-secs` must be less than `--lease-duration-secs` plus `--lease-skew-secs`, so that a renewal shows before its lease runs out. |
| `--dynamodb-ensure-table` | Off | Create the table if it does not exist, and turn on its TTL if it is off. For development. Needs `dynamodb:CreateTable`, and for TTL `dynamodb:DescribeTimeToLive` and `dynamodb:UpdateTimeToLive`; without those two it logs a warning and starts. |

The `--dynamodb-*` flags and `--lease-skew-secs` apply only with `--registration-storage dynamodb`; giving one with the memory store is a startup error.

## Upgrade from a version whose registrations named a model

Routers must learn pools' models from their state streams before the service stops listing them, and
the service must take registrations without a model before Pool Relays stop sending one. Dynamo 1.7.0
is not such a version: its Pool Relays register no pools, and its Global Routers take pools only from
`--pool`. From 1.7.0, set up discovery as for a new deployment.

1. Upgrade the Global Routers.
2. Upgrade the Global Discovery Service.
   - With `karmada` or `mcs`, first remove every flag listed above under the `registrations` and
     `dynamodb` headings: these now stop the service at startup.
   - Also set each cluster's region, or its pools are listed under the cluster's name, not the region
     their Pool Relays registered. A service that reads a member cluster, as
     [`global-discovery-service.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/container/global-discovery-service.yaml)
     does, sees no Karmada `Cluster` objects: give it `--cluster-region` for each cluster, unless the
     MCS implementation labels its slices with the region. One that reads the Karmada control plane
     takes `Cluster.spec.region`, and needs read access to `clusters.cluster.karmada.io` there.
   - Replace every replica at once. Older replicas list only registered pools, under the registered
     regions, so a mix would list different pools from one request to the next. With the manifest,
     scale the Deployment to zero, then apply the new manifest, which sets one replica again:
     `kubectl -n global-discovery scale deployment/global-discovery-service --replicas=0`.
     Elsewhere, use `strategy: Recreate`. Routers and Pool Relays keep their last lists meanwhile.
   - With `registrations`, a rolling update works.
3. Upgrade the Pool Relays: set the new image in each `DynamoPoolRelay`'s Pod template. Until then,
   with `karmada` or `mcs`, each older Pool Relay warns once that it cannot register, and once that
   its registration lapsed, and keeps working.

To roll back, go in the reverse order: Pool Relays, then the Global Discovery Service, then the Global
Routers. An older router does not route a pool listed without a model, and an older service refuses a
registration without one. An older Pool Relay also needs `--routing-pool-region` with
`--routing-pool`, and an older service with `karmada` or `mcs` needs back any store flags it had.

## Environment

| Variable | Meaning |
| --- | --- |
| `DYN_LOG` | Log level, for example `info`. |
| `AWS_REGION` | Region for `--registration-storage dynamodb` when `--dynamodb-region` is unset. |
| AWS credential variables and profile settings | Read by the AWS SDK default chain: environment, shared config, IRSA, EKS Pod Identity, or an instance role. |

## HTTP endpoints

Successful responses carry JSON bodies, and a reader ignores fields it does not know. Errors carry a
short plain-text reason, and `GET /healthz` answers with a status and an empty body.

| Method and path | Purpose | Success | Errors |
| --- | --- | --- | --- |
| `POST /pools` | Register a pool and renew its lease. | 200 with a lease | 501 with `karmada` or `mcs`, which take no registrations; 400 for malformed JSON, 413 for a body over 64 KiB, 415 without `Content-Type: application/json`, 422 for an invalid registration or one the store cannot keep, 503 when the store is unreachable |
| `GET /pools` | List live pools. | 200 with a pool list | 503 when the backend is unreachable. With `dynamodb`, only before the first read of the table. |
| `POST /routers` | Register a router incarnation and renew its lease. | 200 with a lease | 501 unless router registration is enabled; 400/413/415 for request errors, 422 for invalid or conflicting identity/endpoint, 503 on storage errors |
| `GET /routers` | List Global Router replicas, and say whether the service takes pool registrations. | 200 with a router list | 503 when the backend is unreachable |
| `GET /healthz` | Readiness: the backend has answered once. With DynamoDB, that follows the startup check that the table can be read and written. Later backend failures show in the answers to requests, not here. | 200 | 503 until the backend has answered |

### `POST /pools`

Request:

```json
{
  "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
  "location": {"region": "us-west-2", "availability_zone": null, "datacenter": null},
  "frontends": ["http://10.43.1.2:8000/"]
}
```

`frontends` defaults to an empty list. A registration says where the pool is, not what it serves:
Global Routers learn each pool's models from its Pool Relays' state streams. The service ignores
fields it does not know, such as the `model`, `capabilities` and `capacity` that older Pool Relays
send. Reply:

```json
{"heartbeat_interval_s": 5, "lease_duration_s": 15}
```

Register again within `heartbeat_interval_s`. If `lease_duration_s` passes with no successful
registration, the pool drops out of `GET /pools`. The heartbeat interval is above zero and below the lease.
Any caller may overwrite any pool: the last registration for a key wins. So the Pool Relays of one
pool must send the same registration, though lists may come in any order. If they differ, the pool's
listing switches between their versions at every heartbeat, and the service logs a warning that names
the pool.
With `karmada` or `mcs`, the service answers 501 with a plain-text reason: these modes take no
registrations.

### `GET /pools`

```json
{
  "pools": [
    {
      "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
      "location": {"region": "us-west-2", "availability_zone": null, "datacenter": null},
      "frontends": ["http://10.43.1.2:8000/"],
      "relay_identities": []
    }
  ],
  "complete": true
}
```

| Field | Meaning |
| --- | --- |
| `pools[].relay_identities` | The addresses of the pool's Ready Pool Relay pods, such as `ip:10.43.66.175`, for routers that check which Pool Relays stream the pool's state. Routers do not check them yet. Empty in `registrations` mode. |
| `complete` | `true` when the service believes it lists every live pool; `false` when the list may be missing some. It is not an authoritative inventory: with `dynamodb`, an item that cannot be decoded is logged and left out, and another region's writes arrive by replication. With either store it is `false` until one lease has passed after the service starts: a `memory` store has only the registrations it received, and while a rollout moves the service to a DynamoDB table, Pool Relays may still register with the old replicas. With `dynamodb` it is also `false` while the table cannot be reached, and for one lease after, since Pool Relays could not renew meanwhile; and with a Global Table, while a pool from another region is listed past its lease (see `--dynamodb-replication-grace-secs`). In `karmada` and `mcs` modes, it is `false` after about 80 s of MCS watch errors until three minutes pass without one, for a minute after a pool leaves the list, and while the list is empty. |

### `GET /routers`

```json
{
  "routers": [{"replica": "aws-ohio/pod-uid", "url": "http://10.42.0.5:50051/"}],
  "complete": true,
  "accepts_pool_registrations": true
}
```

With `complete: false`, merge the response with known replicas and retain missing ones. Only a
complete response removes missing replicas. With `karmada` and `mcs`, a router list is also partial for
a minute after a replica leaves it and while it is empty. Clients accept older services without this field as
complete. Upgrade Pool Relays before enabling router self-registration, because older clients
ignore `complete`.

`accepts_pool_registrations` is `false` with `karmada` and `mcs`, which take no registrations. A Pool
Relay registers its pool only while it is `true`, and reads an older service without the field as
taking them.

### `POST /routers`

Available with `--pool-discovery registrations --router-discovery registrations`. The default
static directory and MCS/Karmada return 501. Request:

```json
{"replica": "aws-ohio/pod-uid", "url": "http://10.42.0.5:50051/"}
```

`replica` is a nonempty incarnation ID of at most 256 bytes without whitespace or control characters.
`url` is an HTTP(S) authority with no credentials, query, fragment, or path other than `/`;
wildcard/unspecified and multicast addresses and port zero are rejected. It must address one replica's
state listener and be reachable from every Pool Relay. Validation does not verify network reachability.
The reply uses the same lease shape as `POST /pools`; repeat the registration to renew.

The endpoint and DynamoDB writer region are immutable for the row's lifetime. A conflicting registration
returns 422 without changing the directory. Use globally unique incarnation IDs and renew through one
region: DynamoDB Global Table conditions are regional, not global ownership locks. Registration assumes
a trusted network and provides no caller authentication. A stopped router expires by lease; TTL is
only garbage collection. Configure the frontend with paired `--router-replica-id` and
`--router-advertise-url`, together with `--global-discovery-url` and `--relay-view-port`.

Router lists are incomplete for one lease after startup and after the first successful read following
a storage error, and while remote registrations are retained under replication grace. Each router-list
request performs a fresh paginated regional query. A failed page or malformed router row returns 503.

## DynamoDB item layout

With `--registration-storage dynamodb`, the table has one item per pool. The partition key is `pk` (string),
the sort key is `sk` (string), and TTL is enabled on `expires_at`.

| Attribute | Type | Content |
| --- | --- | --- |
| `pk` | S | The constant `pool`. |
| `sk` | S | `<cluster_id>#<namespace>#<dgd_name>`, with `#` escaped as `\#` and `\` as `\\` in each part. At most 1,024 bytes, DynamoDB's limit for a sort key. |
| `registration` | S | The registration as JSON, with `frontends` sorted and deduplicated. |
| `hash` | S | blake3 hex digest of `registration`. Readers detect changes by hashing the registration as they decode it, so a field they do not know is no change. |
| `renewed_at` | N | Unix milliseconds, on the accepting replica's clock, of the last stored registration. |
| `lease_s` | N | The lease duration the accepting replica granted, in seconds. |
| `created_at` | N | Unix milliseconds of the first store of this key since it was last absent. |
| `expires_at` | N | TTL attribute in unix seconds: `renewed_at` plus seven days. Garbage collection only. |
| `written_in` | S | The region of the table replica the last registration was written through. Absent in items written before the service recorded it. |

A pool is live when `renewed_at + lease_s * 1000 + skew >= now` on the reading replica's clock, where
`skew` is `--lease-skew-secs`. A pool whose `written_in` is another region than the reader's stays
listed for `--dynamodb-replication-grace-secs` more, and `complete` is `false` while any pool is listed
that way. The store never expires a pool itself. A registration is one `UpdateItem`
without a condition. Each replica reads the table with one strongly consistent, paginated `Query` on
`pk = "pool"` every
`--dynamodb-poll-interval-secs`, and `GET /pools` answers from the last read, leaving out at once a pool
whose lease has lapsed since.

Router registrations use the same table, with `pk = "ROUTERS"`, `sk = replica`, and attributes
`url`, `writer_region` (strings), `renewed_at_ms`, `lease_ms`, and `expires_at` (numbers).
TTL remains renewal plus seven days. Router writes use conditional `UpdateItem`; router lists
use strongly consistent regional `Query`. Allow the `ROUTERS` partition in IAM policies that
restrict keys. The existing `pool` items and session-affinity table are unchanged.

## IAM actions

| Action | Needed for |
| --- | --- |
| `dynamodb:Query` | Pool polls, `GET /routers`, and the startup check. |
| `dynamodb:UpdateItem` | `POST /pools`, `POST /routers`, and the startup check, whose conditional write never writes. |
| `dynamodb:DescribeTable` | The startup check. |
| `dynamodb:CreateTable` | Only with `--dynamodb-ensure-table`. |
| `dynamodb:DescribeTimeToLive` | Only with `--dynamodb-ensure-table`. |
| `dynamodb:UpdateTimeToLive` | Only with `--dynamodb-ensure-table`. |

Grant them on the table ARN, in each region where the service runs a replica of a Global Table.

## Errors

| Condition | Result |
| --- | --- |
| DynamoDB error, throttling, transport failure, or missing table while serving | `POST /pools` returns 503. `GET /pools` answers from the last read of the table, with `complete: false` until a lease after the table answers again, and drops pools as their leases lapse. `GET /healthz` stays 200. |
| A pool key whose sort key is over 1,024 bytes | 422 from `POST /pools`, with the reason. Nothing is written. |
| An item that fails to decode | Logged and skipped. The rest of the list is served. |
| `--dynamodb-table` missing with `dynamodb` | Startup error. |
| `--dynamodb-poll-interval-secs` plus `--heartbeat-interval-secs` not below `--lease-duration-secs` plus `--lease-skew-secs` | Startup error. |
| A `--dynamodb-table` that DynamoDB would refuse, an empty `--dynamodb-region`, or a `--dynamodb-endpoint` that is not an `http` or `https` URL | Startup error. |
| Table absent at startup without `--dynamodb-ensure-table` | Logged with the table name and the create-table command, then retried. The service does not serve until the table exists. |
| DynamoDB or STS unreachable at startup, or the role cannot describe, read, or write the table | Logged and retried with a backoff of up to 30 seconds. The service does not serve until the check passes. |
| A `--dynamodb-*` flag or `--lease-skew-secs` without `--registration-storage dynamodb` | Startup error. |
| No region from `--dynamodb-region` or the AWS SDK default chain | Startup error. |
| Table whose key schema is not `pk` (HASH, S) and `sk` (RANGE, S) | Logged and retried, as for an absent table. |
