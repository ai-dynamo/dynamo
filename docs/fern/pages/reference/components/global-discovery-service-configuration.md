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
`--registration-storage dynamodb`. A flag left at its default is never refused.

### All modes

| Flag | Default | Meaning |
| --- | --- | --- |
| `--listen <addr>` | `0.0.0.0:8080` | Address to serve on. Use `[::]:8080` on clusters with IPv6 Pods. |
| `--tls-cert-path <path>` | Unset | Serve HTTPS with this certificate. Requires `--tls-key-path`. |
| `--tls-key-path <path>` | Unset | Private key for `--tls-cert-path`. |
| `--pool-discovery <mode>` | `registrations` | Where the pool and router lists come from: `registrations`, `karmada`, or `mcs`. |
| `--registration-storage <store>` | `memory` | Where registrations are kept: `memory` or `dynamodb`. Both work with every `--pool-discovery` value. |
| `--heartbeat-interval-secs <n>` | `5` | How often a Pool Relay registers again. Must be above zero and below the lease. |
| `--lease-duration-secs <n>` | `15` | How long a registration lasts. Longer than the heartbeat interval and at most `86400` (one day). |

### With `--pool-discovery registrations`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--router <id>=<url>` | None | A Global Router replica. Repeat for each replica. Ids must be unique. |

### With `--pool-discovery karmada` or `mcs`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--pool-service-suffix <suffix>` | `-pool-relay` | Each Pool Relay Service is named `<DGD name><suffix>`. Must not be empty. |
| `--frontend-service-suffix <suffix>` | `-frontend` | Each frontend Service is named `<DGD name><suffix>`. Must not be empty or overlap the Pool Relay suffix. |
| `--frontend-port-name <name>` | `http` | The frontend Service's named port for HTTP. Must not be empty. |
| `--router-namespace <ns>` | Required | Namespace of the Global Router Service. |
| `--router-service <name>` | Required | The Global Router Service, exported through MCS. |
| `--router-port-name <name>` | Required | The router Service's named port for state streams. |

These modes join EndpointSlices that MCS imports with registrations from `POST /pools`. A pool is
listed while MCS imports a Ready Pool Relay endpoint for it and its registration lease lasts. While a
DynamoDB table cannot be read, leases lapse on the service's clock, since no Pool Relay can renew.
The registration's `cluster_id` must match the MCS cluster ID or Karmada member cluster name.
MCS supplies relay identities and frontend addresses; the registration supplies the model,
capabilities, capacity, and location. Export each pool's frontend Service to supply its addresses.
The service needs read access to `endpointslices`. `karmada` reads slices named the Karmada way, and `mcs`
reads slices with the standard `multicluster.kubernetes.io` labels.

### With `--registration-storage dynamodb`

These flags work with every `--pool-discovery` value: in each mode the service keeps registrations.

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
| `POST /pools` | Register a pool and renew its lease. | 200 with a lease | 400 for malformed JSON, 413 for a body over 64 KiB, 415 without `Content-Type: application/json`, 422 for an invalid registration or one the store cannot keep, 503 when the store is unreachable |
| `GET /pools` | List live pools. | 200 with a pool list | 503 when the backend is unreachable. With `dynamodb`, only before the first read of the table. |
| `GET /routers` | List Global Router replicas. | 200 with a router list | 503 when the backend is unreachable |
| `GET /healthz` | Readiness: the backend has answered once. With DynamoDB, that follows the startup check that the table can be read and written. Later backend failures show in the answers to requests, not here. | 200 | 503 until the backend has answered |

### `POST /pools`

Request:

```json
{
  "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
  "location": {"region": "us-west-2", "availability_zone": null, "datacenter": null},
  "model": "Qwen/Qwen3-0.6B",
  "capabilities": ["aggregated"],
  "frontends": ["http://10.43.1.2:8000/"],
  "capacity": [{"sku": "H100", "workers": 8}]
}
```

`capabilities`, `frontends`, and `capacity` default to empty lists. `model` must not be blank. Reply:

```json
{"heartbeat_interval_s": 5, "lease_duration_s": 15}
```

Register again within `heartbeat_interval_s`. If `lease_duration_s` passes with no successful
registration, the pool drops out of `GET /pools`. The heartbeat interval is above zero and below the lease.
Any caller may overwrite any pool: the last registration for a key wins. So the Pool Relays of one
pool must send the same registration, though lists may come in any order. If they differ, the pool's
listing switches between their versions at every heartbeat, and the service logs a warning that names
the pool.
In `karmada` and `mcs` modes, the listed frontend addresses come from MCS; the registration's
`frontends` are not used.

### `GET /pools`

```json
{
  "pools": [
    {
      "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
      "location": {"region": "us-west-2", "availability_zone": null, "datacenter": null},
      "model": "Qwen/Qwen3-0.6B",
      "capabilities": ["aggregated"],
      "frontends": ["http://10.43.1.2:8000/"],
      "capacity": [{"sku": "H100", "workers": 8}],
      "relay_identities": []
    }
  ],
  "complete": true
}
```

| Field | Meaning |
| --- | --- |
| `pools[].model` | `null` when the service does not know the pool's model. |
| `pools[].relay_identities` | The identities of the Pool Relay allowed to stream the pool's state, such as `ip:10.43.66.175`. Empty in `registrations` mode. |
| `complete` | `true` when the service believes it lists every live pool; `false` when the list may be missing some. It is not an authoritative inventory: with `dynamodb`, an item that cannot be decoded is logged and left out, and another region's writes arrive by replication. With either store it is `false` until one lease has passed after the service starts: a `memory` store has only the registrations it received, and while a rollout moves the service to a DynamoDB table, Pool Relays may still register with the old replicas. With `dynamodb` it is also `false` while the table cannot be reached, and for one lease after, since Pool Relays could not renew meanwhile; and with a Global Table, while a pool from another region is listed past its lease (see `--dynamodb-replication-grace-secs`). In `karmada` and `mcs` modes, both the imported and registered pool lists must be complete. |

### `GET /routers`

```json
{"routers": [{"replica": "aws-ohio/global-router-0", "url": "http://10.42.0.5:50051/"}]}
```

## DynamoDB item layout

With `--registration-storage dynamodb`, the table has one item per pool. The partition key is `pk` (string),
the sort key is `sk` (string), and TTL is enabled on `expires_at`.

| Attribute | Type | Content |
| --- | --- | --- |
| `pk` | S | The constant `pool`. |
| `sk` | S | `<cluster_id>#<namespace>#<dgd_name>`, with `#` escaped as `\#` and `\` as `\\` in each part. At most 1,024 bytes, DynamoDB's limit for a sort key. |
| `registration` | S | The registration as JSON, with `capabilities`, `frontends`, and `capacity` sorted and deduplicated. |
| `hash` | S | blake3 hex digest of `registration`, used to detect changes. |
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

## IAM actions

| Action | Needed for |
| --- | --- |
| `dynamodb:Query` | Reading the table every poll interval, which `GET /pools` answers from, and the startup check. |
| `dynamodb:UpdateItem` | `POST /pools`, and the startup check, whose conditional write never writes. |
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
