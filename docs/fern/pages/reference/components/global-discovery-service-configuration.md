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

`--help` groups the flags under three headings. A flag under the heading of another
`--pool-discovery` value fails at startup when you set it. A flag left at its default is never refused.

### All modes

| Flag | Default | Meaning |
| --- | --- | --- |
| `--listen <addr>` | `0.0.0.0:8080` | Address to serve on. Use `[::]:8080` on clusters with IPv6 Pods. |
| `--tls-cert-path <path>` | Unset | Serve HTTPS with this certificate. Requires `--tls-key-path`. |
| `--tls-key-path <path>` | Unset | Private key for `--tls-cert-path`. |
| `--tls-client-ca-cert-path <path>` | Unset | Also require callers to present a certificate signed by this CA (mTLS). Requires `--tls-cert-path`. Probes need a certificate too, so probe readiness with TCP or a command. |
| `--pool-discovery <mode>` | `registrations` | Where the pool and router lists come from: `registrations`, `karmada`, or `mcs`. |

### With `--pool-discovery registrations`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--registration-storage <store>` | `memory` | Where registrations are kept: `memory` or `dynamodb`. |
| `--heartbeat-interval-secs <n>` | `5` | How often a Pool Relay registers again. Must be above zero and below the lease. |
| `--lease-duration-secs <n>` | `15` | How long a registration lasts. Longer than the heartbeat interval and at most `86400` (one day). |
| `--router <id>=<url>` | None | A Global Router replica. Repeat for each replica. Ids must be unique. |
| `--lease-skew-secs <n>` | `2` | Clock skew allowed when evaluating leases. Works with `--registration-storage dynamodb`. |
| `--dynamodb-table <name>` | None | Table name. Required with `--registration-storage dynamodb`. |
| `--dynamodb-region <region>` | SDK default chain (`AWS_REGION`) | Region of the table replica this process uses. |
| `--dynamodb-endpoint <url>` | Unset | Endpoint override, for DynamoDB Local. |
| `--dynamodb-poll-interval-secs <n>` | `2` | How often the store polls the table for changes. Must be less than `--heartbeat-interval-secs`. |
| `--dynamodb-ensure-table` | Off | Create the table if it does not exist. For development. Needs `dynamodb:CreateTable` and `dynamodb:UpdateTimeToLive`. |

The `--dynamodb-*` flags and `--lease-skew-secs` apply only with `--registration-storage dynamodb`; giving one with the memory store is a startup error.

### With `--pool-discovery karmada` or `mcs`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--pool-service-suffix <suffix>` | `-pool-relay` | Each Pool Relay Service is named `<DGD name><suffix>`. Must not be empty. |
| `--router-namespace <ns>` | Required | Namespace of the Global Router Service. |
| `--router-service <name>` | Required | The Global Router Service, exported through MCS. |
| `--router-port-name <name>` | Required | The router Service's named port for state streams. |

These modes read EndpointSlices that MCS imports and take no registrations, so `POST /pools` returns 501.
The service needs read access to `endpointslices`. `karmada` reads slices named the Karmada way, and `mcs`
reads slices with the standard `multicluster.kubernetes.io` labels.

## Environment

| Variable | Meaning |
| --- | --- |
| `DYN_LOG` | Log level, for example `info`. |
| `AWS_REGION` | Region for `--registration-storage dynamodb` when `--dynamodb-region` is unset. |
| AWS credential variables and profile settings | Read by the AWS SDK default chain: environment, shared config, IRSA, EKS Pod Identity, or an instance role. |

## HTTP endpoints

All bodies are JSON. A reader ignores fields it does not know.

| Method and path | Purpose | Success | Errors |
| --- | --- | --- | --- |
| `POST /pools` | Register a pool and renew its lease. | 200 with a lease | 422 for an invalid body, 501 in `karmada` and `mcs` modes, 503 when the store is unreachable |
| `GET /pools` | List live pools. | 200 with a pool list | 503 when the backend is unreachable |
| `GET /routers` | List Global Router replicas. | 200 with a router list | 503 when the backend is unreachable |
| `GET /healthz` | Readiness: the backend answers. | 200 | 503 when the backend is unreachable |

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
Any caller may overwrite any pool: the last registration for a key wins.

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
| `complete` | `true` when the service lists every pool. With the `memory` store it is `false` until one lease has passed after start. With `dynamodb` it is `true` at once. |

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
| `sk` | S | `<cluster_id>#<namespace>#<dgd_name>`, with `#` escaped as `\#` and `\` as `\\` in each part. |
| `registration` | S | The registration as JSON, with `capabilities`, `frontends`, and `capacity` sorted and deduplicated. |
| `hash` | S | blake3 hex digest of `registration`, used to detect changes. |
| `renewed_at` | N | Unix milliseconds, on the accepting replica's clock, of the last stored registration. |
| `lease_s` | N | The lease duration the accepting replica granted, in seconds. |
| `created_at` | N | Unix milliseconds of the first store of this key since it was last absent. |
| `expires_at` | N | TTL attribute in unix seconds: `renewed_at` plus seven days. Garbage collection only. |

A pool is live when `renewed_at + lease_s * 1000 + skew >= now` on the reading replica's clock, where
`skew` is `--lease-skew-secs`. The store never expires a pool itself. A registration is one `UpdateItem`
without a condition, and `GET /pools` is one paginated `Query` on `pk = "pool"`.

## IAM actions

| Action | Needed for |
| --- | --- |
| `dynamodb:Query` | `GET /pools` and change polling. |
| `dynamodb:UpdateItem` | `POST /pools`. |
| `dynamodb:DescribeTable` | The startup check. |
| `dynamodb:CreateTable` | Only with `--dynamodb-ensure-table`. |
| `dynamodb:UpdateTimeToLive` | Only with `--dynamodb-ensure-table`. |

Grant them on the table ARN, in each region where the service runs a replica of a Global Table.

## Errors

| Condition | Result |
| --- | --- |
| DynamoDB error, throttling, transport failure, or missing table while serving | 503 from the endpoint, and `GET /healthz` fails. |
| An item that fails to decode | Logged and skipped. The rest of the list is served. |
| `--dynamodb-table` missing with `dynamodb` | Startup error. |
| `--dynamodb-poll-interval-secs` not below `--heartbeat-interval-secs` | Startup error. |
| Table absent at startup without `--dynamodb-ensure-table` | Startup error that names the table and the create-table command. |
| A `--dynamodb-*` flag or `--lease-skew-secs` without `--registration-storage dynamodb` | Startup error. |
| No region from `--dynamodb-region` or the AWS SDK default chain | Startup error. |
| Table whose key schema is not `pk` (HASH, S) and `sk` (RANGE, S) | Startup error. |
