<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Discovery Service

The Global Discovery Service serves the pool and router lists that Global Routers and Pool Relays read
over HTTP. A Pool Relay registers its pool with `POST /pools` and renews it on a heartbeat. A Global
Router reads `GET /pools` for the pools it can route to and `GET /routers` for its peer replicas.
`GET /healthz` reports whether the backend answers.

## Pool discovery

`--pool-discovery` chooses where the pool and router lists come from:

- `registrations` (default): the pools that Pool Relays register while their leases last. Router
  replicas come from repeated `--router <id>=<url>` flags.
- `karmada` or `mcs`: the EndpointSlices that Multi-Cluster Services imports, named the Karmada way or
  the standard way. These modes take no registrations.

## Registration stores

With `--pool-discovery registrations`, `--registration-storage` chooses where registrations are kept:

- `memory` (default): in the process. Run one replica. A restart loses the registrations, and
  `GET /pools` reports `complete: false` until one lease has passed.
- `dynamodb`: in a DynamoDB table, so any number of replicas serve the same pools, a restart loses
  nothing, and `complete` is `true` at once. A Global Table extends it across regions.

The DynamoDB design (item layout, leases, multi-region behavior, errors) is in
[`docs/dynamodb-store.md`](docs/dynamodb-store.md).

## Build and test

```bash
cargo build -p dynamo-global-discovery-service
cargo test -p dynamo-global-discovery-service
```

The DynamoDB tests run against DynamoDB Local and skip when no endpoint is set:

```bash
scripts/dynamodb-local.sh up
export GDS_TEST_DYNAMODB_ENDPOINT=http://localhost:8000
cargo test -p dynamo-global-discovery-service
```

Run the service against the memory store with one router replica:

```bash
cargo run -p dynamo-global-discovery-service -- \
    --pool-discovery registrations --router aws-ohio/global-router-0=http://10.42.0.5:50051
```

## Documentation

- [Registration storage](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/kv-aware-routing/global-discovery-service-storage.md):
  choose a store, create the table, grant access, run and verify.
- [Configuration reference](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/reference/components/global-discovery-service-configuration.md):
  every flag, the HTTP endpoints, and the DynamoDB item layout.
- Example manifests: [`container/global-discovery-service.yaml`](../../container/global-discovery-service.yaml)
  (pools from Karmada) and
  [`container/global-discovery-service-dynamodb.yaml`](../../container/global-discovery-service-dynamodb.yaml)
  (registrations in DynamoDB).
