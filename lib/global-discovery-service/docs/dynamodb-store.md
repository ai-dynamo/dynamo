<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DynamoDB registration storage for the Global Discovery Service

Design for `--registration-storage dynamodb`, an alternative to `memory` for
every `--pool-discovery` value. It keeps pool registrations in one DynamoDB
table so that any number of service replicas, in one region or across regions
through a Global Table, serve the same pool list and survive restarts. With
`karmada` or `mcs`, the service joins these registrations with what MCS
imports, as it does with `memory`.

The service's contract does not change: `POST /pools` still answers a `Lease`,
`GET /pools` still lists live pools with `complete`, and `GET /routers` still
comes from `--router` or MCS. Only where registrations live changes.

## Why

The memory store documents its limits: run one replica, because each process
keeps only the registrations it receives, and a restarted process lists
nothing until PoolRelays register again. A shared store removes both: every
replica reads the same items, so a fresh replica lists every pool at once.

A fresh replica still lists `Estimated` for one lease after it starts, as the
memory store does. The table cannot tell whether every PoolRelay writes to
it: while a rollout moves the service to this table, from memory or from
another table, PoolRelays still register with the old replicas. A replica
that answered `Exact` then would make routers that follow the contract drop
those live pools until the rollout ends.

## What is stored

Only pool registrations. Router replicas have no write path in this service
(they come from `--router` or from MCS EndpointSlices), and relay identities
are only known to the MCS backend, so `relay_identities` stays empty in
registrations mode, as with the memory store.

One item per `PoolKey`:

| Attribute | Type | Content |
|---|---|---|
| `pk` | S | `pool` (constant; see "Partitioning") |
| `sk` | S | `<cluster_id>#<namespace>#<dgd_name>`, each part with `#` escaped as `\#` and `\` as `\\` (see below) |
| `registration` | S | the `PoolRegistration` as JSON, lists sorted and deduplicated (`capabilities`, `frontends`, `capacity`) |
| `hash` | S | blake3 hex of `registration`, for change detection |
| `renewed_at` | N | unix milliseconds on the accepting replica's clock when the last `register` was stored |
| `lease_s` | N | the lease duration the accepting replica granted, in seconds |
| `created_at` | N | unix milliseconds of the first store of this key since it was last absent |
| `expires_at` | N | DynamoDB TTL attribute, unix seconds: `renewed_at + 7 days`. Garbage collection only, never the lease |
| `written_in` | S | the region of the table replica the last `register` was written through; absent in items written before the store recorded it |

A pool is live when `renewed_at + lease_s * 1000 + skew >= now`, evaluated at
read time on the reading replica's clock, where `skew` is
`--lease-skew-secs` (default 2 s). A pool whose `written_in` is another region
than the reader's stays listed for `--dynamodb-replication-grace-secs`
(default 60 s) more, on trust, and lists are `Estimated` while any pool is
listed that way (see "Multi-region"). The store never expires anything
itself; items that nobody renews for a week disappear through TTL.

Item size is a few hundred bytes to a few KB (the `frontends` list dominates).

The sort key escapes with a backslash, not by doubling `#`: with `##`, the
keys `("a#", "b", …)` and `("a", "#b", …)` would both be written `a###b…`.
With `\#`, a part never holds an unescaped `#`, so every pool has one sort
key, and the pools of one cluster share the prefix `<cluster_id>#` (a
per-cluster `begins_with` query stays possible). Parts made of the characters
Kubernetes allows in names sort by cluster, then namespace, then name.

## Partitioning

All items share `pk = "pool"` so that `GET /pools` is one `Query`, paginated
at 1 MB. A single partition is fine for the expected scale: pools are counted
in tens to hundreds, and the write rate is `pools * relay replicas /
heartbeat`, for example 200 pools with 2 relay replicas each at a 5 s
heartbeat is 80 writes per second, well under a partition's 1000 WCU. If a
deployment ever needs more, `pk` can become `pool#<shard>` with a fan-out
Query; the key layout leaves that open and nothing else depends on `pk`.

## Writes

`register(PoolRegistration) -> Lease` is one `UpdateItem`, with every
attribute name behind a placeholder (`hash` is a DynamoDB reserved word):

```text
SET registration = :reg, #h = :hash, renewed_at = :now, lease_s = :lease,
    expires_at = :ttl, created_at = if_not_exists(created_at, :now)
```

No condition. The memory store's rule is that any caller may overwrite any
pool and several PoolRelay replicas register the same pool, so the whole
record is last writer wins, like today.

The PoolRelays of one pool must therefore send the same registration. Lists
in another order or with repeats are the same registration: `register` sorts
and deduplicates them before it hashes and stores. If the content differs,
for example in the frontends each relay sees or in the capacity it counts,
the item switches between their versions at every heartbeat, and every
replica's watchers see the pool change at almost every poll. With `karmada`
or `mcs`, the join takes frontends from MCS, so a difference in frontends
alone does not reach routers. The store warns, once per pool, when a pool's
changes go back to the version before twice in a row.

Whether the registration is published to this replica's watchers is decided
against what those watchers already have (the polled view, below), not
against the old item, so the write asks for no `ReturnValues`. If the view
has the pool with the same `hash`, the call was a renewal and nothing is
published; otherwise an `Upserted` is published at once. Other replicas learn
through polling. Comparing with the old item (`ALL_OLD`) instead would leave a
watcher wrong for good in one race: this replica publishes `X` over the view's
`Y`, another replica writes `Y` again before the next poll, and that poll sees
`Y` equal to its previous read and publishes nothing, though the watcher
holds `X`.

The returned `Lease` is the configured `(heartbeat_interval_secs,
lease_duration_secs)`, unchanged from the memory store. `lease_s` is stored so
that a reader with a different configured lease still evaluates each item with
the lease its writer granted.

Why not `PutItem`: `UpdateItem` keeps `created_at` without a read.

## Reads

Reads are a `Query` on `pk = "pool"` with pagination, strongly consistent
(`ConsistentRead`), so a poll sees every write that its region's replica of
the table took before the poll began. An eventually consistent read could
return an item older than one the replica already has, and make watchers see
a pool change back and forth. A strongly consistent read costs twice the
read capacity, which one poll per replica per poll interval keeps small.
Other regions' writes still arrive by replication (see "Multi-region"). Each
page's items are
decoded and expired ones dropped. `list()` reads the table only before the
first poll; after that it answers from the polled view (below), leaving out
at once a pool whose lease has lapsed since the last poll, so a request costs
no read of its own and read cost scales with replicas, not with clients and
probes. A pool registered through another replica shows within a poll
interval. The result is `Snapshot { items, completeness }`, where `completeness` is `Estimated` until a
lease has passed since the store was made (see "Why"), and `Exact` after
that. A failed call to the table, a read or a write, makes it `Estimated`
again until a lease has passed since the next successful read: no PoolRelay
could renew while the table could not be reached, so items may have lapsed
until their PoolRelays register again. The time a read began decides. A decode failure of one item is
logged and that item skipped, so one bad write cannot take the list down. Any
SDK error (throttling, access denied, transport, missing table) becomes
`DiscoveryError::Unavailable`, which the HTTP layer maps to 503 for
`register`, and for `list` before the first read. `/healthz` stays OK once the
backend has answered.

`list_and_watch()` polls. The store keeps one view per process, shared by
its watchers: the live pools of the last poll, plus this replica's
registrations since. Each call first runs the `Query` (and fails with
`Unavailable` if it does), then starts one poller per store if none runs.
Every `--dynamodb-poll-interval-secs` (default 2 s; plus the heartbeat
interval, it must be less than the lease plus the skew, so a renewal shows in
a poll before its lease runs out) the poller runs the same `Query`, diffs the
live set
against the view by `PoolKey` and `hash`, and emits `Upserted` for new or
changed pools and `Removed` for pools that are gone or expired. The stream
starts with one `Upserted` per pool in the view followed by a `Resync`, as
the memory store does, through the same `watch_stream` helper (it snapshots
the view and subscribes under one lock). The first `Resync` is `Estimated`
until a lease has passed since the store was made; the first poll after that
sends a `Resync(Exact)`, once. Local
`register` calls update the view and emit `Upserted` at once, so the next
poll does not emit the same change again. A poll whose `Query` began before a
local registration may not see it, so for that poll the pool keeps its
registered state; the next poll decides. A failed poll leaves the stream
open, sends `Removed` for each pool whose lease has lapsed on this replica's
clock (nobody can renew while the table cannot be reached), and a
`Resync(Estimated)` if watchers had an `Exact` view; the poller logs it
(warn, then debug) and waits twice as long each time, up to 30 s. The first
successful poll a lease after reads recovered sends `Resync(Exact)` again. The poller stops when the store is dropped,
which also ends every stream; a caller's `CancellationToken` ends its own.
Polling rather than DynamoDB Streams keeps the store to one table and one IAM
policy. With `--pool-discovery registrations` nothing consumes the watch yet:
the HTTP API calls only `list` and `register`. With `karmada` or `mcs`, the
join with MCS watches the store.

## Clock

The store takes a `Clock` (`fn now_ms(&self) -> u64`) so that tests can drive
expiry deterministically; production uses `SystemTime`. Expiry is always
decided on the reader's clock against the writer's `renewed_at`, so replicas
with skewed clocks disagree by at most the skew plus `--lease-skew-secs`; the
only effect is a pool appearing or disappearing a moment earlier or later on
one replica.

## Multi-region

One Global Table with a replica per region; each region's replicas point at
their own regional endpoint (`--dynamodb-region`, or `AWS_REGION`). Global
Tables resolve concurrent writes to one item by last writer wins on the whole
item, which is exactly the store's single-writer-agnostic semantics: whatever
copy wins is a complete, valid registration. There are no cross-region
conditions to break.

Replication is asynchronous, though: usually within a second, but a replica
can fall behind by more than the lease. A reader there would see an old
`renewed_at` for a pool that still heartbeats in another region, drop it as
expired, and call the list `Exact`, so a router would drop a live pool. Each
item therefore records the region of the replica it was written through
(`written_in`, the region of the writing store's client). A reader in
another region keeps such a pool listed for `--dynamodb-replication-grace-secs`
past its lease and the skew, and lists are `Estimated` while it does, so a
router keeps the pool without being told the list is complete. Past the
grace the pool is dropped, which bounds how long a pool that really stopped
in another region stays listed. A pool written through the reader's own
region, or an item without `written_in`, follows the lease alone. Watching
`ReplicationLatency`, or a heartbeat item per region, could narrow the grace
later.

## Configuration

New flags, under their own `With --registration-storage dynamodb` heading,
since they work with every `--pool-discovery` value:

| Flag | Default | Meaning |
|---|---|---|
| `--registration-storage dynamodb` | `memory` | select the store |
| `--dynamodb-table <name>` | none, required with `dynamodb` | table name: 3 to 255 letters, digits, `_`, `-` or `.`, as DynamoDB requires |
| `--dynamodb-region <region>` | SDK default chain (`AWS_REGION`) | region of the table replica this process uses |
| `--dynamodb-endpoint <url>` | none | endpoint override for dynamodb-local; an `http` or `https` URL |
| `--dynamodb-poll-interval-secs <n>` | 2 | poll period, which `list` and watches answer from; plus `--heartbeat-interval-secs`, must be less than `--lease-duration-secs` plus `--lease-skew-secs` |
| `--dynamodb-ensure-table` | off | create the table if it does not exist, and turn on its TTL if it is off, on every start (development; needs `CreateTable`, and for TTL `DescribeTimeToLive` and `UpdateTimeToLive`, without which it warns and starts) |
| `--lease-skew-secs <n>` | 2 | clock skew allowed when evaluating leases; at most 86400 |
| `--dynamodb-replication-grace-secs <n>` | 60 | with a Global Table, how long past its lease a pool written through another region's replica stays listed, as `Estimated`; 0 drops it when its lease lapses; at most 86400 |

The `--dynamodb-*` flags and `--lease-skew-secs` fail at startup without
`--registration-storage dynamodb`, as flags of another `--pool-discovery` do.

Credentials come from the SDK default chain (environment, shared config, IRSA
or EKS Pod Identity, instance role). `aws-config` is already in the workspace
lock; `aws-sdk-dynamodb` is new (1.105.0, the newest that needs no newer
`aws-smithy-*` crates than the lock has, without its default features, which
add a hyper 0.14 and rustls 0.21 client). Requests time out after 3 s to
connect, 5 s per attempt and 15 s with the SDK's retries.

## Table definition

```json
{
  "TableName": "global-discovery-pools",
  "AttributeDefinitions": [
    {"AttributeName": "pk", "AttributeType": "S"},
    {"AttributeName": "sk", "AttributeType": "S"}
  ],
  "KeySchema": [
    {"AttributeName": "pk", "KeyType": "HASH"},
    {"AttributeName": "sk", "KeyType": "RANGE"}
  ],
  "BillingMode": "PAY_PER_REQUEST",
  "TimeToLiveSpecification": {"AttributeName": "expires_at", "Enabled": true}
}
```

Plus point-in-time recovery and, for multi-region, a replica per region. The
`--dynamodb-ensure-table` flag creates this definition (without PITR or
replicas) for development and dynamodb-local.

IAM for the service: `dynamodb:Query`, `dynamodb:UpdateItem`,
`dynamodb:DescribeTable` on the table ARN; add `dynamodb:CreateTable`,
`dynamodb:DescribeTimeToLive` and `dynamodb:UpdateTimeToLive` only where
`--dynamodb-ensure-table` is used. The error for a missing table prints the
`aws` commands that create it, with the `--region` the service uses and, for
dynamodb-local, its `--endpoint-url`.

## Code layout

```text
lib/global-discovery-service/src/backends/dynamodb.rs        pub struct DynamoDbPoolStore, impl PoolRegistrar + Directory<PoolListing, PoolKey>
lib/global-discovery-service/src/backends/dynamodb/keys.rs   sort key encoding and decoding (escaping), item attribute names
lib/global-discovery-service/src/backends/dynamodb/codec.rs  PoolRegistration <-> item, hash, liveness check
lib/global-discovery-service/src/backends/dynamodb/table.rs  ensure_table, DescribeTable readiness
lib/global-discovery-service/src/backends/clock.rs           pub trait Clock: Send + Sync { fn now_ms(&self) -> u64 }, pub struct SystemClock, pub struct ManualClock (set/advance, public so integration tests can use it)
```

`keys`, `codec` and `table` are public modules. `ManualClock` is
`new(now_ms: u64)`, `set(&self, now_ms: u64)` and `advance(&self, by:
Duration)`, and `Default` starts it at 0. `DynamoDbConfig` also has
`new(table)` with the flags' defaults.

Public surface the tests and docs rely on (keep these signatures):

```rust
pub struct DynamoDbConfig {
    pub table: String,
    pub region: Option<String>,
    pub endpoint: Option<Url>,
    pub poll_interval: Duration,
    pub lease_skew: Duration,
    pub replication_grace: Duration,
    pub ensure_table: bool,
}

impl DynamoDbPoolStore {
    /// Builds the SDK client from the default chain plus `config`, optionally
    /// creates the table, and checks it with DescribeTable.
    pub async fn connect(lease: Lease, config: DynamoDbConfig) -> Result<Arc<Self>, ConnectError>;
    /// For tests: a store over an existing client and clock.
    pub fn with_client(lease: Lease, config: DynamoDbConfig, client: aws_sdk_dynamodb::Client, clock: Arc<dyn Clock>) -> Arc<Self>;
}
```

`main.rs` gains the flags, a `RegistrationStorage::Dynamodb` variant and a
match arm in `registration_storage`, which `registrations_backend` and
`mcs_backend` both use. The flags carry the `WITH_DYNAMODB` heading, and
`parse_args` refuses a flag under it without `--registration-storage
dynamodb`, whatever `--pool-discovery` is. The
`every_flag_says_what_it_works_with` test checks that every flag has a
heading or works everywhere.

## Errors

| Source | Result |
|---|---|
| SDK service error, throttling, transport, missing table | `DiscoveryError::Unavailable` (503) |
| A sort key over DynamoDB's 1,024 bytes | `DiscoveryError::Rejected` (422) with the reason, before any write; DynamoDB would answer each write with a `ValidationException` |
| Item that fails to decode | skipped with a warning; the rest of the list is served |
| `--dynamodb-table` missing with `dynamodb` | startup error |
| `--dynamodb-poll-interval-secs + --heartbeat-interval-secs >= --lease-duration-secs + --lease-skew-secs` | usage error |
| A `--dynamodb-table` that DynamoDB would refuse, an empty `--dynamodb-region`, or a `--dynamodb-endpoint` that is not `http` or `https` | usage error |
| A `--dynamodb-*` flag or `--lease-skew-secs` without `--registration-storage dynamodb` | startup error |
| No region from `--dynamodb-region` or the SDK chain | startup error (`ConnectError::Config`) |
| Table absent at startup without `--dynamodb-ensure-table` | logged, naming the table and the create-table command, and retried (`ConnectError::Table`) |
| Table whose key is not `pk` (HASH, S) and `sk` (RANGE, S) | logged and retried (`ConnectError::Table`) |
| DynamoDB or STS unreachable, or access denied, at startup | logged and retried (`ConnectError::Table`) |

`connect` describes the table, then checks that it may read it (a `Query`
of one item) and write it: an `UpdateItem` on a key that is not a pool's,
with a condition that never holds. DynamoDB checks permissions before the
condition, so `ConditionalCheckFailedException` means writes are allowed, and
nothing is written. `main.rs` listens for SIGTERM before it connects, and
retries a `ConnectError::Table` with kube's `DefaultBackoff` (0.8 s doubling
to 30 s, with jitter) instead of exiting, so a short outage at startup does
not leave the Pod in CrashLoopBackOff. Nothing listens until `connect`
succeeds, so the Pod stays unready meanwhile.

## Tests

- Unit (no network): key escaping round trips including `#` and `\` in
  every part; codec round trip; liveness at the boundary with a `ManualClock`
  and skew; diffing of two snapshots into events.
- Conformance: the memory store's behavioural tests (`registering again only
  renews`, `a missed lease hides the pool`, `lists are exact`) rewritten once
  as functions over `Arc<dyn PoolRegistrar>` plus `Arc<PoolDirectory>` and a
  clock, run against both stores. For DynamoDB they run against dynamodb-local
  when `GDS_TEST_DYNAMODB_ENDPOINT` is set and skip otherwise, with a fresh
  table per test.
- Integration against dynamodb-local (`amazon/dynamodb-local` in Docker):
  `ensure_table` creates the table and TTL; two stores over one table see each
  other's registrations through `list` and through the polling watch; an
  item with an unknown attribute set and a corrupt `registration` attribute
  are tolerated; a stopped endpoint turns `list` into `Unavailable`.
- HTTP: the existing `service.rs` tests run unchanged against the memory
  store; one new test wires `DynamoDbPoolStore` behind the router against
  dynamodb-local and asserts `POST /pools` then `GET /pools` on a second
  store instance, with `complete: false` until a lease has passed and `true`
  after that.

## Follow-ups (not in this change)

- Router replica registration through the same table when routers register.
- A sharded `pk` if a deployment exceeds one partition.
- DynamoDB Streams instead of polling if a consumer of `list_and_watch`
  needs sub-second fan-out.

## Router membership

`--router-discovery registrations` shares this table using `pk = "ROUTERS"` and `sk = replica`.
Each item stores `url` (string), `writer_region` (string), `renewed_at_ms` (number), `lease_ms`
(number), and the existing `expires_at` TTL (Unix seconds, renewal plus seven days).
`UpdateItem` requires an absent row or an unchanged URL/writer region and nondecreasing renewal
time. The same incarnation cannot move endpoints or regions; choose a new globally unique ID.
These conditions serialize writes in one region only, not across MREC Global Table replicas.

Every `GET /routers` reads a strongly consistent paginated regional `Query`, rather than using the
pool polling cache. A failed page or invalid row fails the snapshot with 503. Lease skew and
cross-region replication grace match pool semantics. `complete: false` marks startup, one lease
after successful recovery from a storage failure, and remote registrations retained within grace.
Clients keep missing known entries until a complete response confirms removal. A new remote row
cannot be seen before it replicates; completeness is a best-effort lease signal, not global consensus.

No new table or IAM actions are required: Query and UpdateItem on this table, plus DescribeTable
for the existing startup check. If IAM restricts partition keys, allow both `pool` and `ROUTERS`.
Router processes call GDS over HTTP and need no discovery-table credentials.
