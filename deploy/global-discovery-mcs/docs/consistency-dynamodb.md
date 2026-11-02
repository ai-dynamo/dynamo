<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Registry consistency on DynamoDB Global Tables

This document applies the consistency model of `consistency.md` to the
DynamoDB store. The item layout and conditions are in `storage.md`.

## What DynamoDB guarantees

- Within one region, a conditional write is atomic on its item. A strongly
  consistent `GetItem` sees every write the region acknowledged.
- A Global Table (multi-region eventual consistency, the default mode)
  replicates every write asynchronously to the other replicas, usually
  within a second. The `ReplicationLatency` CloudWatch metric shows the
  actual delay per region pair.
- Conflicts are resolved per item by last writer wins: when two regions
  write the same item before seeing each other's write, every replica ends
  up with the write that has the later timestamp, as a whole item.
  Attributes are not merged.
- Conditions are evaluated only against the replica of the region that
  accepts the write. Two regions can both accept conditional writes that
  would have excluded each other in one region.

## How the registry stays correct

The registry does not rely on cross-region conditions. It relies on three
properties of its writers and records:

1. One logical writer per item. A cluster's records (registration,
   exports, endpoint sets) are written only by that cluster's
   `mcs-controller`, which runs one leader at a time. Different writers
   never race on one item.
2. Monotonic generations. Writers send `generation = max(previous + 1,
   now in unix ms)`, so a writer that restarts or fails over to another
   registry region starts above everything it wrote before. Lease epochs
   are minted the same way by the store. Generation order and wall-clock
   order agree, which is the order last writer wins uses.
3. Full-state writes. Every put replaces the whole record (the export spec,
   the full endpoint set), and writers resynchronize periodically. Whatever
   a conflict resolution picks is repaired by the writer's next write.

### Writer failover between regions

A controller writes generation g1 through region A. A fails before g1
replicates. The controller fails over to region B (the next registry URL,
or a DNS name that resolves to B) and writes g2 > g1. B accepts it: its
stored generation is at most g1. When A comes back, g1 replicates to B and
g2 to A; last writer wins keeps g2 everywhere because it was written later.
If g1 never replicates (lost with A), nothing changes.

The residual risk is a write accepted in A after the writer already wrote
the same item in B, for example a delayed retry. Last writer wins can then
keep the older generation. The next write of that writer has a higher
generation than both and repairs the item; the controller's resync bounds
the window.

### Leases and epochs

- A registration in region B mints `max(stored + 1, now in unix ms)`
  from B's copy. Even if B has not seen A's latest registration, the epoch
  is in practice higher, because it is based on a later time.
- Heartbeats are `UpdateItem` with `lease_epoch = :epoch` and replace the
  item as a whole in replication. A heartbeat of the old epoch accepted in A
  concurrently with a re-registration in B can win last writer wins and
  restore the old epoch. The controller then gets `LEASE_EPOCH_MISMATCH` on
  its next heartbeat and registers again; the lease stays alive throughout
  because the re-registration renews it.
- Lease state is evaluated at read time from `lease_renewed_at`, server
  time. Replicas in different regions may disagree on expiry by clock skew
  plus replication delay; `--lease-skew` allows 2 s of skew by default. The
  only effect is an endpoint set that appears or disappears in imports a
  moment earlier or later in one region.

### Deletes

- `DeleteCluster` deletes what its prefix queries find in the accepting
  region. Records the cluster wrote through another region and that have
  not replicated yet survive; running the deletion again, or compaction,
  removes them.
- A delete and a put of the same item in two regions resolve by last writer
  wins like any other write: a registration after a deregistration
  survives, which is the intended order.
- TTL deletions (`expires_at`, 7 days after the last write) are replicated
  as deletes. They only reach records nobody wrote for a week.

### Reads and the view

- `LoadClusterSet` reads eventually consistently. A write acknowledged a
  moment ago may be missing from one refresh in the same region; the
  replica that accepted the write applies it to its own view immediately
  (write-through), and sibling replicas see it on their next refresh.
- Convergence: same region, one refresh interval (2 s); other regions,
  replication delay (about 1 s) plus one refresh interval. The target is
  under 5 s.
- Watch revisions are per `mcs-registry` replica. A client that fails over
  to another replica or region receives a fresh snapshot, so it never
  applies events from two views out of order.

## Operational signals

- Alarm on `ReplicationLatency` well above a few seconds for each replica
  pair; above the lease TTL (30 s by default) leases may expire in the
  lagging region.
- `ConditionalCheckFailedRequests` is expected (stale generations, epoch
  mismatches, lost races); a sudden rise with errors from writers points
  at a writer violating the generation contract.
  `dynamo_mcs_registry_store_conditional_failures_total` counts them per
  operation on the registry side.
- `ThrottledRequests` and `SystemErrors` surface as `STORE_UNAVAILABLE`;
  writers retry.

## Outside this model

The analysis assumes the default multi-region eventual consistency mode of
Global Tables. Other replication modes, and writers that do not follow the
generation contract (hand-written REST clients), are not covered.
