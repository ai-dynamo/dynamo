<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Registry consistency model

Every `mcs-registry` replica serves reads from its own in-memory view of the
store (`internal/view`). This page describes how the view follows the
store, what a client can rely on, and what replicas may disagree on.
`consistency-dynamodb.md` applies it to DynamoDB Global Tables.

## Writes and convergence

- Writes are conditional in the store: generations must increase and
  heartbeats must carry the current lease epoch.
- After a successful store write the handler applies the stored record to
  its replica's view before it responds (write-through). The writer, and
  every client of that replica, read the write at once.
- Sibling replicas in the same region pick the write up on their next
  refresh: every `--refresh-interval` (2 s) a replica reloads each clusterset
  with `LoadClusterSet` and publishes the differences.
- Other regions see the write after DynamoDB Global Table replication (about
  1 s) plus one refresh interval. The target is under 5 s.
- New clustersets are found every `--clusterset-discovery-interval` (30 s),
  or at once when a request names one.

Every item has one logical writer: the `mcs-controller` of its cluster.
Writers use `generation = max(previous + 1, unix milliseconds)`, which stays
monotonic across restarts and region failover, so Global Table
last-writer-wins agrees with generation order. A write with generation 0
asks the handling replica to assign that value from the stored generation;
this suits single writers such as curl sessions, not writers that fail
over between regions mid-write.

## Write-through and refresh

A refresh that started before a local write could undo it on this replica.
To avoid that flap, a record written or deleted through the replica wins
over a snapshot for one refresh interval after the write, unless the
snapshot holds a newer version (higher generation; for clusters also a
newer lease epoch or renewal). A cluster deletion also hides the cluster's
exports and endpoint sets from a snapshot that predates it. After the
window the store is the truth again.

## Leases

- Leases are evaluated at read time from server timestamps: a lease is
  expired once `renewed_at + ttl + --lease-skew` is in the past. Nothing
  writes expiry to the store.
- Each replica re-evaluates every cluster lease every
  `--lease-eval-interval` (1 s) without a store read, and publishes the
  resulting changes. An expiry is counted once per replica in
  `dynamo_mcs_registry_lease_expirations_total{kind="cluster"}`.
- Replicas may disagree on a lease by the skew allowance (2 s), since their
  clocks and refresh times differ. KEP-1645 tolerates that: an endpoint set
  appears or disappears a moment later on one replica than on another.

When a cluster's lease expires:

- its endpoints leave every ServiceImport;
- its exports stop taking part in conflict resolution, so a dead cluster
  cannot pin the winner; its per-cluster status reads
  `Ready=False/ClusterLeaseExpired` and `Conflict=Unknown/ClusterLeaseExpired`;
- an import whose exporting clusters all expired keeps its type and ports,
  with no endpoints and no participants, so derived Services keep their
  virtual IPs while a controller restarts;
- the records stay for `--retention` (24 h), so a returning controller keeps
  its export creation timestamps, then compaction removes them.

`DeregisterCluster` removes everything of the cluster at once.

## Revisions and watches

- A revision is `view_epoch:seq`. The view epoch is minted when the replica
  starts (unix milliseconds and 32 random bits); the sequence counts the
  changes of one clusterset on that replica. Revisions of different
  replicas, or of one replica before and after a restart, are unrelated.
- Each change of a derived object (cluster, ServiceImport) gets the next
  sequence number and goes to a ring buffer of `--watch-history` (10000)
  events per clusterset.
- A watch with a `resume_revision` of the same epoch whose following events
  are all still buffered receives only those events. Any other revision
  (another replica, a restart, too old, malformed, from the future) yields a
  fresh snapshot. A resume never fails and never skips events.
- Deltas are filtered per watcher. An object that stops matching the
  watcher's filter (a ServiceImport whose labels leave the selector, a
  cluster whose lease expires under `include_expired=false`) arrives as
  `DELETE` carrying its new state.
- A `BOOKMARK` every `--bookmark-interval` carries the newest revision the
  watcher has fully processed, including changes its filter skipped, so
  resuming from a bookmark never replays filtered events.
- A watcher whose queue overflows `--watch-queue-size` events is
  re-snapshotted (`SNAPSHOT_START` again) instead of disconnected.
- List responses carry the revision they were taken from; a client may
  watch from it to receive every later change.

## Per-replica derived state

Condition `last_transition_at` is the time this replica saw the change, so
replicas may report different transition times for one condition.

## Compaction

Every replica runs compaction every `--compaction-interval`. It removes a
cluster whose lease expired more than `--retention` ago with everything it
owns, and exports of an unregistered cluster and endpoint sets without an
export that were not updated for `--retention`. Deletes are idempotent, so
concurrent passes agree. The store has no conditional delete: before
deleting an expired cluster the compactor reads it again and keeps it when
it was renewed. A controller that returns between that read and the delete
gets `NOT_FOUND` on its next heartbeat, registers again and resends its
exports.

## Readiness and failures

A replica is ready when the store answers a ping, every clusterset the store
knows was loaded at least once and no view is older than three refresh
intervals. When the store is unreachable, reads keep being served from the
last view while writes fail with `STORE_UNAVAILABLE`. Readiness is checked
every refresh interval: a failed ping, or a view that stops refreshing for
three intervals, makes the replica report not ready, so load balancer and
DNS health checks move clients to healthy replicas or regions.
