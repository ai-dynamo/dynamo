<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Planner decisions and latent etcd races

## Planner decision format

The virtual planner coordinator publishes `num_prefill_workers`,
`num_decode_workers`, and `decision_id` together as a JSON object at
`v1/{namespace}/planner/scaling_decision`. One etcd PUT publishes the entire
decision, so a client or cache watcher cannot observe only part of an update.
The coordinator changes its local state only after that PUT succeeds.
`scaled_decision_id` remains a separate acknowledgement written by the client.

Upgrade the coordinator and client together. Old clients do not read the new
record, and old coordinators do not write it. When the record is absent, new
readers accept the legacy three-key snapshot to preserve counts and decision IDs
across a coordinated upgrade. Once present, the record is authoritative, even
if stale legacy keys remain. An invalid record produces an error rather than
falling back to stale decisions. Mixed-version operation and rollback to legacy
writers require a separate migration plan.

## Deferred races in shared etcd helpers

These are latent hazards based on the current call-site audit, not fixes included
with the planner change. Recheck exposure before adding callers or reusing these
helpers. The shared etcd implementation is unchanged.

### Lock cleanup can remove a replacement owner's lock

In [`DistributedRWLock`](../../lib/runtime/src/transports/etcd/lock.rs), guard
cleanup and writer-acquisition rollback delete keys without checking ownership.
If A's lease expires, B acquires the same writer key, and A's delayed cleanup
then runs, A deletes B's lock. A third writer can acquire while B is still
working. Reader cleanup has the same hazard when reader IDs are reused.

Only test callers were found in the repository. Before production use, make
cleanup and rollback compare a unique acquisition token or creation revision
and delete in the same transaction. Transactional acquisition alone does not
make the lock lifecycle safe.

### Versioned updates can overwrite changes or recreate deleted keys

[`EtcdBucket::update`](../../lib/runtime/src/storage/kv/etcd.rs) reads the key,
checks its version, and then unconditionally writes. A version mismatch only
logs a warning. An intervening update can be lost, and an intervening deletion
can be undone by the PUT.

No production caller supplying a nonzero revision was found; discovery
registration supplies zero, and discovery taint updates use a separate atomic
compare-and-put path. Before using nonzero revisions for concurrency control,
compare the expected key version or revision inside the write transaction and
report conflict or missing-key outcomes instead of overwriting.

### Cache defaults can overwrite a concurrent creation

[`KvCache::new`](../../lib/runtime/src/transports/etcd.rs) decides whether a
default is missing from an earlier snapshot, then uses an unconditional PUT.
A value created after the snapshot can be overwritten by that default.

The production planner caller passes an empty defaults map, so this branch is
currently inactive there. Before supplying defaults in production, use the
existing transactional `kv_create` and load the winning value when another
creator wins. Do not treat absence in a cached snapshot as an atomic create
condition.
