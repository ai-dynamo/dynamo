<!-- SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Concurrent Radix Tree Compressed

`ConcurrentRadixTreeCompressed` is a compressed trie for KV-cache routing. It
keeps the same logical shape as a radix tree, but each non-root node owns a
compressed edge: a vector of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs.
This reduces node count and lets decode append to an existing leaf when the
parent hash is still the covered tail.

The tree supports splitting but not merging. Cleanup may remove stale leaves,
but a split edge is not recompressed later.

## Motivation

The original KV-router indexers optimize different tradeoffs:

- `RadixTree` is simple and exact, but it stores one block per trie node. Long
  shared prompts and decode tails create many nodes, and every append has to
  walk or allocate at block granularity.
- `RadixTreeIndex` improves concurrent event ingestion by sharding workers, but
  each shard is still an ordinary radix tree. Shared prefixes that cross shard
  boundaries are duplicated instead of represented by shared structure.
- `BranchShardedIndexer` routes divergent branches to different shards, but the
  underlying per-shard structure still needs to handle high fanout, decode
  extension, and stale parent lookups cheaply.

`ConcurrentRadixTreeCompressed` is the per-shard structure built for that shape.
Its main differences are:

- **Radix compression**: each non-root node stores a compressed edge instead of a
  single block. A prefill chain can be represented by one node, and decode can
  append directly to a leaf while the node is still childless.
- **Per-worker cutoffs**: worker coverage is tracked as a cutoff inside the
  compressed edge. Removal can shorten one worker's coverage without splitting
  the physical tree for every eviction.
- **Sticky internal nodes**: once a node has had children, it remains logically
  internal even if cleanup removes those children. This avoids reopening old
  fanout points for decode extension after races or cleanup.
- **Lazy lookup repair**: worker-local reverse lookups are repaired only when a
  stale entry is observed. Cross-thread splits do not need to synchronously patch
  every other thread's lookup table.
- **Semi-lock-free structural reads**: child storage uses immutable compact
  snapshots for up to four children and `DashMap` for higher fanout, while the
  edge state is protected separately. Hot read paths do not take the shape gate,
  and shape-sensitive writes use version validation to retry when a plan becomes
  stale.
- **Versioned shape gates**: the node's `shape_gate` and `shape_version` combine
  a small critical section with explicit stale-plan detection. Shared operations,
  such as adding a child under a stable internal node, can proceed with a shared
  gate; structural mutations, such as splits and leaf extensions, take the
  exclusive path.

The intended result is not a fully lock-free tree. It is a compressed tree where
common prefill fanout and decode extension avoid unnecessary exclusive
serialization, while rare structural races are resolved by retrying or lazily
repairing lookup state.

## Node State

Each node contains:

- `edge`: the compressed sequence of local and external block hashes.
- `edge_index`: reverse lookup from `ExternalSequenceBlockHash` to position in
  the edge. Removal uses this to find an evicted block once it has the node.
  A position never moves: edges grow only at the tail, and a split truncates
  them. Edges of at most 16 blocks keep no index and scan from the tail. Longer
  edges keep an open-addressing table of `u32` positions that reads keys back
  through `edge`, so it costs 5 to 11 bytes per block. Since the split prefix
  never grows again, a split keeps its table in O(1) and lookups skip the
  positions that moved to the suffix. The prefix rebuilds only when a table
  half the size would fit, so its table is at most twice the size it needs
  and the rebuilds cost O(blocks split off) over the node's lifetime.
- `full_edge_workers`: workers that cover the full compressed edge.
- `worker_cutoffs`: workers that cover only a prefix of the edge. A cutoff `k`
  means the worker has cached `edge[0..k]`, with `0 < k < edge.len()`.
- `children`: child nodes keyed by the first `LocalBlockHash` of the child
  edge.
- `parent_tail`: the hash of the block before `edge[0]`, or 0 under the root.
  A node's head never moves, so however often the tree splits, its parent is
  the node whose edge ends with this block. See
  [Dead Leaf Reclamation](#dead-leaf-reclamation).
- `shape_gate` and `shape_version`: a per-node shape guard used to validate
  plans across lock gaps.
- `internal`: a sticky marker that becomes true once the node has had children.
  It remains true even if cleanup later removes every physical child, so the
  node is not reopened for leaf extension.
- `anchor`: set only on synthetic branch anchors. They stay registered for the
  tree's lifetime and never change their one-block edge, so the flag identifies a
  store parent as an anchor without probing the anchor map.

Worker lookup tables are not stored on nodes. Each event lane owns a
`LaneLookup` that maps, for each of its workers, external block hashes to the
node that should contain that hash. Entries name nodes by lane-local ids: the
lane holds one `Arc` per distinct node its entries name and counts those entries
locally, so storing or removing a block does not touch the node's shared
reference count. Stale-leaf cleanup still sees every lane that names a node:
such a node carries one extra strong reference per lane that names it.

## Store Paths

The indexer is optimized around two common KV-cache store patterns.

### Prefill Fanout

Many workers may share a prefix and then store different prompt suffixes under
the same parent:

```text
shared prefix parent
many different workers
many different first child hashes
insert new compressed child nodes
```

The desired behavior is to let independent child inserts proceed concurrently
when the parent edge is stable. Child insertion uses the shared shape gate and a
shape-version check, so it can proceed without exclusive edge-shape ownership.

### Decode Extension

During decode, a worker often appends small batches to the tail of its own
compressed edge:

```text
one worker
one leaf compressed node
parent hash is current tail
node has never been internal
append more blocks to the edge
```

This path attempts a direct leaf extension. It is allowed only when the node is
not `internal`, the parent hash is still the edge tail, and the worker covers
that tail. If the node has ever had children, decode falls back to child
insertion or splitting instead of extending the compressed edge.

### Suffix Reuse And Splitting

When a store names a parent hash inside an existing compressed edge, the node can
reuse a matching existing suffix. If the new store diverges from that suffix, the
node is split at the parent position. The suffix becomes a child node and keeps
the original children, so existing descendants remain reachable after the split.

The suffix retains the original child-storage representation. The prefix starts
with compact child storage for its new suffix child, even if it previously had a
sharded map. This avoids allocating a sharded map for a prefix with only a few
children. The prefix remains logically internal and cannot resume leaf extension.

## Removal

Removal updates worker coverage but does not structurally split edges.

When a remove event arrives for worker `w` at edge position `i`:

- `current_cutoff` is `edge.len()` if `w` is in `full_edge_workers`; otherwise
  it is `worker_cutoffs[w]`.
- If `i >= current_cutoff`, the remove is a no-op because the block is already
  beyond that worker's coverage.
- If `i < current_cutoff`, the new cutoff is `i`.
- If the new cutoff is `0`, the worker is removed from the node.
- Otherwise the worker moves to `worker_cutoffs[w] = new_cutoff`.
- Worker lookup entries for the newly uncovered suffix are scrubbed eagerly.

A remove event is applied in runs, each under one exclusive shape gate. A run
starts at the node its first hash's lookup entry names, or the node
[lookup repair](#lookup-repair) resolves that hash to, and takes the following
hashes whose entries name that node. A hash whose entry names another node starts
its own run, because that node may hold the worker's coverage of it: once cleanup
unlinks a subtree the worker's lookup still names, a partial restore can store
some of its hashes on a new live node, and consuming them on the unlinked node
would leave the live coverage behind. The run is scanned under an upgradable
state read, which upgrades to the write lock only when the removal changes the
worker's coverage.

After the coverage update, removal may clear children only when no full-edge
workers remain. Because `internal` is sticky, clearing those children does not
make the node eligible for future leaf extension.

A removal that takes a node's last holder leaves it linked as a dead leaf; the
lane queues it for [reclamation](#dead-leaf-reclamation).

## Dead Leaf Reclamation

Removal changes coverage only, so a leaf whose last holder evicts it stays
linked under its parent until something unlinks it. Under tail-first eviction
almost every dead node is such a leaf, and they can hold most of the tree's
memory. Unlinking one is a structural change on the parent: it takes the
parent's exclusive shape gate and bumps `shape_version`. Neither reclaimer
waits for that gate.

- **Per-lane drain.** When a removal takes a leaf's last holder, the lane finds
  the leaf's parent and queues the pair. The parent is the root, or the node
  whose edge ends with the leaf's `parent_tail`: an anchor, or the node the
  evicting worker's lookup names for that block. The parent is verified by
  identity, and a stale lookup entry is not repaired on this path. The lane
  unlinks queued leaves whenever no task is waiting, at most 256 at a time,
  and all of them before a `Flush` completes. A queue that reaches 4,096
  drains a quarter at once. Each parent's leaves go under one try-locked gate
  and one version bump. A split that races an eviction on another lane can
  produce a suffix that is dead from birth, which no removal will kill; the
  splitting lane queues it with the prefix as its parent.
- **Background sweep.** `ThreadPoolIndexer` runs a `kv-maint` thread that
  sweeps once 1,024 nodes have died since the last sweep or 30 s have passed,
  checking every 50 ms. The walk takes no gates: it records holder-less
  children under the root and every anchor, descending only into nodes with
  children. Each parent's candidates are re-checked and unlinked under one
  try-locked gate, deepest parents first. The sweep catches what the drain
  missed: leaves whose parent was busy or not in the evicting worker's lookup,
  and leaves something still referenced.

Both unlink a child only while it is still attached, holder-less and
childless, its own shape gate is free, and nothing references it but the
parent's map and the unlinker (`Arc::strong_count == 2`). A lane lookup that
still names the leaf, or a reader mid-walk, keeps it for a later sweep. A store
that reaches a dead leaf before it is unlinked re-covers it.

## Lookup Repair

Cross-thread splits can make a worker lookup entry stale: the lookup still points
at the old prefix node even though the requested external hash moved to the new
suffix child. Writers handle this lazily:

```text
worker lookup says hash -> old node
old node no longer contains hash
scan descendants for a node containing hash
repoint entries in the useful covered range that still name the old node
```

Remove and store-parent resolution do not probe the node before using a lookup
entry. The locked operation they already run detects the miss: grouped removal
returns nothing when the run's first hash is not in the edge, and the store's
parent-coverage check reports the parent as missing. Because grouped removal
upgrades its state read only on a coverage change, a miss on a read-hot split
prefix never blocks readers. Only then does `repair_stale` scan descendants, and
the operation retries on the resolved node, not on the lookup entry: remove
repair never rewrites the removed hash's own entry, and store repair leaves it
untouched when the worker no longer covers the parent there. Re-reading the
entry after repair would loop between the stale and resolved nodes. The
stale-parent path inside child insertion still validates through
`resolve_lookup`, because its entry may already point past the split.

Store repair rewrites from the requested parent hash toward the worker's covered
tail. Remove repair rewrites from the head toward the removed hash, excluding the
suffix that removal is about to scrub. This matters for tail-to-head removes:
once one hash in a moved compressed edge repairs the useful head range, later
hashes in that same edge should hit the resolved node directly instead of paying
another subtree scan.

Repair and split lookup updates move only entries that still name the old node,
and never insert missing ones. An entry naming any other node either holds that
worker's coverage there or is stale from an older split; the latter is repaired
on its own next access.

If the scan fails in a store path, the store is rejected with
`ParentBlockNotFound` and logged as a warning. If it fails in a remove path, the
remove treats the block as already gone or stale and skips it.

## Concurrency Model

`ThreadPoolIndexer` sticky-routes events by worker id, so a single worker's KV
events are serialized on one event thread. Different workers can still mutate
shared CRTC nodes concurrently.

Node internals use separate protection for edge state and child maps:

- `CrtcNodeState` is protected by a `parking_lot::RwLock`.
- `children` publishes compact snapshots through `ArcSwap`, promoting to a
  `DashMap` when fanout exceeds four children. A node allocates its `ArcSwap`
  only when it gets its first child, which seeds it directly: dropping or
  swapping one walks every thread's `arc_swap` debt list, and most nodes that
  cleanup frees are leaves that never had a child.
- `shape_gate` and `shape_version` coordinate plans that depend on the relation
  between the edge and child map.

Light shape updates, such as attaching a missing child under a stable parent, use
a shared shape gate plus version validation. Heavy edge-shape updates, such as
splitting an edge, extending a leaf, or moving children to a suffix node, use the
exclusive shape gate.

`find_matches` is best-effort during concurrent shape changes. It reads node
state and child pointers without taking `shape_gate` on the hot step, so it may
observe adjacent tree shapes during a split and undercount. It must not panic,
and apart from a fingerprint collision in the equal-set skip below it must not
return a match past a valid reachable prefix.

### Equal-set skip

After the first node, the walk intersects its active workers with each node's
full-edge coverage. When the two sets are equal it skips that intersection,
which saves a pass over up to all workers at most hops: on the Mooncake replay
with 128 workers, always intersecting raised lookup service p50 by 34%.

Size alone does not prove the sets equal. Removal does not cascade to
children, so after a worker's head blocks are evicted a child can still list
it, and an equal-sized coverage set can then hold a different worker than the
walk carries. Both sets are therefore `WorkerSet`s, which keep the XOR of a
64-bit mix of each member, and the walk skips only when the sizes and these
fingerprints match. The check is probabilistic: unequal sets collide with
probability about 2^-64 per comparison, and a collision overcounts exactly as
the size-only check did. That is the one remaining overcount path.

## Wire Compatibility

- `find_matches` leaves the legacy `OverlapScores.frequencies` field empty.
