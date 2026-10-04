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
- `full`: workers that cover the full compressed edge, as one bit per rank slot
  (see [Rank Slots](#rank-slots)).
- `cutoffs`: workers that cover only a prefix of the edge, keyed by slot. A
  cutoff `k` means the worker has cached `edge[0..k]`, with `0 < k < edge.len()`.
  A slot is never both full and cut off, except briefly while a promote deletes
  its old cutoff.
- `children`: child nodes keyed by the first `LocalBlockHash` of the child
  edge.
- `shape_gate` and `shape_version`: a per-node shape guard used to validate
  plans across lock gaps.
- `internal`: a sticky marker that becomes true once the node has had children.
  It remains true even if cleanup later removes every physical child, so the
  node is not reopened for leaf extension.
- `anchor`: set only on synthetic branch anchors. They stay registered for the
  tree's lifetime and never change their one-block edge, so the flag identifies a
  store parent as an anchor without probing the anchor map.

Worker lookup tables are not stored on nodes. Each event lane owns a
`LaneLookup` that maps, for each of its ranks, external block hashes to the node
that should contain that hash. Entries name nodes by lane-local ids: the lane
holds one `Arc` per distinct node its entries name and counts those entries
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

- `current_cutoff` is `edge.len()` if `w`'s bit is set in `full`; otherwise it
  is `w`'s entry in `cutoffs`, or `0`.
- If `i >= current_cutoff`, the remove is a no-op because the block is already
  beyond that worker's coverage.
- If `i < current_cutoff`, the new cutoff is `i`.
- If the new cutoff is `0`, the worker is removed from the node.
- Otherwise the worker moves to `cutoffs` with `new_cutoff`. Its cutoff is
  written before its full bit is cleared.
- Worker lookup entries for the newly uncovered suffix are scrubbed eagerly.

After the coverage update, removal may clear children only when no full-edge
workers remain. Because `internal` is sticky, clearing those children does not
make the node eligible for future leaf extension.

## Lookup Repair

Cross-thread splits can make a worker lookup entry stale: the lookup still points
at the old prefix node even though the requested external hash moved to the new
suffix child. Writers handle this lazily:

```text
worker lookup says hash -> old node
old node no longer contains hash
scan descendants for a node containing hash
rewrite the useful covered range in the resolved node
```

Remove and store-parent resolution do not probe the node before using a lookup
entry. The locked operation they already run detects the miss: grouped removal
returns nothing when the run's first hash is not in the edge, and the store's
parent-coverage check reports the parent as missing. Grouped removal scans under
an upgradable state read and upgrades only when a hit changes a cutoff (see
[Full-Coverage Writes](#full-coverage-writes)), so a miss on a read-hot split
prefix never blocks readers. Only then does `repair_stale` scan
descendants, and the operation retries on the resolved node, not on the lookup
entry: remove repair never rewrites the removed hash's own entry, and store
repair leaves it untouched when the worker no longer covers the parent there.
Re-reading the entry after repair would loop between the stale and resolved
nodes. The stale-parent path inside child insertion still validates through
`resolve_lookup`, because its entry may already point past the split.

Store repair rewrites from the requested parent hash toward the worker's covered
tail. Remove repair rewrites from the head toward the removed hash, excluding the
suffix that removal is about to scrub. This matters for tail-to-head removes:
once one hash in a moved compressed edge repairs the useful head range, later
hashes in that same edge should hit the resolved node directly instead of paying
another subtree scan.

Repair is batched across the lane's workers, but it only rewrites entries that
still name the node the scan started from. Likewise, a split only rewrites the
lane's entries that still name the split prefix. Any other entry is left for its
own lazy repair. This keeps a lookup from moving onto a node merely because a
recycled slot has stale bits there (see [Rank Slots](#rank-slots)).

If the scan fails in a store path, the store is rejected with
`ParentBlockNotFound` and logged as a warning. If it fails in a remove path, the
remove treats the block as already gone or stale and skips it.

## Concurrency Model

`ThreadPoolIndexer` sticky-routes events by worker id, so a single worker's KV
events are serialized on one event thread. Different workers can still mutate
shared CRTC nodes concurrently.

Node internals use separate protection for edge state and child maps:

- The edge and `cutoffs` are protected by a `parking_lot::RwLock`. The `full`
  bits are atomics outside it that readers load under the read lock, so a reader
  sees bits consistent with the edge. See
  [Full-Coverage Writes](#full-coverage-writes) for who writes them under which
  lock.
- `children` publishes compact snapshots reclaimed through epochs (`crossbeam-epoch`), promoting to a
  `DashMap` when fanout exceeds four children.
- Every child `Arc` a reachable child map owns is released through the epoch:
  retired compact snapshots and `Arc`s a `DashMap` unlinks in place. A dropped
  node frees its own map inline, which is safe because no reader can reach a node
  whose last `Arc` is gone. Dropping the whole tree also frees every map inline,
  since exclusive access rules out readers. `find_matches` pins once per walk and
  borrows each node under that pin instead of cloning its `Arc`, so a node it
  stands on stays allocated even if a writer unlinks it meanwhile. Writers still
  hold `Arc`s.
- Expired epoch garbage is not freed inside the collection that releases it,
  which often runs on a `find_matches` caller. It goes to the collecting thread's
  graveyard when that thread drains one: a lookup frees a small budget after it
  unpins and hands the rest to a shared graveyard, and an event lane frees some
  after each event and all of it when idle or on `Flush`. Garbage collected on
  any other thread (rayon and moka pin the same global collector), overflow, and
  garbage from exiting threads also go to the shared graveyard, which event
  lanes drain. So a detached subtree never stalls one lookup, an idle lookup
  thread holds no garbage, and a lookup touches the shared queue only when its
  collection left more than its budget. A drain releases a retired snapshot's
  reference counts only as it drops each child, so leftovers of a spent budget,
  moved or not, stay counted.
- `shape_gate` and `shape_version` coordinate plans that depend on the relation
  between the edge and child map.

Light shape updates, such as attaching a missing child under a stable parent, use
a shared shape gate plus version validation. Heavy edge-shape updates, such as
splitting an edge, extending a leaf, or moving children to a suffix node, use the
exclusive shape gate.

`find_matches` is best-effort during concurrent shape changes. It reads node
state and child pointers without taking `shape_gate` on the hot step, so it may
observe adjacent tree shapes during a split and undercount. Because reads hold
no reference counts, stale-leaf cleanup can also unlink an empty leaf a read is
standing on; that leaf has no workers, so the read cannot overcount there. It
must not panic, and it must not return a match past a valid reachable prefix.

After the first node, the walk intersects its active slots with each node's
full-edge bits on every hop. An earlier hash-set walk skipped that intersection
when both sets had the same size. That overcounted whenever equal sizes hid
different members: after a head-first eviction a child can still list a worker
whose head blocks are gone, and a concurrent store can promote another worker.
Over slot words the intersection is a few word operations, so the walk always
takes it.

### Full-Coverage Writes

Promoting a worker to full coverage of an edge sets one bit, so it does not take
the state write lock:

- The promote takes the shared shape gate and validates the shape version
  before it looks at the bit. A set bit alone does not prove the plan is
  current: a split also sets it when it promotes the worker's cutoff on the
  prefix, and leaves only a cutoff on the suffix.
- It then sets the bit, unless it is already set. It takes the state write lock
  only when the node has cutoffs, to delete the worker's own stale cutoff, and it
  always sets the bit first, so readers never miss the worker.
- A removal that drops a full worker from the whole edge clears its bit under the
  exclusive shape gate it already holds, with only a read lock on the state. A
  removal that leaves a partial cutoff upgrades that lock to the write lock and
  publishes the cutoff before it clears the bit.

Splits, leaf extensions, sweeps, and cutoff changes keep the state write lock.
Everything that moves bits between nodes or demotes them to cutoffs holds the
exclusive gate, which excludes every shared-gate promote. A reader may see a bit
change between two hops or, past 64 slots, between two words of one hop; each
bit it sees is a coverage state that existed during the walk.

### Rank Slots

Every rank (`WorkerWithDpRank`) that stores into the tree gets a dense slot from
the tree's slot registry, and coverage records slots instead of ranks. Slots
below 256 are bits inline in every node; higher slots use 256-slot chunks that
are installed once and never move. Readers walk with a slot bitset and map
surviving or dropped slots back to ranks through a slot table they load once per
walk. Writers resolve a rank's slot from the current table on every event, under
the epoch guard that covers the event.

- A rank gets the lowest free slot on its first `Stored` event or anchor.
  `Removed` and `Cleared` never allocate. A `Cleared` rank keeps its slot.
- If every slot is taken, the store fails with `CapacityExhausted`: the rank
  stays unindexed and the event metric records it. It is never credited with
  another rank's coverage.
- A slot is released only by the sweeping `RemoveWorker` or `RemoveWorkerDpRank`
  handler, in four steps:
  1. Unmap the rank, so its later events resolve a new slot.
  2. Wait until every thread pinned before the unmap has unpinned, so events
     that resolved the old slot have finished writing its bits.
  3. Sweep the slot out of every node reachable from the root or an anchor.
     The sweep keeps no visited set: the tree has no cycles, and an
     address-keyed set would skip a split suffix that reuses the address of a
     node the sweep already cleared and freed.
  4. Vacate the table entry and free the slot through the epoch, so a reader
     still holding the old table never sees the slot reused.
- A sweeping removal whose ranks another lane has already unmapped, and a
  `Cleared` for a rank that is mid-removal, even one that has stored under a new
  slot since, wait for that lane's step 4 before returning, so an acknowledged
  removal or clear never leaves coverage behind.
- A `Cleared` sweep keeps the rank's slot. It repins every few nodes and stops
  once the rank no longer maps to the slot under the current pin, leaving the
  rest to the removal that unmapped it. A pinned check holds back step 4, so a
  clear never touches a slot after it is recycled to another rank.
- The sweep cannot reach subtrees that removal had already unlinked, so a
  recycled slot can keep stale bits there. Readers reach an unlinked node only
  while pinned from before the unlink, which step 4 waits out. The slot's new
  rank stores only into nodes that were reachable after the sweep, and split and
  repair lookup updates move an entry only off the node they replace (see
  [Lookup Repair](#lookup-repair)), so none of its entries lands on stale bits.
- A dump does not stay pinned across the whole tree, so it can reach an
  unlinked node after its slot was recycled. Like a reader, it credits a rank
  below a parent only if the rank covers the parent's whole edge, and it maps a
  merged chain through one slot table under one pin. The parent check compares
  rank identities captured under an earlier table, so a rank that was removed,
  re-added, and given the recycled slot in between can still be emitted for the
  previous owner's stale bits on an unlinked node. This needs a concurrent
  worker removal during the dump and only affects the dumped events.

## Wire Compatibility

- `find_matches` leaves the legacy `OverlapScores.frequencies` field empty.
