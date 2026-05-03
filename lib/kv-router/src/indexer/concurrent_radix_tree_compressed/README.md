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

## Node State

Each node contains:

- `edge`: the compressed sequence of local and external block hashes.
- `edge_index`: reverse lookup from `ExternalSequenceBlockHash` to position in
  the edge. Removal uses this to find an evicted block in O(1) once it has the
  node.
- `full_edge_workers`: workers that cover the full compressed edge.
- `worker_cutoffs`: workers that cover only a prefix of the edge. A cutoff `k`
  means the worker has cached `edge[0..k]`, with `0 < k < edge.len()`.
- `children`: child nodes keyed by the first `LocalBlockHash` of the child
  edge.
- `shape_gate` and `shape_version`: a per-node shape guard used to validate
  plans across lock gaps.
- `internal`: a sticky marker that becomes true once the node has had children.
  It remains true even if cleanup later removes every physical child, so the
  node is not reopened for leaf extension.

Worker lookup tables are not stored on nodes. Each event worker owns its own
`WorkerLookup`, mapping external block hashes to the node that should contain
that hash.

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

After the coverage update, removal may clear children only when no full-edge
workers remain. Because `internal` is sticky, clearing those children does not
make the node eligible for future leaf extension.

## Lookup Repair

Cross-thread splits can make a worker lookup entry stale: the lookup still points
at the old prefix node even though the requested external hash moved to the new
suffix child. `resolve_lookup` handles this lazily:

```text
worker lookup says hash -> old node
old node no longer contains hash
scan descendants for a node containing hash
rewrite the useful covered range in the resolved node
```

Store repair rewrites from the requested parent hash toward the worker's covered
tail. Remove repair rewrites from the head toward the removed hash, excluding the
suffix that removal is about to scrub. This matters for tail-to-head removes:
once one hash in a moved compressed edge repairs the useful head range, later
hashes in that same edge should hit the resolved node directly instead of paying
another subtree scan.

If the scan fails in a store path, the store is rejected with
`ParentBlockNotFound` and logged as a warning. If it fails in a remove path, the
remove treats the block as already gone or stale and skips it.

## Concurrency Model

`ThreadPoolIndexer` sticky-routes events by worker id, so a single worker's KV
events are serialized on one event thread. Different workers can still mutate
shared CRTC nodes concurrently.

Node internals use separate protection for edge state and child maps:

- `NodeState` is protected by a `parking_lot::RwLock`.
- `children` is a `DashMap`.
- `shape_gate` and `shape_version` coordinate plans that depend on the relation
  between the edge and child map.

Light shape updates, such as attaching a missing child under a stable parent, use
a shared shape gate plus version validation. Heavy edge-shape updates, such as
splitting an edge, extending a leaf, or moving children to a suffix node, use the
exclusive shape gate.

`find_matches` is best-effort during concurrent shape changes. It reads node
state and child pointers without taking `shape_gate` on the hot step, so it may
observe adjacent tree shapes during a split and undercount. It must not panic or
return a match past a valid reachable prefix.

## Limitations Compared With `RadixTree`

- It does not support `expiration_duration` or frequency tracking.
- It does not provide `new_with_frequency()`.
- `find_matches` does not populate `OverlapScores.frequencies`.
