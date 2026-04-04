# Node-Depth Sharding: Design and Hybrid Routing

This document covers the design of `NodeDepthShardedIndexer`, the root cause of
the parent-inheritance imbalance problem, the hybrid depth-aware routing fix,
and benchmark results on the `.prefix` workload.

---

## 1. Background: Why Node-Depth Sharding

### The shared-prefix problem

On workloads with a long universal system prompt (e.g. the `.prefix` trace, where
every conversation starts with the same 15-block / 960-token system prompt),
block-depth routing (`BranchShardedIndexer`, `PrefixShardedIndexer`) collapses:
every request shares the same first `N` blocks, so all traffic hashes to the
same routing key regardless of depth.

### Why CRTC node depth solves it

The `ConcurrentRadixTreeCompressed` (CRTC) path-compresses shared prefixes into
single edges.  The entire 15-block system prompt is **one CRTC node** — one edge
from the root.  Each distinct conversation continuation (new user turn) is a
second CRTC node branching off from the system-prompt node.

Routing at `routing_node_depth=2` therefore sees all distinct conversation
branches immediately, without needing to know the system-prompt length in
advance.

### The shadow trie

A `ShadowNode` trie mirrors the top-`routing_node_depth` levels of the CRTC
structure.  Each edge is a `Vec<LocalBlockHash>` (variable length, matching one
CRTC-node edge).  Edges split when two sequences diverge mid-edge, just as the
CRTC splits its own nodes.

- **`find_matches`** read-locks the trie, walks `routing_node_depth` edges,
  returns the leaf shard.  O(routing_node_depth) hash lookups.
- **`apply_event`** write-locks the trie, walks/creates/splits nodes, assigns
  the least-loaded shard to new leaves.

Routing leaves at `node_depth >= routing_node_depth` are **never split** — they
are permanent routing assignments.

---

## 2. The Imbalance vs. Incoherence Trade-off

### `inherit_parent_shard=true` — imbalance

When a Stored event arrives, the indexer looks up `parent_hash` in
`block_to_shard` and routes all continuation events to the parent's shard.  The
trie is only consulted for root events.

On the `.prefix` workload:

1. The first root event carries the system-prompt blocks → the shadow trie
   creates a depth-1 leaf, assigned to shard 0.
2. Every continuation event's `parent_hash` resolves to a system-prompt block →
   inherits shard 0.
3. Routing diversity at depth 2 (across distinct conversation turns) is never
   exercised at event-routing time.

**Result**: shard 0 owns 100% of all KV blocks.

### `inherit_parent_shard=false` (naive) — incoherence

Without inheritance, every Stored event routes its own incremental blocks
through the shadow trie starting from the root.

On the `.prefix` workload, a continuation event carries only `[t2b0, t2b1]` (the
new user-turn blocks), not the full prefix.  `insert_and_get_shard` sees
`[t2b0, t2b1]` with `seq_pos=0` — it looks for `t2b0` among the root's
children, finds none (the root only has the system-prompt branch keyed on
`sp0`), and creates a **new depth-1 leaf** as a sibling of the system-prompt
branch.

```
Before:  root → [sp0..sp14]  depth=1  shard=X
After:   root → [sp0..sp14]  depth=1  shard=X
         root → [t2b0,t2b1]  depth=1  shard=Y   ← phantom branch
```

Now `find_matches` for a query `[sp0..sp14, t2b0, t2b1, ...]` routes via the
system-prompt path → shard X.  But `t2b0` and `t2b1` are stored on shard Y.
The shard CRTC on shard X has no knowledge of turn-2 blocks → KV cache miss.

**The incoherence**: the routing key (trie path) and the actual storage location
(`block_to_shard`) disagree for continuation blocks.

---

## 3. Root Cause: `insert_and_get_shard` always starts at `seq_pos=0`

The naive code:

```rust
} else {
    // No parent inheritance — always route by the event's own blocks.
    let local: Vec<LocalBlockHash> =
        store_data.blocks.iter().map(|b| b.tokens_hash).collect();
    self.assign_via_trie(&local)   // starts from root every time
};
```

`assign_via_trie` calls `insert_and_get_shard(&mut trie, local_hashes, 0, ...)`.
The `0` is `seq_pos` — always the trie root.  The function has no way to know
that `[t2b0, t2b1]` belongs under the system-prompt node; it only sees the
incremental blocks in isolation.

The `block_to_shard` map knows which shard `sp14` is on, but has no pointer to
`sp14`'s position inside the shadow trie.  The shadow trie has no reverse index
of `block_hash → *ShadowNode`.

---

## 4. Hybrid Depth-Aware Routing

### Key insight: routing leaves are permanent

The code already ensures that nodes at `node_depth >= routing_node_depth` are
never split (guarded at the split site).  Once a sequence has walked
`routing_node_depth` CRTC nodes, its routing leaf assignment is permanent.

This means:

- **Parent at routing depth** → future continuations can inherit the shard
  unconditionally — no trie access needed.
- **Parent is shallow** (below routing depth) → we must reconstruct the full
  prefix and re-walk the trie so the continuation is inserted as a child of the
  correct node rather than a phantom root sibling.

### Why node depth is unreliable to track directly

Splits can promote nodes from depth D to depth D+1, invalidating any stored
depth integer.  Example:

```
routing_depth = 2

Insert [a,b,c,d,e]:  root → [a,b,c,d,e]  depth=1  (stored depth=1 for 'e')
Insert [a,b,c,a]:    split →
    root → [a,b,c]       depth=1  interior
           → [d,e]        depth=2  ('e' now at depth=2 — stored depth stale!)
           → [a]          depth=2
```

Because routing leaves are never split, the flag "has this sequence reached a
routing leaf" is **monotonically stable**: it can flip from `false → true` via a
split promotion, but never `true → false`.  We can safely store this boolean
without invalidation.

### Data structure: `last_block_to_path`

```rust
last_block_to_path: DashMap<u64, Option<Vec<LocalBlockHash>>, FxBuildHasher>
```

Keyed on the `ExternalSequenceBlockHash.0` of the **last block** of each stored
event (the only block ever referenced as `parent_hash` in continuations):

| Value | Meaning |
|-------|---------|
| `None` | Sequence reached a routing leaf — inherit shard, no trie access |
| `Some(path)` | Sequence is shallow — `path` is the full prefix from the trie root |

### Modified `insert_and_get_shard`

Return type changed from `usize` to `(usize, bool)`.  The `bool` is `true`
only at the `node.node_depth >= routing_depth` base case — reaching a permanent
routing leaf.

```
Base case (routing leaf):   return (shard, true)
Base case (seq exhausted):  return (shard, false)   ← still shallow
New leaf at depth D+1:      return (shard, D+1 >= routing_depth)
Split → new leaf at D+1:    return (shard, D+1 >= routing_depth)
```

### `apply_event` logic (hybrid path)

```
Continuation event, parent_hash = P:

  1. Look up P in last_block_to_path:
     ├── None (reached routing depth):
     │     inherit shard from block_to_shard[P]
     │     store last_block → None  (still settled)
     │
     ├── Some(prefix) (shallow):
     │     full_seq = prefix + incremental_blocks
     │     (shard, reached) = assign_via_trie(full_seq)
     │     store last_block → if reached { None } else { Some(full_seq) }
     │
     └── not found (OOO / evicted / predates tracking):
           (shard, reached) = assign_via_trie(incremental_blocks)
           store last_block → if reached { None } else { Some(incremental_blocks) }

Root event (no parent_hash):
    (shard, reached) = assign_via_trie(incremental_blocks)
    store last_block → if reached { None } else { Some(incremental_blocks) }
```

### Memory behaviour

`Some(path)` entries are only live while a conversation is in the routing
region (depth < routing_node_depth).  On the `.prefix` workload
(`routing_node_depth=2`):

- Root event (system prompt, 15 blocks): `reached=false` → stores
  `Some([sp0..sp14])` — 15 hashes.
- First continuation (new user turn): full_seq = `[sp0..sp14] + [t2b0,t2b1]`
  → trie creates depth-2 leaf → `reached=true` → stores `None`.
- All further continuations: parent has `None` → inherit directly, no path
  stored.

After the second event, memory overhead drops to zero per conversation.
`Some(path)` entries are bounded to the few events that cross the
routing-depth boundary.

### Cleanup

- **Removed events**: `last_block_to_path.remove(bh)` alongside
  `block_to_shard.remove(bh)`.
- **Cleared events**: both maps are cleared.

---

## 5. Benchmark Results

**Workload**: `.prefix` synthetic trace —
`conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl`
(15-block universal system prompt, ~11M events, ~374k ops/s offered load)

**Config**: `node-depth-sharded-crtc`, `--no-parent-inheritance`,
`--routing-node-depth 2`, `--num-event-workers-per-shard 4`,
`--benchmark-duration-ms 30000`.
Total cores = shards × 4 (linear scaling).

### Timing breakdown

| Shards | Total cores | Avg routing overhead | Avg shard time | Total (approx) | p99 latency |
|--------|-------------|----------------------|----------------|----------------|-------------|
| 2      | 8           | 1607 ns              | 94 µs          | ~95.6 µs       | 135 µs      |
| 4      | 16          | 2198 ns              | 65 µs          | ~67.2 µs       | 100 µs      |
| 8      | 32          | 2111 ns              | 43 µs          | ~45.1 µs       | 71 µs       |

- **Routing overhead** (shadow trie lookup): <2.5 µs in all cases — negligible
  vs. shard work.
- **Shard time** halves (roughly) with each doubling of shards, consistent with
  half the CRTC nodes per shard.

### Load balance (routing leaves per shard)

| Shards | Distribution |
|--------|-------------|
| 2      | shard[0]=56995, shard[1]=56994 — exactly 50/50 |
| 4      | 28497 each — exactly 25% |
| 8      | 14248–14249 each — exactly 12.5% |

Perfect balance confirms the hybrid is working: each depth-2 routing leaf is
assigned by the least-loaded algorithm at creation time, and no single shard
monopolises the system-prompt root.

### CRTC node scaling

| Shards | Total nodes across shards | Avg hashes/node | p99 hashes/node |
|--------|--------------------------|-----------------|-----------------|
| 2      | 1,620,047                | 3.0             | 64              |
| 4      | 693,700                  | 4.2             | 136             |
| 8      | 337,329                  | 5.1             | 195             |

Node count halves per shard doubling.  The increasing `p99 hashes/node` reflects
that with more shards each shard's CRTC is deeper and less path-compressed
(fewer sequences share full prefixes within a shard).

### Miss rate

83.3% of `find_matches` calls are trie misses (sequences not yet seen by any
worker).  This is expected for the `.prefix` trace structure and is unchanged
across shard counts — the miss path is a fast early return.

---

## 6. Implementation File

`lib/kv-router/src/indexer/node_depth_sharded.rs`

Key changes relative to the original implementation:

1. `insert_and_get_shard` — return type `usize` → `(usize, bool)`
2. `assign_via_trie` — return type `usize` → `(usize, bool)`
3. `NodeDepthShardedIndexer` — added `last_block_to_path` field
4. `apply_event` / Stored — replaced naive trie-from-root with hybrid
   `Resolution::Inherit` / `Resolution::TrieInsert` logic
5. `apply_event` / Removed — clears `last_block_to_path` entries
6. `apply_event` / Cleared — clears both maps
