# ⚡ FlashIndexer — KV Router Index Data Structures

This document explains the KV cache index implementations: `RadixTree` (and its concurrent variant `ConcurrentRadixTree`) and `PositionalIndexer` (NestedMap).

The concurrent indexers achieve a combined throughput of over **10 million events + requests per second** with **p99 latency under 10 microseconds**.

## Motivation: The Four Block Identifiers

Every cached KV block in a distributed LLM system needs four pieces of information:

### 1. Local Block Hash (`LocalBlockHash`, u64)

**What**: Hash of the tokens *within* a single block (e.g., 64 tokens).

**Why**: Identifies the content of this specific block, independent of context. Two blocks with the same tokens have the same local hash.

```
Block at position 5: tokens [101, 102, 103, ...]
LocalBlockHash = hash(tokens) = 0xABCD1234
```

### 2. External Sequence Block Hash (`ExternalSequenceBlockHash`, u64)

**What**: Cumulative hash of the entire sequence up to and including this block.

**Why**: Uniquely identifies a block's position in a *specific* sequence history. Two blocks with the same local content but different prefixes have different sequence hashes.

```
Sequence A: [block0, block1, block2]
Sequence B: [block0', block1', block2]  // block2 has same content but different prefix

block2 in A: seq_hash = hash(hash(hash(block0) || block1) || block2) = 0x1111
block2 in B: seq_hash = hash(hash(hash(block0') || block1') || block2) = 0x2222
```

**Computation**: `seq_hash[i] = hash(seq_hash[i-1] || local_hash[i])` where `seq_hash[0] = local_hash[0]`

> **Important: Engine-Provided Hashes**
>
> In practice, the `ExternalSequenceBlockHash` may come directly from the inference engine (e.g., vLLM, TensorRT-LLM) using a rolling hash algorithm that we don't know or control. The engine computes these hashes internally and reports them via KV cache events.
>
> **Implications for index implementations:**
>
> - **RadixTree**: Can handle engine-provided hashes because it traverses the tree structure using `LocalBlockHash` for navigation and only uses `ExternalSequenceBlockHash` as an opaque identifier for lookups. It doesn't need to recompute hashes.
>
> - **NestedMap**: Requires the ability to compute `ExternalSequenceBlockHash` incrementally for its lazy hash optimization in `find_matches`. To use NestedMap, one of the following is required:
>   1. **Force a known hasher**: Configure the engine to use a specific hashing algorithm that the router can replicate, OR
>   2. **Recompute on the relay**: Have the publisher/relay layer recompute the rolling hash using a known algorithm before forwarding events to the router.
>
> Without this, NestedMap's `find_matches` will fail when encountering `SeqEntry::Multi` cases (multiple seq_hashes at the same position+local_hash) because it cannot disambiguate which entry to use.

### 3. Worker ID (`WorkerWithDpRank`)

**What**: Identifies which worker (inference server) has this block cached.

**Why**: The router needs to know which workers can serve a request based on their cached blocks.

### 4. Position (`usize`)

**What**: The block's index in the sequence (0, 1, 2, ...).

**Why**: Enables efficient prefix matching. Position 0 is the first block, position N-1 is the last.

---

## The Core Operations

Both data structures support three operations:

| Operation | Description |
|-----------|-------------|
| `store_blocks` | Add blocks for a worker (background) |
| `remove_blocks` | Remove blocks for a worker (background) |
| `find_matches` | Find workers with matching prefix (per-request) |

**Read vs write cost and frequency**: When the radix tree has little or no shared prefix for a request, `find_matches` can exit after a single root-level lookup (first block miss)—reads then do less work than writes (which traverse and update multiple nodes). Vice versa, with large prefix overlap, reads traverse deeper and can be invoked more often than writes. We consider both extremes; the proposed data structures are designed to handle them.

---

## RadixTree: Tree-Based Index

### Structure

```
RadixTree
├── root: SharedRadixBlock (Rc<RefCell<RadixBlock>>)
└── lookup: HashMap<Worker, HashMap<SeqHash, SharedRadixBlock>>

RadixBlock
├── children: HashMap<LocalBlockHash, SharedRadixBlock>
├── workers: HashSet<Worker>
├── block_hash: Option<SeqHash>
└── recent_uses: VecDeque<Instant>
```

### Visual Representation

```
                    [root]
                   /      \
            local=0xA    local=0xB
               ↓            ↓
           [block0]     [block0']
           workers:     workers:
           {W0,W1}      {W2}
              |
         local=0xC
              ↓
          [block1]
          workers:
          {W0,W1}
              |
         local=0xD
              ↓
          [block2]
          workers:
          {W0}         ← W1 diverged here
```

### How Operations Work

**store_blocks(worker, parent_hash, blocks)**:
1. Find parent via `lookup[worker][parent_hash]`
2. For each block, traverse/create child nodes using `local_hash`
3. Add worker to each node's `workers` set
4. Update `lookup[worker][seq_hash] = node`

**remove_blocks(worker, block_hashes)**:
1. For each hash, find node via `lookup[worker][hash]`
2. Remove worker from node's `workers` set
3. If `workers` empty, clear children (cascading cleanup)
4. Remove from `lookup[worker]`

**find_matches(local_hashes, early_exit)**:
1. Start at root with all workers as candidates
2. For each position, traverse to child matching `local_hash`
3. Intersect candidates with node's `workers`
4. Track depth where each worker drops out
5. Return `{worker -> depth}` scores

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| store_blocks (N blocks) | O(N) | O(N) nodes |
| remove_blocks (N blocks) | O(N) | - |
| find_matches (depth D) | O(D × W) | O(W) |

Where W = number of workers.

---

## ConcurrentRadixTree: Thread-Safe Variant

`ConcurrentRadixTree` adapts the `RadixTree` for concurrent access. The key change is replacing `Rc<RefCell<>>` with `Arc<RwLock<>>` per node, and using a `DashMap` for the per-worker lookup table:

```
ConcurrentRadixTree
├── root: SharedBlock (Arc<RwLock<Block>>)
└── lookup: DashMap<Worker, RwLock<HashMap<SeqHash, SharedBlock>>>
```

The `DashMap` distributes lock contention across shards, while each worker's block map is behind its own `RwLock`. This means `find_matches` only takes read locks — on the tree nodes and on the lookup — so multiple reads can proceed in parallel without blocking each other.

Writes (`store_blocks`, `remove_blocks`) take write locks on the affected nodes using hand-over-hand locking (parent before child). To avoid write–write contention, `ConcurrentRadixTree` is designed to be wrapped in a `ThreadPoolIndexer`, which uses per-worker sticky routing: each `WorkerId` is assigned to a dedicated OS thread via a `DashMap<WorkerId, usize>` mapping, and events are dispatched through per-thread `flume` channels. Since KV events for a given worker always land on the same thread, writes to that worker's subtree are serialized without cross-thread locking.

```
                   ┌──────────────────────────────────┐
find_matches() ──→ │   Arc<ConcurrentRadixTree>       │ ← reads go inline
                   │                                  │
 KV events ──→ flume[0] ──→ thread 0 (W0, W3) ──→     │
           ──→ flume[1] ──→ thread 1 (W1, W4) ──→     │ ← writes via sticky
           ──→ flume[2] ──→ thread 2 (W2, W5) ──→     │   worker assignment
                   └──────────────────────────────────┘
```

This same pattern — inline reads on the caller thread, sticky-routed writes through a thread pool — is shared with `PositionalIndexer` (see below). Both implement the `SyncIndexer` trait and are wrapped in `ThreadPoolIndexer`.

One trade-off: `ConcurrentRadixTree` drops the `recent_uses` frequency tracking from `RadixTree`, keeping `find_matches` fully read-only (no mutable state updates on the read path).

---

## PositionalIndexer (NestedMap): Position-First HashMap Index

### Structure

```
PositionalIndexer
├── index: DashMap<(Position, LocalHash), SeqEntry>
├── worker_blocks: DashMap<Worker, RwLock<HashMap<SeqHash, (Position, LocalHash)>>>
└── jump_size: usize

SeqEntry (enum for memory optimization)
├── Single(SeqHash, HashSet<Worker>)  // Common case: one seq_hash
└── Multi(HashMap<SeqHash, HashSet<Worker>>)  // Rare: multiple prefixes
```

`PositionalIndexer` implements `SyncIndexer` and is thread-safe via `DashMap` (sharded
concurrent map) and `RwLock`. It is designed to be wrapped in a `ThreadPoolIndexer` which
routes write events to dedicated OS threads and executes reads inline.

The `index` uses a flat compound key `(position, local_hash)` in a `DashMap`, which
distributes lock contention across shards while enabling O(1) random-position access for
the jump optimization. The `worker_blocks` reverse lookup uses `DashMap` for the outer
per-worker map and `RwLock<HashMap>` for each worker's block set, since writes to a
given worker are serialized by sticky routing in `ThreadPoolIndexer`.

### Visual Representation

```
index (DashMap with compound keys):
┌──────────────────────┬──────────────────────────────────────┐
│ (pos=0, local=0xA)   │ Single(seq=0x1111, {W0,W1})          │
│ (pos=0, local=0xB)   │ Single(seq=0x2222, {W2})             │
│ (pos=1, local=0xC)   │ Single(seq=0x3333, {W0,W1})          │
│ (pos=2, local=0xD)   │ Multi{                               │
│                      │    seq=0x4444 → {W0},                │
│                      │    seq=0x5555 → {W1}   ← diverged    │
│                      │  }                                   │
└──────────────────────┴──────────────────────────────────────┘

worker_blocks (DashMap<Worker, RwLock<HashMap>>):
┌─────────┬─────────────────────────────────────────────────┐
│ W0      │ seq=0x1111 → (pos=0, local=0xA)                 │
│         │ seq=0x3333 → (pos=1, local=0xC)                 │
│         │ seq=0x4444 → (pos=2, local=0xD)                 │
├─────────┼─────────────────────────────────────────────────┤
│ W1      │ seq=0x1111 → (pos=0, local=0xA)                 │
│         │ seq=0x3333 → (pos=1, local=0xC)                 │
│         │ seq=0x5555 → (pos=2, local=0xD)                 │
└─────────┴─────────────────────────────────────────────────┘
```

### How Operations Work

**store_blocks(worker, parent_hash, blocks)**:
1. Find starting position: `pos = worker_blocks[worker][parent_hash].position + 1`
2. For each block at position `i`:
   - Insert into `index[(pos+i, local_hash)]` → add worker to SeqEntry
   - Insert into `worker_blocks[worker][seq_hash] = (pos+i, local_hash)`

**remove_blocks(worker, block_hashes)**:
1. For each hash, lookup `(pos, local_hash) = worker_blocks[worker][hash]`
2. Remove worker from `index[(pos, local_hash)]`
3. Remove from `worker_blocks[worker]`
4. Cleanup empty SeqEntry entries from the DashMap

**find_matches(local_hashes, early_exit)** with Jump Optimization:
1. Start at position 0, initialize candidates from first block
2. **Jump**: Skip ahead by `jump_size` positions (e.g., 32)
3. At each jump point, check if candidates still match (count-only, no clone)
4. If workers dropped, **scan back** to find exact drain points
5. Continue until sequence exhausted or one worker remains

```
Query: [b0, b1, b2, ..., b63, b64, ..., b127, ...]
        ↑                   ↑                  ↑
       pos=0              pos=64             pos=128
        │                   │                  │
        └── jump ──────────→└── jump ─────────→│
                           all match?         some dropped?
                              ↓                   ↓
                           continue          scan [64,128]
```

**Lazy Hash Optimization**:
- Most (position, local_hash) pairs have only ONE seq_hash (SeqEntry::Single)
- Skip seq_hash computation entirely in this case
- Only compute when disambiguation needed (SeqEntry::Multi)

**dump_events()**:
1. Iterate `worker_blocks`, collecting all blocks per worker
2. Sort each worker's blocks by position (parents before children)
3. Emit one single-block `RouterEvent::Stored` per block, synthesizing
   `parent_hash` from any seq_hash at the prior position
4. Events can be replayed into a fresh `PositionalIndexer` to reconstruct
   the same index state

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| store_blocks (N blocks) | O(N) | O(N) entries |
| remove_blocks (N blocks) | O(N) | - |
| find_matches (depth D) | O(D/J) | O(W) |

Where J = jump_size, W = number of workers. The jump optimization reduces D sequential lookups to D/J jumps, with occasional linear scans over skipped positions when workers drop out at a jump point.

---

## Comparison

| Aspect | RadixTree | PositionalIndexer |
|--------|-----------|-------------------|
| **Structure** | Tree with Rc<RefCell<>> nodes | DashMap with compound keys |
| **Concurrent variant** | ConcurrentRadixTree (Arc<RwLock<>> + DashMap) | Thread-safe by default (DashMap + RwLock) |
| **find_matches** | O(D×W) tree traversal | O(D/J) with jump optimization |
| **store_blocks** | O(N) node creation | O(N) DashMap inserts |
| **remove_blocks** | O(N) with cascading cleanup | O(N) with entry cleanup |
| **dump_events** | BFS traversal of tree | Sort by position per worker |
| **Memory** | Higher (Rc/Arc overhead per node) | Lower (flat entries) |
| **Cache locality** | Poor (pointer chasing) | Better (position-first) |

---

## Why Position Matters for PositionalIndexer

The compound key `(position, local_hash)` in the DashMap enables the jump optimization:

```rust
// Without position-first: must traverse entire tree
for pos in 0..depth {
    node = node.children[local_hashes[pos]];  // O(depth) traversals
}

// With position-first: can jump directly to any position
let workers_at_64 = index.get(&(64, local_hashes[64]));  // O(1) lookup
let workers_at_128 = index.get(&(128, local_hashes[128]));  // O(1) lookup
// Skip positions 1-63, 65-127 entirely!
```
