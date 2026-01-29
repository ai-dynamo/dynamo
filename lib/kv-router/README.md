# KV Router Index Data Structures

This document explains the two KV cache index implementations: `RadixTree` and `NestedMap`.

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

block2 in A: seq_hash = hash(hash(hash(block0) + block1) + block2) = 0x1111
block2 in B: seq_hash = hash(hash(hash(block0') + block1') + block2) = 0x2222
```

**Computation**: `seq_hash[i] = hash(seq_hash[i-1] || local_hash[i])` where `seq_hash[0] = local_hash[0]`

### 3. Worker ID (`WorkerWithDpRank`)

**What**: Identifies which worker (inference server) has this block cached.

**Why**: The router needs to know which workers can serve a request based on their cached blocks.

### 4. Position (`u64`)

**What**: The block's index in the sequence (0, 1, 2, ...).

**Why**: Enables efficient prefix matching. Position 0 is the first block, position N-1 is the last.

---

## The Core Operations

Both data structures support three operations:

| Operation | Description | Hot Path? |
|-----------|-------------|-----------|
| `store_blocks` | Add blocks for a worker | No (background) |
| `remove_blocks` | Remove blocks for a worker | No (background) |
| `find_matches` | Find workers with matching prefix | **Yes** (per-request) |

The key insight: **reads (find_matches) are far more frequent than writes (store/remove)**. This motivates different structural tradeoffs.

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

## NestedMap: Position-First HashMap Index

### Structure

```
NestedMap
├── index: HashMap<Position, HashMap<LocalHash, SeqEntry>>
├── worker_blocks: HashMap<Worker, HashMap<SeqHash, (Position, LocalHash)>>
└── jump_size: u64

SeqEntry (enum for memory optimization)
├── Single(SeqHash, HashSet<Worker>)  // Common case: one seq_hash
└── Multi(HashMap<SeqHash, HashSet<Worker>>)  // Rare: multiple prefixes
```

### Visual Representation

```
index:
┌─────────┬─────────────────────────────────────────────────┐
│ pos=0   │ local=0xA → Single(seq=0x1111, {W0,W1})        │
│         │ local=0xB → Single(seq=0x2222, {W2})           │
├─────────┼─────────────────────────────────────────────────┤
│ pos=1   │ local=0xC → Single(seq=0x3333, {W0,W1})        │
├─────────┼─────────────────────────────────────────────────┤
│ pos=2   │ local=0xD → Multi{                             │
│         │               seq=0x4444 → {W0},               │
│         │               seq=0x5555 → {W1}   ← diverged   │
│         │             }                                   │
└─────────┴─────────────────────────────────────────────────┘

worker_blocks:
┌─────────┬─────────────────────────────────────────────────┐
│ W0      │ seq=0x1111 → (pos=0, local=0xA)                │
│         │ seq=0x3333 → (pos=1, local=0xC)                │
│         │ seq=0x4444 → (pos=2, local=0xD)                │
├─────────┼─────────────────────────────────────────────────┤
│ W1      │ seq=0x1111 → (pos=0, local=0xA)                │
│         │ seq=0x3333 → (pos=1, local=0xC)                │
│         │ seq=0x5555 → (pos=2, local=0xD)                │
└─────────┴─────────────────────────────────────────────────┘
```

### How Operations Work

**store_blocks(worker, parent_hash, blocks)**:
1. Find starting position: `pos = worker_blocks[worker][parent_hash].position + 1`
2. For each block at position `i`:
   - Insert into `index[pos+i][local_hash]` → add worker to SeqEntry
   - Insert into `worker_blocks[worker][seq_hash] = (pos+i, local_hash)`

**remove_blocks(worker, block_hashes)**:
1. For each hash, lookup `(pos, local_hash) = worker_blocks[worker][hash]`
2. Remove worker from `index[pos][local_hash]`
3. Remove from `worker_blocks[worker]`
4. Cleanup empty nested maps

**find_matches(local_hashes, early_exit)** with Jump Optimization:
1. Start at position 0, initialize candidates from first block
2. **Jump**: Skip ahead by `jump_size` positions (e.g., 32)
3. At each jump point, check if candidates still match
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

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| store_blocks (N blocks) | O(N) | O(N) entries |
| remove_blocks (N blocks) | O(N) | - |
| find_matches (depth D) | O(D/J + J×W) | O(W) |

Where J = jump_size, W = number of workers. The jump optimization reduces D iterations to D/J jumps plus occasional scans.

---

## Comparison

| Aspect | RadixTree | NestedMap |
|--------|-----------|-----------|
| **Structure** | Tree with Rc<RefCell<>> nodes | Nested HashMaps |
| **find_matches** | O(D×W) tree traversal | O(D/J) with jump optimization |
| **store_blocks** | O(N) node creation | O(N) HashMap inserts |
| **remove_blocks** | O(N) with cascading cleanup | O(N) with map cleanup |
| **Memory** | Higher (Rc overhead per node) | Lower (flat entries) |
| **Cache locality** | Poor (pointer chasing) | Better (position-first) |

### Benchmark Results (1M blocks, depth 1024, 128 workers)

| Operation | RadixTree | NestedMap | Winner |
|-----------|-----------|-----------|--------|
| STORE_BLOCK | 90µs | 98µs | RadixTree (1.1x) |
| REMOVE_BLOCK | 91µs | 233µs | RadixTree (2.5x) |
| FIND_MATCHES (HIT) | 227µs | **44µs** | **NestedMap (5.2x)** |
| FIND_MATCHES (PARTIAL) | 216µs | **44µs** | **NestedMap (4.9x)** |

**Recommendation**: Use NestedMap for read-heavy workloads (typical router usage).

---

## Why Position Matters for NestedMap

Position-first nesting enables the jump optimization:

```rust
// Without position-first: must traverse entire tree
for pos in 0..depth {
    node = node.children[local_hashes[pos]];  // O(depth) traversals
}

// With position-first: can jump directly to any position
let workers_at_64 = index[64][local_hashes[64]];  // O(1) lookup
let workers_at_128 = index[128][local_hashes[128]];  // O(1) lookup
// Skip positions 1-63, 65-127 entirely!
```

---

## SeqEntry Optimization

The innermost level uses an enum to avoid HashMap allocation in the common case:

```rust
enum SeqEntry {
    // Common: one prefix leads to this (position, local_hash)
    Single(SeqHash, HashSet<Worker>),

    // Rare: different prefixes converge on same (position, local_hash)
    Multi(HashMap<SeqHash, HashSet<Worker>>),
}
```

**When does Multi occur?**

Only when two different sequences have:
1. Same local block content at position P
2. Different prefix histories (different seq_hash)

Example:
```
Sequence A: [tok1, tok2, tok3] → positions 0,1,2
Sequence B: [tok4, tok5, tok3] → positions 0,1,2
                       ^^^^
                 Same local content at pos=2
                 but different seq_hash!
```

This is rare in practice, so `Single` saves ~48 bytes per entry.
