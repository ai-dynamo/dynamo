# Design: KVBM Per-Block Multimodal Salt (R3 — Final)

## Goal
Match vLLM's native prefix caching: text-prefix blocks shared, multimodal blocks differentiated.

## Constraints
- **[User]** Must match vLLM native prefix caching semantics
- **[System]** `TokenBlockSequence` uses single `SaltHash` (u64)
- **[System]** Block hashing is chained: `seq_hash[N] = h(seq_hash[N-1], block_hash[N], salt)`
- **[System]** `split_tokens()` uses Rayon for parallel block creation
- **[System]** vLLM's `mm_features` sorted by prompt position

## Hash Formula (Two-Stage, Collision-Safe)

```rust
// Stage 1: hash tokens (identical to current behavior)
let token_hash = compute_hash_v2(cast_slice(&tokens), salt_hash);

// Stage 2: combine with extra_hash in separate domain (if present)
let block_hash = match extra_hash {
    Some(extra) => compute_hash_v2(
        cast_slice(&[token_hash, extra]),
        salt_hash  // tagged combiner, separate from token domain
    ),
    None => token_hash,  // unchanged — backward compat
};

// sequence_hash: base salt only (no extra_hash)
let sequence_hash = match parent_sequence_hash {
    Some(parent) => compute_hash_v2(
        cast_slice(&[parent, block_hash]),
        salt_hash
    ),
    None => block_hash,
};
```

**Why collision-safe**: Stage 1 produces a u64 from token data. Stage 2 combines two u64s (token_hash + extra_hash). These are in completely different domains — no token sequence can collide with a `[u64, u64]` pair under the same hash function.

**Equivalence classes**:
- Text-only: `block_hash = token_hash` → identical to current → **shared** ✓
- Same image: same extra_hash → same block_hash → **shared** ✓
- Different image: different extra_hash → different block_hash → **unique** ✓

## Approach

### Step 1: Extend `TokenBlockChunk` in Rust

**File**: `lib/llm/src/tokens.rs`

```rust
impl TokenBlockChunk {
    fn from_tokens(
        tokens: &[u32],
        salt_hash: SaltHash,
        extra_hash: Option<u64>,  // NEW
    ) -> Self {
        let token_hash = compute_hash_v2(cast_slice(tokens), salt_hash);
        let block_hash = match extra_hash {
            Some(extra) => compute_hash_v2(
                cast_slice(&[token_hash, extra]),
                salt_hash,
            ),
            None => token_hash,
        };
        Self { tokens, salt_hash, block_hash }
    }
}
```

`TokenBlock::from_chunk()` — unchanged (uses base salt for sequence_hash).

### Step 2: Extend `TokenBlockSequence`

```rust
pub struct TokenBlockSequence {
    blocks: Vec<TokenBlock>,
    current_block: PartialTokenBlock,
    salt_hash: SaltHash,
    block_size: usize,
    extra_block_hashes: Vec<Option<u64>>,  // NEW — stored for extend()
}
```

`new()` and `split_tokens()` accept `extra_block_hashes: Vec<Option<u64>>`. In `split_tokens()`, Rayon parallel map passes `extra_block_hashes[i]` to each chunk.

`extend()` for decode-phase: new blocks get `None` extra_hash (text-only).

`PartialTokenBlock`: stores its extra_hash for correct `commit()`.

### Step 3: Thread through Slot and create_slot

**Files**: `slot.rs`, `connector/leader.rs`

```rust
fn create_slot(
    &mut self,
    request: KvbmRequest,
    tokens: Vec<u32>,
    extra_block_hashes: Vec<Option<u64>>,
) -> anyhow::Result<()>
```

**Validation**: when non-empty, `extra_block_hashes.len() == ceil(tokens.len() / block_size)`.

### Step 4: Centralized Python helper

**New file**: `lib/bindings/kvbm/python/kvbm/vllm_integration/mm_block_hashes.py`

```python
import json
import hashlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.v1.request import Request


def compute_mm_block_hashes(
    request: "Request",
    block_size: int,
) -> list[Optional[int]]:
    mm_features: Optional[list] = getattr(request, "mm_features", None)
    if not mm_features:
        return []

    num_tokens: int = len(request.all_token_ids)
    num_blocks: int = (num_tokens + block_size - 1) // block_size
    result: list[Optional[int]] = [None] * num_blocks

    for block_idx in range(num_blocks):
        start: int = block_idx * block_size
        end: int = min(start + block_size, num_tokens)

        mm_ids: list[str] = []
        for f in mm_features:
            mm_start: int = f.mm_position.offset
            mm_end: int = mm_start + f.mm_position.length
            if start < mm_end and end > mm_start:
                identifier: Optional[str] = getattr(f, "identifier", None)
                if not identifier:
                    raise ValueError(
                        "KVBM: mm_features without stable identifier"
                    )
                mm_ids.append(identifier)

        if mm_ids:
            canonical: str = json.dumps(mm_ids, separators=(",", ":"))
            digest: bytes = hashlib.sha256(canonical.encode()).digest()[:8]
            result[block_idx] = int.from_bytes(digest, "little")

    return result
```

### Step 5: Update `_create_slot` in both Python files

Import from shared helper. Remove mm rejection guard. Pass extra_block_hashes.

## Block-Level Example

```
Tokens: [sys×32, "What is", img_pad×48, "?"]
block_size = 16

Block 0: [sys×16]        extra=None   block_hash=h(tokens,salt)          SHARED ✓
Block 1: [sys×16]        extra=None   block_hash=h(tokens,salt)          SHARED ✓
Block 2: ["What",img×10] extra=0xF1   block_hash=h(h(tokens,salt),0xF1)  UNIQUE ✓
Block 3: [img×16]        extra=0xF1   block_hash=h(h(tokens,salt),0xF1)  UNIQUE ✓
Block 4: [img×16]        extra=0xF1   block_hash=h(h(tokens,salt),0xF1)  UNIQUE ✓
Block 5: [img×6,"?"]     extra=0xF1   block_hash=h(h(tokens,salt),0xF1)  UNIQUE ✓

Chained seq_hash: S0→S1 shared prefix, S2 diverges.
```

## Files to Change

| File | Change |
|------|--------|
| `lib/llm/src/tokens.rs` | Two-stage hash in `TokenBlockChunk`. `TokenBlockSequence` stores extra vec. |
| `lib/bindings/kvbm/src/block_manager/vllm/slot.rs` | Thread extra_block_hashes |
| `lib/bindings/kvbm/src/block_manager/vllm/connector/leader.rs` | Accept + validate extra_block_hashes |
| `lib/bindings/kvbm/python/kvbm/vllm_integration/mm_block_hashes.py` | **New** shared helper |
| `lib/bindings/kvbm/python/kvbm/_core.pyi` | Update type stub |
| `lib/bindings/kvbm/python/kvbm/vllm_integration/connector_leader.py` | Use helper |
| `lib/bindings/kvbm/python/kvbm/vllm_integration/kv_cache_manager.py` | Same |
