# Request Sequence

Low-level request primitive with direct RAII block lifecycle management.

[`RequestSequence`] composes [`BlockSequence`](crate::BlockSequence),
[`LogicalBlockAssignments`](crate::LogicalBlockAssignments), and
[`BlockManager`](crate::BlockManager) into a single type that exposes
individual block lifecycle operations without opinionation about
scheduling policy.

For a structured two-phase schedule/apply layer built on top of this,
see [`SchedulableSequence`](crate::SchedulableSequence).

## When to use

Use `RequestSequence` directly when you need full control over block
allocation, staging, and registration timing. Use `SchedulableSequence`
when you want state-machine enforcement of the prefill/decode protocol.

## Block lifecycle

Blocks flow through three phases:

1. **Unassigned** -- freshly allocated `MutableBlock`s waiting to be paired
   with token data
2. **Staged** -- `CompleteBlock`s paired with token data but not yet
   committed to the registry
3. **Assigned** -- `ImmutableBlock`s registered in the block manager,
   visible for prefix matching

## Basic usage

```ignore
use kvbm_logical::{RequestSequence, BlockManager};

// 1. Construct with tokens only (no manager interaction)
let tokens: Vec<u32> = (0..8).collect();
let mut seq = RequestSequence::<MyMeta>::new(tokens, 10, 4);
// total_tokens=8, num_blocks=2, nothing allocated yet

// 2. Prefix match against the cache
let matched = seq.match_prefix(&manager);
let matched_count = matched.len();
if !matched.is_empty() {
    seq.add_matched_blocks(matched).unwrap();
}

// 3. Allocate blocks for the rest
let remaining = seq.num_blocks() - matched_count;
seq.allocate_blocks(remaining, &manager);

// 4. Stage and register
seq.complete_and_register_pending(&manager);
// Now: assigned_blocks() == num_blocks()
```

## Generation loop

After initial setup, generate tokens one at a time. Each `append_token`
call returns `Some(block_index)` when a block boundary is crossed,
signaling that `complete_and_register_pending` should be called and a
new generation block allocated.

```ignore
while !seq.is_complete() {
    let token = model.forward(&seq);
    let crossed = seq.append_token(token);
    if crossed.is_some() {
        seq.complete_and_register_pending(&manager);
        seq.allocate_blocks(1, &manager);
    }
}
```

## Preemption and reacquire

Release all blocks (RAII returns them to pools), then re-acquire later.
Prefix-matched blocks may come from cache, saving re-computation.

```ignore
// Preempt
seq.release();
assert_eq!(seq.assigned_blocks(), 0);

// Later: reacquire
let success = seq.reacquire(&manager);
// Prefix cache hits are reflected in prefix_matched_blocks()
```

## Key accessors

| Method                    | Description                                    |
|---------------------------|------------------------------------------------|
| `total_tokens()`          | Input + generated token count                  |
| `num_input_tokens()`      | Original input token count                     |
| `generated_tokens()`      | Tokens appended via `append_token`             |
| `num_blocks()`            | Complete token blocks in the sequence          |
| `assigned_blocks()`       | Registered/cache-matched blocks                |
| `staged_blocks()`         | Completed but not yet registered               |
| `unassigned_blocks()`     | Allocated but not yet paired with token data   |
| `prefix_matched_blocks()` | Blocks matched from cache                      |
| `is_complete()`           | `generated_tokens >= max_output_tokens`         |
| `new_tokens_for_prefill()`| Tokens not covered by cache hits               |
