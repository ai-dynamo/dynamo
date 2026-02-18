# Schedulable Sequence

Two-phase schedule/apply layer for LLM inference on top of
[`RequestSequence`](crate::RequestSequence).

[`SchedulableSequence`] enforces a state-machine protocol for prefill,
decode, and speculative decode operations, tracks KV position, and
maintains an append-only event history for observability.

## State machine

```text
         schedule_prefill        apply_prefill
 Idle ──────────────────► PrefillScheduled ─────► Idle
  │                                                 │
  │    schedule_decode          apply_decode         │
  ├──────────────────► DecodeScheduled ─────────► Idle
  │                                                 │
  │   schedule_speculative    apply_speculative      │
  └──────────────────► SpeculativeScheduled ────► Idle
                  │
                  │   revert_schedule
                  └──────────────────────────► Idle
```

Every `schedule_*` call validates preconditions and pre-allocates blocks.
Every `apply_*` call commits the operation (appends tokens, registers
blocks). `revert_schedule` undoes a schedule without applying,
LIFO-releasing pre-allocated blocks.

## Dangling token tracking

`SchedulableSequence` tracks which tokens have had their KV computed via
`kv_position`. The difference `total_tokens - kv_position` gives the
**tail token count** -- tokens whose KV hasn't been computed yet.

After prefill with a generated token, `tail_tokens() == 1` (the first
generated token is "dangling"). After each decode or speculative step,
the count remains 1 (the newest token replaces the old dangling one).

`schedule_decode` and `schedule_speculative` enforce exactly 1 tail
token as a precondition.

## Typical lifecycle

```ignore
use kvbm_logical::SchedulableSequence;

let tokens: Vec<u32> = (0..8).collect();
let mut seq = SchedulableSequence::<MyMeta>::new(tokens, 10, 4);

// 1. Optional prefix matching
let matched = seq.match_and_add_prefix(&manager)?;

// 2. Prefill (single chunk)
seq.schedule_prefill(8 - matched * 4, &manager)?;
seq.apply_prefill(Some(first_generated_token), &manager)?;
// kv_position = 8, tail_tokens = 1

// 3. Decode loop
while !seq.is_complete() {
    seq.schedule_decode(&manager)?;
    let token = model.forward(&seq);
    let outcome = seq.apply_decode(token, &manager)?;
    // outcome: Continue | BlockCompleted | MaxLength | BlockCompletedAndMaxLength
}

// 4. Release
seq.release()?;
```

## Chunked prefill

Split prefill across multiple chunks. Only the **final** chunk (the one
that reaches `num_input_tokens`) must provide a generated token.

```ignore
// Chunk 1 (non-final): no token
seq.schedule_prefill(4, &manager)?;
seq.apply_prefill(None, &manager)?;

// Chunk 2 (final): must provide first generated token
seq.schedule_prefill(4, &manager)?;
seq.apply_prefill(Some(first_token), &manager)?;
```

## Speculative decode

Schedule a batch of draft tokens, then accept a prefix of them.
Excess pre-allocated blocks are automatically released.

```ignore
seq.schedule_speculative(5, &manager)?;
// Model verifies draft tokens, accepts first 3
let outcome = seq.apply_speculative(&[tok1, tok2, tok3], &manager)?;
// Excess blocks LIFO-dropped, tail_tokens still 1
```

## Preemption

Release and later reacquire blocks. Prefix cache hits reduce
re-computation cost.

```ignore
seq.release()?;
// ... later ...
let success = seq.reacquire(&manager)?;
// Reacquire does not allocate a generation block;
// the next schedule_decode handles that.
seq.schedule_decode(&manager)?;
```

## Error handling

| Error                     | When                                              |
|---------------------------|---------------------------------------------------|
| `ScheduleError::NotIdle`  | `schedule_*` called while already scheduled        |
| `ScheduleError::PrefillNotComplete` | Decode/speculative before prefill done  |
| `ScheduleError::PrefillComplete` | `schedule_prefill` after all input processed |
| `ScheduleError::PrefillOverrun` | Chunk would exceed input token count        |
| `ScheduleError::AllocationFailed` | Not enough blocks in the manager          |
| `ScheduleError::GenerationComplete` | Already hit `max_output_tokens`         |
| `ScheduleError::WrongDanglingCount` | Tail tokens != 1 for decode/speculative |
| `ApplyError::WrongState`  | `apply_*` called in wrong state                   |
| `ApplyError::TokenOnNonFinalChunk` | Token provided on non-final prefill chunk |
| `ApplyError::MissingTokenOnFinalChunk` | Final prefill chunk missing token    |
| `ApplyError::AcceptedExceedsScheduled` | More accepted than draft tokens     |

## Event history

Every lifecycle transition is recorded in an append-only
`Vec<SequenceEvent>`, accessible via `history()`. Events include
`Created`, `PrefillScheduled`, `PrefillApplied`, `DecodeScheduled`,
`DecodeApplied`, `SpeculativeScheduled`, `SpeculativeApplied`,
`ScheduleReverted`, `UnassignedDropped`, `Released`, and `Reacquired`.
