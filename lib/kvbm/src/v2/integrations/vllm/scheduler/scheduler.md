SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
SPDX-License-Identifier: Apache-2.0

## Rust Scheduler (v1) – Working Design

This document describes a Rust-native scheduler that mirrors vLLM v1 concepts but adjusts the control flow to better suit Dynamo’s architecture and concurrency model.

### Goals

* Extract constants from vLLM configs and keep them in `SchedulerConstants`.
* Make Rust-native data models for requests, outputs, and intermediate scheduler products.
* Separate Python boundary conversions from hot-path Rust logic.
* Parallelize `update_from_output` handling using Rust-owned `OwnedModelRunnerOutput`.
* Introduce a staging thread that prepares “ready-to-execute” requests ahead of scheduling steps.

### Key Differences from vLLM Python Scheduler

* We only process requests for the current forward pass and a `ready_to_execute` queue.
* Expensive block matches (both device prefix-cache and external connector) happen in a separate staging thread.
* The Python boundary hands off `ModelRunnerOutput` as a Rust-owned structure; we then fan out per request in parallel (future: rayon or task pools) to amortize Python <-> Rust overhead.

### Data Models (Rust)

* `RequestState`: tracks per-request state used by the scheduler. Key fields:
  * token sequences: `prompt_token_ids`, `output_token_ids`, `all_token_ids`
  * counters: `num_computed_tokens`, `num_cached_tokens`, `num_output_placeholders`
  * speculative: `spec_token_ids`
  * mm inputs: `mm_positions`, `has_encoder_inputs`
  * LoRA, structured output, `client_index`, `priority`
* `SchedulerOutput`: mirrors Python’s `SchedulerOutput` but in Rust types, split into `new_requests` and `cached_requests`, plus per-request token schedules and batch signals (finished ids, prefix blocks, etc.).
* `OwnedModelRunnerOutput`: Rust-owned copy of the Python `ModelRunnerOutput`, holding Vecs/Maps that are GIL-less.

#### vLLM docstrings summary (relevant parts)

- `Scheduler.schedule()` (Python): produces per-step scheduling for a single forward pass; respects `max_num_batched_tokens`, `max_num_seqs`, `max_model_len`; handles preemption, prefix caching, speculative decoding, structured output, encoder inputs; may include finished request IDs and encoder free lists.
- `Scheduler.update_from_output()`: consumes generated token IDs, adjusts for spec decoding rejections, handles stop conditions, frees requests (including encoder/KV cache), and returns grouped `EngineCoreOutputs` per client with optional stats.
- `Request`: maintains `num_tokens_with_spec = prompt + output + speculative`, `num_computed_tokens` advanced post-schedule, structured-output FSM hooks, `cache_salt` for hashing, and status transitions.
- `SchedulerOutput`: two lists (`scheduled_new_reqs`, `scheduled_cached_reqs`) plus maps of scheduled tokens, spec tokens, encoder inputs, grammar bitmask, finished IDs, and connector metadata.

### Interfaces

* `Scheduler` trait: `schedule`, `update_from_output`, `add_request`, `finish_requests`, `get_request_counts`, `has_finished_requests`, `reset_prefix_cache`, `shutdown`, `get_kv_connector`.
* `KvCacheManager` trait: abstract device KV operations (allocate slots, free, cache blocks, prefix computations).
* `KvConnector` trait: optional p/d disaggregation hooks, including matched-token queries and post-alloc updates.
* `StructuredOutputManager` trait: grammar bitmask and FSM advancement hook.

### Control Flow

1. Staging Thread (separate):
   - On new requests or when recovering from preemption, perform:
     - local prefix cache matches (`KvCacheManager.get_computed_blocks`)
     - external matches (`KvConnector.get_num_new_matched_tokens`)
   - Update prepared state (e.g., `num_computed_tokens`) and enqueue `ready_to_execute` item.

2. `schedule()`:
   - Consume token budget for currently `running` requests first.
   - Then pop from `ready_to_execute` to schedule “new” prepared requests.
   - Allocate blocks via `KvCacheManager.allocate_slots` with spec lookahead when enabled.
   - Build `SchedulerOutput`, set `num_common_prefix_blocks`, optionally add connector metadata.
   - Advance `num_computed_tokens` after building the output.

3. `update_from_output()`:
   - For each scheduled request, merge generated token ids, adjust for speculative rejections, check stop, and emit `EngineCoreOutputs` grouped by `client_index`.
   - Update connector finished-sending/receiving state and free blocks as needed.
   - Attach `finished_requests` sets if configured.

### Parallelizing ModelRunnerOutput

* Python side converts its `ModelRunnerOutput` into a Rust-owned `OwnedModelRunnerOutput` (no torch tensors crossing the boundary for the hot path).
* In Rust, process each request’s outputs in parallel (future: `rayon` or work-stealing executor). The current sketch processes sequentially but isolates the loop so parallelization is straightforward.
* Alternative: Python-side channel enqueue per-request items; Rust side drains and processes concurrently.

### Open TODOs

* Implement stop conditions (EOS, max length, guided decoding).
* Integrate real logprobs and pooling tensors as zero-copy or pinned views.
* Add encoder-cache budget management and free-on-advance logic.
* Implement preemption policy under allocation pressure.
* Wire structured output manager and grammar bitmask generation.
* Provide concrete KvCacheManager and KvConnector adapters to existing components.
* Add feature-gated rayon parallelism for `update_from_output`.

### Testing Plan
### Worst-Case Projections and Offload Strategy

We introduce a projection subsystem to anticipate KV block pressure and plan proactive offload to host memory.

- Inputs:
  - `block_size`, `gpu_total_blocks` from cache config.
  - Per-request `current_tokens`, `max_tokens`, `num_computed_tokens`.
- Definitions:
  - `wc_until_first_complete`: in worst-case (1 token/request/step), minimum steps until any request completes.
  - `wc_until_block_starvation`: the first future step where total required blocks would exceed `gpu_total_blocks`, assuming no early termination.
  - `predicted_blocks_per_pass`: total blocks trajectory across the next K steps.
- Heuristics:
  - Model decoding with `tokens_per_pass_per_request=1`; increase to approximate chunked prefills.
  - Use ceil division `ceil(seq_len / block_size)` to estimate blocks.
  - If a starvation is predicted at pass `k`, we target freeing enough blocks before `k` by offloading one or more requests to host.
  - Selection policy: greedy on fewest blocks first to minimize movement and ensure complete offload(s). Prefer pausing on block boundaries.
- Data model (Rust):
  - `ProjectionParams`, `RequestProjection`, `WorstCaseProjection`, `OffloadCandidate`, `OffloadPlan`.
- Lifecycle:
  - Compute projections at the start of `schedule()` or periodically.
  - If `wc_until_block_starvation` is Some and less than a small threshold, build an `OffloadPlan`, pause and offload selected requests, and re-run scheduling.

This allows maintaining steady throughput by avoiding mid-iteration allocation failures and by freeing GPU KV blocks predictably.


* Unit tests for:
  * token budgeting and `num_computed_tokens` progression
  * staging updates and ready queue semantics
  * connector callbacks on alloc/finish
* Integration tests against a thin Python harness using vLLM fixtures.


