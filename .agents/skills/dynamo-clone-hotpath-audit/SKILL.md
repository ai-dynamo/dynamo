---
name: dynamo-clone-hotpath-audit
description: Audit Dynamo Rust hot-path `.clone()` calls, explain which clones are removable and why, and only apply clone-removal patches when explicitly requested.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - rust
    - performance
    - code-review
    - allocation
  permissions:
    - file_read
    - file_write
---

# Dynamo Clone Hotpath Audit

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Purpose

Find `.clone()` calls in Dynamo Rust request, scheduling, KV, block-manager, and
runtime hot paths that can be removed without changing ownership semantics. This
is an audit-first workflow, not a blanket clone-removal tool.

Default behavior is read-only audit. Do not edit files, commit, or open a PR
unless the user explicitly asks to fix, patch, apply, implement, or create an
MR/PR. A skill invocation such as `$dynamo-clone-hotpath-audit`, "audit",
"check", or "scan" is not patch permission.

## Prerequisites

- Rust source checkout of `ai-dynamo/dynamo`.
- Python 3.10+ for the inventory script.
- Ability to run targeted Rust validation commands for touched crates.
- Subagent tooling if available. If subagents are unavailable, run the same
  roles as separate, explicit review passes and say they were not independent.

## Instructions

### 1. Build The Inventory

Run the read-only scanner from the repository root:

```bash
python3 .agents/skills/dynamo-clone-hotpath-audit/scripts/clone_inventory.py \
  --only-actionable \
  --limit 80
```

Use narrower paths when the user names a subsystem:

```bash
python3 .agents/skills/dynamo-clone-hotpath-audit/scripts/clone_inventory.py \
  --paths lib/kv-router/src/scheduling lib/llm/src/backend \
  --only-actionable
```

Treat the scanner as a triage aid. It ranks likely expensive clones but does not
prove removability.

### 2. Split The Audit By Hot Path

Prioritize in this order:

1. per-request LLM paths: `lib/llm/src/backend.rs`, preprocessor, HTTP/gRPC
   services, protocol conversion, migration
2. KV routing and scheduling: `lib/kv-router/src/scheduling`,
   `lib/kv-router/src/sequences`, `lib/kv-router/src/indexer`
3. KV/block manager event paths: `lib/llm/src/block_manager`,
   `lib/bindings/kvbm/src/block_manager`, `lib/kvbm-engine/src/offload`
4. runtime request/transport paths: `lib/runtime/src/component`,
   `lib/runtime/src/pipeline`, `lib/runtime/src/transports`
5. tests, examples, debug paths, and benchmark setup only after hot paths

### 3. Use Independent Review Roles

For non-trivial audits, use subagents. Give each subagent only the inventory
slice and relevant source files, not your intended answer.

Required roles:

- candidate finder: identify high-value clones and classify cheap/required ones
- ownership refactorer: propose concrete borrow, move, `Arc`, or `mem::take`
  changes
- correctness adversary: reject changes that alter sharing, extend lock
  lifetimes, borrow across `.await`, break spawned task ownership, or make APIs
  less clear
- validation planner: choose the smallest tests, clippy commands, or benchmarks
  that prove the touched behavior

If two roles disagree, keep the clone unless you can write down a precise
ownership proof and validation plan.

### 4. Classify Every Candidate

Use these buckets:

- cheap required: `Arc`, sender, cancellation token, runtime handle, watch
  receiver, metrics handle, or tracing span clone needed to share ownership
- semantic required: the original value must stay available for a later use,
  retry, fan-out, or async task
- cold/test-only: outside production hot paths
- removable move: value is cloned immediately before its final ownership use
- removable borrow: callee does not need ownership and can accept `&T`, `&str`,
  slice, or iterator
- structural refactor: requires changing data layout, e.g. `Vec<T>` to
  `Arc<[T]>`, or changing an API family

Do not patch candidates in the first three buckets.

### 5. Patch In Small Batches

Only run this section when the user explicitly asks for fixes or a patch. If the
user only asked for an audit, stop after the report and list recommended patch
batches as follow-up work.

Keep each patch batch narrow: one subsystem or one repeated pattern. Avoid a
single repository-wide clone cleanup PR unless the user explicitly asks for it.

Preferred fixes:

- move a value when the clone is immediately consumed and not used afterward
- pass `&T`, `&str`, `&[T]`, or an iterator when the callee only reads
- consume batches with `into_iter()` instead of indexing and cloning
- extract small `Copy` fields before moving a large event
- use `std::mem::take` only when leaving the source value empty is part of the
  intended semantics
- use `Arc` only when shared ownership is semantically right, not just to avoid
  borrow-checker work

Do not:

- remove clones that cross `tokio::spawn`, channel send, callback storage, or
  task lifetime boundaries without proving ownership
- extend mutex/RwLock guard lifetimes to avoid a clone
- borrow across `.await` unless the borrow is local and compiler-verified
- trade a clear cheap clone for a confusing lifetime-heavy API
- make public APIs borrow data whose ownership was intentionally independent

### 6. Required Finding Format

For every proposed change, report:

- `file:line`
- hot-path rationale
- cloned value and likely clone cost
- bucket and recommended change
- removal rationale: why this clone is unnecessary, not only why it is expensive
- required-clone check: why it is not cheap required or semantic required
- correctness proof: why the old and new ownership semantics match
- validation command

Example:

```text
lib/llm/src/preprocessor.rs:123
Hot path: every embeddings request.
Cost: clones Vec<String> before moving into spawn_blocking.
Bucket: removable move.
Change: move input_strs into the closure.
Removal rationale: the clone feeds the only owned consumer and the original is
not read after closure construction.
Required-clone check: no fan-out, retry, async task sharing, or later logging
uses the original value.
Proof: the closure receives the same owned Vec<String>; all later behavior reads
from that moved value.
Validation: cargo test -p dynamo-llm preprocessor
```

### 7. Validate Before Finishing

If patches were made, always run formatting and the narrowest relevant Rust
tests. For broad API changes, also run clippy or crate-level tests for each
touched crate. If no patches were made, do not run tests just to make the audit
look validated; report the inventory command and any read-only review checks.

Return:

- clone candidates reviewed
- candidates intentionally left alone and why
- patches made, or `none: audit-only`
- tests run and results, or `not run: no files changed`
- residual high-value candidates that should be separate PRs

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/clone_inventory.py` | Rank Rust `.clone()` call sites by hot-path likelihood and removal potential | `--paths`, `--limit`, `--format`, `--only-actionable`, `--include-tests` |

Invoke via the agentskills.io `run_script()` protocol:

```python
run_script("scripts/clone_inventory.py", args=["--only-actionable", "--limit", "80"])
```

## Output Contract

Return a concise audit report or PR summary with:

- scope and inventory command
- top findings with buckets
- per finding: removal rationale, required-clone check, correctness proof, and
  validation command
- exact refactors made, or `none: audit-only`
- independent review role outcomes
- validation commands and status
- follow-up candidates excluded from the current audit or patch

## Limitations

- The scanner is heuristic and line-based. It can miss macro-expanded clones,
  multi-line ownership patterns, or clones hidden behind helper methods.
- Some expensive-looking clones are required for async task ownership, fan-out,
  retries, or API clarity.
- Performance impact is inferred unless backed by benchmarks or allocation
  profiles.
