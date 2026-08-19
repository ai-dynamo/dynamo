<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Workflows

Multi-agent orchestration scripts for Claude Code's workflow runner. A workflow
deterministically fans work out across many subagents — to be comprehensive, to
be confident (independent perspectives + adversarial verification before
committing), or to take on scale one context can't hold.

Workflows live **canonically here** (`.agents/workflows/`); `.claude/workflows/`
is a symlink to this directory, so the runner discovers them while the source of
truth stays under `.agents/`. **Edit only the canonical copy.**

Each workflow is a self-contained JavaScript file that begins with
`export const meta = { name, description, phases }` (a pure literal) followed by
the script body using `agent()` / `pipeline()` / `parallel()` / `phase()`.

| Workflow | What it does |
|----------|--------------|
| [`code-review`](code-review.js) | Runs an adversarial framing pass, then fans out **pluggable expert reviewers** (built-ins: correctness, concurrency, distributed, simplicity, perf, comment-hygiene, tests/API — add more in [`reviewers/`](reviewers/)), reviews the diff, adversarially verifies every finding, and returns a ranked report. Optional args: `{ base, reviewers, votes, exhaustive }` (default base `origin/main`). |

## Adding a workflow

1. Add `<name>.js` here with a pure-literal `meta` block; use the same phase
   titles in `meta.phases` as in the `phase()` calls.
2. List it in the table above.
3. Scripts run in a sandboxed JS context: no filesystem or shell access, and
   `Date.now()` / `Math.random()` are unavailable. Do repo inspection (git,
   file reads) inside `agent()` subagents, which have tool access.
