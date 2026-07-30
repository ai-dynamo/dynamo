<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code-review experts

Each file here defines **one expert reviewer** for the [`code-review`](../code-review.js)
workflow. Drop a new file to add an expert — the workflow's loader picks it up
and fans out a reviewer for it. No change to the orchestration is needed.

The loader **merges** these files with the workflow's built-in experts by `key`:
a file whose `key` matches a built-in (`correctness`, `concurrency`,
`distributed`, `simplicity`, `perf`, `comment-hygiene`, `tests-and-api`)
**overrides** that built-in's focus; a new `key` **adds** an expert.

## File format

```markdown
---
key: security            # required, kebab-case, unique. Same key as a built-in overrides it.
name: Security           # optional, human label
applies_to: "**/*.rs, **/*.py, **/Dockerfile"   # optional globs; expert runs only when the diff touches a match
---
The review focus: the concrete concerns this expert looks for, phrased as a
checklist an agent can apply to a diff. This body becomes the reviewer's lens.
```

- **`key`** is the only required field.
- **`applies_to`** is a relevance pre-filter. Omit it to let the loader decide
  relevance from the change itself (it already skips experts with no surface —
  e.g. `perf` on a docs-only diff).
- The body should describe *material* concerns; the workflow already enforces
  the shared "report only material, changed-line issues" discipline.

## Findings contract

Every reviewer returns findings shaped as
`{ file, line, severity: blocking|major|minor|nit, claim, why, suggestion? }`.
Each finding is then adversarially verified before it reaches the report, so an
expert should surface candidates freely and let verification filter them.
