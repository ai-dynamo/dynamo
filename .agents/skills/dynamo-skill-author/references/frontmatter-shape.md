<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontmatter Shape

Strict spec for the YAML frontmatter that opens every `SKILL.md` under
`.agents/skills/dynamo-<name>/`. Validated mechanically by
`scripts/check-frontmatter.py`.

## Required Fields

| Field | Type | Constraints |
|---|---|---|
| `name` | string | Lowercase, hyphen-separated. Matches the directory name exactly. Must begin with `dynamo-`. |
| `description` | string (YAML folded scalar `>-`) | 50-500 chars, opens with a verb, lists trigger phrases inline, no angle-bracket placeholders. |
| `version` | string | Matches the Dynamo release line targeted by this skill (e.g. `1.2.0`). Not arbitrary semver. |
| `author` | string | `NVIDIA` (institutional voice; no personal attribution). |
| `tags` | list[string] | Must include `dynamo` plus the lifecycle stage. 3-6 tags total. |
| `tools` | list[string] | Minimal set the skill body invokes; typically `Shell`, `Read`, `Write`. |

## Canonical Shape

```yaml
---
name: dynamo-<your-name>
description: >-
  One-paragraph description. Opens with a verb. Lists trigger phrases
  inline so the agent's loader matches the user's actual prompt. 50-500
  chars. No angle-bracket placeholders — they get parsed as XML by
  NV-ACES and fail evaluation. Use lowercase placeholder words instead:
  backend, model, namespace.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - <lifecycle-stage>
  - <optional-tag-1>
  - <optional-tag-2>
tools:
  - Shell
  - Read
  - Write
---
```

## Field-by-Field

### `name`

- Lowercase, hyphen-separated, must match the directory name exactly. The
  validator compares `name` against the parent directory name.
- Must begin with `dynamo-` to keep the naming namespace clean. The seven
  existing skills follow this rule; the validator enforces it.
- Verb-first for Dynamo-owned components (`dynamo-plan`, `dynamo-deploy`).
  Product-name first is reserved for external NVIDIA brands with
  independent identity (would be e.g. `dynamo-aiperf` if AIPerf were its
  own skill rather than a section in `dynamo-benchmark`).

### `description`

- YAML folded scalar (`>-` opener), one paragraph, 50-500 characters.
- Opens with a verb in the imperative ("Author …", "Deploy …", "Plan …").
- Lists trigger phrases inline. The agent's loader matches on the
  description text, so the phrases the user is likely to say must appear
  verbatim. Use the form: `Trigger phrases include "...", "...", "..."`.
- **No angle-bracket placeholders.** `<backend>`, `<model>`, `<name>`
  inside the folded scalar get parsed as XML by NV-ACES and fail with an
  error. Use lowercase placeholder words instead: `backend`, `model`,
  `namespace`. The validator scans for `<[a-z][^>]*>` and rejects matches.

### `version`

- Matches the Dynamo release line this skill targets. Format `MAJOR.MINOR.PATCH`.
- Bumped during the per-release refresh procedure (see the directory
  README §Per-Release Refresh).
- Not arbitrary semver. The validator checks the format; the human reviewer
  checks the alignment with the active release line.

### `author`

- `NVIDIA`. Institutional voice; no personal attribution in the
  frontmatter. Individual contributors are credited via `git log` and the
  DCO sign-off.

### `tags`

- Must include `dynamo`.
- Must include the lifecycle stage as a tag (one of: `plan`, `optimize`,
  `serve`, `deploy`, `frontend`, `troubleshoot`, `benchmark`, or the new
  stage you are introducing).
- 3-6 tags total. More than 6 dilutes the loader's matching.
- Acceptable additional tags: `kubernetes`, `inference`, `meta`,
  `authoring`, `gateway`, `kv-cache`, `multimodal`, `quantization`,
  `nixl`, `aiperf`, `modelopt`, `aiconfigurator`.

### `tools`

- Minimum set the skill body invokes. The cross-client loader uses this
  list to pre-check tool availability.
- Typical set: `Shell`, `Read`, `Write`. Add more only if the body
  explicitly invokes them.

## Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| NV-ACES Tier 1 reports XML parse error on `description`. | Angle-bracket placeholder inside the folded scalar. | Replace `<backend>` with `backend`, `<model>` with `model`, etc. |
| `check-frontmatter.py` reports name mismatch. | `name:` value does not match the parent directory. | Rename one or the other so they match exactly. |
| `check-frontmatter.py` reports missing tag. | `tags:` list does not include `dynamo` or the lifecycle stage. | Add both. The validator enforces both. |
| `check-frontmatter.py` reports `version` malformed. | Used a calendar version, a git SHA, or something not matching `\d+\.\d+\.\d+`. | Use `MAJOR.MINOR.PATCH` matching the Dynamo release line. |
| `check-frontmatter.py` reports `description` too short or too long. | Outside the 50-500 char range. | Tighten or expand; aim for ~300-450 chars in practice. |
