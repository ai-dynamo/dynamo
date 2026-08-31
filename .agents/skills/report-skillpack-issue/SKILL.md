---
name: report-skillpack-issue
description: Reports a defect in the optimization skillpack itself (the skills under .agents/skills/ and the documents under agent-docs/) as a GitHub issue on ai-dynamo/dynamo. Use when a pack rule contradicts another pack rule, a cross-reference points at a file or section that does not exist, an instruction cannot be executed as written, a factual claim about a tool or flag is wrong, or a rule repeatedly fights the observed environment. Do not use for bugs in Dynamo itself, for engagement-specific problems, or to request new features.
license: Apache-2.0
user-invocable: true
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - skillpack
    - telemetry
    - github
---

# Skill: Report a Skillpack Defect

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Issues filed against the skillpack are the maintainers' primary field telemetry. A defect you hit and route around
silently gets hit by every other user's agent too. Filing takes two minutes; the report is more valuable than the
workaround.

## Step 1: Confirm the defect is in the pack, not the environment

Re-read the suspect rule or skill file in full before reporting. Classify the defect:

- `contradiction` — two pack statements cannot both be followed.
- `dead-reference` — a cross-reference targets a file, section, or field that does not exist.
- `unexecutable` — an instruction cannot be carried out as written (missing input, impossible ordering, undefined
  term with no declared source).
- `factual-error` — a claim about a tool, flag, default, or format is wrong; verify against the tool's own
  source or help output first, and quote that verification in the report.
- `environment-mismatch` — the rule assumes something (paths, permissions, cluster shape) that the target
  environment class does not satisfy; report only when the mismatch looks general, not site-specific.
- `missing-coverage` — the pack gives no guidance for a component or situation it plainly should cover.

If the problem disappears on a careful re-read, or is specific to one site's configuration, do not file.

## Step 2: Record the exact pack version

```bash
git -C <repo-root> rev-parse HEAD
```

If the pack was vendored without git history, record the release or image tag it came from. A report that cannot
say which version it observed cannot be triaged.

## Step 3: Sanitize

The issue lands in a public repository. The body must not contain: hostnames, IPs, endpoints, URLs of internal
systems, tokens or credentials of any kind, cluster or namespace names, customer or user identifiers, proprietary
model or deployment configurations, or pasted logs. Quote pack text freely; describe your environment only in
generic terms (GPU class, backend, harness). When in doubt, leave it out — the rule file, version, and defect class
are usually enough to reproduce.

## Step 4: Check for an existing report

```bash
gh issue list --repo ai-dynamo/dynamo --search "skillpack: <rule-or-skill filename> in:title" --state all
```

If a matching issue exists, add a comment confirming the defect at your pack version instead of filing a duplicate.
Comments require the same operator approval as filing (step 6).

## Step 5: Draft the issue

Title: `skillpack: <file path relative to repo root>: <one-line defect>`.

Body template:

```markdown
## Defect class
<contradiction | dead-reference | unexecutable | factual-error | environment-mismatch | missing-coverage>

## Pack version
<commit SHA, or release/image tag>

## Location
<file path and the section heading or quoted sentence>

## What the pack says
<exact quote(s); for a contradiction, quote both sides>

## What was observed
<the generic, sanitized observation that contradicts it, and how it was verified>

## Effect on the engagement
<what the agent could not do, did wrong, or had to route around>

## Suggested fix (optional)
<one sentence; omit if unsure>

Filed by an agent running the optimization skillpack, with operator approval.
```

## Step 6: Get operator approval, then file

Filing a public issue is an external side effect. Show the operator the complete drafted title and body and file
only on their approval:

```bash
gh issue create --repo ai-dynamo/dynamo --title "<title>" --body "<body>"
```

Do not attempt to set labels; maintainers triage by the `skillpack:` title prefix.

## Fallback: no GitHub access or no approval

Write the complete draft to the engagement's final artifacts (for optimization engagements,
`EXP_ROOT/final/known_limitations.md` under a `## Skillpack defects observed` heading) so the operator can file it
later. A drafted-but-unfiled report is still telemetry; a silent workaround is not.
