---
name: report-skillpack-issue
description: Reports a defect in the optimization skillpack itself (the skills under .agents/skills/ and the documents under agent-docs/) as a GitHub issue on ai-dynamo/dynamo, using the repository's agent-reported issue conventions. Use when a pack rule contradicts another pack rule, a cross-reference points at a file or section that does not exist, an instruction cannot be executed as written, a factual claim about a tool or flag is wrong, or a rule repeatedly fights the observed environment. Do not use for bugs in Dynamo itself, for engagement-specific problems, or to request new features.
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
SPDX-License-Identifier: Apache-2.0
-->

Issues filed against the skillpack are the maintainers' primary field telemetry. A defect you hit and route around
silently gets hit by every other user's agent too. Filing takes two minutes; the report is more valuable than the
workaround.

This skill follows the repository's agent-reported issue conventions
(`.github/ISSUE_TEMPLATE/agent-reported.yml`): the `[AGENT]: ` title prefix, the `agent-reported` label, a
declared agent identity, and at most one issue per session.

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

## Step 2: Record your agent identity and the exact pack version

The report must state the driver model and the skills commit it was running (for example
`claude-opus-5, skills @ abc1234`):

```bash
git -C <repo-root> rev-parse --short HEAD
```

If the pack was vendored without git history, record the release or image tag it came from. A report that cannot
say which version it observed cannot be triaged.

## Step 3: Sanitize

The issue lands in a public repository. The body must not contain: hostnames, IPs, endpoints, URLs of internal
systems, tokens or credentials of any kind, cluster or namespace names, customer or user identifiers, workload or
traffic specifics, proprietary model or deployment configurations, or pasted logs. Quote pack text freely; describe
your environment only in generic terms (GPU class, backend, harness). When in doubt, leave it out — the rule file,
version, and defect class are usually enough to reproduce.

## Step 4: Check for an existing report

```bash
gh issue list --repo ai-dynamo/dynamo --label agent-reported --search "<rule-or-skill filename> in:title" --state all
```

If a matching issue exists, add a comment confirming the defect at your pack version instead of filing a duplicate.
Comments require the same operator approval as filing (step 6). File at most one new issue per session; if the
session surfaced several defects, put the most impactful one in the issue and list the rest briefly in its body.

## Step 5: Draft the issue

Title: `[AGENT]: <file path relative to repo root>: <one-line defect>`.

Write the body to a scratch file (never inline in a shell command), following the agent-reported template's
structure:

```markdown
### Agent identity

<driver model>, skills @ <commit or tag>

### What the instructions said vs what you verified

Defect class: <contradiction | dead-reference | unexecutable | factual-error | environment-mismatch | missing-coverage>
Location: <file path and the section heading or quoted sentence>

<exact quote(s) of what the pack says; for a contradiction, quote both sides>

<the generic, sanitized observation that contradicts it, how it was verified, and what the agent could not do,
did wrong, or had to route around>

### Suggested correction

<the wording you would have needed; omit the section if unsure>
```

## Step 6: Get operator approval, then file

Filing a public issue is an external side effect. Show the operator the complete drafted title and body and file
only on their approval. Pass the body as a file and the title as a quoted variable so that Markdown containing
backticks or `$()` is never interpreted by the shell:

```bash
gh issue create --repo ai-dynamo/dynamo --title "$ISSUE_TITLE" --body-file <draft file> --label agent-reported
```

If label assignment is rejected for lack of permissions, file without the label; maintainers also triage by the
`[AGENT]: ` title prefix.

## Fallback: no GitHub access or no approval

Write the complete draft to a file the operator will find and tell them where it is: inside an optimization
engagement, append it to `EXP_ROOT/final/known_limitations.md` under a `## Skillpack defects observed` heading;
outside one, write `skillpack-issue-draft-<n>.md` at the root of the current working directory. A
drafted-but-unfiled report is still telemetry; a silent workaround is not.
