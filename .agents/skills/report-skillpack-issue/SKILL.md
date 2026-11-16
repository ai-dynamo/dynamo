---
name: report-skillpack-issue
description: Reports a defect in the optimization skillpack itself (the skills under .agents/skills/, the documents under agent-docs/, the role contracts under agents/, and the repository AGENTS.md files) as a GitHub issue on ai-dynamo/dynamo, using the repository's agent-reported issue conventions. Use when a pack rule contradicts another pack rule, a cross-reference points at a file or section that does not exist, an instruction cannot be executed as written, a factual claim about a tool or flag is wrong, or a rule repeatedly fights the observed environment. Do not use for bugs in Dynamo itself, for engagement-specific problems, or to request new features.
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
git rev-parse --short HEAD
```

Run that from the repository root. In every command in this skill, replace placeholders BEFORE running
and never leave angle brackets in a shell line: `<...>` parses as redirection and can truncate files.

If the pack was vendored without git history, record the release or image tag it came from. A report that cannot
say which version it observed cannot be triaged.

## Step 3: Sanitize

The issue lands in a public repository. Nothing you emit, in the title, the body, a comment, a search term,
or a version string, may contain: Quote pack text from the public repository freely (not from a locally modified or vendored copy, which may
carry private details); describe your environment only in generic terms (GPU class, backend, harness). When in doubt, leave it out — the rule file,
version, and defect class are usually enough to reproduce.

## Step 4: Check for an existing report

Search titles AND bodies in one bounded, body-inclusive query. A labeled query adds nothing (`--label`
ANDs into the search, so it is a strict subset of the unlabeled one), the default limit of 30 hides real
matches for common keys such as `README.md`, and `--json` returns bodies inline instead of costing one
`gh issue view` per hit. Use the bare filename as the search term (no backticks):

```bash
gh issue list --repo ai-dynamo/dynamo --state all --limit 100 --json number,title,body \
  --search 'benchmark-isolation.md in:title,body'
```

Review every hit whose title starts with `[AGENT]: ` or whose body mentions the same file and
defect class, and read each candidate's comments too, because confirmations of an existing defect
are filed as comments (below) and do not appear in title or body search:

```bash
ISSUE_NUMBER=123
gh issue view "$ISSUE_NUMBER" --repo ai-dynamo/dynamo --comments
```

If a matching issue exists, do not file a duplicate: draft the same body (step 5), route it through
step 6 with the target recorded as "comment on #123", and on approval run the command with BOTH
variables assigned in the same shell invocation (nothing persists between harness shell calls):

```bash
ISSUE_NUMBER=123
BODY_FILE=$EXP_ROOT/analysis/skillpack-issue-drafts/001-perf-analyzer-body.md
gh issue comment "$ISSUE_NUMBER" --repo ai-dynamo/dynamo --body-file "$BODY_FILE"
```

File at most one new issue per session; if the session surfaced several defects, put the most
impactful one in the issue and list the rest briefly in its body.

## Step 5: Draft the issue

Title: `[AGENT]: <file path relative to repo root>: <one-line defect>`.

Write the body with a file-writing tool, never by echoing it through a shell, to a run-scoped path that
names the invoking role, because several roles can hit defects in the same engagement window: inside an
engagement `<EXP_ROOT>/analysis/skillpack-issue-drafts/NNN-<role>-body.md` (NNN increasing), otherwise
`./skillpack-issue-drafts/NNN-<role>-body.md` under the current working directory. Never use a shared
fixed path such as `/tmp`, where another role's write can replace an approved body before it is
submitted. Delete the draft file once its issue or comment is filed, so a later invocation cannot inherit
a stale body. Follow the
agent-reported template's structure:

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

## Step 6: Who files, and how

Filing a public issue or comment is an external side effect that needs the operator's approval of the
exact drafted text. Two cases:

- **Dispatched sub-role** (interviewer, deployer, analyzer, generator, challenger): you do not talk to
  the operator, so you never run `gh issue create` or `gh issue comment`. Record the draft in
  `<EXP_ROOT>/analysis/skillpack-defects.md` (dated section, defect class, location, draft title,
  draft body path, proposed target: new issue or comment on #N, status `unfiled`) and return that
  path to your parent. The top-level agent batches every unfiled draft, presents them to the operator,
  and on approval files at most ONE new issue per session (secondary defects go in its body) plus any
  approved comments, then updates each draft's status to `filed` with the URL. This is what makes
  "one issue per session" hold across roles.
- **Standalone invocation by an operator**: the operator is present; show the complete title and body,
  file only on approval.

Whoever files: assign the title from a quoted heredoc in the same shell invocation as the command
(harness shells do not persist variables between calls), and pass the approved body file unchanged.
Quoting `"$ISSUE_TITLE"` at the call site protects nothing on its own: backticks and `$()` in a drafted
title are interpreted when the variable is *assigned*, and the pack's own style backticks every flag.
Do not edit the body between approval and filing; file the same path the operator saw.

```bash
BODY_FILE=$EXP_ROOT/analysis/skillpack-issue-drafts/001-perf-analyzer-body.md
ISSUE_TITLE=$(cat <<'EOF'
[AGENT]: agent-docs/rules/benchmarking/benchmark-isolation.md: one-line defect
EOF
)
gh issue create --repo ai-dynamo/dynamo --title "$ISSUE_TITLE" --body-file "$BODY_FILE" --label agent-reported
```

Labels behave two ways depending on the client path: the REST API silently drops labels the
reporter may not set ("Only users with push access can set labels for new issues. Labels are silently
dropped otherwise."), while `gh` may instead fail before creating the issue when it cannot resolve a
label. Handle both: if the command fails on the label, retry without `--label`; if it succeeds,
verify:

```bash
ISSUE_NUMBER=123
gh issue view "$ISSUE_NUMBER" --repo ai-dynamo/dynamo --json labels --jq '[.labels[].name]'
```

If `agent-reported` is absent, tell the operator so a maintainer can add it; maintainers also triage
by the `[AGENT]: ` title prefix, and the step 4 search covers unlabeled reports.

## Fallback: no GitHub access or no approval

Write the complete draft to an append-only artifact the invoking role owns and tell the operator
where it is. Inside an optimization engagement that is `<EXP_ROOT>/analysis/skillpack-defects.md`
(one dated section per draft, shaped like an `asks.jsonl` entry: defect class, location, draft
title, draft body, status `unfiled`); never write under `final/`, whose three artifacts belong to
`hypothesis-generator` and are written only at stop-request time. Outside an engagement, write
`skillpack-issue-draft-<n>.md` at the root of the current working directory. A
drafted-but-unfiled report is still telemetry; a silent workaround is not.
