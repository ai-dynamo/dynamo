# Skill: Create a DEP as a GitHub Issue

## Purpose

Create a new Dynamo Enhancement Proposal (DEP) as a GitHub Issue on
`ai-dynamo/dynamo`. The issue number becomes the DEP number. Also
handles adding implementation plans and retroactive DEPs for existing
work.

## When to Use

When the user wants to propose a new feature, architecture change, or
process improvement via the issue-based DEP workflow. Also when adding
an implementation plan to an existing DEP, or filing a retroactive DEP
for work already merged.

## Workflow

### Create a New DEP

1. **Gather required fields** from the user (prompt if missing):
   - **Summary**: One-paragraph description of the proposal
   - **Motivation**: Why this change is needed
   - **Proposal**: Detailed description of the proposed change

2. **Determine the area label** based on proposal content. Area labels
   are bare names (e.g., `frontend`, `router`, `backend-vllm`) that
   correspond to CODEOWNERS teams.

3. **Decide template**: full or lightweight.
   Use lightweight if only Summary, Motivation, and Proposal are needed.

4. **Create the issue**:

```bash
gh issue create \
  --repo ai-dynamo/dynamo \
  --title "DEP: <short descriptive title>" \
  --label "dep:draft" \
  --label "<area>" \
  --body "$(cat <<'EOF'
## Summary
<summary>

## Motivation
<motivation>

## Proposal
<proposal>

## Alternate Solutions
<alternates or omit for lightweight>

## Requirements
<requirements or omit for lightweight>

## References
<references or omit for lightweight>
EOF
)"
```

5. **Report** the created issue number and URL to the user.

### Add an Implementation Plan

1. **Read the DEP issue** and its discussion:

```bash
gh issue view <number> --repo ai-dynamo/dynamo
gh issue view <number> --repo ai-dynamo/dynamo --comments
```

2. **Draft the plan** with phases, tasks, effort estimates,
   dependencies, risks, and testing strategy.

3. **Post as a comment**:

```bash
gh issue comment <number> --repo ai-dynamo/dynamo --body-file /tmp/plan.md
```

### Retroactive DEP

For work already merged without a DEP, file with `dep:implementing`
or `dep:done` and reference the existing PRs.

## Notes

- The issue body IS the spec — treat it as a living document.
- `dep:draft` is applied automatically. PIC changes to
  `dep:under-review` when ready.
- For lightweight DEPs, use `dep:lightweight` label and omit optional
  sections.
- For plan revisions, post a new comment with a changelog at the top.
  Do not edit the original — preserve the timeline.
