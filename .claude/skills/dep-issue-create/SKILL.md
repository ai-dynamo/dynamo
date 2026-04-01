# Skill: Create a DEP as a GitHub Issue

## Purpose

Create a new Dynamo Enhancement Proposal (DEP) as a GitHub Issue on
`ai-dynamo/dynamo`. The issue number becomes the DEP number.

## When to Use

When the user wants to propose a new feature, architecture change, or
process improvement via the issue-based DEP workflow.

## Workflow

1. **Gather required fields** from the user (prompt if missing):
   - **Summary**: One-paragraph description of the proposal
   - **Motivation**: Why this change is needed
   - **Proposal**: Detailed description of the proposed change

2. **Determine the area label** based on proposal content:
   `area/frontend`, `area/router`, `area/backend`, `area/kvbm`,
   `area/bindings`, `area/deployment`, `area/observability`,
   `area/cicd`, `area/process`, `area/cross-cutting`

3. **Decide template**: full or lightweight.
   Use lightweight if only Summary, Motivation, and Proposal are needed.

4. **Create the issue**:

```bash
gh issue create \
  --repo ai-dynamo/dynamo \
  --title "DEP: <short descriptive title>" \
  --label "dep:draft" \
  --label "area/<area>" \
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

## Notes

- The issue body IS the spec — treat it as a living document.
- For detailed specs, the author can attach a `dep.md` file or link to
  a Google Doc. The issue body should always contain at minimum the
  summary and a link to the full spec.
- `dep:draft` is applied automatically. PIC changes to
  `dep:under-review` when ready.
- For lightweight DEPs, use `dep:lightweight` label and omit optional
  sections.
