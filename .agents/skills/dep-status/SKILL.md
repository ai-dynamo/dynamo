---
name: dep-status
description: Check Dynamo Enhancement Proposal issue status, list DEPs by lifecycle state or area, and find related DEP issues in ai-dynamo/dynamo.
---

# Skill: Check DEP Status

## Purpose

List DEP issues with their current status, area, PIC, and approval
state. Find related DEPs for a given topic or component.

## When to Use

When the user wants to see the status of one or more DEPs, check what's
pending review, find DEPs related to a component, or get a triage
summary.

## Workflow

1. **List open DEP issues**:

```bash
gh issue list --repo ai-dynamo/dynamo \
  --search 'label:"dep:draft","dep:under-review","dep:approved","dep:implementing"' \
  --json number,title,labels,assignees,createdAt,updatedAt
```

2. **Filter by area** (if requested):

```bash
gh issue list --repo ai-dynamo/dynamo \
  --label "<area>" \
  --json number,title,labels,assignees
```

3. **Filter by status** (if requested):

```bash
gh issue list --repo ai-dynamo/dynamo \
  --label "dep:<status>" \
  --json number,title,labels,assignees
```

4. **Format as a summary table**:

```text
| # | Title | Status | Area | PIC | Updated |
|---|-------|--------|------|-----|---------|
| 42 | DEP: KV router scheduling | dep:under-review | router | @pic | 2026-03-28 |
```

5. **Find related DEPs** by searching issue titles and bodies:

```bash
gh issue list --repo ai-dynamo/dynamo \
  --search 'DEP <keyword> label:"dep:draft","dep:under-review","dep:approved","dep:implementing","dep:done"' \
  --json number,title,labels,state
```

6. **Include closed DEPs** if requested:

```bash
gh issue list --repo ai-dynamo/dynamo \
  --state closed \
  --search 'label:"dep:done","dep:deferred","dep:rejected","dep:replaced"' \
  --json number,title,labels,assignees,closedAt
```

## Notes

- For a full triage view, include both open and recently closed DEPs.
- Cross-reference with `dep:lightweight` label to distinguish full vs.
  lightweight DEPs.
- Area labels are bare names (e.g., `frontend`, `router`) — no prefix.
