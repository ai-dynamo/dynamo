# Skill: Update DEP Lifecycle

## Purpose

Update DEP status through its lifecycle — triage, review, approve,
defer, or close. Covers the PIC workflow from initial assignment
through final approval.

## When to Use

When triaging DEP issues, reviewing a DEP as PIC or reviewer,
approving a DEP that is under review, or updating DEP status.

## Workflow

### Triage (assign PIC)

1. **List unassigned DEPs**:

```bash
gh issue list --repo ai-dynamo/dynamo \
  --label "dep:draft" \
  --json number,title,labels,assignees \
  --jq '.[] | select(.assignees | length == 0)'
```

2. **Assign PIC** based on the area label:

```bash
gh issue edit <number> --repo ai-dynamo/dynamo \
  --add-assignee "<github-username>"
```

3. **Move to review** when the spec is ready:

```bash
gh issue edit <number> --repo ai-dynamo/dynamo \
  --remove-label "dep:draft" \
  --add-label "dep:under-review"
```

### Review

1. **Read the DEP issue and discussion**:

```bash
gh issue view <number> --repo ai-dynamo/dynamo
gh issue view <number> --repo ai-dynamo/dynamo --comments
```

2. **Post review feedback** as comments on the issue.

3. **Request changes** or clarifications from the author.

### Approve

1. **Verify the issue is under review**:

```bash
gh issue view <number> --repo ai-dynamo/dynamo --json labels
```

2. **Post the approval comment**:

```bash
gh issue comment <number> --repo ai-dynamo/dynamo --body "/approve"
```

3. **If this is the PIC approving** (or all required reviewers have
   approved), update the label:

```bash
gh issue edit <number> --repo ai-dynamo/dynamo \
  --remove-label "dep:under-review" \
  --add-label "dep:approved"
```

## Notes

- For straightforward DEPs, the PIC's `/approve` is sufficient.
- For multi-reviewer DEPs, the PIC maintains a pinned approval
  checklist and updates the label only when all required approvals are
  collected.
- `/approve` comments are searchable for audit:
  `gh search issues --repo ai-dynamo/dynamo "/approve" in:comments`
- Area labels are bare names (e.g., `frontend`, `router`) — no prefix.
