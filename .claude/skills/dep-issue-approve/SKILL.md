# Skill: Approve a DEP Issue

## Purpose

Post an approval on a DEP issue and update its status label.

## When to Use

When a reviewer or PIC is ready to approve a DEP that is under review.

## Workflow

1. **Verify the issue is under review**:

```bash
gh issue view <number> --repo ai-dynamo/dynamo --json labels
```

   Confirm `dep:under-review` label is present.

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

4. **Report** the approval status to the user.

## Notes

- For straightforward DEPs, the PIC's `/approve` is sufficient — no
  checklist needed.
- For multi-reviewer DEPs, the PIC maintains a pinned approval
  checklist and updates the label only when all required approvals are
  collected.
- `/approve` comments are searchable across the repo for audit:
  `gh search issues --repo ai-dynamo/dynamo "/approve" in:comments`
