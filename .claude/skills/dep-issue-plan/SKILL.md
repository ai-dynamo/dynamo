# Skill: Add Implementation Plan to a DEP Issue

## Purpose

Draft and post an implementation plan as a comment on an existing DEP
issue. The plan is kept separate from the spec (issue body) to allow
independent iteration.

## When to Use

When a DEP spec is solid enough to plan implementation. Can be invoked
by the author, PIC, or an AI agent.

## Workflow

1. **Read the DEP issue** to understand the spec:

```bash
gh issue view <number> --repo ai-dynamo/dynamo
```

2. **Read the discussion** for reviewer feedback and constraints:

```bash
gh issue view <number> --repo ai-dynamo/dynamo --comments
```

3. **Draft the plan** with this structure:
   - **Phases**: Sequential phases with clear deliverables
   - **Tasks**: Concrete work items per phase (can become PRs)
   - **Effort**: T-shirt sizes (S/M/L/XL) per phase
   - **Dependencies**: Other DEPs, external systems, prerequisites
   - **Risks**: Technical risks and mitigations
   - **Testing strategy**: How the implementation will be validated

4. **Write plan to a temp file and post as a comment**:

```bash
cat <<'EOF' > /tmp/plan.md
# Implementation Plan

## Phase 1: <name>
**Effort**: <estimate>
**Tasks**:
- [ ] Task 1
- [ ] Task 2

## Phase 2: <name>
**Effort**: <estimate>
**Tasks**:
- [ ] Task 3
- [ ] Task 4

## Dependencies
- <dependency list>

## Risks
- <risk and mitigation>

## Testing Strategy
- <how to validate>
EOF

gh issue comment <number> --repo ai-dynamo/dynamo --body-file /tmp/plan.md
```

5. **Report** the comment URL to the user.

## Notes

- For revisions, post a new comment with the updated plan and a
  changelog at the top. Do not edit the original — preserve the
  timeline.
- Task checkboxes can be updated as implementation progresses.
