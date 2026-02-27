# AI Agent Instructions for Offload Module

This document provides governance rules for AI agents (LLMs, copilots, etc.) that modify files in this offload module.

## Before Making Any Changes

1. **Read [README.md](README.md)** - Understand the documented policies
2. **Evaluate alignment** - Does your proposed change align with the policies?
3. **Consult [DEVELOPER.md](DEVELOPER.md)** - Understand implementation details

## Policy Alignment Check

Before implementing any change to this module, you must:

1. Identify which policy statements in README.md are affected by your change
2. Determine if the change is:
   - **On-policy**: Aligns with documented behavior
   - **Off-policy**: Contradicts or extends documented behavior

## Decision Flow

```
┌─────────────────────────────────────────────────┐
│             Proposed Change                     │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│   Does it align with README.md policies?        │
│                                                 │
│   Review policies P1-P6 in README.md            │
└─────────────────────────────────────────────────┘
          │                         │
         YES                        NO
          │                         │
          ▼                         ▼
┌───────────────────┐   ┌─────────────────────────────────────┐
│ Proceed with      │   │ STOP: Prompt user to decide         │
│ implementation    │   │                                     │
│                   │   │ Ask: "This change conflicts with    │
│ Update            │   │ policy [Pn]. Should I:              │
│ DEVELOPER.md      │   │                                     │
│ with details      │   │ 1. Update the README policy, OR     │
│                   │   │ 2. Adapt the code to align with     │
│                   │   │    the existing policy?"            │
└───────────────────┘   └─────────────────────────────────────┘
```

## On-Policy Changes

If the change aligns with documented policies:

1. **Implement the change**
2. **Update DEVELOPER.md** with relevant implementation details
3. **Do NOT modify README.md** policy statements

Examples of on-policy changes:
- Bug fixes that maintain documented behavior
- Performance optimizations within policy bounds
- Adding tests for existing functionality
- Refactoring that preserves semantics

## Off-Policy Changes

If the change contradicts or extends documented policies:

1. **Do not implement** without user approval
2. **Present the conflict** to the user with:
   - The exact policy being affected (quote from README.md)
   - How the proposed change conflicts
   - The decision options
3. **Wait for user decision**:
   - If updating policy: Modify README.md first, then implement
   - If adapting code: Revise implementation to align with policy

Example prompt to user:

```
This change would allow cancellation after the upgrade step, which conflicts
with policy P3 in README.md:

> **P3: Upgrade is the Commitment Boundary**
> The upgrade step (Weak → Strong) is the point of no return. After upgrade,
> cancellation no longer applies.

Should I:
1. Update README.md to allow post-upgrade cancellation, OR
2. Adapt the code to perform cancellation before upgrade?
```

## Key Policies to Check

When modifying this module, specifically verify alignment with:

| Policy | Summary | Watch For |
|--------|---------|-----------|
| **P1** | Container is unit of cancellation | Don't add per-block cancellation |
| **P2** | Token travels with container | Don't strip token before upgrade |
| **P3** | Upgrade is commitment boundary | Don't cancel after upgrade |
| **P4** | Sweep before upgrade | Don't skip the final sweep |
| **P5** | Flat map after upgrade | Don't preserve container identity post-upgrade |
| **P6** | PreconditionAwaiter uses select | Don't block without cancel check |

## Documentation Update Requirements

| Change Type | README.md | DEVELOPER.md |
|-------------|-----------|--------------|
| Bug fix (on-policy) | No change | Update if relevant |
| Feature (on-policy) | No change | Add implementation section |
| Policy change | Update affected policies | Update accordingly |
| Refactor (on-policy) | No change | Update if relevant |
| New destination type | No change | Add to TransferDestination section |
| New policy filter | No change | Add to Extension Rules section |

## Common Scenarios

### Adding a New Pipeline Stage

1. Check: Does this affect cancellation boundaries? (P3, P4)
2. Check: How does the token propagate through this stage? (P2)
3. If on-policy: Implement and document in DEVELOPER.md
4. If off-policy: Ask user about policy updates

### Modifying Cancellation Behavior

1. This likely affects P1-P4
2. Always ask user before implementing
3. Quote the specific policies being changed

### Changing the Batcher

1. Check: Does batching preserve per-container cancellability? (P1)
2. Check: Is sweep still performed before upgrade? (P4)
3. Verify containers maintain their tokens

### Modifying TransferExecutor

1. Check: Is upgrade still the commitment point? (P3)
2. Check: Is sweep called before upgrade? (P4)
3. Check: Does flat map occur after upgrade? (P5)

## Error Recovery

If you realize mid-implementation that your change is off-policy:

1. **Stop immediately**
2. **Inform the user** of the policy conflict discovered
3. **Offer to**:
   - Revert partial changes and redesign
   - Complete with policy update (if user approves)
   - Adapt approach to be on-policy

## Governance Updates

If the governance process itself needs updating (this file):

1. Propose changes to the user explicitly
2. Explain why current governance is insufficient
3. Only modify AGENTS.md with explicit approval





