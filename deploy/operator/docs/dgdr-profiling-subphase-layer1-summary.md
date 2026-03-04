# Layer 1: Controller Prep for DGDR Profiling Sub-Phase Tracking

> Implementation summary for the first layer of `dgdr-profiling-subphase-tracking.md`.

## Goal

When `kubectl get dgdr` shows a DGDR in `Profiling` phase, users should see **which sub-phase** is active. When profiling fails, the failure should be **attributed to the specific sub-phase** that failed, not a generic "ProfilingFailed".

## Design Principle: Profiler Owns the Vocabulary

The profiler input is the DGDR spec verbatim as JSON. The same principle applies to output: **the profiler writes `phase` and `message`, the controller copies them through with zero translation.**

- **`phase`** — profiler writes it (Layer 3), controller copies to `status.profilingPhase`
- **`message`** — profiler writes it (already does today), controller copies to `condition.Message`
- **`reason`** — derived mechanically: `string(phase)` when running, `string(phase) + "Failed"` when failed

This works because `ProfilingPhase` string values and `ProfilingReason` string values are identical by design (e.g., `ProfilingPhaseSweepingDecode = "SweepingDecode" = ProfilingReasonSweepingDecode`), and failure reasons follow the convention `"<Phase>Failed"`.

## What Changed

### File: `deploy/operator/internal/controller/dynamographdeploymentrequest_controller.go`

#### Phase-to-reason derivation (replaces 3 switch-based mapping functions)

Two simple derivation functions replace the original 63-line switch statements:

| Function | Logic | Fallback |
|---|---|---|
| `profilingPhaseReason(phase)` | `string(phase)` (identity cast) | `"Unknown"` for empty, `"Completed"` for Done |
| `profilingPhaseFailureReason(phase)` | `string(phase) + "Failed"` | `"ProfilingFailed"` for empty |

No message mapping exists — the message comes from the profiler via the progress ConfigMap.

#### Bug fix: don't clear profilingPhase on failure

**Before:** `dgdr.ClearProfilingPhase()` was called on Job failure, erasing where profiling died.

**After:** Removed. The `profilingPhase` remains set on failure. Still cleared on success.

#### Use sub-phase reason when entering Profiling

**Before:** Generic `"ProfilingRunning"` reason.
**After:** `"Initializing"` reason with message `"Profiling job created, entering initialization phase"`.

#### Failure phase attribution

- **Failure path** reads `dgdr.Status.ProfilingPhase` directly — it's already current because `updateProfilingSubPhase()` runs first (reads progress ConfigMap) and it was set to `Initializing` on entry. No separate lookup function needed.
- Derives failure reason mechanically: `string(phase) + "Failed"` (e.g., `"SweepingDecodeFailed"`).

#### Enhanced Succeeded condition on failure

`setSucceededConditionWithDetails(dgdr, phase, reason, message)` sets a sub-phase-specific reason on the Succeeded condition instead of generic `"Failed"`.

The profiling failure path now sets phase, Profiling condition, and Succeeded condition directly (bypassing `updatePhaseWithCondition` which would use the generic `setSucceededCondition`).

#### Sub-phase update from output ConfigMap

- **`updateProfilingSubPhase(ctx, dgdr)`** — reads the existing output ConfigMap (`dgdr-output-<name>`), copies `phase` to `status.profilingPhase`, copies `message` to condition Message, derives Reason from phase. No-op until the sidecar writes phase/message to the ConfigMap.

Called at the top of `handleProfilingPhase()`, before checking Job completion.

### File: `deploy/operator/internal/controller/dynamographdeploymentrequest_controller_test.go`

| Test Suite | Tests |
|---|---|
| **Phase Derivation Functions** | `profilingPhaseReason`: identity for 6 phases, Completed for Done, Unknown for empty, passthrough for unrecognized. `profilingPhaseFailureReason`: `<Phase>Failed` for all phases, generic for empty |
| **Progress ConfigMap Naming** | Correct name generation; prefix consistency |
| **setSucceededConditionWithDetails** | Failed → `ConditionFalse` with custom reason; Ready/Deployed → `ConditionTrue`; Profiling → `ConditionFalse` |
| **Failure attribution** | Preserves `profilingPhase`, uses `<Phase>Failed` on both conditions; generic when no sub-phase |
| **Profiling entry reason** | Uses `Initializing` reason (not `ProfilingRunning`) |
| **updateProfilingSubPhase** | Updates from ConfigMap (phase + message); no-op without ConfigMap; skips when unchanged |

## Data Flow (Full Pipeline)

```
profiler_status.yaml          ConfigMap (sidecar relays)       DGDR status (controller copies)
─────────────────────         ──────────────────────────       ────────────────────────────────
phase: SweepingDecode    →    data.phase: SweepingDecode   →  status.profilingPhase: SweepingDecode
message: "Sweeping TP=4" →   data.message: "Sweeping TP=4" → condition.Message: "Sweeping TP=4"
                                                               condition.Reason: SweepingDecode  (derived)
```

On failure:
```
                                                               condition.Reason: SweepingDecodeFailed  (phase + "Failed")
                                                               condition.Message: "profiling job failed: OOM..."  (from Job)
```

## Behavioral Changes

### Before Layer 1
```yaml
status:
  phase: Failed
  profilingPhase: ""             # cleared (bug)
  conditions:
  - type: Profiling
    reason: ProfilingFailed      # generic
  - type: Succeeded
    reason: Failed               # generic
```

### After Layer 1 (without Layers 2-3)
```yaml
status:
  phase: Failed
  profilingPhase: Initializing   # preserved (bug fixed)
  conditions:
  - type: Profiling
    reason: InitializingFailed   # derived: phase + "Failed"
  - type: Succeeded
    reason: InitializingFailed   # sub-phase-specific
```

### After All 3 Layers
```yaml
# During profiling
status:
  phase: Profiling
  profilingPhase: SweepingDecode
  conditions:
  - type: Profiling
    reason: SweepingDecode
    message: "Sweeping TP=4 DEP=2, measuring ITL at concurrency 32"  # from profiler

# On failure
status:
  phase: Failed
  profilingPhase: SweepingDecode
  conditions:
  - type: Profiling
    reason: SweepingDecodeFailed
    message: "profiling job failed: CUDA out of memory..."
  - type: Succeeded
    reason: SweepingDecodeFailed
```

## Graceful Degradation

- **Without Layers 2-3:** `profilingPhase` stays at `Initializing`. `updateProfilingSubPhase()` is a no-op. On failure, attribution uses `Initializing` → `InitializingFailed`.
- **With all layers:** Full sub-phase visibility. Messages come from the profiler.

## Files Changed

| File | Changes |
|---|---|
| `deploy/operator/internal/controller/dynamographdeploymentrequest_controller.go` | +2 derivation functions (replacing 3 switch statements), +1 helper method for sub-phase updates, bug fix, updated entry reason, enhanced failure handling |
| `deploy/operator/internal/controller/dynamographdeploymentrequest_controller_test.go` | +4 Describe blocks with tests for derivation, naming, conditions, failure attribution, sub-phase updates |
| `deploy/operator/api/v1beta1/dynamographdeploymentrequest_types.go` | No changes needed |
