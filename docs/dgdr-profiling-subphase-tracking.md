# DGDR Profiling Sub-Phase Tracking: Implementation Plan

> **Updated** — reflects the current implementation state. Layer 1 (controller prep) is **done**. The key design change from the original plan: **no mapping logic** in the controller. The profiler owns the vocabulary — it writes `phase` and `message` to `profiler_status.yaml`, and the controller copies them through verbatim.

---

## Goal

When a user runs `kubectl get dgdr`, they should see **which profiling step is active** (not just "Profiling"), and when profiling fails, the failure should be **attributed to the specific sub-phase** that failed.

```
NAME           PHASE      PROFILING        REASON               AGE
my-deploy      Profiling  SweepingDecode   SweepingDecode       12m
my-deploy      Failed     SweepingDecode   SweepingDecodeFailed  15m
```

### Sub-phases

`Initializing → SweepingPrefill → SweepingDecode → SelectingConfig → BuildingCurves → GeneratingDGD → Done`

### Status mechanisms involved

| Field / Condition | Purpose | Status |
|---|---|---|
| `status.phase` | High-level lifecycle (Pending/Profiling/Ready/Deploying/Deployed/Failed) | ✅ Done |
| `Succeeded` condition | Aggregate success/progress/failure | ✅ Done (sub-phase-aware) |
| `status.profilingPhase` | Current profiling sub-phase | ✅ Done (controller reads from ConfigMap) |
| `Profiling` condition reason | Mirrors sub-phase | ✅ Done (derived from phase) |

---

## Design Principle: Profiler Owns the Vocabulary

The profiler writes `profiler_status.yaml` with `phase` and `message` fields. The controller copies them through with **zero translation**:

- **`phase`** → copied to `status.profilingPhase` and used as `condition.Reason`
- **`message`** → copied to `condition.Message`
- **failure reason** → derived mechanically: `string(phase) + "Failed"`

This works because `ProfilingPhase` string values and `ProfilingReason` string values are identical by design (e.g., `ProfilingPhaseSweepingDecode = "SweepingDecode" = ProfilingReasonSweepingDecode`).

No switch-based mapping functions exist — two simple derivation functions handle everything:

| Function | Logic | Fallback |
|---|---|---|
| `profilingPhaseReason(phase)` | `string(phase)` (identity cast) | `"Unknown"` for empty, `"Completed"` for Done |
| `profilingPhaseFailureReason(phase)` | `string(phase) + "Failed"` | `"ProfilingFailed"` for empty |

---

## Architecture: How Sub-Phases Flow

The profiling Job is a **black box** to the controller — it creates a Job and watches for completion. Sub-phase reporting piggybacks on the **existing sidecar** (`output-copier`) and the **existing output ConfigMap** (`dgdr-output-<name>`).

Today the sidecar is "wait for profiler to terminate, then copy output." The change: make it a **continuous poller** that relays `phase` and `message` from `profiler_status.yaml` → ConfigMap during execution, then writes the final profiling results as the last step after the profiler terminates. No new ConfigMap — reuse `dgdr-output-<name>`.

```
Profiler Pod               Sidecar Container              Controller
(profile_sla.py)           (output-copier)                (Go)
     │                          │                              │
     │ write_profiler_status()  │                              │
     │   phase: SweepingDecode  │                              │
     │   message: "Sweeping..." │                              │
     │ → profiler_status.yaml   │                              │
     │ (shared emptyDir)        │                              │
     │─────────────────────────►│                              │
     │                          │ polls every 10s, writes      │
     │                          │ phase+message to existing    │
     │                          │ dgdr-output-<name> ConfigMap │
     │                          │─────────────────────────────►│
     │                          │                              │ updateProfilingSubPhase()
     │                          │                              │ reads dgdr-output-<name>
     │                          │                              │ copies verbatim:
     │                          │                              │   phase → status.profilingPhase
     │                          │                              │   phase → condition.Reason
     │                          │                              │   message → condition.Message
     │  (terminates)            │                              │
     │─────────────────────────►│                              │
     │                          │ final poll: writes           │
     │                          │ final_config.yaml +          │
     │                          │ profiler_status.yaml to      │
     │                          │ same ConfigMap               │
     │                          │─────────────────────────────►│
     │                          │                              │ generateDGDSpec()
     │                          │                              │ reads final_config.yaml
     │                          │                              │ from same ConfigMap
```

Data flows verbatim end-to-end:

```
profiler_status.yaml           ConfigMap dgdr-output-<name>      DGDR status (controller copies)
─────────────────────          ────────────────────────────       ────────────────────────────────
phase: SweepingDecode     →    data.phase: SweepingDecode    →   status.profilingPhase: SweepingDecode
message: "Sweeping TP=4"  →    data.message: "Sweeping TP=4" →  condition.Message: "Sweeping TP=4"
                                                                  condition.Reason: SweepingDecode  (= phase)
```

On failure:
```
                                                                  condition.Reason: SweepingDecodeFailed  (phase + "Failed")
                                                                  condition.Message: "profiling job failed: OOM..."  (from Job)
```

Each layer degrades gracefully: if the profiler doesn't write phases, the sidecar relays nothing new. If the sidecar hasn't written yet, the controller falls back to `Initializing`. Everything still works — you just don't get sub-phase progression.

---

## What's Done

### Layer 1: Controller Prep ✅ COMPLETE

**File**: `deploy/operator/internal/controller/dynamographdeploymentrequest_controller.go`

All of the following are implemented and tested:

#### Phase derivation functions (no mapping)

Two simple functions replace the originally-planned 63-line switch statements:

```go
func profilingPhaseReason(phase nvidiacomv1beta1.ProfilingPhase) string {
    if phase == nvidiacomv1beta1.ProfilingPhaseDone {
        return nvidiacomv1beta1.ProfilingReasonCompleted
    }
    if phase == "" {
        return "Unknown"
    }
    return string(phase) // identity — profiler vocabulary passes through
}

func profilingPhaseFailureReason(phase nvidiacomv1beta1.ProfilingPhase) string {
    if phase == "" {
        return "ProfilingFailed"
    }
    return string(phase) + "Failed"
}
```

#### Bug fix: don't clear profilingPhase on failure

`ClearProfilingPhase()` is no longer called on Job failure. The sub-phase remains set so users can see where profiling died. Still cleared on success.

#### Initializing reason on entry

When entering the Profiling phase, uses `Initializing` reason (not the old generic `ProfilingRunning`):

```go
dgdr.SetProfilingPhase(nvidiacomv1beta1.ProfilingPhaseInitializing)
return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseProfiling,
    nvidiacomv1beta1.ConditionTypeProfiling, metav1.ConditionFalse,
    nvidiacomv1beta1.ProfilingReasonInitializing,
    "Profiling job created, entering initialization phase")
```

#### Failure phase attribution

The failure path reads `dgdr.Status.ProfilingPhase` directly (already current because `updateProfilingSubPhase()` runs first) and derives the failure reason mechanically:

```go
failureReason := "ProfilingFailed"
if dgdr.Status.ProfilingPhase != "" {
    failureReason = profilingPhaseFailureReason(dgdr.Status.ProfilingPhase)
}
```

Both the `Profiling` and `Succeeded` conditions are set with the sub-phase-specific failure reason directly (bypassing `updatePhaseWithCondition` which would use the generic `setSucceededCondition`).

#### Sub-phase update from output ConfigMap

- `updateProfilingSubPhase(ctx, dgdr)` — reads the existing output ConfigMap (`dgdr-output-<name>`), copies `phase` to `status.profilingPhase`, copies `message` to condition Message, derives Reason from phase identity. Called at top of `handleProfilingPhase()`, before checking Job completion. No-op until the sidecar writes phase/message to the ConfigMap (Layer 2).

> **Note:** Layer 1 was originally built with a separate `dgdr-progress-<name>` ConfigMap. This will be simplified in Layer 2 to read from the existing `dgdr-output-<name>` ConfigMap instead — no separate progress ConfigMap needed. The `ProgressConfigMapPrefix` constant and `getProgressConfigMapName()` helper will be removed.

#### Tests

| Test Suite | Coverage |
|---|---|
| Phase Derivation | `profilingPhaseReason`: identity for 6 phases, Completed for Done, Unknown for empty, passthrough for unrecognized. `profilingPhaseFailureReason`: `<Phase>Failed` for all phases, generic for empty |
| ConfigMap Naming | Correct name generation; prefix consistency |
| Failure Attribution | Preserves `profilingPhase`, uses `<Phase>Failed` on both conditions; generic when no sub-phase |
| Profiling Entry | Uses `Initializing` reason (not `ProfilingRunning`) |
| updateProfilingSubPhase | Updates from ConfigMap (phase + message); no-op without ConfigMap; skips when unchanged |

---

## What Remains: 1 Layer of Work

### Layer 3: Profiler Phase Writes (Python)

~15 lines across 2 files. Adds `phase` parameter to `write_profiler_status()` and calls it at each stage boundary. Lights up the sub-phase visibility.

---

## Layer 2: Sidecar Continuous Polling ✅ COMPLETE

**File**: `deploy/operator/internal/controller/dynamographdeploymentrequest_controller.go`

The existing sidecar (`output-copier`) is a "wait for termination, then copy" script. We restructure it into a **continuous poller** that relays phase/message during execution and writes the final output as the last step. Same ConfigMap (`dgdr-output-<name>`), no new resources.

### 2.1 Rewrite sidecar script

Replace the current `sidecarScriptTemplate` with a unified poll loop:

```bash
set -e
set -o pipefail

STATUS_FILE="{{.OutputPath}}/profiler_status.yaml"
LAST_PHASE=""

# Poll profiler_status.yaml and relay phase+message to the output ConfigMap.
# When the profiler terminates, do one final poll, then copy the profiling output.
while true; do
  # --- Phase relay (runs every iteration, including the final one) ---
  if [ -f "$STATUS_FILE" ]; then
    PHASE=$(grep "^phase:" "$STATUS_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")
    MESSAGE=$(grep "^message:" "$STATUS_FILE" | sed 's/^message: *//' | tr -d '"' | tr -d "'")
    if [ -n "$PHASE" ] && [ "$PHASE" != "$LAST_PHASE" ]; then
      # Build a minimal ConfigMap with just phase+message (preserves existing keys on apply)
      cat >/tmp/progress.yaml <<PEOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{.ConfigMapName}}
  namespace: {{.Namespace}}
  labels:
    dgdr.nvidia.com/name: {{.DGDRName}}
    nvidia.com/managed-by: dynamo-operator
data:
  phase: "$PHASE"
  message: "$MESSAGE"
PEOF
      kubectl apply -f /tmp/progress.yaml
      LAST_PHASE="$PHASE"
    fi
  fi

  # --- Check if profiler container has terminated ---
  CONTAINER_STATUS=$(kubectl get pod $HOSTNAME -n {{.Namespace}} \
    -o jsonpath='{.status.containerStatuses[?(@.name=="profiler")].state}' 2>/dev/null || echo "")
  if echo "$CONTAINER_STATUS" | grep -q "terminated"; then
    echo "Profiler terminated"
    break
  fi

  sleep 10
done

# --- Final output copy (same as before) ---

# Verify profiler status
if [ ! -f "$STATUS_FILE" ]; then
  echo "ERROR: Status file not found"
  exit 1
fi

STATUS=$(grep "^status:" "$STATUS_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")
case "$STATUS" in
  success)
    echo "Profiler succeeded"
    ;;
  failed)
    ERROR=$(grep "^error:" "$STATUS_FILE" | sed 's/^error: *//' | tr -d '"' | tr -d "'")
    echo "ERROR: Profiler failed: ${ERROR}"
    exit 1
    ;;
  *)
    echo "ERROR: Unexpected status: $STATUS"
    exit 1
    ;;
esac

# Append final_config.yaml and profiler_status.yaml to the ConfigMap
echo "Writing profiling output to ConfigMap..."
cat >/tmp/cm.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{.ConfigMapName}}
  namespace: {{.Namespace}}
  labels:
    dgdr.nvidia.com/name: {{.DGDRName}}
    nvidia.com/managed-by: dynamo-operator
data:
  {{.OutputFile}}: |
EOF
sed 's/^/    /' {{.OutputPath}}/{{.OutputFile}} >> /tmp/cm.yaml

if [ -f {{.OutputPath}}/profiler_status.yaml ]; then
  echo "  profiler_status.yaml: |" >> /tmp/cm.yaml
  sed 's/^/    /' {{.OutputPath}}/profiler_status.yaml >> /tmp/cm.yaml
fi

kubectl apply -f /tmp/cm.yaml
echo "Saved profiling output to ConfigMap {{.ConfigMapName}}"
```

Key changes from the current sidecar:
- **Continuous polling** instead of blocking wait — relays `phase` and `message` to the output ConfigMap every time the phase changes
- **Same ConfigMap** (`dgdr-output-<name>`) for both progress and final output — no separate progress ConfigMap
- **Final output** is appended to the ConfigMap after termination (same as before, just uses `kubectl apply` instead of `kubectl create`)
- No new template variables needed — `ConfigMapName`, `Namespace`, `DGDRName`, `OutputPath`, `OutputFile` already exist

### 2.2 Update `updateProfilingSubPhase()` to read from output ConfigMap

Change `updateProfilingSubPhase()` to read from the output ConfigMap (`dgdr-output-<name>`) instead of a separate progress ConfigMap:

```go
func (r *DynamoGraphDeploymentRequestReconciler) updateProfilingSubPhase(
    ctx context.Context,
    dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) {
    logger := log.FromContext(ctx)
    outputCMName := getOutputConfigMapName(dgdr) // dgdr-output-<name>

    cm := &corev1.ConfigMap{}
    if err := r.Get(ctx, types.NamespacedName{
        Name: outputCMName, Namespace: dgdr.Namespace,
    }, cm); err != nil {
        return // No ConfigMap yet — skip
    }

    phase, exists := cm.Data["phase"]
    if !exists || phase == "" {
        return
    }

    profilingPhase := nvidiacomv1beta1.ProfilingPhase(phase)
    if dgdr.Status.ProfilingPhase == profilingPhase {
        return // No change
    }

    logger.Info("Profiling sub-phase updated", "phase", phase)
    dgdr.SetProfilingPhase(profilingPhase)

    reason := profilingPhaseReason(profilingPhase)
    message := cm.Data["message"]

    meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
        Type:               nvidiacomv1beta1.ConditionTypeProfiling,
        Status:             metav1.ConditionFalse,
        ObservedGeneration: dgdr.Generation,
        Reason:             reason,
        Message:            message,
    })
    meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
        Type:               nvidiacomv1beta1.ConditionTypeSucceeded,
        Status:             metav1.ConditionFalse,
        ObservedGeneration: dgdr.Generation,
        Reason:             reason,
        Message:            message,
    })

    if err := r.Status().Update(ctx, dgdr); err != nil {
        logger.Error(err, "Failed to update profiling sub-phase in status")
    }
}
```

### 2.3 Remove progress ConfigMap infrastructure

Layer 1 introduced `ProgressConfigMapPrefix` and `getProgressConfigMapName()` as a placeholder. Now that we piggyback on the output ConfigMap, remove them:

- Delete `ProgressConfigMapPrefix` constant
- Delete `getProgressConfigMapName()` function

### 2.4 Add ConfigMap watch in `SetupWithManager()`

Watch the output ConfigMap for phase updates during profiling. Filter to only ConfigMaps with the DGDR label:

```go
Watches(
    &corev1.ConfigMap{},
    handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
        cm := obj.(*corev1.ConfigMap)
        dgdrName, hasName := cm.Labels[nvidiacomv1beta1.LabelDGDRName]
        dgdrNamespace, hasNamespace := cm.Labels[nvidiacomv1beta1.LabelDGDRNamespace]
        if !hasName || !hasNamespace {
            return nil
        }
        return []ctrl.Request{{
            NamespacedName: types.NamespacedName{
                Name:      dgdrName,
                Namespace: dgdrNamespace,
            },
        }}
    }),
    builder.WithPredicates(predicate.Funcs{
        CreateFunc:  func(ce event.CreateEvent) bool { return true },
        UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
        DeleteFunc:  func(de event.DeleteEvent) bool { return false },
        GenericFunc: func(ge event.GenericEvent) bool { return false },
    }),
),
```

Note: The label filter (`dgdr.nvidia.com/name`) ensures we only trigger on ConfigMaps created by the sidecar, not random cluster ConfigMaps.

### 2.5 Verify RBAC

No change expected — the sidecar already has ConfigMap create/update permissions for `dgdr-output-<name>`. Using `kubectl apply` instead of `kubectl create` just requires the same permissions.

### Layer 2 verification

1. Manually create ConfigMap `dgdr-output-<name>` with `phase: SweepingPrefill`, `message: "Sweeping TP configs"` → controller picks it up, updates `profilingPhase` and conditions with verbatim values
2. Update ConfigMap to `phase: SweepingDecode`, `message: "Sweeping decode concurrency"` → status updates
3. Complete profiling → same ConfigMap now has `final_config.yaml` key (appended by sidecar); controller reads it in `generateDGDSpec()` as before
4. Without Layer 3 (profiler doesn't write phases): sidecar finds no `phase:` key in `profiler_status.yaml`, never writes phase to ConfigMap. Everything still works — you just see `Initializing` for the entire run.

---

## Layer 3: Profiler Phase Writes (Python)

**Files**:
- `components/src/dynamo/profiler/utils/profiler_status.py`
- `components/src/dynamo/profiler/profile_sla.py`

~15 lines total. Independent of Layers 1-2 (writes to a file that's read by the sidecar).

### 3.1 Add `phase` parameter to `write_profiler_status()`

```python
def write_profiler_status(
    output_dir: str,
    status: ProfilerStatus,
    message: str = "",
    error: str = "",
    outputs: dict | None = None,
    phase: str | None = None,        # NEW — optional profiling sub-phase
) -> None:
    ...
    if phase:
        status_data["phase"] = phase
    ...
```

Fully backward-compatible — existing callers don't pass `phase`, so nothing breaks.

### 3.2 Add phase transitions in `run_profile()`

Add `phase=` kwarg to the existing `write_profiler_status()` calls, and insert new calls at stage boundaries:

| Location in `profile_sla.py` | Phase | Call |
|---|---|---|
| Line 307 (existing `RUNNING` call) | `Initializing` | Add `phase=ProfilingPhase.Initializing` |
| Before `_execute_strategy()` | `SweepingPrefill` | New call with descriptive message |
| Inside `_execute_strategy()` before decode sweep | `SweepingDecode` | New call |
| After strategy completes, before config selection | `SelectingConfig` | New call |
| Before `run_interpolation()` (line 376) | `BuildingCurves` | New call |
| Before `_assemble_final_config()` (line 394) | `GeneratingDGD` | New call |
| Line 278 (existing `SUCCESS` call) | `Done` | Add `phase=ProfilingPhase.Done` |
| Line 412 (existing `FAILED` call) | *(current phase)* | Add `phase=<current>` to preserve context |

The profiler already has the `ProfilingPhase` enum in `dgdr_v1beta1_types.py`:
```python
class ProfilingPhase(str, Enum):
    Initializing = "Initializing"
    SweepingPrefill = "SweepingPrefill"
    SweepingDecode = "SweepingDecode"
    SelectingConfig = "SelectingConfig"
    BuildingCurves = "BuildingCurves"
    GeneratingDGD = "GeneratingDGD"
    Done = "Done"
```

The **message** written by the profiler is what users see in `kubectl describe dgdr`. It should be descriptive, e.g.:
- `"Loading model architecture and detecting GPU hardware"`
- `"Sweeping TP=4 PP=1 DEP=2, measuring TTFT at concurrency 32"`
- `"Filtering 12 configs against SLA: TTFT ≤ 2000ms, ITL ≤ 30ms"`

The controller copies the message verbatim — no controller-side message mapping needed.

### Example `profiler_status.yaml` output

During sweeping:
```yaml
status: running
timestamp: "2026-03-03T12:34:56Z"
message: "Sweeping TP=4 PP=1, measuring TTFT at concurrency 32"
phase: SweepingPrefill
```

On success:
```yaml
status: success
timestamp: "2026-03-03T13:45:00Z"
message: "Profiler completed successfully"
phase: Done
outputs:
  final_config: final_config.yaml
```

On failure:
```yaml
status: failed
timestamp: "2026-03-03T13:00:00Z"
message: "Profiler failed with exception: OutOfMemoryError"
error: "CUDA out of memory during decode sweep with TP=4"
phase: SweepingDecode
```

### Layer 3 verification

1. Run profiler locally → `profiler_status.yaml` has `phase` and `message` fields at each stage
2. With Layers 1-2 deployed → `kubectl get dgdr -w` shows phase progression with profiler-written messages
3. Force a failure at a specific phase → `kubectl describe dgdr` shows `<Phase>Failed` with the profiler's error message

---

## Dependency Graph

```
Layer 1: Controller Prep ✅ DONE             Layer 3: Profiler Phase Writes
  (derivation functions, bug fix,              (add phase= to write_profiler_status,
   failure attribution, sub-phase               write descriptive messages)
   reading from output ConfigMap)
         │                                         │
         ▼                                         │
Layer 2: Sidecar Continuous Polling ✅ DONE        │
  (rewrite sidecar to poll during                  │
   execution, relay phase+message                  │
   to existing output ConfigMap,                   │
   ConfigMap watch in SetupWithManager)            │
         │                                         │
         └──────────── both needed for ────────────┘
                    full E2E visibility
```

- **Layer 1** ✅ — independently useful: fixes the bug where `profilingPhase` was cleared on failure, uses proper condition reasons, reads phase/message from output ConfigMap verbatim
- **Layer 2** ✅ — independently useful with Layer 1: restructures sidecar to poll continuously, writing phase/message to the existing output ConfigMap; controller watches ConfigMap for updates
- **Layer 3** — independently safe: writes to a file, changes nothing if Layers 1-2 aren't deployed
- **All three together** = full sub-phase visibility in `kubectl`, with profiler-authored messages

---

## Files Changed

| Layer | File | Language | Status |
|---|---|---|---|
| 1 | `deploy/operator/internal/controller/dynamographdeploymentrequest_controller.go` | Go | ✅ Done |
| 1 | `deploy/operator/internal/controller/dynamographdeploymentrequest_controller_test.go` | Go | ✅ Done |
| 2 | `deploy/operator/internal/controller/dynamographdeploymentrequest_controller.go` (sidecar template rewrite + ConfigMap watch + remove progress CM) | Go | ✅ Done |
| 3 | `components/src/dynamo/profiler/utils/profiler_status.py` | Python | TODO |
| 3 | `components/src/dynamo/profiler/profile_sla.py` | Python | TODO |

---

## Expected `kubectl` Output After All Layers

### During profiling
```
NAME           PHASE      PROFILING        REASON             DGD   AGE
my-deploy      Profiling  SweepingDecode   SweepingDecode           12m
```

### `kubectl describe dgdr` during profiling
```yaml
Status:
  Phase:            Profiling
  Profiling Phase:  SweepingDecode
  Conditions:
  - Type:    Profiling
    Status:  False
    Reason:  SweepingDecode
    Message: Sweeping TP=4 PP=1, measuring ITL at concurrency 32   # from profiler
  - Type:    Succeeded
    Status:  False
    Reason:  SweepingDecode
    Message: Sweeping TP=4 PP=1, measuring ITL at concurrency 32   # from profiler
```

### After deployment (autoApply=true)
```
NAME           PHASE     PROFILING   REASON     DGD              AGE
my-deploy      Deployed              Deployed   my-deploy-dgd    45m
```

### Profiling complete (autoApply=false)
```
NAME           PHASE   PROFILING   REASON           DGD   AGE
my-deploy      Ready                SpecGenerated         30m
```

### Failure during profiling
```yaml
Status:
  Phase:            Failed
  Profiling Phase:  SweepingDecode     # preserved (not cleared)
  Conditions:
  - Type:    Succeeded
    Status:  False
    Reason:  SweepingDecodeFailed      # derived: phase + "Failed"
    Message: profiling job failed: CUDA out of memory...
  - Type:    Profiling
    Status:  False
    Reason:  SweepingDecodeFailed
    Message: profiling job failed: CUDA out of memory...
```

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Sidecar ConfigMap write fails (RBAC) | Graceful degradation — phase stays at `Initializing`, everything else works. Same RBAC as existing sidecar (already creates ConfigMap). |
| Race between sidecar write and controller read | Controller catches up on next reconcile (ConfigMap watch triggers it) |
| Profiler phases don't match v1beta1 enum values | Profiler imports `ProfilingPhase` from `dgdr_v1beta1_types.py` — same enum values |
| ConfigMap watch causes reconcile churn | Label predicate filters to only DGDR-managed ConfigMaps; sidecar only writes on phase change |
| `kubectl apply` overwrites final_config.yaml during polling | During polling, only `phase` and `message` keys are written. The final output (with `final_config.yaml`) is written as a single apply after termination, replacing the progress-only ConfigMap. |
| Profiler message too long for condition | Kubernetes conditions have no hard limit; condition messages are typically ≤32KB. Profiler messages should be concise (1-2 sentences). |
