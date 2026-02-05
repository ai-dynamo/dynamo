# Design: Fine-Grained Profiling Status for DGDR

## Problem Statement

When a `DynamoGraphDeploymentRequest` (DGDR) is in the `Profiling` state, users have no visibility into which phase of profiling is currently executing. The profiler runs 6-7 distinct phases that can take 60-90+ minutes total, but the DGDR status only shows:

```yaml
status:
  state: Profiling
  conditions:
  - type: Profiling
    status: "False"
    reason: ProfilingRunning
    message: Profiling is in progress
```

Users must manually `kubectl logs` the profiler pod to understand progress.

## Current Architecture

```
┌─────────────────┐     creates      ┌─────────────────┐
│      DGDR       │ ──────────────► │  Profiling Job  │
│   Controller    │                  │    (Pod)        │
└─────────────────┘                  └─────────────────┘
        │                                    │
        │ watches Job status                 │ runs profile_sla.py
        │ (complete/failed only)             │ (7 phases, 60-90 min)
        ▼                                    ▼
┌─────────────────┐                  ┌─────────────────┐
│  DGDR Status    │                  │  Pod Logs       │
│  (coarse)       │                  │  (detailed)     │
└─────────────────┘                  └─────────────────┘
```

The operator only sees Job completion status, not internal profiler phases.

## Profiling Phases

Based on `benchmarks/profiler/profile_sla.py`, these are the phases:

| Phase | Name | Description | Typical Duration |
|-------|------|-------------|------------------|
| 1 | `PrefillSweep` | Profile prefill across GPU counts (1, 2, 4, ...) | 5-15 min |
| 2 | `DecodeSweep` | Profile decode across GPU counts with batch sweeps | 15-30 min |
| 3 | `ConfigSelection` | Select best prefill/decode configs meeting SLA | <1 min |
| 4 | `PrefillInterpolation` | Sweep ISL values for selected prefill config | 5-10 min |
| 5 | `DecodeInterpolation` | Sweep batch sizes for selected decode config | 15-30 min |
| 6 | `PlannerGeneration` | Run SLA planner to compute replica counts | <1 min |
| 7 | `Complete` | Profiling finished, output ready | - |

## Proposed Solutions

### Option A: Profiler writes status to ConfigMap (Recommended)

The profiler periodically updates a ConfigMap with its current phase. The operator watches this ConfigMap and updates DGDR conditions.

**Profiler changes:**
```python
# In profile_sla.py, add status reporting
async def update_profiling_status(phase: str, message: str, progress: dict = None):
    """Write current phase to status ConfigMap."""
    status = {
        "phase": phase,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "progress": progress or {}
    }
    # Write to /data/profiling_status.json
    with open(f"{args.output_dir}/profiling_status.json", "w") as f:
        json.dump(status, f)
```

**Status file example:**
```json
{
  "phase": "DecodeInterpolation",
  "message": "Profiling decode with batch size 496 (4/6)",
  "timestamp": "2026-01-24T02:08:42Z",
  "progress": {
    "currentStep": 4,
    "totalSteps": 6,
    "gpuConfig": "1 GPU, TP=1"
  }
}
```

**Operator changes:**
1. Add sidecar or modify existing sidecar to write status ConfigMap
2. Watch the status ConfigMap and update DGDR conditions
3. Add new condition type `ProfilingPhase` or update `Profiling` condition message

**New DGDR status:**
```yaml
status:
  state: Profiling
  profilingPhase: DecodeInterpolation  # NEW field
  conditions:
  - type: Profiling
    status: "False"
    reason: ProfilingRunning
    message: "Phase 5/7: Profiling decode interpolation (batch 496, 4/6 complete)"
  - type: ProfilingPhase  # NEW condition type
    status: "True"
    reason: DecodeInterpolation
    message: "Profiling decode with batch size 496 (4/6)"
```

**Pros:**
- Decoupled: profiler doesn't need K8s API access
- Flexible: can add arbitrary progress info
- Low overhead: file write is fast

**Cons:**
- Requires sidecar modification to watch file and update ConfigMap
- Slight delay in status propagation

---

### Option B: Profiler updates ConfigMap directly via kubectl

The profiler uses kubectl (already available in the container) to update a status ConfigMap.

**Profiler changes:**
```python
import subprocess

def update_profiling_status(phase: str, message: str):
    """Update status ConfigMap via kubectl."""
    status_json = json.dumps({
        "phase": phase,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })

    subprocess.run([
        "kubectl", "patch", "configmap", f"dgdr-status-{dgdr_name}",
        "-n", namespace,
        "--type=merge",
        "-p", f'{{"data":{{"status":"{status_json}"}}}}'
    ], check=True)
```

**Pros:**
- Direct update, minimal delay
- No sidecar changes needed

**Cons:**
- Requires kubectl and RBAC for ConfigMap updates
- Couples profiler to K8s API
- More error handling needed

---

### Option C: Pod annotations via Downward API + Controller watching

The profiler writes to a file that gets reflected as pod annotations via the Downward API... but this doesn't work (Downward API is read-only from pod perspective).

**Not recommended.**

---

### Option D: Structured logging + Log parsing

The operator parses profiler logs in real-time to extract phase information.

**Profiler changes:**
```python
# Emit structured log lines
logger.info("PROFILING_PHASE::DecodeInterpolation::Profiling decode with batch size 496")
```

**Operator changes:**
1. Stream pod logs via K8s API
2. Parse for `PROFILING_PHASE::` markers
3. Update DGDR conditions

**Pros:**
- No new infrastructure
- Logs are already being written

**Cons:**
- Log streaming adds complexity
- Log parsing is fragile
- Resource intensive for long-running jobs

---

## Recommended Implementation: Option A

### Changes Required

#### 1. Profiler (`benchmarks/profiler/profile_sla.py`)

Add a `ProfilingStatusReporter` class:

```python
# benchmarks/profiler/utils/status_reporter.py

import json
import os
from datetime import datetime
from enum import Enum

class ProfilingPhase(Enum):
    INITIALIZING = "Initializing"
    PREFILL_SWEEP = "PrefillSweep"
    DECODE_SWEEP = "DecodeSweep"
    CONFIG_SELECTION = "ConfigSelection"
    PREFILL_INTERPOLATION = "PrefillInterpolation"
    DECODE_INTERPOLATION = "DecodeInterpolation"
    PLANNER_GENERATION = "PlannerGeneration"
    COMPLETE = "Complete"
    FAILED = "Failed"

class ProfilingStatusReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.status_file = os.path.join(output_dir, "profiling_status.json")

    def update(self, phase: ProfilingPhase, message: str,
               current_step: int = None, total_steps: int = None,
               details: dict = None):
        status = {
            "phase": phase.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if current_step is not None and total_steps is not None:
            status["progress"] = {
                "currentStep": current_step,
                "totalSteps": total_steps,
                "percentage": round(current_step / total_steps * 100, 1)
            }
        if details:
            status["details"] = details

        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)
```

Integrate into `profile_sla.py`:

```python
from benchmarks.profiler.utils.status_reporter import ProfilingStatusReporter, ProfilingPhase

async def run_profile(args):
    status = ProfilingStatusReporter(args.output_dir)

    status.update(ProfilingPhase.INITIALIZING, "Initializing profiling job")

    # Phase 1: Prefill sweep
    status.update(ProfilingPhase.PREFILL_SWEEP,
                  f"Profiling prefill across {len(profile_num_gpus)} GPU configurations")
    for i, num_gpus in enumerate(profile_num_gpus):
        status.update(ProfilingPhase.PREFILL_SWEEP,
                      f"Profiling prefill with {num_gpus} GPUs",
                      current_step=i+1, total_steps=len(profile_num_gpus))
        # ... existing profiling code ...

    # Phase 2: Decode sweep
    status.update(ProfilingPhase.DECODE_SWEEP,
                  f"Profiling decode across {len(profile_num_gpus)} GPU configurations")
    # ... etc ...

    # Final
    status.update(ProfilingPhase.COMPLETE, "Profiling completed successfully")
```

#### 2. Sidecar Script (operator controller)

Modify the sidecar script to also watch the status file:

```bash
# In sidecar script, add status ConfigMap updates
STATUS_CM_NAME="dgdr-status-{{.DGDRName}}"

# Background job to update status
while true; do
  if [ -f {{.OutputPath}}/profiling_status.json ]; then
    # Create/update status ConfigMap
    kubectl create configmap $STATUS_CM_NAME \
      --from-file=status.json={{.OutputPath}}/profiling_status.json \
      -n {{.Namespace}} \
      --dry-run=client -o yaml | kubectl apply -f -
  fi
  sleep 10
done &
```

#### 3. DGDR Types (`api/v1alpha1/dynamographdeploymentrequest_types.go`)

Add new status field:

```go
// DynamoGraphDeploymentRequestStatus represents the observed state
type DynamoGraphDeploymentRequestStatus struct {
    // ... existing fields ...

    // ProfilingPhase indicates the current phase of profiling when State is "Profiling".
    // Possible values: Initializing, PrefillSweep, DecodeSweep, ConfigSelection,
    // PrefillInterpolation, DecodeInterpolation, PlannerGeneration, Complete, Failed
    // +kubebuilder:validation:Optional
    ProfilingPhase string `json:"profilingPhase,omitempty"`

    // ProfilingProgress provides detailed progress information for the current phase.
    // +kubebuilder:validation:Optional
    ProfilingProgress *ProfilingProgress `json:"profilingProgress,omitempty"`
}

// ProfilingProgress contains detailed progress information
type ProfilingProgress struct {
    // CurrentStep is the current step within the phase
    CurrentStep int `json:"currentStep,omitempty"`
    // TotalSteps is the total number of steps in the phase
    TotalSteps int `json:"totalSteps,omitempty"`
    // Percentage is the completion percentage (0-100)
    Percentage float64 `json:"percentage,omitempty"`
    // Message is a human-readable description of current activity
    Message string `json:"message,omitempty"`
    // LastUpdated is the timestamp of the last status update
    LastUpdated metav1.Time `json:"lastUpdated,omitempty"`
}
```

#### 4. DGDR Controller

Add ConfigMap watcher and status synchronization:

```go
// Watch status ConfigMap
Watches(
    &corev1.ConfigMap{},
    handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
        cm := obj.(*corev1.ConfigMap)
        if !strings.HasPrefix(cm.Name, "dgdr-status-") {
            return nil
        }
        dgdrName := strings.TrimPrefix(cm.Name, "dgdr-status-")
        return []ctrl.Request{{
            NamespacedName: types.NamespacedName{
                Name:      dgdrName,
                Namespace: cm.Namespace,
            },
        }}
    }),
),

// In handleProfilingState, sync status from ConfigMap
func (r *DynamoGraphDeploymentRequestReconciler) syncProfilingStatus(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) error {
    statusCM := &corev1.ConfigMap{}
    err := r.Get(ctx, types.NamespacedName{
        Name:      fmt.Sprintf("dgdr-status-%s", dgdr.Name),
        Namespace: dgdr.Namespace,
    }, statusCM)

    if apierrors.IsNotFound(err) {
        return nil // Status not yet available
    }
    if err != nil {
        return err
    }

    statusJSON, exists := statusCM.Data["status.json"]
    if !exists {
        return nil
    }

    var status struct {
        Phase    string `json:"phase"`
        Message  string `json:"message"`
        Progress struct {
            CurrentStep int     `json:"currentStep"`
            TotalSteps  int     `json:"totalSteps"`
            Percentage  float64 `json:"percentage"`
        } `json:"progress"`
    }

    if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
        return err
    }

    dgdr.Status.ProfilingPhase = status.Phase
    dgdr.Status.ProfilingProgress = &nvidiacomv1alpha1.ProfilingProgress{
        CurrentStep: status.Progress.CurrentStep,
        TotalSteps:  status.Progress.TotalSteps,
        Percentage:  status.Progress.Percentage,
        Message:     status.Message,
        LastUpdated: metav1.Now(),
    }

    // Update condition message with phase info
    meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
        Type:    ConditionTypeProfiling,
        Status:  metav1.ConditionFalse,
        Reason:  "ProfilingRunning",
        Message: fmt.Sprintf("[%s] %s", status.Phase, status.Message),
    })

    return nil
}
```

#### 5. Add kubectl print column

```go
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.profilingPhase`,priority=1
```

### Result

After implementation, users see:

```bash
$ kubectl get dgdr sla-online
NAME         MODEL             BACKEND   STATE       PHASE                  DGD-STATE   AGE
sla-online   Qwen/Qwen3-0.6B   vllm      Profiling   DecodeInterpolation                74m

$ kubectl get dgdr sla-online -o yaml
status:
  state: Profiling
  profilingPhase: DecodeInterpolation
  profilingProgress:
    currentStep: 4
    totalSteps: 6
    percentage: 66.7
    message: "Profiling decode with batch size 496"
    lastUpdated: "2026-01-24T02:08:42Z"
  conditions:
  - type: Profiling
    status: "False"
    reason: ProfilingRunning
    message: "[DecodeInterpolation] Profiling decode with batch size 496 (4/6, 66.7%)"
```

### Migration & Compatibility

- **Backward compatible**: New fields are optional, old profilers work without status reporting
- **Graceful degradation**: If status ConfigMap doesn't exist, DGDR shows "Profiling is in progress" (current behavior)

### Testing

1. Unit tests for `ProfilingStatusReporter`
2. Integration test: verify status ConfigMap creation and content
3. E2E test: verify DGDR status reflects profiler phase updates

## Alternatives Considered

| Option | Complexity | Latency | Coupling | Chosen |
|--------|------------|---------|----------|--------|
| A: File → ConfigMap | Medium | ~10s | Low | Yes |
| B: Direct kubectl | Low | <1s | High | No |
| C: Downward API | N/A | N/A | N/A | No (not possible) |
| D: Log parsing | High | Real-time | Medium | No |

## Open Questions

1. **Status update frequency**: Every phase change, or also within-phase progress?
   - Recommendation: Both - update on phase change AND every N iterations within a phase

2. **Cleanup**: When should the status ConfigMap be deleted?
   - Recommendation: Delete when DGDR transitions out of Profiling state

3. **Error reporting**: Should profiler errors be surfaced through this mechanism?
   - Recommendation: Yes, set phase to "Failed" with error message in details
