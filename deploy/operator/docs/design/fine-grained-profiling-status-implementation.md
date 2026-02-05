# Design: Fine-Grained Profiling Status for DGDR

## Problem Statement

When a `DynamoGraphDeploymentRequest` (DGDR) is in the `Profiling` state, users have no visibility into which phase of profiling is currently executing. The profiler runs 6-7 distinct phases that can take 60-120+ minutes total, but the DGDR status only shows:

```yaml
status:
  state: Profiling
  conditions:
  - type: Profiling
    status: "False"
    reason: ProfilingRunning
    message: Profiling is in progress
```

Users must manually `kubectl logs` the profiler pod to understand progress, which is a poor user experience for a potentially 2-hour operation.

## Goals

1. **Phase visibility**: Users can see which of the 7 profiling phases is currently executing
2. **Progress tracking**: Users can see step progress within phases (e.g., "2/4 GPU configs tested")
3. **Time estimation**: Provide enough information for users to estimate remaining time
4. **Low latency**: Status updates should appear within ~10-15 seconds of phase changes
5. **Backward compatible**: Existing DGDRs and profilers should continue to work

## Non-Goals

- Real-time log streaming to DGDR status
- Detailed performance metrics in status
- Cancellation or pause functionality (separate feature)

## Profiling Phases

Based on `benchmarks/profiler/profile_sla.py`, these are the distinct phases:

| # | Phase ID | Description | Typical Duration | Progress Trackable |
|---|----------|-------------|------------------|-------------------|
| 1 | `Initializing` | Loading config, validating inputs | <1 min | No |
| 2 | `PrefillSweep` | Profile prefill across GPU counts | 5-15 min | Yes (GPU configs) |
| 3 | `DecodeSweep` | Profile decode across GPU counts + batch sizes | 15-30 min | Yes (GPU configs x batch sizes) |
| 4 | `ConfigSelection` | Analyze results, select best configs | <1 min | No |
| 5 | `PrefillInterpolation` | Build ISL→TTFT curve for selected config | 5-10 min | Yes (ISL values) |
| 6 | `DecodeInterpolation` | Build batch size curves for selected config | 15-30 min | Yes (batch sizes) |
| 7 | `PlannerGeneration` | Run SLA planner, generate DGD config | <1 min | No |
| 8 | `Complete` | Profiling finished successfully | - | - |
| 9 | `Failed` | Profiling failed with error | - | - |

**Key insight**: Phases 2, 3, 5, and 6 have nested loops where we can track meaningful progress. Each iteration involves deploying a temporary DGD, running benchmarks, and cleaning up.

## Architecture

### Current State

```
┌─────────────────┐     creates      ┌─────────────────────────────┐
│      DGDR       │ ──────────────► │      Profiling Job          │
│   Controller    │                  │  ┌─────────┐  ┌───────────┐ │
└─────────────────┘                  │  │profiler │  │  sidecar  │ │
        │                            │  │container│  │ container │ │
        │ watches Job status         │  └─────────┘  └───────────┘ │
        │ (complete/failed only)     └─────────────────────────────┘
        ▼
┌─────────────────┐
│  DGDR Status    │
│  (coarse)       │
└─────────────────┘
```

### Proposed Architecture

```
┌─────────────────┐                  ┌─────────────────────────────┐
│      DGDR       │ ◄─── watches ─── │      Profiling Job          │
│   Controller    │    ConfigMap     │  ┌─────────┐  ┌───────────┐ │
└─────────────────┘                  │  │profiler │  │  sidecar  │ │
        │                            │  │container│──│ container │ │
        │                            │  └────┬────┘  └─────┬─────┘ │
        ▼                            │       │             │       │
┌─────────────────┐                  │       │  writes     │       │
│  DGDR Status    │                  │       ▼             │       │
│  (fine-grained) │                  │  ┌──────────┐       │       │
│                 │                  │  │status.json│ ─────┘       │
│ profilingPhase  │                  │  └──────────┘  reads &      │
│ profilingProgress│                 │                updates CM   │
└─────────────────┘                  └─────────────────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ dgdr-status-*   │
                                     │   ConfigMap     │
                                     └─────────────────┘
```

**Data Flow**:
1. Profiler writes status to `/data/profiling_status.json` at phase boundaries and within loops
2. Sidecar watches the file and updates a ConfigMap (`dgdr-status-{name}`) every 10 seconds
3. Controller watches the status ConfigMap and syncs to DGDR status fields
4. Users see phase and progress via `kubectl get dgdr`

## Detailed Design

### 1. Status File Format

The profiler writes a JSON file with the following structure:

```json
{
  "phase": "DecodeSweep",
  "phaseNumber": 3,
  "totalPhases": 7,
  "message": "Profiling decode with 4 GPUs, TP=2/DP=2",
  "timestamp": "2026-01-23T14:30:00Z",
  "progress": {
    "currentStep": 5,
    "totalSteps": 12,
    "percentage": 41.7,
    "subPhase": "Benchmarking batch size 256"
  },
  "startedAt": "2026-01-23T14:00:00Z",
  "gpuConfig": {
    "currentGpus": 4,
    "parallelMapping": "TP=2/DP=2"
  }
}
```

### 2. Python Status Reporter (`benchmarks/profiler/utils/status_reporter.py`)

```python
import json
import os
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional


class ProfilingPhase(Enum):
    INITIALIZING = ("Initializing", 1)
    PREFILL_SWEEP = ("PrefillSweep", 2)
    DECODE_SWEEP = ("DecodeSweep", 3)
    CONFIG_SELECTION = ("ConfigSelection", 4)
    PREFILL_INTERPOLATION = ("PrefillInterpolation", 5)
    DECODE_INTERPOLATION = ("DecodeInterpolation", 6)
    PLANNER_GENERATION = ("PlannerGeneration", 7)
    COMPLETE = ("Complete", 8)
    FAILED = ("Failed", 9)

    def __init__(self, phase_id: str, phase_number: int):
        self.phase_id = phase_id
        self.phase_number = phase_number


@dataclass
class ProfilingProgress:
    current_step: int
    total_steps: int
    sub_phase: Optional[str] = None

    @property
    def percentage(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return round(self.current_step / self.total_steps * 100, 1)


@dataclass
class GpuConfig:
    current_gpus: int
    parallel_mapping: str


class ProfilingStatusReporter:
    """Reports profiling status to a JSON file for the sidecar to pick up."""

    TOTAL_PHASES = 7  # Excluding Complete/Failed

    def __init__(self, output_dir: str, enabled: bool = True):
        self.output_dir = output_dir
        self.status_file = os.path.join(output_dir, "profiling_status.json")
        self.enabled = enabled
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._current_phase: Optional[ProfilingPhase] = None

    def update(
        self,
        phase: ProfilingPhase,
        message: str,
        progress: Optional[ProfilingProgress] = None,
        gpu_config: Optional[GpuConfig] = None,
    ) -> None:
        """Write current status to the status file."""
        if not self.enabled:
            return

        self._current_phase = phase

        status = {
            "phase": phase.phase_id,
            "phaseNumber": phase.phase_number,
            "totalPhases": self.TOTAL_PHASES,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "startedAt": self.started_at,
        }

        if progress:
            status["progress"] = {
                "currentStep": progress.current_step,
                "totalSteps": progress.total_steps,
                "percentage": progress.percentage,
            }
            if progress.sub_phase:
                status["progress"]["subPhase"] = progress.sub_phase

        if gpu_config:
            status["gpuConfig"] = {
                "currentGpus": gpu_config.current_gpus,
                "parallelMapping": gpu_config.parallel_mapping,
            }

        # Atomic write: write to temp file then rename
        temp_file = self.status_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(status, f, indent=2)
        os.rename(temp_file, self.status_file)

    def complete(self, message: str = "Profiling completed successfully") -> None:
        """Mark profiling as complete."""
        self.update(ProfilingPhase.COMPLETE, message)

    def fail(self, error: str) -> None:
        """Mark profiling as failed."""
        self.update(ProfilingPhase.FAILED, f"Profiling failed: {error}")
```

### 3. Integration into `profile_sla.py`

Key integration points with status reporting calls:

```python
# At the start of run_profile()
status = ProfilingStatusReporter(args.output_dir)
status.update(ProfilingPhase.INITIALIZING, "Loading configuration and validating inputs")

# Phase 1: Prefill Sweep (around line 224)
total_prefill_configs = sum(
    len(get_candidate_parallel_mappings(n, args.model_info, EngineType.PREFILL))
    for n in profile_num_gpus
)
prefill_config_idx = 0

for num_gpus in profile_num_gpus:
    candidate_mappings = get_candidate_parallel_mappings(...)
    for mapping in candidate_mappings:
        prefill_config_idx += 1
        status.update(
            ProfilingPhase.PREFILL_SWEEP,
            f"Profiling prefill with {num_gpus} GPUs ({mapping.label()})",
            progress=ProfilingProgress(
                current_step=prefill_config_idx,
                total_steps=total_prefill_configs,
                sub_phase="Deploying" | "Benchmarking" | "Cleanup"
            ),
            gpu_config=GpuConfig(num_gpus, mapping.label())
        )
        # ... existing profiling code ...

# Phase 2: Decode Sweep (around line 332)
# Similar pattern with nested loop tracking

# Phase 3: Config Selection (around line 487)
status.update(ProfilingPhase.CONFIG_SELECTION, "Analyzing results and selecting optimal configurations")

# Phase 4: Prefill Interpolation (around line 561)
status.update(
    ProfilingPhase.PREFILL_INTERPOLATION,
    f"Building TTFT curve for {best_prefill_gpus} GPU config",
    progress=ProfilingProgress(current_step=i, total_steps=len(isl_values)),
    gpu_config=GpuConfig(best_prefill_gpus, best_prefill_mapping.label())
)

# Phase 5: Decode Interpolation (around line 647)
status.update(
    ProfilingPhase.DECODE_INTERPOLATION,
    f"Building throughput curves for {best_decode_gpus} GPU config",
    progress=ProfilingProgress(current_step=i, total_steps=len(batch_sizes)),
    gpu_config=GpuConfig(best_decode_gpus, best_decode_mapping.label())
)

# Phase 6: Planner Generation (around line 735)
status.update(ProfilingPhase.PLANNER_GENERATION, "Running SLA planner and generating deployment config")

# At the end
status.complete()

# In exception handler
except Exception as e:
    status.fail(str(e))
    raise
```

### 4. Sidecar Script Enhancement

Modify the sidecar script template to also watch and sync status:

```bash
# In sidecarScriptTemplate, add status syncing loop

STATUS_CM_NAME="dgdr-status-{{.DGDRName}}"
STATUS_FILE="{{.OutputPath}}/profiling_status.json"
LAST_STATUS_HASH=""

# Function to update status ConfigMap
update_status_cm() {
  if [ -f "$STATUS_FILE" ]; then
    CURRENT_HASH=$(md5sum "$STATUS_FILE" | cut -d' ' -f1)
    if [ "$CURRENT_HASH" != "$LAST_STATUS_HASH" ]; then
      kubectl create configmap "$STATUS_CM_NAME" \
        --from-file=status.json="$STATUS_FILE" \
        -n {{.Namespace}} \
        --dry-run=client -o yaml | kubectl apply -f -
      LAST_STATUS_HASH="$CURRENT_HASH"
      echo "Updated status ConfigMap"
    fi
  fi
}

# Background status sync loop (runs every 10 seconds)
(
  while true; do
    update_status_cm
    sleep 10
  done
) &
STATUS_SYNC_PID=$!

# ... existing sidecar logic for waiting for profiler completion ...

# Cleanup status sync on exit
trap "kill $STATUS_SYNC_PID 2>/dev/null" EXIT
```

### 5. DGDR API Types Update

Add new status fields to `api/v1alpha1/dynamographdeploymentrequest_types.go`:

```go
// ProfilingProgress contains detailed progress information for profiling
type ProfilingProgress struct {
    // Phase is the current profiling phase
    // +kubebuilder:validation:Enum=Initializing;PrefillSweep;DecodeSweep;ConfigSelection;PrefillInterpolation;DecodeInterpolation;PlannerGeneration;Complete;Failed
    Phase string `json:"phase,omitempty"`

    // PhaseNumber is the current phase number (1-7, or 8 for Complete, 9 for Failed)
    PhaseNumber int `json:"phaseNumber,omitempty"`

    // TotalPhases is the total number of main phases (7)
    TotalPhases int `json:"totalPhases,omitempty"`

    // Message is a human-readable description of current activity
    Message string `json:"message,omitempty"`

    // CurrentStep is the current step within the phase (for phases with loops)
    // +optional
    CurrentStep int `json:"currentStep,omitempty"`

    // TotalSteps is the total number of steps in the current phase
    // +optional
    TotalSteps int `json:"totalSteps,omitempty"`

    // Percentage is the completion percentage within the current phase (0-100)
    // +optional
    Percentage float64 `json:"percentage,omitempty"`

    // SubPhase provides additional detail about what's happening within a step
    // +optional
    SubPhase string `json:"subPhase,omitempty"`

    // StartedAt is when profiling started
    // +optional
    StartedAt *metav1.Time `json:"startedAt,omitempty"`

    // LastUpdated is the timestamp of the last status update from the profiler
    // +optional
    LastUpdated *metav1.Time `json:"lastUpdated,omitempty"`

    // GpuConfig describes the current GPU configuration being profiled
    // +optional
    GpuConfig *GpuConfigStatus `json:"gpuConfig,omitempty"`
}

// GpuConfigStatus describes the GPU configuration being profiled
type GpuConfigStatus struct {
    // CurrentGpus is the number of GPUs in the current configuration
    CurrentGpus int `json:"currentGpus,omitempty"`

    // ParallelMapping is the parallelization strategy (e.g., "TP=2/DP=2")
    ParallelMapping string `json:"parallelMapping,omitempty"`
}

// In DynamoGraphDeploymentRequestStatus, add:
type DynamoGraphDeploymentRequestStatus struct {
    // ... existing fields ...

    // ProfilingProgress provides detailed progress information when State is "Profiling".
    // This field is updated periodically based on profiler status reports.
    // +optional
    ProfilingProgress *ProfilingProgress `json:"profilingProgress,omitempty"`
}
```

### 6. Controller Updates

Add ConfigMap watching and status sync:

```go
// Add to SetupWithManager - watch status ConfigMaps
Watches(
    &corev1.ConfigMap{},
    handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
        cm := obj.(*corev1.ConfigMap)
        // Only watch status ConfigMaps
        if !strings.HasPrefix(cm.Name, "dgdr-status-") {
            return nil
        }
        dgdrName := strings.TrimPrefix(cm.Name, "dgdr-status-")
        // Check if DGDR exists in same namespace
        return []ctrl.Request{{
            NamespacedName: types.NamespacedName{
                Name:      dgdrName,
                Namespace: cm.Namespace,
            },
        }}
    }),
    builder.WithPredicates(predicate.Funcs{
        CreateFunc:  func(ce event.CreateEvent) bool { return true },
        DeleteFunc:  func(de event.DeleteEvent) bool { return false },
        UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
        GenericFunc: func(ge event.GenericEvent) bool { return true },
    }),
),

// Add to handleProfilingState, before checking job status:
if err := r.syncProfilingProgress(ctx, dgdr); err != nil {
    logger.Error(err, "Failed to sync profiling progress (non-fatal)")
    // Continue - this is best-effort status sync
}

// New function to sync status from ConfigMap
func (r *DynamoGraphDeploymentRequestReconciler) syncProfilingProgress(
    ctx context.Context,
    dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest,
) error {
    logger := log.FromContext(ctx)

    statusCMName := fmt.Sprintf("dgdr-status-%s", dgdr.Name)
    cm := &corev1.ConfigMap{}
    err := r.Get(ctx, types.NamespacedName{
        Name:      statusCMName,
        Namespace: dgdr.Namespace,
    }, cm)

    if apierrors.IsNotFound(err) {
        // Status not yet available - profiler may not have started writing
        return nil
    }
    if err != nil {
        return err
    }

    statusJSON, exists := cm.Data["status.json"]
    if !exists {
        return nil
    }

    // Parse status JSON
    var status struct {
        Phase       string  `json:"phase"`
        PhaseNumber int     `json:"phaseNumber"`
        TotalPhases int     `json:"totalPhases"`
        Message     string  `json:"message"`
        Timestamp   string  `json:"timestamp"`
        StartedAt   string  `json:"startedAt"`
        Progress    *struct {
            CurrentStep int     `json:"currentStep"`
            TotalSteps  int     `json:"totalSteps"`
            Percentage  float64 `json:"percentage"`
            SubPhase    string  `json:"subPhase"`
        } `json:"progress,omitempty"`
        GpuConfig *struct {
            CurrentGpus     int    `json:"currentGpus"`
            ParallelMapping string `json:"parallelMapping"`
        } `json:"gpuConfig,omitempty"`
    }

    if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
        logger.Error(err, "Failed to parse profiling status JSON")
        return nil // Non-fatal
    }

    // Update DGDR status
    progress := &nvidiacomv1alpha1.ProfilingProgress{
        Phase:       status.Phase,
        PhaseNumber: status.PhaseNumber,
        TotalPhases: status.TotalPhases,
        Message:     status.Message,
    }

    if status.Progress != nil {
        progress.CurrentStep = status.Progress.CurrentStep
        progress.TotalSteps = status.Progress.TotalSteps
        progress.Percentage = status.Progress.Percentage
        progress.SubPhase = status.Progress.SubPhase
    }

    if t, err := time.Parse(time.RFC3339, status.Timestamp); err == nil {
        progress.LastUpdated = &metav1.Time{Time: t}
    }

    if t, err := time.Parse(time.RFC3339, status.StartedAt); err == nil {
        progress.StartedAt = &metav1.Time{Time: t}
    }

    if status.GpuConfig != nil {
        progress.GpuConfig = &nvidiacomv1alpha1.GpuConfigStatus{
            CurrentGpus:     status.GpuConfig.CurrentGpus,
            ParallelMapping: status.GpuConfig.ParallelMapping,
        }
    }

    dgdr.Status.ProfilingProgress = progress

    // Update condition message with phase info
    meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
        Type:    ConditionTypeProfiling,
        Status:  metav1.ConditionFalse,
        Reason:  "ProfilingRunning",
        Message: fmt.Sprintf("[Phase %d/%d: %s] %s",
            status.PhaseNumber, status.TotalPhases, status.Phase, status.Message),
    })

    return r.Status().Update(ctx, dgdr)
}

// Cleanup status ConfigMap when profiling completes (add to handleProfilingState after completion)
func (r *DynamoGraphDeploymentRequestReconciler) cleanupStatusConfigMap(
    ctx context.Context,
    dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest,
) {
    logger := log.FromContext(ctx)
    statusCMName := fmt.Sprintf("dgdr-status-%s", dgdr.Name)

    cm := &corev1.ConfigMap{}
    err := r.Get(ctx, types.NamespacedName{
        Name:      statusCMName,
        Namespace: dgdr.Namespace,
    }, cm)

    if err == nil {
        if err := r.Delete(ctx, cm); err != nil && !apierrors.IsNotFound(err) {
            logger.Error(err, "Failed to cleanup status ConfigMap", "name", statusCMName)
        }
    }
}
```

### 7. kubectl Print Column

Add a print column for phase visibility:

```go
// In DynamoGraphDeploymentRequest type annotation:
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.profilingProgress.phase`,priority=1
// +kubebuilder:printcolumn:name="Progress",type=string,JSONPath=`.status.profilingProgress.message`,priority=1
```

## User Experience

### Before (Current State)

```bash
$ kubectl get dgdr my-deployment
NAME            MODEL             BACKEND   STATE       AGE
my-deployment   Qwen/Qwen3-0.6B   vllm      Profiling   74m

$ kubectl describe dgdr my-deployment
# ... lots of yaml ...
Status:
  State: Profiling
  Conditions:
    - Type: Profiling
      Status: False
      Reason: ProfilingRunning
      Message: Profiling is in progress   # <-- No useful info!
```

### After (With Fine-Grained Status)

```bash
$ kubectl get dgdr my-deployment
NAME            MODEL             BACKEND   STATE       PHASE                  AGE
my-deployment   Qwen/Qwen3-0.6B   vllm      Profiling   DecodeInterpolation    74m

$ kubectl get dgdr my-deployment -o wide
NAME            MODEL             BACKEND   STATE       PHASE                  PROGRESS                                          AGE
my-deployment   Qwen/Qwen3-0.6B   vllm      Profiling   DecodeInterpolation    Profiling decode (batch 496, 4/6 complete)        74m

$ kubectl describe dgdr my-deployment
# ...
Status:
  State: Profiling
  Profiling Progress:
    Phase: DecodeInterpolation
    Phase Number: 6
    Total Phases: 7
    Message: Profiling decode with batch size 496
    Current Step: 4
    Total Steps: 6
    Percentage: 66.7
    Sub Phase: Running AIPerf benchmark
    Started At: 2026-01-23T13:00:00Z
    Last Updated: 2026-01-23T14:14:42Z
    Gpu Config:
      Current Gpus: 2
      Parallel Mapping: TP=2/DP=1
  Conditions:
    - Type: Profiling
      Status: False
      Reason: ProfilingRunning
      Message: "[Phase 6/7: DecodeInterpolation] Profiling decode with batch size 496 (4/6, 66.7%)"
```

### JSON Output for Scripting

```bash
$ kubectl get dgdr my-deployment -o jsonpath='{.status.profilingProgress}'
{
  "phase": "DecodeInterpolation",
  "phaseNumber": 6,
  "totalPhases": 7,
  "message": "Profiling decode with batch size 496",
  "currentStep": 4,
  "totalSteps": 6,
  "percentage": 66.7,
  "startedAt": "2026-01-23T13:00:00Z",
  "lastUpdated": "2026-01-23T14:14:42Z"
}
```

## Implementation Plan

### Phase 1: Python Status Reporter (Week 1)

1. Create `benchmarks/profiler/utils/status_reporter.py`
2. Integrate into `profile_sla.py` at all phase boundaries
3. Add sub-phase tracking within loops (Deploying/Benchmarking/Cleanup)
4. Unit tests for status reporter

### Phase 2: Sidecar Enhancement (Week 1)

1. Update sidecar script template to sync status file to ConfigMap
2. Test ConfigMap creation/update in isolation
3. Ensure atomic updates and proper error handling

### Phase 3: API Types & Controller (Week 2)

1. Add `ProfilingProgress` type to DGDR API
2. Run `make generate manifests` to update CRDs
3. Add ConfigMap watcher to controller
4. Implement `syncProfilingProgress()` function
5. Add cleanup logic for status ConfigMap
6. Integration tests

### Phase 4: Polish & Documentation (Week 2)

1. Add kubectl print columns for better UX
2. Update DGDR documentation
3. Add examples showing status output
4. E2E test with real profiling run

## Testing Strategy

### Unit Tests

1. `ProfilingStatusReporter` writes correct JSON format
2. Status file atomic write (no partial reads)
3. Controller parses status JSON correctly
4. Status sync handles missing/malformed ConfigMaps gracefully

### Integration Tests

1. Sidecar correctly syncs file changes to ConfigMap
2. Controller updates DGDR status from ConfigMap
3. Status ConfigMap cleaned up after profiling completes
4. Multiple DGDRs don't interfere with each other's status

### E2E Tests

1. Full profiling run with status updates visible throughout
2. Status updates continue after controller restart
3. `kubectl get dgdr` shows phase and progress columns

## Migration & Compatibility

### Backward Compatibility

- **Old profilers (no status file)**: Controller sees no status ConfigMap, shows generic "Profiling is in progress" message (current behavior)
- **New profilers, old controller**: Status file is written but ignored; no impact
- **New DGDR API with old DGDRs**: `profilingProgress` field is optional and nil

### Upgrade Path

1. Deploy new CRDs (adds `profilingProgress` field)
2. Deploy new controller (starts watching status ConfigMaps)
3. Deploy new profiler image (starts writing status files)
4. Existing DGDRs in Profiling state will start showing progress on next profiler run

## Alternatives Considered

### Option A: Direct kubectl from Profiler

Profiler uses kubectl to update ConfigMap directly.

**Pros**: Lower latency (<1s), no sidecar changes
**Cons**: Couples profiler to K8s API, requires more RBAC

### Option B: Log Parsing

Controller streams and parses profiler logs for status markers.

**Pros**: No new files or ConfigMaps
**Cons**: Complex, fragile, resource-intensive for long-running jobs

### Option C: gRPC/HTTP Endpoint

Profiler exposes status via API, controller polls it.

**Pros**: Real-time updates
**Cons**: Complex, requires networking setup, pod-to-pod communication

**Chosen: File + ConfigMap approach** - Simple, decoupled, reliable, uses existing patterns.

## Open Questions

1. **Update frequency**: Currently 10 seconds. Should this be configurable?
   - Recommendation: Fixed at 10s, adequate for UX without excessive API calls

2. **Status retention after completion**: Keep or delete status ConfigMap?
   - Recommendation: Delete after transition from Profiling state to reduce clutter

3. **Error details**: How much error context to include in Failed status?
   - Recommendation: First 500 chars of error message in status; full logs available in pod

4. **Estimated time remaining**: Should we estimate completion time?
   - Recommendation: Future enhancement; requires historical data and is model-dependent
