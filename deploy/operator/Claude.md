# Dynamo Operator - Claude Context

## Overview

The Dynamo Operator manages DynamoGraphDeployment (DGD) resources, which deploy ML inference graphs consisting of frontends, workers (prefill/decode), planners, and routers.

## Key Components

### Controllers
- **DynamoGraphDeployment Controller** (`internal/controller/dynamographdeployment_controller.go`): Main controller that reconciles DGD resources
- **DynamoComponentDeployment Controller** (`internal/controller/dynamocomponentdeployment_controller.go`): Manages individual component deployments (DCDs)

### Rolling Update Implementation
Located in `internal/controller/dynamographdeployment_rollout.go` and `internal/dynamo/graph.go`.

## Rolling Update Architecture

### Trigger Condition
Rolling updates are triggered when the **worker spec hash** changes. The hash is computed from worker components only (prefill, decode, worker types). Frontend-only changes do NOT trigger rolling updates.

### DCD Naming Convention
- **Frontend DCDs**: `{dgd-name}-{service-name}` (no hash suffix, stable across rollouts)
- **Worker DCDs**: `{dgd-name}-{service-name}-{workerHash[:8]}` (hash suffix for isolation)

Example: `myapp-frontend`, `myapp-decode-abc12345`

### Namespace Strategy
- **All components**: Use base namespace (`<k8s-ns>-<dgd-name>`) for `DYN_NAMESPACE`
- **Frontend**: Additionally gets `DYN_NAMESPACE_PREFIX` env var for prefix-based worker discovery
- **Workers**: Additionally get `DYN_NAMESPACE_WORKER_SUFFIX=<hash>` so new runtime composes effective namespace as `<base>-<suffix>` for discovery isolation during rolling updates
- **Backwards compatibility**: Old runtime images ignore the unknown env vars and use `DYN_NAMESPACE` directly

### Traffic Routing
The frontend uses **prefix-based discovery** via the MultiPoolManager runtime component:
- Frontend discovers all workers whose namespace starts with the base prefix
- Workers register in namespaces composed from `DYN_NAMESPACE` + `DYN_NAMESPACE_WORKER_SUFFIX`
- Automatically load balances across workers in both old and new pools during rollout
- No external proxy required - traffic shifting is handled natively by the frontend

### Resources Generated During Rolling Update

**Frontend (stable):**
- Single frontend DCD with base namespace
- Discovers workers via prefix matching

**New workers (target):**
- Worker DCDs with new hash suffix in name, labeled with `KubeLabelDynamoWorkerHash`
- Gradually scaled up: `min(desired, newReady + 1)`

**Old workers (current):**
- Existing worker DCDs with old hash suffix, identified by worker hash label
- Gradually scaled down: `max(0, desired - newReady)`
- Deleted after rollout completes (found by `KubeLabelDynamoWorkerHash` label)

### Rollout Phases
1. **None** → **Pending**: Rollout detected, status initialized
2. **Pending** → **InProgress**: Begin scaling new workers
3. **InProgress**: Monitor worker readiness, scale old down as new become ready
4. **InProgress** → **Completed**: All new workers ready, old workers scaled to 0
5. **Completed**: Cleanup old worker DCDs (by hash label), update active hash

### Key Functions

```go
// Detect if rolling update needed
shouldTriggerRollingUpdate(dgd) bool

// Build context with hashes and replica calculations
buildRolloutContext(ctx, dgd) *RolloutContext

// Generate all DCDs (all use base namespace, workers get suffix env var)
GenerateDynamoComponentsDeployments(ctx, dgd, ..., rolloutCtx) map[string]*DCD

// Orchestrate rollout state machine
reconcileRollingUpdate(ctx, dgd) error

// Clean up old worker DCDs by worker hash label
deleteOldDCDs(ctx, dgd, oldWorkerHash) error
```

### RolloutContext Structure
```go
type RolloutContext struct {
    InProgress        bool
    OldWorkerHash     string           // "abc12345" (8 chars)
    NewWorkerHash     string           // "def67890" (8 chars)
    OldWorkerReplicas map[string]int32 // per-service old replica counts
    NewWorkerReplicas map[string]int32 // per-service new replica counts
}
```

### Replica Scaling Heuristic
```
newReplicas = min(desiredReplicas, newReadyReplicas + 1)  // Always one ahead
oldReplicas = max(0, desiredReplicas - newReadyReplicas)  // Scale down as new ready
```

This ensures:
1. New deployment starts with 1 replica
2. Once ready, old decrements by 1, new increments by 1
3. Continues until new has all replicas, old has 0

## Unsupported Pathways

Rolling updates are NOT supported for:
- **Grove pathway**: Uses different orchestration (PodCliqueSets)
- **Multinode deployments**: Uses LeaderWorkerSet (LWS)

These pathways log a warning and update the hash without rolling update.

## Important Labels and Annotations

```go
// Labels
KubeLabelDynamoNamespace           = "nvidia.com/dynamo-namespace"
KubeLabelDynamoGraphDeploymentName = "nvidia.com/dynamo-graph-deployment-name"
KubeLabelDynamoComponent           = "nvidia.com/dynamo-component"
KubeLabelDynamoComponentType       = "nvidia.com/dynamo-component-type"
KubeLabelDynamoWorkerHash          = "nvidia.com/dynamo-worker-hash"  // Worker DCDs only, for rollout cleanup

// Annotations
AnnotationCurrentWorkerHash = "nvidia.com/current-worker-hash"
```

## Environment Variables

```go
// Set on all components
DynamoNamespaceEnvVar = "DYN_NAMESPACE"  // Base dynamo namespace (<k8s-ns>-<dgd-name>)

// Set on frontend only (for prefix-based discovery)
DynamoNamespacePrefixEnvVar = "DYN_NAMESPACE_PREFIX"  // Base prefix for worker discovery

// Set on workers only (for discovery isolation during rolling updates)
DynamoNamespaceWorkerSuffixEnvVar = "DYN_NAMESPACE_WORKER_SUFFIX"  // Hash suffix, new runtime composes <base>-<suffix>
```

## Testing

Controller tests are in `internal/controller/dynamographdeployment_controller_test.go` and `internal/controller/dynamographdeployment_rollout_test.go`.

Run tests:
```bash
go test ./internal/controller/... -count=1
go test ./internal/dynamo/... -count=1
```

## Common Patterns

### Checking Component Type
```go
if dynamo.IsWorkerComponent(component.ComponentType) {
    // worker, prefill, or decode
}
if component.ComponentType == consts.ComponentTypeFrontend {
    // frontend
}
```

### Computing Hashes and Namespaces
```go
hash := dynamo.ComputeWorkerSpecHash(dgd)                         // 8-char hash from worker specs
baseNamespace := dynamo.ComputeBaseDynamoNamespace(dgd)            // For all components: <k8s-ns>-<dgd-name>
```
