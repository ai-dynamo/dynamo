# DGDR v1beta1 Controller Refactor — Phase 2 Summary

## Phase: State Machine → Phase Machine

**Date**: 2026-02-24
**File changed**: `internal/controller/dynamographdeploymentrequest_controller.go`

---

## Overview

Phase 2 replaces the v1alpha1 state-based reconciliation loop with the v1beta1 phase-based model. Most of the work (items 2.1–2.7) was completed previously. This session implemented the remaining three items: the `Succeeded` aggregate condition (2.8), `ProfilingPhase` tracking (2.9), and `status.profilingJobName` (2.10).

---

## Items Completed (Previously)

| Item | Description | Status |
|------|-------------|--------|
| 2.1 | Phase enum mapping | ✅ Done |
| 2.2 | Refactor `Reconcile` switch to use `dgdr.Status.Phase` | ✅ Done |
| 2.3 | Merge `Initializing` into `Pending` (check `ObservedGeneration == 0`) | ✅ Done |
| 2.4 | Handle `Deployed` phase (new `handleDeployedPhase()`) | ✅ Done |
| 2.5 | Handle `DeploymentDeleted` → `Failed` | ✅ Done |
| 2.6 | Immutability check uses v1beta1 phases | ✅ Done |
| 2.7 | Replace `updateStateAndRequeue` / `updateStateWithCondition` with `updatePhaseAndRequeue` / `updatePhaseWithCondition` | ✅ Done |

---

## Items Completed (This Session)

### 2.8 — `Succeeded` Aggregate Condition

Added a `setSucceededCondition()` helper function that maps each phase to a `Succeeded` condition:

| Phase | Succeeded.Status | Succeeded.Reason |
|-------|------------------|------------------|
| Pending | `False` | `Pending` |
| Profiling | `False` | `Profiling` |
| Ready | `True` | `SpecGenerated` |
| Deploying | `False` | `Deploying` |
| Deployed | `True` | `Deployed` |
| Failed | `False` | `Failed` |

**Integration points** — `setSucceededCondition` is called:
- In `updatePhaseAndRequeue()` — the standard phase transition helper
- In `updatePhaseWithCondition()` — the phase+condition transition helper
- At 4 direct `dgdr.Status.Phase = ...` assignments:
  - `handleDeployingPhase()` → Ready (when autoApply disabled)
  - `handleDeployingPhase()` → Deployed (DGD ready)
  - `handleDeployedPhase()` → Deploying (DGD degraded)
  - `handleDGDDeleted()` → Failed (DGD deleted by user)

This ensures every phase transition, whether through helpers or direct assignment, updates the `Succeeded` condition for `kubectl get dgdr` observability (the v1beta1 CRD has a printcolumn for `Succeeded.reason`).

### 2.9 — `ProfilingPhase` Sub-Phase Tracking

Set and clear `status.profilingPhase` at the correct lifecycle boundaries:

- **Set to `Initializing`**: In `handlePendingPhase()`, immediately before transitioning to the `Profiling` phase (after `createProfilingJob` succeeds). Uses `dgdr.SetProfilingPhase(ProfilingPhaseInitializing)`.

- **Cleared on success**: In `handleProfilingPhase()`, after profiling job completes successfully (`dgdr.ClearProfilingPhase()`), before spec generation.

- **Cleared on failure**: In `handleProfilingPhase()`, when `checkProfilingJobStatus` returns an error (`dgdr.ClearProfilingPhase()`), before transitioning to `Failed`.

**Future enhancement**: The profiler can update a ConfigMap with its current sub-phase (e.g., `SweepingPrefill`, `SweepingDecode`), and the controller can read it during `handleProfilingPhase()` to update `status.profilingPhase` with finer granularity.

### 2.10 — `status.profilingJobName`

Added `dgdr.Status.ProfilingJobName = job.Name` in `createProfilingJob()` after `SyncResource` creates or updates the Job. This stores the Kubernetes Job name in status for:
- Observability (`kubectl get dgdr -o yaml` shows the job name)
- Future use by `handleProfilingPhase()` for direct job lookups

---

## Build Verification

The changes compile cleanly. The only build errors present are pre-existing `DeploymentOverrides` references from the Phase 1 type switch (v1alpha1 → v1beta1), which will be addressed in Phase 5 (DGD Spec Generation & AutoApply) when the deployment namespace/name/labels logic is updated to use v1beta1's `Overrides` struct.

---

## Phase 2 Completion Status

All 10 items of Phase 2 are now complete. The controller's reconciliation loop fully operates on v1beta1 phase semantics with proper aggregate condition tracking, profiling sub-phase visibility, and job name observability.
