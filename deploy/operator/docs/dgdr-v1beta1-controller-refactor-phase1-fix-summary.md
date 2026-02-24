# DGDR v1beta1 Controller Refactor — Phase 1 Fix Summary

## Phase: Switch Import & Core Types — Compilation Fix

**Date**: 2026-02-24
**File changed**: `internal/controller/dynamographdeploymentrequest_controller.go`

---

## Problem

Phase 1's stated goal was "Make the controller **compile** against v1beta1 types" (mechanical
changes only). However, the initial Phase 1 implementation switched the DGDR type from
v1alpha1 to v1beta1 without updating all field access patterns, leaving the controller in
a non-compiling state. Three categories of broken references remained:

1. **`DeploymentOverrides`** (11 references) — v1alpha1 had `Spec.DeploymentOverrides` with
   `.Name`, `.Namespace`, `.Labels`, `.Annotations`, `.WorkersImage` fields. v1beta1 removed
   this struct entirely.
2. **`UseMocker`** (1 reference) — v1alpha1 had `Spec.UseMocker` (bool). v1beta1 moved this
   to `Spec.Features.Mocker.Enabled`.
3. **`yaml.Marshal`** (1 reference) — The `k8s.io/apimachinery/pkg/util/yaml` package only
   provides decode functions, not `Marshal`. Should use `sigs.k8s.io/yaml` (`sigsyaml`).

---

## Fixes Applied

### `DeploymentOverrides` → defaults (11 references removed)

In v1beta1, there is no `DeploymentOverrides` struct. The DGD defaults to:
- **Name**: from the generated DGD (profiler output)
- **Namespace**: same as the DGDR namespace

Phase 5 may add namespace/name override support via `overrides.dgd` if needed.

| Location | v1alpha1 pattern | v1beta1 fix |
|---|---|---|
| `handleProfilingPhase()` — additional resources namespace | `DeploymentOverrides.Namespace` | `dgdr.Namespace` |
| `handleDeployingPhase()` — DGD lookup namespace | `DeploymentOverrides.Namespace` | `dgdr.Namespace` |
| `handleDeployedPhase()` — DGD lookup namespace | `DeploymentOverrides.Namespace` | `dgdr.Namespace` |
| `createDGD()` — DGD name override | `DeploymentOverrides.Name` | Use generated name |
| `createDGD()` — DGD namespace override | `DeploymentOverrides.Namespace` | `dgdr.Namespace` |
| `createDGD()` — custom labels merge | `DeploymentOverrides.Labels` | Removed (only managed labels) |
| `createDGD()` — custom annotations merge | `DeploymentOverrides.Annotations` | Removed (only generated annotations) |

Each fix includes a `// Phase 1 fix:` comment in the code explaining the change and
noting that Phase 5 may refine the pattern.

### `UseMocker` → `Features.Mocker.Enabled` (1 reference)

| Location | v1alpha1 | v1beta1 |
|---|---|---|
| `generateDGDSpec()` — output file selection | `dgdr.Spec.UseMocker` | `dgdr.Spec.Features != nil && dgdr.Spec.Features.Mocker != nil && dgdr.Spec.Features.Mocker.Enabled` |

### `yaml.Marshal` → `sigsyaml.Marshal` (1 reference)

| Location | Before | After |
|---|---|---|
| `generateDGDSpec()` — serialize DGD to annotation | `yaml.Marshal(dgd)` | `sigsyaml.Marshal(dgd)` |

---

## Build Status

- **`go build ./...`**: ✅ Clean (zero errors)
- **`go vet ./...`**: ✅ Controller clean (1 pre-existing test file error in Phase 7 scope)

## Remaining test file issue

`dynamographdeploymentrequest_controller_test.go:114` references `nvidiacomv1beta1.ProfilingConfigSpec`
which doesn't exist in v1beta1. This is a Phase 7 (Tests) item — the test fixtures need to be
rewritten to use v1beta1 spec patterns.
