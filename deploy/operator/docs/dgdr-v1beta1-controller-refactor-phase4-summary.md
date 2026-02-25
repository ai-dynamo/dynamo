# DGDR v1beta1 Controller Refactor — Phase 4 Summary

## Phase: Profiling Job Creation

**Date**: 2026-02-24
**File changed**: `internal/controller/dynamographdeploymentrequest_controller.go`

---

## Overview

Phase 4 ensures `createProfilingJob()` uses v1beta1 spec fields and supports the new
override mechanism, while maintaining backward compatibility with v1alpha1 resources
that are converted to v1beta1 via the hub conversion layer.

Most Phase 4 items were already completed as part of earlier phases. This session
implemented the two remaining items: annotation-based fallbacks for `ConfigMapRef`
(4.4) and `OutputPVC` (4.5) to support round-tripped v1alpha1 resources.

---

## Items — Status

### 4.1 — Spec access mapping (reference table)

Not code — just a reference mapping. All mappings are now implemented:

| v1alpha1 field | v1beta1 equivalent | Implemented in |
|---|---|---|
| `ProfilingConfig.ProfilerImage` | `dgdr.Spec.Image` | Phase 3 |
| `ProfilingConfig.Config.Raw` | `json.Marshal(dgdr.Spec)` | Phase 3 |
| `prepareProfilingConfig()` (150-line YAML builder) | `marshalDGDRSpec()` | Phase 3 |
| `--profile-config <yaml>` | `--config <json>` | Phase 3 |
| `python -m dynamo.profiler.profile_sla` | `python -m dynamo.profiler` | Phase 3 |
| `ProfilingConfig.Resources` | `Overrides.ProfilingJob...Resources` | Phase 3 |
| `ProfilingConfig.Tolerations` | `Overrides.ProfilingJob...Tolerations` | Phase 3 |
| `ProfilingConfig.NodeSelector` | `Overrides.ProfilingJob...NodeSelector` | Phase 3 |
| `ProfilingConfig.ConfigMapRef` | Annotation fallback (this phase) | **Phase 4** |
| `ProfilingConfig.OutputPVC` | Annotation fallback (this phase) | **Phase 4** |
| `DeploymentOverrides.WorkersImage` | `dgdr.Spec.Image` | Phase 1 fix |
| `Spec.UseMocker` | `Spec.Features.Mocker.Enabled` | Phase 1 fix |

### 4.2 — Profiler image & command — ✅ Done (Phase 3)

### 4.3 — Resource/tolerations/nodeSelector — ✅ Done (Phase 3)

### 4.4 — ConfigMapRef from annotation — ✅ Done (this session)

Added `configMapRefFromAnnotation()` helper that reads the JSON-serialized
`ConfigMapKeySelector` from annotation `nvidia.com/dgdr-config-map-ref`.

When present (round-tripped v1alpha1 resource), the controller:
- Mounts the referenced ConfigMap as a volume at `/config`
- The profiler reads the DGD base config from this mount

When absent (native v1beta1 resource), no ConfigMap is mounted — the DGD base
config comes from `overrides.dgd` in the spec JSON.

**New type**: `configMapKeySelector` struct (controller-local, mirrors v1alpha1 type)

### 4.5 — OutputPVC from annotation — ✅ Done (this session)

Added `outputPVCFromAnnotation()` helper that reads the PVC name from annotation
`nvidia.com/dgdr-output-pvc`.

When present (round-tripped v1alpha1 resource), profiling output uses the PVC.
When absent (native v1beta1 resource), profiling output uses `emptyDir` — the
sidecar copies results to a ConfigMap, which is the canonical output mechanism.

### 4.6 — Store job name in status — ✅ Done (Phase 2)

### 4.7 — UseMocker source — ✅ Done (Phase 1 fix)

---

## Constants added

| Constant | Value | Purpose |
|---|---|---|
| `AnnotationConfigMapRef` | `nvidia.com/dgdr-config-map-ref` | v1alpha1 round-trip annotation for ConfigMap ref |
| `AnnotationOutputPVC` | `nvidia.com/dgdr-output-pvc` | v1alpha1 round-trip annotation for output PVC |
| `VolumeNameProfilingConfig` | `profiling-config` | Volume name for ConfigMap mount (re-added) |
| `ProfilingConfigMountPath` | `/config` | Mount path for ConfigMap volume |
| `ProfilingConfigDefaultKey` | `disagg.yaml` | Default key in ConfigMap |

---

## Build Status

- **`go build ./...`**: ✅ Clean
- **`go vet ./...`**: ✅ Controller clean (1 pre-existing test file error, Phase 7 scope)
