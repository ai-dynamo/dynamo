# DGDR v1beta1 Controller Refactor — Phase 3 Summary

## Phase: Structured Config — Eliminate the JSON Blob

**Date**: 2026-02-24
**File changed**: `internal/controller/dynamographdeploymentrequest_controller.go`

---

## Overview

Phase 3 eliminates the fragile `map[string]interface{}` JSON blob parsing that was the core
of the v1alpha1 profiling config preparation. The controller now marshals the typed v1beta1
spec directly to JSON and passes it to the profiler unchanged. This is the highest-value
change in the refactor — it removes ~200 lines of brittle key-by-key mapping code and
replaces it with `json.Marshal(dgdr.Spec)`.

---

## Items Completed

### 3.1 — `prepareProfilingConfig()` → `marshalDGDRSpec()`

**Deleted**: The entire `prepareProfilingConfig()` function (~95 lines) which:
- Parsed `dgdr.Spec.ProfilingConfig.Config.Raw` into `map[string]interface{}`
- Manually injected `deployment.namespace`, `deployment.model`, `engine.backend`
- Merged GPU discovery results key-by-key into the hardware section
- Re-serialized to YAML

**Added**: `marshalDGDRSpec()` — a 6-line function that calls `json.Marshal(dgdr.Spec)`.
The profiler receives the DGDR spec verbatim as JSON via `--config`.

**Added**: `enrichHardwareFromDiscovery()` — fills in `spec.hardware.gpuSku`, `vramMb`,
and `numGpusPerNode` from cluster GPU discovery if the user didn't set them. Called
before `marshalDGDRSpec()` so discovered values are included in the JSON. Mutates the
in-memory spec only (never persisted via status update).

### 3.2 — `extractModelCachePVCConfig()` — typed struct access

**Before**: Parsed `dgdr.Spec.ProfilingConfig.Config.Raw` → `config["deployment"]["modelCache"]["pvcName"]`  
**After**: Reads `dgdr.Spec.ModelCache.PVCName` and `dgdr.Spec.ModelCache.PVCMountPath` directly.

| Old blob path | New typed field |
|---|---|
| `config["deployment"]["modelCache"]["pvcName"]` | `dgdr.Spec.ModelCache.PVCName` |
| `config["deployment"]["modelCache"]["mountPath"]` | `dgdr.Spec.ModelCache.PVCMountPath` |

### 3.3 — `isOnlineProfiling()` — simplified

**Before**: Parsed `dgdr.Spec.ProfilingConfig.Config.Raw` → `config["sweep"]["use_ai_configurator"]`  
**After**: Always returns `true`. The profiler decides online vs AIC mode internally.
The distinction only affected the Job label (`dynamo-profiler` vs `aic-profiler`),
which is now always `dynamo-profiler`.

### 3.4 — `validateGPUHardwareInfo()` — typed struct validation

**Before**: ~80 lines parsing `config["hardware"]` and `config["engine"]` from the JSON blob,
using `toFloat64()` helper for type-unsafe numeric conversions.

**After**: ~25 lines checking `dgdr.Spec.Hardware.GPUSKU`, `.VRAMMB`, `.NumGPUsPerNode` directly.
Error messages now reference v1beta1 field paths (`spec.hardware.gpuSku`) instead of blob paths.

**Deleted**: `toFloat64()` helper — no longer needed with typed fields.

### 3.5 — Removed `ConfigKey*` constants

All 19 `ConfigKey*` constants deleted:
```
ConfigKeyDeployment, ConfigKeyModelCache, ConfigKeyPVCName, ConfigKeyPVCPath,
ConfigKeyMountPath, ConfigKeyHardware, ConfigKeyEngine, ConfigKeyOutputDir,
ConfigKeyNumGpusPerNode, ConfigKeyGPUModel, ConfigKeyGPUVramMib, ConfigKeySystem,
ConfigKeyMinNumGpusPerEng, ConfigKeyMaxNumGpusPerEng, ConfigKeyBackend,
ConfigKeyConfig, ConfigKeyNamespace, ConfigKeyModel, ConfigKeyDGDImage
```

Also removed unused volume constants: `VolumeNameProfilingConfig`, `ProfilingConfigPath`,
`ProfilingConfigFile` (these were for ConfigMapRef volume mounting, which is removed).

### 3.6 — `overrides.profilingJob` merge

Replaced direct `ProfilingConfig.{Resources, Tolerations, NodeSelector}` field access with
structured merge from `dgdr.Spec.Overrides.ProfilingJob`:

| v1alpha1 field | v1beta1 override source |
|---|---|
| `ProfilingConfig.Resources` | `Overrides.ProfilingJob.Template.Spec.Containers[0].Resources` |
| `ProfilingConfig.Tolerations` | `Overrides.ProfilingJob.Template.Spec.Tolerations` |
| `ProfilingConfig.NodeSelector` | `Overrides.ProfilingJob.Template.Spec.NodeSelector` |
| *(new)* | `Overrides.ProfilingJob.Template.Spec.ImagePullSecrets` |
| *(new)* | `Overrides.ProfilingJob.Template.Spec.ServiceAccountName` |
| *(new)* | `Overrides.ProfilingJob.BackoffLimit` |

### 3.7 — `overrides.dgd` (noted)

The `Overrides.DGD` field (`*runtime.RawExtension`) replaces the old `ConfigMapRef` pattern.
The profiler merges profiling results on top of this template. No additional controller
changes needed in this phase — the profiler reads the DGD template from the spec JSON.

### 3.8 — Profiler invocation update

| Aspect | Before (v1alpha1) | After (v1beta1) |
|---|---|---|
| Command | `python -m dynamo.profiler.profile_sla` | `python -m dynamo.profiler` |
| Args | `--profile-config <yaml_string>` | `--config <json_string>` |
| Image source | `dgdr.Spec.ProfilingConfig.ProfilerImage` | `dgdr.Spec.Image` |
| Config format | Bespoke YAML with injected keys | `json.Marshal(dgdr.Spec)` |

### `validateSpec()` — ConfigMapRef validation removed

Removed the `ProfilingConfig.ConfigMapRef` validation block (ConfigMap existence check,
key existence check). In v1beta1, the DGD base template comes from `Overrides.DGD`,
not a ConfigMap. Model cache PVC validation now reads from `dgdr.Spec.ModelCache.PVCName`.

### OutputPVC removal

The `ProfilingConfig.OutputPVC` field has no v1beta1 equivalent. Profiling output now
always uses `emptyDir` — the sidecar copies results to a ConfigMap, which is the
canonical output mechanism.

---

## Net code change

- **~200 lines deleted** (prepareProfilingConfig, blob parsing in validateGPUHardwareInfo,
  extractModelCachePVCConfig blob version, ConfigKey constants, toFloat64, ConfigMapRef
  validation, OutputPVC/ConfigMapRef volume logic)
- **~80 lines added** (marshalDGDRSpec, enrichHardwareFromDiscovery, typed
  extractModelCachePVCConfig, overrides merge, typed validateGPUHardwareInfo)
- **Net: ~120 lines removed**

## Build verification

No new compile errors introduced. Only pre-existing `DeploymentOverrides` errors remain
(11 references in DGD creation/deployment handling — will be addressed in Phase 5).

## Remaining `ProfilingConfig` references

Zero. All `dgdr.Spec.ProfilingConfig.*` references have been eliminated from the controller.

## Remaining `DeploymentOverrides` references (Phase 5)

11 references in `handleDeployingPhase`, `handleDeployedPhase`, `createDGD`, and
`handleProfilingPhase` (target namespace resolution). These will be updated in Phase 5
to use the v1beta1 deployment namespace/name/labels patterns.
