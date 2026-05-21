# DGDR Shape (v1beta1)

Annotated field reference for `DynamoGraphDeploymentRequest` v1beta1, the
deploy-by-intent Custom Resource served by the Dynamo operator under group
`nvidia.com` (per `citations.md`).

Source files:

- CRD: `deploy/operator/config/crd/bases/nvidia.com_dynamographdeploymentrequests.yaml`
- Go types: `deploy/operator/api/v1beta1/dynamographdeploymentrequest_types.go`
- Narrative: `docs/kubernetes/dgdr.md`
- Canonical sample: `deploy/operator/config/samples/nvidia.com_v1beta1_dynamographdeploymentrequest.yaml`

All paths are relative to the Dynamo source on the release branch the skill
targets. Re-derive against `origin/release/<X.Y.Z>` per `SKILL_AUTHORING.md`
§11 when bumping the skill `version`.

---

## 1. Resource Identifiers

| Field | Value | Citation |
|---|---|---|
| `apiVersion` | `nvidia.com/v1beta1` | |
| `kind` | `DynamoGraphDeploymentRequest` | |
| Plural | `dynamographdeploymentrequests` | |
| Singular | `dynamographdeploymentrequest` | |
| Short name | `dgdr` | |
| Scope | `Namespaced` | |

`v1alpha1` is also served and is marked `deprecated: true` with a
`deprecationWarning` directing new manifests at `v1beta1` (per). The
conversion webhook bridges the two; existing v1alpha1 manifests continue
to work transparently.

---

## 2. Top-Level Spec Fields

The v1beta1 DGDR spec has exactly one required field — `model` — and
accepts the following top-level fields (per):

| Field | Required | Type | Default | Purpose |
|---|---|---|---|---|
| `model` | Yes | string | — | HuggingFace model ID or local path (e.g. `Qwen/Qwen3-0.6B`) |
| `image` | No | string | — | Container image for the profiling Job. Dynamo >= 1.1.0 uses `dynamo-planner`; earlier versions used `dynamo-frontend` (per). |
| `backend` | No | enum | `auto` | Inference engine choice (see §3) |
| `searchStrategy` | No | enum | `rapid` | Profiling depth (see §4) |
| `autoApply` | No | bool | `true` | Auto-create the DGD when profiling finishes |
| `hardware` | No | object | auto-detect | GPU SKU, count, layout (see §5) |
| `workload` | No | object | see §6 | Expected ISL / OSL / concurrency |
| `sla` | No | object | — | TTFT / ITL / e2eLatency targets (see §7) |
| `modelCache` | No | object | — | Pre-cached weights via PVC (see §8) |
| `features` | No | object | — | Optional features: planner, mocker |
| `overrides` | No | object | — | Customisations applied to the generated DGD |

---

## 3. `backend`

Enum, default `auto`, per:

```
auto, sglang, trtllm, vllm
```

| Value | Behavior |
|---|---|
| `auto` | Operator picks the engine based on model family (default). |
| `vllm` | Pin to vLLM. |
| `sglang` | Pin to SGLang. |
| `trtllm` | Pin to TensorRT-LLM. |

Per-release container tags for each backend live in
`container/context.yaml` on the target release branch. See
`SKILL_AUTHORING.md` §4.

---

## 4. `searchStrategy`

Enum, default `rapid`, per:

| Value | Cost | Behavior |
|---|---|---|
| `rapid` | ~30 s | Uses the AIC simulator to sweep configurations without consuming real GPUs. Default. |
| `thorough` | 2-4 h | Runs real-GPU sweeps over the prefill and decode parallelism space. Reserves the GPUs in `hardware` for the duration. |

The current 1.2.0 RC tracker flagged that the `rapid` mode does not emit
the `planner-profile-data` ConfigMap (NVBug 6189270 / DYN-3063); skills
using `rapid` for SLA-bounded production should verify the resulting DGD
has planner integration before relying on it.

---

## 5. `hardware`

| Field | Type | Default | Notes |
|---|---|---|---|
| `gpuSku` | enum (see §5.1) | auto-detected | Required in namespace-restricted operator installs (operator cannot enumerate nodes). |
| `vramMb` | int | auto-detected | GPU VRAM in MiB. |
| `totalGpus` | int | auto-detected (capped at 32) | Total GPUs available across the cluster. |
| `numGpusPerNode` | int | auto-detected | GPUs per node. |
| `interconnect` | string | auto-detected | E.g. `nvlink`, `pcie`, `roce`, `infiniband`. |
| `rdma` | bool | auto-detected | Whether RDMA is available between worker nodes. |

In cluster-scoped operator installs, all six fields are auto-detected
from GPU Feature Discovery node labels. In namespace-restricted installs
the operator does not have `list nodes` RBAC; `gpuSku`, `vramMb`, and
`numGpusPerNode` must be set explicitly or validation fails.

### 5.1 `hardware.gpuSku` Enum

All 16 supported values (per):

```
Blackwell:  gb200_sxm, b200_sxm
Hopper:     h200_sxm, h100_sxm, h100_pcie
Ampere:     a100_sxm, a100_pcie, a30
Ada:        l40s, l40, l4
Older:      v100_sxm, v100_pcie, t4
AMD:        mi200, mi300
```

Use lowercase-underscore exactly. The strings `H100-SXM5-80GB`,
`A100-SXM4-80GB`, etc. (vendor-style identifiers) are rejected by the
schema.

**PCIe caveat (per).** `h100_pcie`, `a100_pcie`, and `v100_pcie`
are admitted by the CRD but the AIC profiler does not ship training data
for them. Submitting a DGDR with a PCIe SKU is allowed; the operator
will accept it but profiler-assisted sizing falls back to defaults.

---

## 6. `workload`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `isl` | int | `4000` | Average input sequence length (tokens). |
| `osl` | int | `1000` | Average output sequence length (tokens). |
| `concurrency` | float | — | Target concurrent requests. Required (or `requestRate`) when the planner is disabled. |
| `requestRate` | float | — | Target requests per second. Required (or `concurrency`) when the planner is disabled. |

The defaults match a generic chat workload. Production deploys should
set ISL/OSL to match the actual workload distribution; the profiler uses
them to pick a config that hits the SLA at those operating points.

---

## 7. `sla`

| Field | Type | Notes |
|---|---|---|
| `ttft` | float (ms) | Time to first token. |
| `itl` | float (ms) | Inter-token latency. |
| `e2eLatency` | float (ms) | End-to-end latency. Cannot be combined with explicit `ttft`/`itl` — pick one composition or the other. |
| `optimizationType` | enum | `latency` or `throughput`. |

`ttft + itl` composition is the recommended path; `e2eLatency` is
appropriate when the workload is highly variable in ISL/OSL and total
latency is the actual user-facing SLA.

---

## 8. `modelCache`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `pvcName` | string | — | Name of a `ReadWriteMany` PVC containing cached model weights. |
| `pvcModelPath` | string | — | Path to the model directory inside the PVC. |
| `pvcMountPath` | string | `/opt/model-cache` | Mount path inside containers. |

When `pvcName` is set, workers skip the HuggingFace download and load
weights from the PVC. Saves minutes-to-hours of pod-startup time on
large models. The PVC must already exist and be populated; the operator
does not provision it.

---

## 9. `features`

Optional features controlled by raw JSON blocks under `features`:

| Sub-field | Notes |
|---|---|
| `features.planner` | Enable the SLA-aware Planner. Raw JSON config. Disabled by default. |
| `features.mocker` | Enable mocker mode for testing without a real model. Disabled by default. |

When `features.planner` is set, the generated DGD includes a Planner pod
that scales worker replicas to hit the configured TTFT/ITL targets.

---

## 10. `overrides`

| Sub-field | Type | Purpose |
|---|---|---|
| `overrides.profilingJob` | `batchv1.JobSpec` | Customise the profiling Job (tolerations, node selectors, resource requests). |
| `overrides.dgd` | partial DGD | Merged into the profiler-generated `DynamoGraphDeployment` after a config is selected. |

`overrides.profilingJob` only affects the profiling Job. To inject
settings into the deployed worker pods, use `overrides.dgd` — which
accepts a partial `nvidia.com/v1alpha1` DGD object that is overlaid on
the generated deployment. Common uses:

- Inject an env var into all generated services (`spec.envs`).
- Inject an env var into a single named service (`spec.services.<name>.envs`).
- Override resource requests, image tags, or sidecar containers.

---

## 11. Lifecycle Phases

Top-level phase enum (per):

```
Pending → Profiling → Ready → Deploying → Deployed
                                          ↘ Failed (terminal from any state)
```

| Phase | What is happening |
|---|---|
| `Pending` | Spec validated; operator is discovering GPU hardware and preparing the profiling Job. |
| `Profiling` | Profiling Job running. See §12 for sub-phases. |
| `Ready` | Profiling complete; optimal config stored in `.status.profilingResults.selectedConfig`. Terminal when `autoApply: false`. |
| `Deploying` | Operator is creating the `DynamoGraphDeployment` (only when `autoApply: true`). |
| `Deployed` | DGD is running and healthy. |
| `Failed` | Unrecoverable error. Profiling failures are not retried (`backoffLimit: 0`); check events and conditions. |

---

## 12. Profiling Sub-Phases

When the top-level phase is `Profiling`, `status.profilingPhase` reports
which step is running:

```
Initializing → SweepingPrefill → SweepingDecode → SelectingConfig
            → BuildingCurves → GeneratingDGD → Done
```

| Sub-phase | Activity |
|---|---|
| `Initializing` | Loading the DGD template, detecting GPU hardware, resolving model architecture from HuggingFace. |
| `SweepingPrefill` | Sweeping parallelization (TP/TEP/DEP) across GPU counts for prefill, measuring TTFT. |
| `SweepingDecode` | Sweeping parallelization and concurrency levels for decode, measuring ITL. |
| `SelectingConfig` | Filtering against SLA targets, selecting the most cost-efficient config. |
| `BuildingCurves` | Building interpolation curves (ISL→TTFT, KV-usage×context→ITL) for planner integration. |
| `GeneratingDGD` | Packaging profiling data into a ConfigMap and generating the final DGD YAML. |
| `Done` | Profiling complete. |

---

## 13. Status Conditions

The operator maintains the following conditions on `.status.conditions`
(per):

| Condition | Meaning |
|---|---|
| `Validation` | Spec validation passed or failed. |
| `Profiling` | Profiling Job state. `Reason` mirrors the sub-phase (per §12). On failure, the `Reason` is `<Phase>Failed`. |
| `SpecGenerated` | Generated DGD spec is available on `.status.profilingResults.selectedConfig`. |
| `DeploymentReady` | DGD is deployed and reports healthy. |
| `Succeeded` | Aggregate condition. True when the DGDR reaches its target state (Ready with `autoApply: false`, or Deployed otherwise). |

---

## 14. Event Reasons

Events emitted by the controller (verbatim from
`dynamographdeploymentrequest_types.go`):

```
Initialized, ValidationFailed, ProfilingJobCreated, ProfilingJobFailed,
AIConfiguratorFailed, SpecGenerated, SpecChangeRejected,
DeploymentCreated, DeploymentReady, DeploymentDegraded,
DeploymentDeleted, ImagePullFailed
```

Read with:

```bash
kubectl get events -n <ns> \
  --field-selector involvedObject.name=<dgdr-name>,involvedObject.kind=DynamoGraphDeploymentRequest \
  --sort-by='.lastTimestamp'
```

---

## 15. Labels

The operator stamps these labels on the generated DGD (per):

| Label | Value |
|---|---|
| `dgdr.nvidia.com/name` | Name of the source DGDR |
| `dgdr.nvidia.com/namespace` | Namespace of the source DGDR |
| `app.kubernetes.io/managed-by` | `dynamo-operator` |

The DGD does **not** carry an `ownerReference` back to the DGDR — the
DGD is intentionally orphaned so deleting the DGDR does not delete the
running deployment. The relationship is recoverable through the labels
alone.

---

## 16. Canonical Sample

Upstream sample (per):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: example-llm-sla
spec:
  model: Qwen/Qwen3-0.6B
  backend: trtllm
  image: "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1"
  searchStrategy: thorough
  hardware:
    numGpusPerNode: 8
    gpuSku: h200_sxm
  workload:
    isl: 3000
    osl: 500
  sla:
    ttft: 50.0
    itl: 10.0
  autoApply: true
```

---

## 17. v1alpha1 / v1beta1 Compatibility

Both API versions are served on the 1.2.0 release line. v1alpha1 is
marked `deprecated: true` with a deprecation warning; the storage version
for DGDR is v1beta1. The conversion webhook is responsible for the
bridge.

Skills authored for 1.2.0 write v1beta1 in new manifests. Existing
v1alpha1 manifests applied to a 1.2.0 cluster are converted on apply and
stored as v1beta1. Per `SKILL_AUTHORING.md` §4, never claim
v1alpha1 is removed or broken.

The other Dynamo CRDs follow different patterns:

| CRD | v1alpha1 | v1beta1 | Storage version |
|---|---|---|---|
| `DynamoGraphDeploymentRequest` | served (deprecated) | served | v1beta1 |
| `DynamoGraphDeployment` | served (deprecated) | served | v1beta1 |
| `DynamoComponentDeployment` | served (deprecated) | served | v1beta1 |
| `DynamoGraphDeploymentScalingAdapter` | served, storage | served | **v1alpha1** (not yet promoted) |
| `DynamoModel` | served, storage | — | v1alpha1 (no v1beta1) |
| `DynamoCheckpoint` | served, storage | — | v1alpha1 (no v1beta1) |
| `DynamoWorkerMetadata` | served, storage | — | v1alpha1 (no v1beta1) |

Skills must not author `nvidia.com/v1beta1` manifests for `DynamoModel`,
`DynamoCheckpoint`, or `DynamoWorkerMetadata` — there is no v1beta1
schema for those kinds.
