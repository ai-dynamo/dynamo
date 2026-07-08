---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DGDR Reference
subtitle: Field reference for DynamoGraphDeploymentRequest, the deploy-by-intent generator that profiles and produces a DGD.
---

A `DynamoGraphDeploymentRequest` (DGDR) is Dynamo's deploy-by-intent generator
for [`DynamoGraphDeployment`](api-reference.md#dynamographdeployment) (DGD)
resources. You describe what you want to run and your performance targets; the
profiler determines a configuration and produces the DGD that serves traffic.

For the full deployment mental model — including DGD, DCD, DGDR, recipes,
strategy selection, model caching, planner setup, and common pitfalls — see the
[Deployment Overview](model-deployment-guide.md).

## DGDR, DGD, and Recipes

Dynamo provides two Custom Resources for deploying inference graphs:

| | DGD (canonical live deployment) | DGDR (generator/profiler) |
|---|---|---|
| **You provide** | Full deployment spec (services, parallelism, replicas, resource limits, etc.) | Model, backend, workload, hardware, and optional SLA targets |
| **What happens** | The operator reconciles the DGD into `DynamoComponentDeployment` resources and pods | The profiler generates a DGD; with `autoApply: true`, the operator creates it |
| **Best for** | Known-good configs, tuned recipes, or full manual control | New model/hardware combinations, SLA-driven sizing, or generated DGD YAML |
| **Persistence** | Persists and serves traffic | Reaches a terminal state after generation/deploy |

Use DGD directly when you have a hand-crafted configuration for a specific
model/hardware combination. Most
[recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) are tuned DGD
manifests. Use DGDR when you want Dynamo to generate the DGD for you.

For DGD deployment details, see [Creating Deployments](deployment/create-deployment.md).

## Spec Reference

### Minimal Example

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: Qwen/Qwen3-0.6B
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
```

### Field Reference

| Field | Required | Default | Purpose |
|---|---|---|---|
| `model` | Yes | — | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) |
| `image` | No | — | Container image for the profiling job. Dynamo >= 1.1.0: use `dynamo-planner`; earlier versions: use `dynamo-frontend`. |
| `backend` | No | `auto` | Inference engine: `auto`, `vllm`, `sglang`, `trtllm` |
| `searchStrategy` | No | `rapid` | Profiling depth: `rapid` (AIC-backed DynoSim-style modeling, ~30s) or `thorough` (real GPU, 2–4h) |
| `autoApply` | No | `true` | Automatically deploy the profiler's recommended config |
| `sla.ttft` | No | — | Target time to first token (ms) |
| `sla.itl` | No | — | Target inter-token latency (ms) |
| `sla.e2eLatency` | No | — | Target end-to-end latency (ms). Cannot be combined with explicit `ttft`/`itl`. |
| `workload.isl` | No | `4000` | Expected average input sequence length |
| `workload.osl` | No | `1000` | Expected average output sequence length |
| `workload.requestRate` | No | — | Target requests per second |
| `workload.concurrency` | No | — | Target concurrent requests |
| `hardware.gpuSku` | No | auto-detected | GPU SKU (see [SKU Format](#sku-format)) |
| `hardware.vramMb` | No | auto-detected | GPU VRAM in MB |
| `hardware.totalGpus` | No | auto-detected (capped at 32) | Total GPUs available to the deployment |
| `hardware.numGpusPerNode` | No | auto-detected | GPUs per node |
| `hardware.interconnect` | No | auto-detected | Interconnect type |
| `hardware.rdma` | No | auto-detected | Whether RDMA is available |
| `modelCache.pvcName` | No | — | Name of a `ReadWriteMany` PVC containing cached model weights |
| `modelCache.pvcModelPath` | No | — | Path to the model directory inside the PVC |
| `modelCache.pvcMountPath` | No | `/opt/model-cache` | Mount path inside containers |
| `features.planner` | No | disabled | PlannerConfig passed to the Planner service; when present, the generated DGD includes Planner service/configuration |
| `features.mocker` | No | disabled | Enable mocker mode for testing |
| `overrides.profilingJob` | No | — | `batchv1.JobSpec` overrides for the profiling job (e.g., tolerations) |
| `overrides.dgd` | No | — | Raw DGD override base applied to the generated deployment |

For the complete CRD spec, see the [API Reference](api-reference.md).

### Planner

DGDR supports Planner through `spec.features.planner`. Set this field to a
PlannerConfig object to have DGDR pass that configuration to the profiler and
generate Planner support in the final DGD. DGDR passes this PlannerConfig
through without field-level validation; the Planner service validates it when it
starts.

When Planner is enabled, the generated output may include a `Planner` service in
the DGD plus supporting Planner configuration resources, such as a
`planner-config-*` ConfigMap. Depending on profiling mode and Planner settings,
DGDR may also generate profiling-data resources for Planner bootstrap data.

Minimal Planner-enabled DGDR:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-planner
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
  features:
    planner:
      mode: disagg
      backend: vllm
```

To evaluate Planner recommendations without applying scaling changes, enable
advisory mode in the same `features.planner` object:

```yaml
spec:
  features:
    planner:
      mode: disagg
      backend: vllm
      advisory: true
```

For Planner behavior, scaling modes, and the full PlannerConfig field reference,
see the [Planner overview](../components/planner/README.md) and
[Planner Guide](../components/planner/planner-guide.md).
For additional generated-deployment examples, see
[DGDR Examples](dgdr-examples.md).

`spec.overrides.dgd` is not required to enable Planner. Use
`spec.features.planner` for Planner enablement and configuration. Use
`spec.overrides.dgd` only when you need to customize the generated DGD after
DGDR has assembled it.

> [!NOTE]
> DGDR does not currently expose a `features.kvRouter` field. To configure
> router mode or KV-aware routing details, use a direct DGD, a tuned recipe, or
> `overrides.dgd` when you still want DGDR to generate the base deployment.

### Generated DGD Overrides

Use `spec.overrides.dgd` when the generated `DynamoGraphDeployment` needs a
field that DGDR does not expose directly. The value is a partial
`nvidia.com/v1alpha1` DGD object that is merged into the profiler-generated
deployment after Dynamo selects a configuration.

For example, to inject an environment variable into every generated service:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-sglang
spec:
  model: Qwen/Qwen3-30B-A3B
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1
      kind: DynamoGraphDeployment
      spec:
        envs:
          - name: TRITON_PTXAS_PATH
            value: /usr/local/cuda/bin/ptxas
```

Use `spec.envs` for variables that should apply to all generated services. To
target a single service, override that service's `envs` entry instead:

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1
      kind: DynamoGraphDeployment
      spec:
        services:
          decode:  # replace with the generated service name
            envs:
              - name: CUSTOM_WORKER_ENV
                value: "enabled"
```

> [!NOTE]
> `overrides.profilingJob` only customizes the profiling Job. Use
> `overrides.dgd` for settings that must appear on the deployed worker pods.

### Routing

DGDR-generated deployments include a standalone `Frontend` service. That
frontend runs Dynamo's embedded router and defaults to `round-robin` routing,
which is often not optimal. Because DGDR does not yet expose a first-class
router feature, configure the generated frontend with `spec.overrides.dgd`.

For the full router mode and environment variable reference, see
[Router Guide](../components/router/router-guide.md) and
[Router Configuration](../components/router/router-configuration.md).

For example, enable KV-aware routing on the generated frontend:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-kv-router
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1  # v1beta1 not yet supported for overrides
      kind: DynamoGraphDeployment
      spec:
        services:
          Frontend:
            envs:
              - name: DYN_ROUTER_MODE
                value: kv
```

Use the same `Frontend` override for other frontend router modes, such as
`random`, `least-loaded`, or `device-aware-weighted`. For normal DGDR
deployments, use `kv` when you want prefix-cache-aware routing and
`round-robin` or `least-loaded` when you only want load balancing. Use
`direct` only when an external router supplies explicit worker IDs in the
request routing hints. For detailed mode definitions, see
[Router Guide](../components/router/router-guide.md#routing-modes-router-mode).

KV-aware routing can use event-driven prefix-cache state or approximate
prefix matching. The frontend still runs in `kv` mode in both cases. If you
do not configure worker KV-event publication, set
`DYN_ROUTER_USE_KV_EVENTS=false` to use approximate KV mode:

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1  # v1beta1 not yet supported for overrides
      kind: DynamoGraphDeployment
      spec:
        services:
          Frontend:
            envs:
              - name: DYN_ROUTER_MODE
                value: kv
              - name: DYN_ROUTER_USE_KV_EVENTS
                value: "false"
```

For event-driven prefix-cache state, enable worker event publication only
where prefill happens: the single worker in aggregated serving, or prefill
workers in disaggregated serving. Decode workers are scored by load
(`dyn-decode-scorer`), not prefix overlap (`dyn-prefill-scorer`), so vLLM
decode workers omit both `--enable-prefix-caching` and `--kv-events-config`.
Service names depend on the selected backend and topology, so inspect the
generated DGD first, especially when `autoApply: false`.

For example, a generated vLLM disaggregated deployment may contain a
`VllmPrefillWorker` service. This override appends the vLLM KV-event publishing
arguments to that service while enabling the frontend KV router:

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1  # v1beta1 not yet supported for overrides
      kind: DynamoGraphDeployment
      spec:
        services:
          Frontend:
            envs:
              - name: DYN_ROUTER_MODE
                value: kv
          VllmPrefillWorker:
            extraPodSpec:
              mainContainer:
                args:
                  - --enable-prefix-caching
                  - --kv-events-config
                  - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

Worker KV-event flags are backend-specific. For cross-backend behavior, see
[Router Operations](../components/router/router-operations.md#additional-notes).

| Backend | Detailed docs | Worker-side event publishing |
|---|---|---|
| vLLM | [vLLM Reference Guide](../backends/vllm/vllm-reference-guide.md#argument-reference), [vLLM Examples](../backends/vllm/vllm-examples.md#aggregated-serving-with-kv-routing) | `--enable-prefix-caching` and `--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'` on the aggregated worker or disaggregated prefill worker |
| SGLang | [SGLang KV Events](../backends/sglang/sglang-reference-guide.md#kv-events), [SGLang Examples](../backends/sglang/sglang-examples.md#aggregated-serving-with-kv-routing) | `--kv-events-config` with the SGLang event endpoint |
| TRT-LLM | [TRT-LLM DP Rank Routing](../backends/trtllm/trtllm-dp-rank-routing.md#enabling-dp-rank-routing), [TRT-LLM Observability](../backends/trtllm/trtllm-observability.md) | `--publish-events-and-metrics` |

In Kubernetes deployments the Dynamo runtime normally uses Kubernetes
discovery and the NATS event plane. Some backends, such as vLLM and SGLang,
emit raw KV events over ZMQ; the Dynamo worker consumes those backend events
and republishes router events through the Dynamo event plane. For the event
plane model, see [Event Plane](../design-docs/event-plane.md).

### EPP and Gateway Routing

EPP/Gateway routing is a different topology from the standalone frontend that
DGDR generates:

```text
client -> Gateway -> EPP selects worker -> worker frontend sidecar -> engine
```

In this mode the EPP owns worker selection. The worker-local frontend sidecar
must run with `--router-mode direct` so it honors the worker IDs selected by
EPP. In the normal Gateway path, the selected endpoint and the frontend sidecar
are the same worker pod; if they differ, direct mode can still forward to the
worker ID supplied by EPP.

DGDR does not currently generate EPP components or frontend sidecars. Also,
`overrides.dgd` only patches services that already exist in the generated DGD,
so it cannot be used to add a missing `Epp` service to a DGDR-generated
deployment. Use a direct DGD manifest or a GAIE recipe for EPP deployments.
For manifests, `frontendSidecar` configuration, direct routing, EPP routing
variables such as `DYN_USE_KV_EVENTS`, and route setup, see
[Gateway API Inference Extension](inference-gateway.md). The same guide also
documents the optional [Rust EPP](inference-gateway.md#4b-build-rust-epp-image-optional--experimental),
which is currently experimental.

### SKU Format

When providing hardware configuration manually, use lowercase underscore format:

| Correct | Incorrect |
|---|---|
| `h100_sxm` | `H100-SXM5-80GB` |
| `h200_sxm` | `H200-SXM-141GB` |
| `a100_sxm` | `A100-SXM4-80GB` |
| `a30` | `A30` |
| `l40s` | `L40S` |

All supported values: `gb200_sxm`, `b200_sxm`, `h200_sxm`, `h100_sxm`,
`h100_pcie`, `a100_sxm`, `a100_pcie`, `a30`, `l40s`, `l40`, `l4`,
`v100_sxm`, `v100_pcie`, `t4`, `mi200`, `mi300`.

> [!NOTE]
> Not all SKUs are supported by the AIC profiler for `rapid` mode. See
> [AIC Support Matrix](model-deployment-guide.md#aic-support-matrix) for details.

> [!IMPORTANT]
> **PCIe variants not yet supported by profiler.** The CRD admits PCIe SKUs
> (`h100_pcie`, `a100_pcie`, `v100_pcie`), but the profiler does not currently
> ship training data for them. You can submit a DGDR with a PCIe value; the
> operator will accept it but profiler-assisted sizing will fall back to
> defaults. Profiler support for PCIe SKUs is tracked as an engineering
> follow-up.

## Lifecycle

When you create a DGDR, it progresses through these phases:

| Phase | What is happening |
|---|---|
| `Pending` | Spec validated; operator is discovering GPU hardware and preparing the profiling job |
| `Profiling` | Profiling job running — sub-phases: `Initializing`, `SweepingPrefill`, `SweepingDecode`, `SelectingConfig`, `BuildingCurves`, `GeneratingDGD`, `Done` |
| `Ready` | Profiling complete; optimal config stored in `.status.profilingResults.selectedConfig`. Terminal state when `autoApply: false`. |
| `Deploying` | Creating the `DynamoGraphDeployment` (only when `autoApply: true`) |
| `Deployed` | DGD is running and healthy |
| `Failed` | Unrecoverable error — profiling failures are not retried (`backoffLimit: 0`); check events and conditions for details |

### Conditions

The operator maintains these conditions on the DGDR status:

| Condition | Meaning |
|---|---|
| `Validation` | Spec validation passed or failed |
| `Profiling` | Profiling job is running, succeeded, or failed |
| `SpecGenerated` | Generated DGD spec is available |
| `DeploymentReady` | DGD is deployed and healthy |
| `Succeeded` | Aggregate condition — true when the DGDR has reached its target state |

### Monitoring

```bash
# Watch phase transitions
kubectl get dgdr my-model -n $NAMESPACE -w

# Detailed status, conditions, and events
kubectl describe dgdr my-model -n $NAMESPACE

# Profiling sub-phase
kubectl get dgdr my-model -n $NAMESPACE -o jsonpath='{.status.profilingPhase}'

# Profiling job logs
kubectl get pods -n $NAMESPACE -l nvidia.com/dgdr-name=my-model
kubectl logs -f <profiling-pod-name> -n $NAMESPACE

# View generated DGD spec (when autoApply: false)
kubectl get dgdr my-model -n $NAMESPACE \
  -o jsonpath='{.status.profilingResults.selectedConfig}' | python3 -m json.tool
```

### Resource Ownership

- The DGDR does **not** set an owner reference on the DGD it creates. Deleting
  a DGDR does not delete the DGD — it persists independently so it can continue
  serving traffic.
- The relationship is tracked via labels: `dgdr.nvidia.com/name` and
  `dgdr.nvidia.com/namespace`.
- Additional resources (planner ConfigMaps) are created in the same namespace
  and labeled with `dgdr.nvidia.com/name`.

## Known Issues

- **`pareto_analysis.py` produces NaN for some configurations.** Tracked as an
  engineering follow-up. Workaround: re-run with a narrower sweep; narrow
  sweeps bypass the NaN path in practice.
- **PCIe profiler data not yet available.** See the PCIe callout under
  [SKU Format](#sku-format).

## Further Reading

- [Deployment Overview](model-deployment-guide.md) — DGD, DCD, DGDR, recipes, strategy selection, and common pitfalls
- [Profiler Guide](../components/profiler/profiler-guide.md) — Profiling algorithms, picking modes, gate checks
- [Profiler Examples](../components/profiler/profiler-examples.md) — Ready-to-use YAML for SLA targets, private models, MoE, overrides
- [Planner Guide](../components/planner/planner-guide.md) — Scaling modes, PlannerConfig reference
- [API Reference](api-reference.md) — Complete CRD field specifications
- [Creating Deployments](deployment/create-deployment.md) — DGD spec for full manual control
