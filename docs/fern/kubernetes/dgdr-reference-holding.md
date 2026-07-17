<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DGDR reference — extracted prose (holding doc)

**Not published.** This file is a temporary holding area for prose and examples
removed from `dgdr-reference.mdx` when it was converted to a pure CRD reference.
Sort each block into one of:

- the [DGDR walkthrough](dgdr-guide.md),
- a new subpage under the DGDR section (for example a "DGDR routing" or "DGDR
  planner" how-to), or
- delete it (already covered by the Router Guide, Planner Guide, or DGDR
  examples).

Delete this file once its contents have been re-homed.

---

## Planner (from `## Planner`)

> DGDR supports the Planner through `spec.features.planner`. Set this field to a
> PlannerConfig object to have DGDR pass that configuration to the profiler and
> generate Planner support in the final DGD. DGDR passes the PlannerConfig
> through without field-level validation; the Planner service validates it when
> it starts.
>
> When the Planner is enabled, the generated output may include a `Planner`
> component in the DGD plus supporting configuration resources, such as a
> `planner-config-*` ConfigMap. Depending on profiling mode and Planner
> settings, DGDR may also generate profiling-data resources for Planner
> bootstrap data.

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

To evaluate Planner recommendations without applying scaling changes, enable advisory mode in the same `features.planner` object:

```yaml
spec:
  features:
    planner:
      mode: disagg
      backend: vllm
      advisory: true
```

> For Planner behavior, scaling modes, and the full PlannerConfig field
> reference, see the [Planner overview](../components/planner/README.md) and
> [Planner Guide](../components/planner/planner-guide.md). For additional
> generated-deployment examples, see [DGDR Examples](dgdr-examples.md).
>
> `spec.overrides.dgd` is not required to enable the Planner. Use
> `spec.features.planner` for Planner enablement and configuration. Use
> `spec.overrides.dgd` only when you need to customize the generated DGD after
> DGDR has assembled it.

---

## Generated DGD overrides (extra examples from `## Generated DGD overrides`)

The reference now keeps only the field definition and a single minimal example. The examples below were the extra prose-heavy walkthrough content.

> Use `spec.overrides.dgd` when the generated DGD needs a field that DGDR does
> not expose directly. The value is a partial `nvidia.com/v1alpha1` DGD object
> merged into the profiler-generated deployment after Dynamo selects a
> configuration.

For example, to inject an environment variable into every generated component:

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

Use `spec.envs` for variables that should apply to all generated components. To target a single component, override that component's `envs` entry instead:

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1
      kind: DynamoGraphDeployment
      spec:
        services:
          decode:  # replace with the generated component name
            envs:
              - name: CUSTOM_WORKER_ENV
                value: "enabled"
```

> `overrides.profilingJob` only customizes the profiling Job. Use
> `overrides.dgd` for settings that must appear on the deployed worker pods.

---

## Routing (entire `## Routing` section)

> DGDR-generated deployments include a standalone `Frontend` component. That
> frontend runs Dynamo's embedded router and defaults to `round-robin` routing,
> which is often not optimal. Because DGDR does not yet expose a first-class
> router feature, configure the generated frontend with `spec.overrides.dgd`.
>
> For the full router mode and environment variable reference, see
> [Router Guide](../components/router/router-guide.md) and
> [Router Configuration](../components/router/router-configuration.md).

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

> Use the same `Frontend` override for other frontend router modes, such as
> `random`, `least-loaded`, or `device-aware-weighted`. For normal DGDR
> deployments, use `kv` when you want prefix-cache-aware routing and
> `round-robin` or `least-loaded` when you only want load balancing. Use
> `direct` only when an external router supplies explicit worker IDs in the
> request routing hints. For detailed mode definitions, see
> [Router Guide](../components/router/router-guide.md#routing-modes-router-mode).
>
> KV-aware routing can use event-driven prefix-cache state or approximate prefix
> matching. The frontend still runs in `kv` mode in both cases. If you do not
> configure worker KV-event publication, set `DYN_ROUTER_USE_KV_EVENTS=false` to
> use approximate KV mode:

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

> For event-driven prefix-cache state, enable worker event publication only
> where prefill happens: the single worker in aggregated serving, or prefill
> workers in disaggregated serving. Decode workers are scored by load
> (`dyn-decode-scorer`), not prefix overlap (`dyn-prefill-scorer`), so vLLM
> decode workers omit both `--enable-prefix-caching` and `--kv-events-config`.
> Component names depend on the selected backend and topology, so inspect the
> generated DGD first, especially when `autoApply: false`.

For example, a generated vLLM disaggregated deployment may contain a `VllmPrefillWorker` component. This override appends the vLLM KV-event publishing arguments to that component while enabling the frontend KV router:

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

> Worker KV-event flags are backend-specific. For cross-backend behavior, see
> [Router Operations](../components/router/router-operations.md#additional-notes).

| Backend | Detailed docs | Worker-side event publishing |
|---|---|---|
| vLLM | [vLLM Reference Guide](../backends/vllm/vllm-reference-guide.md#argument-reference), [vLLM Examples](../backends/vllm/vllm-examples.md#aggregated-serving-with-kv-routing) | `--enable-prefix-caching` and `--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'` on the aggregated worker or disaggregated prefill worker |
| SGLang | [SGLang KV Events](../backends/sglang/sglang-reference-guide.md#kv-events), [SGLang Examples](../backends/sglang/sglang-examples.md#aggregated-serving-with-kv-routing) | `--kv-events-config` with the SGLang event endpoint |
| TRT-LLM | [TRT-LLM DP Rank Routing](../backends/trtllm/trtllm-dp-rank-routing.md#enabling-dp-rank-routing), [TRT-LLM Observability](../backends/trtllm/trtllm-observability.md) | `--publish-events-and-metrics` |

> In Kubernetes deployments the Dynamo runtime normally uses Kubernetes
> discovery and the NATS event plane. Some backends, such as vLLM and SGLang,
> emit raw KV events over ZMQ; the Dynamo worker consumes those backend events
> and republishes router events through the Dynamo event plane. For the event
> plane model, see [Event Plane](../design-docs/event-plane.md).

### EPP and gateway routing (from `### EPP and gateway routing`)

> EPP/Gateway routing is a different topology from the standalone frontend that
> DGDR generates:

```text
client -> Gateway -> EPP selects worker -> worker frontend sidecar -> engine
```

> In this mode the EPP owns worker selection. The worker-local frontend sidecar
> must run with `--router-mode direct` so it honors the worker IDs selected by
> EPP. In the normal Gateway path, the selected endpoint and the frontend
> sidecar are the same worker pod; if they differ, direct mode can still forward
> to the worker ID supplied by EPP.
>
> DGDR does not currently generate EPP components or frontend sidecars. Also,
> `overrides.dgd` only patches components that already exist in the generated
> DGD, so it cannot add a missing `Epp` component to a DGDR-generated
> deployment. Use a direct DGD manifest or a GAIE recipe for EPP deployments.
> For manifests, `frontendSidecar` configuration, direct routing, EPP routing
> variables such as `DYN_USE_KV_EVENTS`, and route setup, see
> [Gateway API Inference Extension](inference-gateway.md). The same guide also
> documents the optional
> [Rust EPP](inference-gateway.md#4b-build-rust-epp-image-optional--experimental),
> which is currently experimental.

---

## SKU format (prose from `## SKU format`)

The reference keeps the field's allowed-values list and the two callouts. The comparison table and intro prose below were the "how to write it" walkthrough content.

> When providing hardware configuration manually, use lowercase underscore
> format:

| Correct | Incorrect |
|---|---|
| `h100_sxm` | `H100-SXM5-80GB` |
| `h200_sxm` | `H200-SXM-141GB` |
| `a100_sxm` | `A100-SXM4-80GB` |
| `a30` | `A30` |
| `l40s` | `L40S` |
