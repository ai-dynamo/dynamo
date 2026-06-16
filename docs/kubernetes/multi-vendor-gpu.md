---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Running Dynamo on non-NVIDIA and shared GPUs
---

Dynamo started life assuming one device: a single NVIDIA GPU exposed as
`nvidia.com/gpu`. The operator still defaults to that path, but the
`DynamoComponentDeploymentSharedSpec.device` field lets you ask for any
accelerator the cluster can advertise — including AMD ROCm, Intel XPU, and
NVIDIA GPUs sliced through [HAMi](https://github.com/Project-HAMi/HAMi).

The change is small on purpose. The operator does not know what HAMi is, and
it does not parse vendor-specific keys. It just copies the four Kubernetes
primitives you set into the generated pod spec and skips the NVIDIA defaults.

## What the field looks like

```yaml
spec:
  components:
  - name: VllmDecodeWorker
    device:
      resources:
        nvidia.com/gpu: "1"
        nvidia.com/gpumem: "3000"
        nvidia.com/gpucores: "50"
      schedulerName: hami-scheduler
      nodeSelector:
        gpu: "on"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
    podTemplate:
      spec:
        containers:
        - name: main
          # do NOT set resources.limits.nvidia.com/gpu here — the operator
          # merges device.resources into the main container at build time.
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
```

The four primitives, and what the operator does with each:

| Field           | Effect                                                                                |
| --------------- | ------------------------------------------------------------------------------------- |
| `resources`     | merged into the main container's `Limits` and `Requests`; req == lim is forced        |
| `tolerations`   | appended to the pod's tolerations, deduplicated on `key+operator+effect+value`        |
| `nodeSelector`  | merged with the pod template's selector; existing keys win                            |
| `schedulerName` | set on the pod if the pod template did not already pick one                           |

When the field is unset, the operator keeps the current NVIDIA-default path:
auto-injected `nvidia.com/gpu` toleration, the GMS DRA claim, and the LWS
worker GPU validation all behave exactly as they do today.

## HAMi (shared NVIDIA GPUs)

[HAMi](https://github.com/Project-HAMi/HAMi) is a CNCF sandbox project that
slices a physical GPU into multiple isolated pods. It exposes three extended
resource keys on top of `nvidia.com/gpu`:

- `nvidia.com/gpumem` — memory slice in MB
- `nvidia.com/gpucores` — share of SMs (0–100, or a device-plugin-specific unit)
- plus any task-specific keys your device plugin adds

A full example lives at
[`examples/backends/vllm/deploy/v1beta1/disagg_hami.yaml`](https://github.com/ai-dynamo/dynamo/blob/bdeb46ca87a2de1037808cc23a569c8f0e89cbe1/examples/backends/vllm/deploy/v1beta1/disagg_hami.yaml).
The same shape, in v1alpha1 syntax, is at
[`examples/backends/vllm/deploy/disagg_hami.yaml`](https://github.com/ai-dynamo/dynamo/blob/bdeb46ca87a2de1037808cc23a569c8f0e89cbe1/examples/backends/vllm/deploy/disagg_hami.yaml).

Install steps on the cluster side:

1. Install the HAMi device plugin (DaemonSet) following the
   [HAMi install guide](https://github.com/Project-HAMi/HAMi#installation).
   Confirm `kubectl get nodes -o json | jq '.items[].status.allocatable'`
   shows `nvidia.com/gpumem` and `nvidia.com/gpucores`.
2. Confirm the scheduler is reachable: `kubectl get pods -n kube-system -l
   app=hami-scheduler`.
3. Taint your GPU nodes so only HAMi-managed pods land there, and add the
   matching toleration in `device.tolerations`.
4. Submit the DynamoGraphDeployment. The operator hands `device` through and
   `hami-scheduler` picks a slice.

> [!WARNING]
> The shipped examples have not been exercised against a real HAMi cluster.
> Validate against a dev cluster before using in production.

## AMD ROCm

The AMD GPU device plugin advertises `amd.com/gpu`. Set `device.resources`
to that key and add the matching toleration:

```yaml
device:
  resources:
    amd.com/gpu: "1"
  tolerations:
  - key: amd.com/gpu
    operator: Exists
```

You will also need an AMD-capable backend image (ROCm vLLM, ROCm SGLang, etc.)
in `podTemplate`. The operator change does not touch backend runtimes — swap
the image you already trust.

> [!NOTE]
> We do not ship a checked-in `disagg_amd.yaml` yet because the Dynamo
> vLLM/SGLang ROCm build path is not validated end-to-end. Once it is, the
> example will land as `examples/backends/<backend>/deploy/v1beta1/disagg_amd.yaml`.

## Intel XPU

Intel XPU clusters can run Dynamo today through
[Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
(DRA), using the pattern in
[`examples/backends/vllm/deploy/v1beta1/disagg_xpu_dra.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/v1beta1/disagg_xpu_dra.yaml).
That path does not use the `device` field — DRA wires the GPU through a
`ResourceClaimTemplate` instead. The two paths are complementary: use `device`
when the cluster only has a simple device-plugin (HAMi, AMD, etc.) and use
DRA when the cluster has a full DRA driver (Intel GPU DRA driver today,
NVIDIA DRA driver in the future).

## What the operator does NOT do

To keep the surface small and the API stable, the `device` field does not
configure:

- **Backend runtime flags.** Switching from CUDA vLLM to ROCm vLLM still
  means shipping the right image and the right `args`. The operator only
  wires scheduling and resource requests.
- **DRA claim generation.** When DRA is available, prefer the
  `ResourceClaimTemplate` path in
  [`disagg_xpu_dra.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/v1beta1/disagg_xpu_dra.yaml).
  The `device` field is the lightweight alternative.
- **GPU Memory Service.** When `device` is set, the operator skips the
  intra-pod GMS injection path. GMS is wired to the NVIDIA DRA driver and is
  not applicable to AMD / Intel / HAMi-sliced layouts.
- **LeaderWorkerSet GPU validation.** The check switched from a hard
  `nvidia.com/gpu` lookup to a vendor-agnostic `HasAnyGPUResource` so a
  device-only worker is accepted.

If you find a case that genuinely needs a fifth primitive, open an issue and
we'll talk about extending the field rather than adding more `podTemplate`
workarounds.

## Troubleshooting

The pod stays `Pending` with `nvidia.com/gpumem` / `nvidia.com/gpucores`
"not found" — the HAMi device plugin is not running on the target node, or
the node was tainted after the DaemonSet was deployed. Check
`kubectl describe node <node>`.

The pod is admitted but the container crashes with `CUDA error: no
CUDA-capable device` — the image you shipped is the CUDA vLLM, not the
ROCm/XPU build. This is a backend runtime mismatch; the operator cannot
detect it from `device` alone.

The pod is admitted, scheduled, and the main container exits cleanly, but
the `device.tolerations` you wrote are missing from the rendered pod — the
pod template already had the same toleration key. The operator dedupes on
`key+operator+effect+value` and keeps the first occurrence.