---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Rolling Updates
---

# Rolling Updates for DynamoGraphDeployments

This guide covers how rolling updates work for `DynamoGraphDeployment` (DGD) resources. Rolling updates allow you to update worker configurations (images, resources, environment variables, etc.) with minimal downtime by gradually replacing old pods with new ones.

The behavior of rolling updates depends on the backing resource type of your deployment. DGDs backed by Kubernetes Deployments benefit from **managed rolling updates** with namespace isolation, while Grove and LWS-backed deployments use their native update mechanisms.

## Example

Consider a disaggregated deployment with separate prefill and decode workers. You want to update the tensor parallelism of the decode worker to 2.

**Before** — original deployment:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    VllmDecodeWorker:
      componentType: worker
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --disaggregation-mode
            - decode
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --disaggregation-mode
            - prefill
```

**After** — updated with new image and parallelism tuning:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    VllmDecodeWorker:
      componentType: worker
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --disaggregation-mode
            - decode
            - --tensor-parallelism
            - "2"
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --disaggregation-mode
            - prefill
```

Apply the update:

```bash
kubectl apply -f vllm-disagg.yaml
```

Monitor rolling update progress:

```bash
kubectl get dgd vllm-disagg -n dynamo -o jsonpath='{.status.rollingUpdate}'
```

## Default Behavior (Grove and LWS)

For DGDs backed by **Grove** (PodCliques, PodCliqueSets) or **LWS** (LeaderWorkerSets), the operator does not manage rolling updates directly. Instead, these deployments rely on the native rolling update mechanisms of their underlying resources.

### What Happens

- A modification to the pod spec of a service triggers the rolling update behavior of the backing resource. In the example above, the modification to the pod spec of the decode worker triggers the rolling update of just the decode worker.
- For Grove, PodCliques and PodCliqueScalingGroups use a static rolling update strategy of `maxUnavailable: 1` and `maxSurge: 0`. LWS follows the same `maxUnavailable: 1` and `maxSurge: 0` strategy.
- **Old and new workers operate within the same Dynamo namespace.** This means old and new workers can discover each other through service discovery.

The following diagram illustrates the rolling update of the decode worker in a Grove PodCliqueSet (PCS). Only the decode PodClique is updated — the frontend and prefill PodCliques are unaffected:

```
┌─ PodCliqueSet: vllm-disagg ───────────────────────────────────────────────────────┐
│                                                                                    │
│  ┌─ PCLQ: Frontend ──────┐  ┌─ PCLQ: VllmPrefillWorker ─┐                        │
│  │                        │  │                            │                        │
│  │  ┌──────────────────┐  │  │  ┌──────────────────────┐  │                        │
│  │  │ Pod (v1) ✓       │  │  │  │ Pod (v1) ✓           │  │   No changes —        │
│  │  └──────────────────┘  │  │  └──────────────────────┘  │   not rolling          │
│  │                        │  │                            │                        │
│  └────────────────────────┘  └────────────────────────────┘                        │
│                                                                                    │
│  ┌─ PCLQ: VllmDecodeWorker ──────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │  maxUnavailable: 1, maxSurge: 0                                                │ │
│  │                                                                                │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                            │ │
│  │  │ Pod (v2) ✓ NEW       │  │ Pod (v1) Terminating │  ← rolling one at a time   │ │
│  │  └──────────────────────┘  └──────────────────────┘                            │ │
│  │                                                                                │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
│                        ┌──────────────────────────────────┐                        │
│                        │  Dynamo Namespace: vllm-disagg   │                        │
│                        │                                  │                        │
│                        │  All v1 and v2 pods registered   │                        │
│                        │  and discoverable by each other  │                        │
│                        └──────────────────────────────────┘                        │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Implications for Disaggregated Deployments

Because old and new workers share the same Dynamo namespace, they are visible to each other. In a disaggregated setup, this can lead to cross-generation communication — for example, a newly deployed prefill worker may discover and route requests to an old decode worker (or vice versa). If the old and new versions are incompatible, this can result in errors.

> [!WARNING]
> For Grove and LWS deployments with disaggregated prefill/decode workers, be aware that during a rolling update, new workers may communicate with old workers. Ensure that your worker versions are backward-compatible, or consider using Deployment-backed DGDs which provide namespace isolation during updates.

> [!NOTE]
> Managed rolling updates with namespace isolation are planned for Grove and LWS-backed deployments in a future release. See [Future Work](#future-work) for details.

## Managed Rolling Updates (Deployments)

For DGDs backed by Kubernetes **Deployments** (single-node, non-multinode services), the Dynamo operator implements managed rolling updates with namespace isolation. This is tracked in the DGD status and provides stronger guarantees for disaggregated deployments.

### How It Works

1. **Spec change detection** — The operator computes a hash of all worker service specs (prefill, decode, and worker component types). When this hash changes, a rolling update is triggered.

2. **Namespace isolation** — New worker `DynamoComponentDeployments` (DCDs) are created with the spec hash appended to their Dynamo namespace. This means new workers register in a different Dynamo namespace than old workers, preventing cross-generation discovery. A new prefill worker will only discover and route to new decode workers, avoiding compatibility issues.

3. **Gradual replacement** — The operator gradually scales up new worker DCDs and scales down old ones, respecting `maxSurge` and `maxUnavailable` constraints. When a worker service is updated (all new replicas are ready, all old replicas are terminated), it is marked as completed.

4. **Cleanup** — Once all worker services have completed the transition, old worker DCDs are deleted and the rolling update is marked as completed.

```
┌─ DynamoGraphDeployment: vllm-disagg ──────────────────────────────────────────────┐
│                                                                                    │
│  ┌─ DCD: Frontend ──────────┐                                                      │
│  │                          │                                                      │
│  │  ┌────────────────────┐  │   No changes —                                       │
│  │  │ Pod (v1) ✓         │  │   not a worker component                             │
│  │  └────────────────────┘  │                                                      │
│  │                          │                                                      │
│  └──────────────────────────┘                                                      │
│                                                                                    │
│  ┌─ OLD DCDs (hash: a1b2c3d4) ──────────────────────────────────────────────────┐  │
│  │                                                                               │  │
│  │  ┌─ DCD: VllmDecodeWorker-a1b2c3d4 ──┐  ┌─ DCD: VllmPrefillWorker-a1b2c3d4 ┐│  │
│  │  │                                    │  │                                   ││  │
│  │  │  ┌──────────────────────┐          │  │  ┌─────────────────────┐          ││  │
│  │  │  │ Pod (v1) Terminating │          │  │  │ Pod (v1) Terminating│          ││  │
│  │  │  └──────────────────────┘          │  │  └─────────────────────┘          ││  │
│  │  │                                    │  │                                   ││  │
│  │  │  Dynamo Namespace: vllm-disagg     │  │  Dynamo Namespace: vllm-disagg    ││  │
│  │  │                  -a1b2c3d4         │  │                  -a1b2c3d4        ││  │
│  │  └────────────────────────────────────┘  └───────────────────────────────────┘│  │
│  │                                                                               │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                    │
│  ┌─ NEW DCDs (hash: f5e6d7c8) ──────────────────────────────────────────────────┐  │
│  │                                                                               │  │
│  │  ┌─ DCD: VllmDecodeWorker-f5e6d7c8 ──┐  ┌─ DCD: VllmPrefillWorker-f5e6d7c8 ┐│  │
│  │  │                                    │  │                                   ││  │
│  │  │  ┌──────────────────────┐          │  │  ┌─────────────────────┐          ││  │
│  │  │  │ Pod (v2) ✓ NEW      │          │  │  │ Pod (v2) ✓ NEW     │          ││  │
│  │  │  └──────────────────────┘          │  │  └─────────────────────┘          ││  │
│  │  │                                    │  │                                   ││  │
│  │  │  Dynamo Namespace: vllm-disagg     │  │  Dynamo Namespace: vllm-disagg    ││  │
│  │  │                  -f5e6d7c8         │  │                  -f5e6d7c8        ││  │
│  │  └────────────────────────────────────┘  └───────────────────────────────────┘│  │
│  │                                                                               │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                    │
│  Old and new workers are in different Dynamo namespaces —                           │
│  new prefill only discovers new decode, preventing cross-generation routing.        │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> Only worker component types (`worker`, `prefill`, `decode`) participate in managed rolling updates. Non-worker components like `frontend` are updated in-place without namespace isolation.

### Rolling Update Phases

The rolling update progress is tracked in `.status.rollingUpdate` with the following phases:

| Phase | Description |
|-------|-------------|
| `Pending` | A spec change was detected and the rolling update has been initialized. |
| `InProgress` | New worker DCDs are being scaled up and old ones are being scaled down. |
| `Completed` | All worker services have transitioned to new replicas. Old DCDs have been cleaned up. |

The status also tracks:
- `startTime` — When the rolling update began.
- `endTime` — When the rolling update completed.
- `updatedServices` — List of worker services that have completed the transition.

### Configuring maxSurge and maxUnavailable

You can configure the rolling update strategy per service using annotations:

| Annotation | Description | Default |
|------------|-------------|---------|
| `nvidia.com/deployment-rolling-update-max-surge` | Maximum number of extra pods that can be created above the desired count during the update. | `25%` |
| `nvidia.com/deployment-rolling-update-max-unavailable` | Maximum number of pods that can be unavailable during the update. | `25%` |

Values can be absolute integers (e.g., `"1"`, `"2"`) or percentages (e.g., `"25%"`, `"50%"`). Percentages are resolved against the desired replica count — rounding up for `maxSurge` and rounding down for `maxUnavailable`. The operator ensures at least one of `maxSurge` or `maxUnavailable` is greater than zero to guarantee forward progress.

**Example** — zero-downtime update with surge capacity:

```yaml
Prefill:
  componentType: prefill
  replicas: 4
  annotations:
    nvidia.com/deployment-rolling-update-max-surge: "1"
    nvidia.com/deployment-rolling-update-max-unavailable: "0"
```

This ensures that all 4 existing prefill replicas remain available while 1 new replica is brought up at a time.

**Example** — fast update allowing temporary capacity reduction:

```yaml
Decode:
  componentType: decode
  replicas: 8
  annotations:
    nvidia.com/deployment-rolling-update-max-surge: "0"
    nvidia.com/deployment-rolling-update-max-unavailable: "2"
```

This avoids creating extra pods but allows up to 2 decode replicas to be unavailable at a time, speeding up the transition.

### Worker Hash and DCD Naming

Worker DCDs always include a hash suffix derived from the worker specs: `{dgd-name}-{service-name}-{hash}` (e.g., `vllm-disagg-vllmdecodeworker-a1b2c3d4`). During a rolling update, the new worker DCDs are created with the new spec hash while the old DCDs retain the previous hash, allowing both generations to coexist:

- **Old worker DCD:** `vllm-disagg-vllmdecodeworker-a1b2c3d4` (previous hash)
- **New worker DCD:** `vllm-disagg-vllmdecodeworker-f5e6d7c8` (new hash)

The hash is computed from a SHA-256 digest of all worker service specs (excluding non-pod-template fields like `replicas`, `autoscaling`, and `ingress`). This means:

- Scaling changes (replica count) do **not** trigger a rolling update.
- Pod template changes (image, resources, env vars, volumes, etc.) **do** trigger a rolling update.
- The hash covers **all** worker services together — changing any single worker's spec triggers a rolling update for all workers.

The current worker hash is stored as the annotation `nvidia.com/current-worker-hash` on the DGD resource, and individual worker DCDs are labeled with `nvidia.com/dynamo-worker-hash` for filtering.

### Status During Rolling Updates

During a rolling update, the DGD status aggregates information from both old and new worker DCDs:

- **Replicas** — Total count across old and new.
- **ReadyReplicas** — Aggregate ready count across old and new.
- **UpdatedReplicas** — Only new worker replicas.

This provides a holistic view of the deployment's health during the transition.

## Comparison

| Aspect | Grove / LWS | Deployments (Managed) |
|--------|-------------|----------------------|
| Update mechanism | Native resource rolling update | Operator-managed with DCD lifecycle |
| Namespace isolation | No — old and new share the same namespace | Yes — hash-based namespace separation |
| Cross-generation discovery | Possible — old and new workers can see each other | Prevented — new workers only discover new workers |
| maxSurge / maxUnavailable | Fixed (`maxUnavailable: 1`, `maxSurge: 0` for Grove) | Configurable per service via annotations |
| Status tracking | Native resource status | DGD `.status.rollingUpdate` with phase and per-service tracking |
| Multinode support | Yes | No (single-node only) |

## Future Work

The following enhancements are planned for future releases:

- **Managed rolling updates for Grove and LWS** — Extending managed rolling updates with namespace isolation to Grove and LWS-backed deployments, providing the same cross-generation discovery protection that Deployment-backed DGDs have today.
- **Coordinated worker updates** — Currently, prefill and decode workers are updated independently, which can result in an imbalance between old and new sets during the transition. Future releases will coordinate the rollout across worker types.
- **Partitioned rollouts** — The ability to roll out updates to a percentage of workers (e.g., 30%), pause, observe metrics, and then continue. This enables canary-style deployments for safer rollouts.
- **DGD-level rolling update configuration** — The ability to configure `maxSurge` and `maxUnavailable` at the DGD API level, regardless of the backing resource type.
