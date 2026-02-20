---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Topology Aware Scheduling
---

Topology Aware Scheduling (TAS) lets you control where Dynamo places inference workload pods relative to the cluster's network topology. By packing related pods within the same rack, block, or other topology domain, you reduce inter-node latency and improve throughput — especially for disaggregated serving where prefill, decode, and routing components communicate frequently.

TAS is **opt-in**. Existing deployments without topology constraints continue to work unchanged.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Grove** | Installed on the cluster. See the [Grove Installation Guide](https://github.com/ai-dynamo/grove/blob/main/docs/installation.md). |
| **ClusterTopology CR** | A cluster-scoped `ClusterTopology` resource configured by the cluster admin, mapping topology domain names to node labels. See [Grove documentation](https://github.com/ai-dynamo/grove) for setup instructions. |
| **KAI Scheduler** | [KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler) is required by Grove for topology-aware pod placement. |
| **Dynamo operator** | The latest Dynamo operator Helm chart includes read-only RBAC for `clustertopologies.grove.io` via a dedicated ClusterRole. This works for both cluster-wide and namespace-restricted operator deployments — no extra configuration is needed. |

## Topology Domains

Dynamo defines a fixed set of abstract topology domains, ordered from broadest to narrowest:

| Domain | Description |
|--------|-------------|
| `region` | Cloud region or geographic area (broadest) |
| `zone` | Availability zone within a region |
| `datacenter` | Physical datacenter |
| `block` | Network block within a datacenter |
| `rack` | Server rack |
| `host` | Individual host / node |
| `numa` | NUMA node within a host (narrowest) |

These are Dynamo's own abstract terms. They map 1:1 to Grove's topology domain names today. For future non-Grove frameworks, a translation layer handles the mapping without changing the DGD API.

When you specify a `packDomain`, the scheduler packs all replicas of the constrained component within a single instance of that domain. For example, `packDomain: rack` means "place all pods within the same rack."

## Enabling TAS on a DGD

Add a `topologyConstraint` field to your `DynamoGraphDeployment` at the deployment level, at the service level, or both. Each constraint specifies a `packDomain`.

### Example 1: Deployment-Level Constraint (Services Inherit)

All services inherit the deployment-level constraint. This is the simplest configuration when you want uniform topology packing.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
spec:
  topologyConstraint:
    packDomain: zone
  services:
    VllmWorker:
      dynamoNamespace: my-llm
      componentType: worker
      replicas: 2
      envFromSecret: hf-token-secret
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: my-image
          command: ["/bin/sh", "-c"]
          args:
            - python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B
    Frontend:
      dynamoNamespace: my-llm
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: my-image
          command: ["/bin/sh", "-c"]
          args:
            - python3 -m dynamo.frontend
```

### Example 2: Service-Level Constraint Only

Only the specified service gets topology packing. Other services are scheduled without topology constraints.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
spec:
  services:
    VllmWorker:
      dynamoNamespace: my-llm
      componentType: worker
      replicas: 2
      multinode:
        nodeCount: 4
      topologyConstraint:
        packDomain: rack
      envFromSecret: hf-token-secret
      resources:
        limits:
          gpu: "8"
      extraPodSpec:
        mainContainer:
          image: my-image
          command: ["/bin/sh", "-c"]
          args:
            - python3 -m dynamo.vllm --model meta-llama/Llama-4-Maverick-17B-128E
    Frontend:
      dynamoNamespace: my-llm
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: my-image
          command: ["/bin/sh", "-c"]
          args:
            - python3 -m dynamo.frontend
```

### Example 3: Mixed (Deployment-Level Default + Per-Service Override)

Set a broad constraint at the deployment level and a narrower override on specific services. Service-level constraints must be **equal to or narrower than** the deployment-level constraint.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
spec:
  topologyConstraint:
    packDomain: zone
  services:
    VllmWorker:
      dynamoNamespace: my-llm
      componentType: worker
      replicas: 2
      multinode:
        nodeCount: 4
      topologyConstraint:
        packDomain: block    # narrower than zone — valid
      envFromSecret: hf-token-secret
      resources:
        limits:
          gpu: "8"
      extraPodSpec:
        mainContainer:
          image: my-image
          command: ["/bin/sh", "-c"]
          args:
            - python3 -m dynamo.vllm --model meta-llama/Llama-4-Maverick-17B-128E
    Frontend:
      dynamoNamespace: my-llm
      componentType: frontend
      replicas: 1
      # inherits zone from spec.topologyConstraint
      extraPodSpec:
        mainContainer:
          image: my-image
          command: ["/bin/sh", "-c"]
          args:
            - python3 -m dynamo.frontend
```

## Hierarchy Rules

When **both** a deployment-level and a service-level `topologyConstraint` are set, the service's `packDomain` must be **equal to or narrower** than the deployment-level `packDomain`. The Dynamo webhook rejects the DGD if a service constraint is broader than the deployment constraint.

When only one level is set (deployment-level only or service-level only), no hierarchy check applies.

| Configuration | Behavior |
|---------------|----------|
| `spec.topologyConstraint` set, service has none | Service inherits the deployment-level constraint |
| `spec.topologyConstraint` set, service also set | Both applied; service must be narrower or equal |
| `spec.topologyConstraint` not set, service set | Only that service gets a topology constraint |
| Neither set | No topology constraints (default) |

## Multinode Considerations

For multinode services (services with a `multinode` section), the topology constraint is applied at the **scaling group** level rather than on individual worker pods. This is important because a multinode service spawns `replicas × nodeCount` pods — for example, 2 replicas with `nodeCount: 4` produces 8 pods across 8 nodes. Applying the constraint at the scaling group level means the scheduler packs each replica's set of nodes within the requested domain, without over-constraining individual pods to a single host.

For example, with this configuration:

```yaml
VllmWorker:
  replicas: 2
  multinode:
    nodeCount: 4
  topologyConstraint:
    packDomain: rack
```

Each replica's 4 nodes are packed within a single rack. The two replicas may land in different racks (the constraint applies per-replica, not across all replicas).

**Recommendation:** For multinode services, use `rack` or `block` as the `packDomain` to keep workers within a high-bandwidth domain while still allowing the scheduler to spread them across hosts within that domain. Avoid `host` for multinode services, as packing multiple nodes onto one host is not meaningful.

## Immutability

Topology constraints **cannot be changed after the DGD is created**. This includes:

- Adding a topology constraint to a DGD or service that did not have one
- Removing an existing topology constraint
- Changing the `packDomain` value

To change topology constraints, **delete and recreate** the DGD. This matches the behavior of the underlying framework, which enforces immutability on topology constraints for generated resources.

## Monitoring Topology Enforcement

When any topology constraint is set, the DGD status includes a `TopologyLevelsAvailable` condition that reports whether the topology levels referenced by your constraints still exist in the cluster topology.

**Healthy state:**

```yaml
status:
  conditions:
    - type: Ready
      status: "True"
    - type: TopologyLevelsAvailable
      status: "True"
      reason: AllTopologyLevelsAvailable
      message: "All required topology levels are available in the cluster topology"
```

**Degraded state** (e.g., an admin removed a topology level from the `ClusterTopology` CR after deployment):

```yaml
status:
  conditions:
    - type: Ready
      status: "True"
    - type: TopologyLevelsAvailable
      status: "False"
      reason: TopologyLevelsUnavailable
      message: "Topology level 'rack' is no longer available in the cluster topology"
```

When topology levels become unavailable, Dynamo emits a **Warning** event on the DGD. The deployment may still appear `Ready` because the underlying framework keeps pods running, but topology placement is no longer guaranteed.

## Troubleshooting

### DGD rejected: "ClusterTopology not found"

The Dynamo webhook validates that the framework's topology resource exists when any topology constraint is set. If it cannot read the `ClusterTopology` CR:

- Verify that the cluster admin has created a `ClusterTopology` resource. See the [Grove documentation](https://github.com/ai-dynamo/grove) for setup.
- Verify that the Dynamo operator has RBAC to read `clustertopologies.grove.io` (included in the default Helm chart).

### DGD rejected: "packDomain not found in cluster topology"

The specified `packDomain` does not exist as a level in the cluster's `ClusterTopology` CR. Check which domains are defined:

```bash
kubectl get clustertopology -o yaml
```

Ensure the domain you are requesting (e.g., `rack`) is configured in the `ClusterTopology` with a corresponding node label.

### Pods stuck in Pending

The scheduler cannot satisfy the topology constraint. Common causes:

- Not enough nodes within a single instance of the requested domain (e.g., requesting 8 GPUs packed in one rack, but no rack has 8 available GPUs).
- Node labels do not match the `ClusterTopology` configuration.

Inspect scheduler events for details:

```bash
kubectl describe pod <pod-name> -n <namespace>
```

### TopologyLevelsAvailable is False

The DGD was deployed successfully, but the topology definition has since changed. The underlying framework detected that one or more required topology levels are no longer available.

- Check the condition message for specifics.
- Inspect the `ClusterTopology` CR to see if a domain was removed or renamed.
- If the topology was intentionally changed, delete and recreate the DGD to pick up the new topology.

### DGD rejected: hierarchy violation

A service-level `packDomain` is broader than the deployment-level `packDomain`. For example, `spec.topologyConstraint.packDomain: rack` with a service specifying `zone` (which is broader) is invalid.

Ensure service-level constraints are equal to or narrower than the deployment-level constraint.
