---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multinode Orchestration
---

Multinode deployments require either Grove + KAI Scheduler or an alternative orchestrator setup (LeaderWorkerSet + Volcano) to enable gang scheduling for workloads that span multiple nodes.

## Option 1: Grove + KAI Scheduler

Grove is the default and recommended orchestrator for multinode deployments. It requires KAI Scheduler as well. There are two ways to enable Grove and KAI Scheduler, either dynamo can install it automatically (recommended for development and testing), or you can install them separately (recommended for production).

<Tabs>
  <Tab title="Managed Installation" value="managed">

  The managed installation is recommended for development and testing. It is the simplest path, and allows Dynamo to manage the lifecycle of Grove and KAI Scheduler as bundled subcharts. Run the following command to install Dynamo with Grove and KAI Scheduler:

  ```bash
  helm upgrade --install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "global.grove.install=true" \
  --set "global.kai-scheduler.install=true" \
  --set-string "kai-scheduler.scheduler.args.default-staleness-grace-period=-1s"
  ```
  </Tab>
  <Tab title="External Installation" value="external">
  The external installation is recommended for production or if it's already installed on your system. It allows you to install Grove and KAI Scheduler separately, and manage their lifecycle independently or share them across namespaces.

  See the [Grove installation guide](https://github.com/NVIDIA/grove/blob/main/docs/installation.md) and [KAI Scheduler deployment guide](https://github.com/NVIDIA/KAI-Scheduler) for instructions.

  Then, run the following command to install or configure Dynamo to use the existing Grove and KAI Scheduler:

  ```bash
  helm upgrade --install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "global.grove.enabled=true" \
  --set "global.kai-scheduler.enabled=true"
  ```

  > [!NOTE]
  > If you install Grove and KAI Scheduler externally, ensure that the versions are compatible with the Dynamo Platform version you are installing. The following table shows the minimum required versions for each component:
  >
  > | dynamo-platform | kai-scheduler | Grove |
  > |-----------------|---------------|-------|
  > | 1.0.x           | >= v0.13.0    | >= v0.1.0-alpha.6 |
  > | 1.1.x           | >= v0.13.4    | >= v0.1.0-alpha.8 |
  > | 1.5.x           | >= v0.17.0    | >= v0.1.0-alpha.13 |
  > | 1.6.x           | >= v0.17.0    | >= v0.1.0-alpha.14 |

  </Tab>
</Tabs>


## Option 2: LeaderWorkerSet + Volcano

If you are not using Grove for multinode, you can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/installation/) with [Volcano](https://github.com/volcano-sh/volcano#quick-start-guide) for gang scheduling. Use an LWS release that serves `disaggregatedset.x-k8s.io/v1` (the current Dynamo operator dependency is LWS `v0.10.0`) when enabling DS; the standard DCD pathway supports LWS `v0.7.0` or newer. Both must be installed before deploying multinode workloads.

1. Install Volcano:

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

2. Install LWS with Volcano gang scheduling enabled. Use an LWS release that serves `disaggregatedset.x-k8s.io/v1` (the current Dynamo operator dependency is LWS `v0.10.0`) when enabling DS; the standard DCD pathway supports LWS `v0.7.0` or newer:

```bash
export LWS_VERSION=0.10.0
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=$LWS_VERSION \
  --namespace lws-system \
  --create-namespace \
  --set gangSchedulingManagement.schedulerProvider=volcano \
  --wait --timeout 300s
```

See the [LWS docs](https://lws.sigs.k8s.io/docs/) and [Volcano docs](https://github.com/volcano-sh/volcano#quick-start-guide) for configuration options.
## Role-Specific Pod Templates

**Available since Dynamo 1.6.0.** By default, a multinode component uses one component-level
`podTemplate` for its leader and worker roles, and Dynamo generates the backend-specific launch
details. Use `roles[].podTemplate` when the roles need different images, resources, placement, or
commands.

Role Pod templates are complete inputs rather than patches. Omit the component-level `podTemplate`
and provide one template for both `leader` and `worker`. `multinode.nodeCount` remains the source of
truth for cardinality: admission defaults the leader to one Pod and the worker role to
`nodeCount - 1` Pods.

The following vLLM component runs one leader and one worker on different node pools. Each Pod uses
one GPU, and the two Pods form one TP-2 engine:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: role-specific-multinode
spec:
  backendFramework: vllm
  components:
  - name: decode
    type: decode
    replicas: 1
    multinode:
      nodeCount: 2
    roles:
    - name: leader
      podTemplate:
        spec:
          nodeSelector:
            example.com/engine-role: leader
          containers:
          - name: main
            image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
            command: [python3, -m, dynamo.vllm]
            args:
            - --model
            - Qwen/Qwen3-8B
            - --tensor-parallel-size
            - "2"
            - --distributed-executor-backend
            - mp
            - --nnodes
            - "2"
            - --node-rank
            - $(DYNAMO_RANK)
            - --master-addr
            - $(DYNAMO_LEADER_ADDRESS)
            resources:
              limits:
                nvidia.com/gpu: "1"
    - name: worker
      podTemplate:
        spec:
          nodeSelector:
            example.com/engine-role: worker
          containers:
          - name: main
            image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
            command: [python3, -m, dynamo.vllm]
            args:
            - --model
            - Qwen/Qwen3-8B
            - --tensor-parallel-size
            - "2"
            - --distributed-executor-backend
            - mp
            - --nnodes
            - "2"
            - --node-rank
            - $(DYNAMO_RANK)
            - --master-addr
            - $(DYNAMO_LEADER_ADDRESS)
            - --headless
            resources:
              limits:
                nvidia.com/gpu: "1"
```

Complete role templates transfer ownership of the role-dependent backend command to the manifest.
Dynamo preserves the authored vLLM, SGLang, or TensorRT-LLM topology arguments instead of generating
Leader and Worker launch commands. It still owns Services, labels, volumes, portable topology
variables, provider resources, rollout, readiness, status, and scaling. For vLLM multiprocessing,
Dynamo also adds `--master-port=29500` and keeps the worker wait-for-leader wiring on that port.
For TensorRT-LLM, the authored worker command must start `sshd` on port `2222`; Dynamo retains the
worker readiness probe on that port and mounts the MPI SSH key.

The two template-source modes are exclusive:

```text
Valid:   component.podTemplate
Valid:   leader.podTemplate + worker.podTemplate
Invalid: component.podTemplate + any role.podTemplate
Invalid: a podTemplate on only one required role
```

> [!IMPORTANT]
> A DGD that includes a Planner and uses role Pod templates requires Planner runtime 1.6.0 or later.
> Set the Planner's `runtimeVersionOverride` when its image tag does not identify that version.
> Role Pod templates cannot currently be combined with GPU Memory Service (GMS) or failover.

See the [`roles` API reference](../../reference/kubernetes-api/dynamo-component-deployment.mdx#shared-component-spec)
for validation and lifecycle details.

## Portable Topology Environment Variables

**Available since Dynamo 1.6.0.** The operator injects two provider-independent environment
variables into the `main` container of each new multinode DynamoGraphDeployment (DGD) component:

| Variable | Meaning | Grove source | LWS source |
|---|---|---|---|
| `DYNAMO_RANK` | Zero-based engine node rank. The leader is `0`; workers use `1` through `multinode.nodeCount - 1`. | `GROVE_PCSG_POD_INDEX` | `LWS_WORKER_INDEX` |
| `DYNAMO_LEADER_ADDRESS` | DNS hostname of the engine leader, without a port. | PodCliqueScalingGroup leader DNS name | `LWS_LEADER_ADDRESS` |

For an inter-pod GPU Memory Service (GMS) deployment, the operator sets `DYNAMO_RANK` from the
engine role instead of the PCSG-wide pod index because the scaling group also contains weight-server
cliques.

The operator owns these variables and replaces values with the same names from the component's
`podTemplate`. Use the aliases instead of provider-specific `GROVE_*` or `LWS_*` variables when a
custom command needs the component topology.

Kubernetes expands `$(NAME)` references in a container's `command` and `args`. A shell process that
runs inside the container reads the same environment variables with `$NAME` or `${NAME}`. For
example, an engine command can receive the topology as separate arguments:

```yaml
args:
- --master-addr
- $(DYNAMO_LEADER_ADDRESS)
- --master-port
- "29500"
- --node-rank
- $(DYNAMO_RANK)
```

> [!NOTE]
> Do not add these flags to standard automatically configured vLLM or SGLang components. In vLLM
> multiprocessing mode, the operator uses the aliases for `--master-addr` and `--node-rank`. For
> SGLang, it uses them for `--dist-init-addr` and `--node-rank`. Custom launch commands can reference
> the aliases when they need the same topology values.

The immutable operator origin version controls this behavior. An operator-only upgrade does not add
the variables or change the command line of an existing DGD, so it does not roll that workload. To
use the aliases, create the DGD with Dynamo Operator 1.6.0 or later. When Grove is managed outside
the Dynamo platform chart, install Grove `v0.1.0-alpha.14` or later before creating the DGD.
