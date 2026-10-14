---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Install with AICR
subtitle: Install the Dynamo Platform on an existing Kubernetes cluster with NVIDIA AI Cluster Runtime (AICR).
---

[NVIDIA AI Cluster Runtime (AICR)](https://github.com/NVIDIA/aicr) generates one version-locked **recipe** for the whole GPU and Dynamo stack, renders it into a deployable **bundle**, and validates the running cluster against the recipe. It is an alternative to the manual steps in [Install Dynamo](install-dynamo.md).

This guide is written against AICR v1.0.0, whose Dynamo recipes install the `dynamo-platform` Helm chart 1.4.2 into the `dynamo-system` namespace. The commands target Amazon EKS with H100 GPUs and Ubuntu nodes; change the `--service`, `--accelerator`, and `--os` values to match your cluster.

> [!NOTE]
> AICR is a separately maintained project. It configures an existing cluster and does not create one. For the full command reference, see the [AICR documentation](https://github.com/NVIDIA/aicr/tree/v1.0.0/docs/user).

## Check for a Dynamo Recipe

AICR covers more environments than Dynamo does. List the combinations that have a Dynamo recipe instead of assuming yours does:

```bash
aicr recipe list --intent inference --platform dynamo
```

Narrow the list with `--service`, `--accelerator`, and `--os`. Each row is a leaf recipe, with two columns that describe it:

- `STATUS`: the structural verdict (`pass`, `warn`, `fail`, or `unknown`). `pass` means the recipe resolves and its charts are pinned. It does not mean the recipe was validated on hardware.
- `COVERAGE`: the validation checks the recipe declares, in the form `R:n D:n P:n C:n` for the readiness, deployment, performance, and conformance phases.

If no row matches your cluster, use the manual steps in [Install Dynamo](install-dynamo.md).

## Prerequisites

- A Kubernetes 1.34 or later cluster, which the Dynamo recipes need for Dynamic Resource Allocation (DRA), with GPU nodes that match the recipe. To create an EKS cluster, see [Amazon EKS](managed-kubernetes/eks/eks-setup.mdx).
- `kubectl` and Helm access with cluster-admin permissions.
- Image pull access from the GPU nodes to the registries the bundle and your Dynamo runtime images use.
- A cluster without existing installs of components the recipe manages. Run `helm list -A` and remove or plan around an existing GPU Operator, cert-manager, kube-prometheus-stack, or `dynamo-platform` release, because the bundle installs its own.

## Install the CLI

```bash
brew tap NVIDIA/aicr
brew install aicr
```

Or use the install script:

```bash
curl -sfL https://get.aicr.run | bash -s --
```

Verify the installation:

```bash
aicr --version
```

## Generate the Recipe

```bash
aicr recipe --service eks --accelerator h100 --os ubuntu \
  --intent inference --platform dynamo -o recipe.yaml
```

Review the resolved components and their versions in `recipe.yaml` before you deploy.

## Render and Deploy the Bundle

Render the recipe into a Helm bundle:

```bash
aicr bundle --recipe recipe.yaml --deployer helm --output ./bundles
```

Verify the bundle, then run its `deploy.sh`, which installs each component in dependency order:

```bash
(cd bundles && aicr verify . && chmod +x deploy.sh && ./deploy.sh)
```

> [!IMPORTANT]
> Do not edit files in the generated bundle. `aicr verify` fails when a file no longer matches the bundle's checksums. To change a platform setting, pass `--set` to `aicr bundle` and regenerate the bundle.

To deploy with Argo CD, Flux, or Helmfile, see [Generating Bundles](https://github.com/NVIDIA/aicr/blob/v1.0.0/docs/user/bundling.md) in the AICR documentation.

## Validate with AICR

Validate the cluster against the bundle's recipe, which records what was deployed:

```bash
aicr validate --recipe ./bundles/recipe.yaml --phase deployment --phase conformance
```

`aicr validate` first runs a readiness pre-flight and exits with code 2 if the cluster does not meet the recipe's constraints. It then runs the requested phases:

| Phase | What it checks |
|-------|----------------|
| `deployment` | The recipe's components are installed and healthy. |
| `conformance` | The cluster supports the capabilities the recipe declares, such as DRA, gang scheduling, and metrics collection. |
| `performance` | The cluster meets a throughput threshold with an inference benchmark. Run it with `--phase performance` before production use. |

For the phases a recipe declares, see [Validating a Cluster](https://github.com/NVIDIA/aicr/blob/v1.0.0/docs/user/validation.md).

## Check the Dynamo Components

AICR validation does not replace a direct check of the Dynamo components. Confirm that the CRDs exist and the pods are running:

```bash
kubectl get crd | grep -E "dynamo|grove"
kubectl get pods -n dynamo-system
kubectl get pods -n kai-scheduler
```

The `dynamo-system` namespace holds the Dynamo operator and Grove. Every pod should be `Running` and `Ready`.

Confirm that the GPU nodes advertise their GPUs:

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu'
```

Each GPU node should report a nonzero GPU count.

## Open Network Access on EKS

The Dynamo recipes do not install NATS. The request plane uses TCP and KV events use ZMQ, so frontend and worker pods connect to each other directly. If you place frontend and worker components in different node groups, allow this traffic between their security groups:

| Source | Destination | Port | Purpose |
|--------|-------------|------|---------|
| Frontend | Worker | TCP 1024-65535 | Request plane and KV events |
| Worker | Frontend | TCP 1024-65535 | Response streams |

The `ai-service-metrics` conformance check also needs TCP 9090 access to Prometheus from every node group that can run validation jobs, even when frontend and worker share a node group. For the complete rules and an example, see [EKS Dynamo Networking Prerequisites](https://github.com/NVIDIA/aicr/blob/v1.0.0/docs/integrator/eks-dynamo-networking.md).

Disaggregated deployments also transfer KV cache between prefill and decode workers; see [EFA on AWS](rdma-setup/efa-on-aws.mdx).

## Next Steps

Your cluster is ready. Follow the [Model Deployment guide](../model-deployment/introduction.mdx) to deploy a model with a DynamoGraphDeployment. The bundle owns the platform: change platform settings by regenerating it with `aicr bundle --set`, and put model and serving settings in your DynamoGraphDeployment.

To upgrade a deployed stack, run `aicr upgrade-check` and follow [Upgrading a Deployed Stack](https://github.com/NVIDIA/aicr/blob/v1.0.0/docs/user/upgrading.md). After an upgrade that changes the platform version, update each DynamoGraphDeployment's runtime image (vLLM, SGLang, or TensorRT-LLM) to the matching Dynamo version. AICR v1.0.0 does not document an uninstall path for the Helm deployer. To remove the Dynamo CRDs, see **Clean uninstall?** in [Install Dynamo troubleshooting](install-dynamo.md#troubleshooting).

## Troubleshooting

**`aicr recipe` fails with `requires additional criteria`.** Run `aicr recipe list --intent inference --platform dynamo` and use values from a listed row. A partial query such as `--os ubuntu --accelerator h100` can be ambiguous, so also set `--service`.

**`aicr validate` exits with code 2 before running any phase.** The readiness pre-flight failed. The output names the failed constraint; this recipe requires Kubernetes 1.34 or later.

**AICR jobs stay `Pending`.** The snapshot agent and the inner validation workloads target GPU nodes. If your GPU nodes carry non-standard labels or taints, pass `--node-selector` and `--toleration` to `aicr validate`. These flags do not affect the validator orchestrator Job.

**The `ai-service-metrics` check fails or requests time out between pods.** Security groups can block traffic between node groups. See [Open Network Access on EKS](#open-network-access-on-eks).
