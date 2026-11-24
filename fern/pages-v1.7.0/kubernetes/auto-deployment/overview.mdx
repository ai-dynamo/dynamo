---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Auto Deployment
subtitle: Generate a DynamoGraphDeployment from model, hardware, workload, and latency requirements.
---

Dynamo auto deployment starts with a `DynamoGraphDeploymentRequest` (DGDR). Instead of specifying
every worker, parallelism setting, and replica count, you describe the model and deployment intent.
Dynamo profiles the available options and generates the `DynamoGraphDeployment` (DGD) that serves
traffic.

Auto deployment does not replace the DGD. It provides a way to generate one when you do not already
have a known-good deployment configuration.

## When to Use Auto Deployment

Use a DGDR when:

- You are deploying a new model and hardware combination and do not know the best configuration.
- You want Dynamo to choose parallelism, topology, and replica counts.
- You want to size a deployment for an expected workload or latency target.
- You want to inspect and customize a generated DGD before deploying it.
- You want profiling data to initialize optional Planner autoscaling.

Use a DGD directly when:

- A [recipe or backend template](../model-deployment/introduction.mdx#create-a-dgd) already
  matches your deployment.
- You already know the topology, parallelism, and replica counts.
- You need complete control over the workers and Kubernetes configuration.
- Your model, backend, or hardware combination is not adequately supported by the profiler.

## How DGDR Creates a Deployment

<Steps>
  <Step title="Describe the deployment intent">
    Create a DGDR with the model and optional backend, workload, latency targets, hardware limits,
    model cache, and Planner configuration.
  </Step>
  <Step title="Discover the available hardware">
    The operator discovers the cluster's GPU type, memory, GPU count, and node capacity. You can
    override the discovered values when you want to constrain the search or when the operator runs
    without cluster-wide node permissions.
  </Step>
  <Step title="Profile candidate configurations">
    The Profiler uses AIConfigurator to enumerate and evaluate deployment configurations. The
    `rapid` strategy uses performance estimates, while `thorough` deploys candidates and measures
    them on real GPUs.
  </Step>
  <Step title="Generate the DGD">
    The Profiler selects a configuration and produces a complete DGD containing the worker
    topology, parallelism, replica counts, resources, model settings, and enabled platform features.
  </Step>
  <Step title="Review or deploy the DGD">
    With `autoApply: true`, the operator creates the generated DGD automatically. With
    `autoApply: false`, the DGDR stops at `Ready` so you can inspect, modify, and apply the DGD
    yourself.
  </Step>
  <Step title="Scale at runtime with Planner (optional)">
    When `features.planner` is set, the generated DGD includes the Planner. The Planner observes the
    running deployment and adjusts prefill and decode replicas as traffic changes.
  </Step>
</Steps>

## Auto Deployment Components

| Component | Role |
|---|---|
| **DGDR** | Stores the deployment intent and reports the profiling and generation lifecycle. |
| **Profiler** | Evaluates candidate configurations and generates the DGD before serving begins. |
| **AIConfigurator** | Supplies system metadata, candidate generation, performance estimates, and configuration ranking to the Profiler. |
| **Planner** | Optionally adjusts replica counts after the generated DGD begins serving traffic. |
| **Mocker** | Optionally replaces model-serving workers with simulated workers for testing and experimentation. |

The Profiler and Planner serve different phases. The Profiler runs before deployment to choose a
configuration. The Planner runs with the deployed DGD and changes capacity over time. You can use a
DGDR without enabling the Planner.

## Profiling Strategies

| Strategy | How it works | Cost | Use it when |
|---|---|---|---|
| `rapid` | Evaluates candidates with AIConfigurator performance estimates. | About 30 seconds and no profiling GPUs. | You are getting started, iterating, or using a supported model and hardware combination. |
| `thorough` | Deploys and benchmarks candidate configurations on real GPUs. | Typically 2–4 hours and real GPU capacity. | You need measured results or are tuning a production deployment. |

See the [Profiler Guide](../../developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md) for support, gate checks, picking
behavior, and profiling artifacts.

## Next Steps

<CardGroup cols={3}>
  <Card title="Auto Deploy with DGDR" icon="regular wand-magic-sparkles" href="auto-deploy-with-dgdr.md">
    Build a request step by step and generate your first DGD.
  </Card>
  <Card title="DGDR Templates" icon="regular file-code" href="../../recipes/kubernetes-templates/dgdr.mdx">
    Start from copyable requests for rapid, thorough, Planner, mocker, and override workflows.
  </Card>
  <Card title="Profiler" icon="regular magnifying-glass-chart" href="dynamo-profiler.md">
    Understand candidate evaluation, ranking, and generated profiling data.
  </Card>
  <Card title="Planner" icon="regular chart-line" href="dynamo-planner.mdx">
    Configure optional runtime autoscaling for the generated deployment.
  </Card>
  <Card title="DGDR Reference" icon="regular book" href="../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx">
    Review fields, defaults, lifecycle phases, status, and validation behavior.
  </Card>
  <Card title="Deploy with DGD" icon="regular pen-to-square" href="../model-deployment/deploy-with-dgd.md">
    Write the deployment directly when you already know the desired configuration.
  </Card>
</CardGroup>
