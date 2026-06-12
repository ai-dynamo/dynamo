---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: API Reference
---

# Kubernetes API Reference

The Dynamo Kubernetes API reference is generated from the operator CRDs with
`kubectl-doc`. Each resource page renders an interactive YAML-shaped schema with
foldable fields, required/default/validation metadata, and field details.

<CardGroup cols={2}>
  <Card
    title="DynamoGraphDeployment"
    icon="regular diagram-project"
    href="/dynamo/dev/kubernetes-deployment/api-reference/dynamo-graph-deployment"
  >
    Author the model-serving graph and its component-level pod defaults.
  </Card>
  <Card
    title="DynamoGraphDeploymentRequest"
    icon="regular wand-magic-sparkles"
    href="/dynamo/dev/kubernetes-deployment/api-reference/dynamo-graph-deployment-request"
  >
    Request a Dynamo deployment through planner and profiling workflows.
  </Card>
  <Card
    title="DynamoComponentDeployment"
    icon="regular cubes"
    href="/dynamo/dev/kubernetes-deployment/api-reference/dynamo-component-deployment"
  >
    Inspect the per-component workload resources reconciled from a graph.
  </Card>
  <Card
    title="DynamoGraphDeploymentScalingAdapter"
    icon="regular chart-line"
    href="/dynamo/dev/kubernetes-deployment/api-reference/dynamo-graph-deployment-scaling-adapter"
  >
    Configure autoscaling adapters for Dynamo graph deployments.
  </Card>
  <Card
    title="DynamoCheckpoint"
    icon="regular clock-rotate-left"
    href="/dynamo/dev/kubernetes-deployment/api-reference/dynamo-checkpoint"
  >
    Describe checkpoint resources used to restore pods to a warm state.
  </Card>
  <Card
    title="DynamoModel"
    icon="regular database"
    href="/dynamo/dev/kubernetes-deployment/api-reference/dynamo-model"
  >
    Register model metadata consumed by Dynamo model deployment flows.
  </Card>
  <Card
    title="DynamoWorkerMetadata"
    icon="regular tags"
    href="/dynamo/dev/kubernetes-deployment/api-reference/dynamo-worker-metadata"
  >
    Store worker metadata discovered and consumed by Dynamo controllers.
  </Card>
</CardGroup>
