---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multinode Orchestration
---

Multinode deployments require either Grove + KAI Scheduler or an alternative orchestrator setup (LeaderWorkerSet + Volcano) to enable gang scheduling for workloads that span multiple nodes. See the [Multinode Deployment Guide](./deployment/multinode-deployment.md) for details on orchestrator selection and configuration.



## Option 1: Grove + KAI Scheduler

Grove is the default and recommended orchestrator for multinode deployments. It requires KAI Scheduler as well. There are two ways to enable Grove and KAI Scheduler, either dynamo can install it automatically (recommended for development and testing), or you can install them separately (recommended for production). 

<Tabs>
  <Tab title="Managed Installation" value="managed">

  The managed installation is recommended for development and testing. It is the simplest path, and allows Dynamo to manage the lifecycle of Grove and KAI Scheduler as bundled subcharts. Run the following command to install Dynamo with Grove and KAI Scheduler:

  ```
  helm upgrade --install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "global.grove.install=true" \
  --set "global.kai-scheduler.install=true"
  ```
  </Tab>
  <Tab title="External Installation" value="external">
  The external installation is recommended for production or if it's already installed on your system. It allows you to install Grove and KAI Scheduler separately, and manage their lifecycle independently or share them across namespaces. 
  
  See the [Grove installation guide](https://github.com/NVIDIA/grove/blob/main/docs/installation.md) and [KAI Scheduler deployment guide](https://github.com/NVIDIA/KAI-Scheduler) for instructions.

  Then, run the following command to install or configure Dynamo to use the existing Grove and KAI Scheduler:

  ```
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

  </Tab>
</Tabs>


## Option 2: LeaderWorkerSet + Volcano

If you are not using Grove for multinode, you can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/installation/) (>= v0.7.0) with [Volcano](https://github.com/volcano-sh/volcano#quick-start-guide) for gang scheduling. Both must be installed before deploying multinode workloads.

1. Install Volcano:

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

2. Install LWS (>= v0.7.0) with Volcano gang scheduling enabled:

```bash
export LWS_VERSION=0.8.0
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=$LWS_VERSION \
  --namespace lws-system \
  --create-namespace \
  --set gangSchedulingManagement.schedulerProvider=volcano \
  --wait --timeout 300s
```

See the [LWS docs](https://lws.sigs.k8s.io/docs/) and [Volcano docs](https://github.com/volcano-sh/volcano#quick-start-guide) for configuration options, and the [Multinode Deployment Guide](./deployment/multinode-deployment.md) for orchestrator selection.
