<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR deployment examples

This directory holds manifests and scripts that deploy two aggregated vLLM workers on separate GPU
nodes with KVCR as a secondary KV-cache tier over RDMA.

For prerequisites, deployment steps, and configuration, see
[Deploy KVCR](../../../../../docs/fern/pages/kubernetes/kv-cache-offloading/kvcr.md).

## Contents

| Path | Description |
|---|---|
| `agg.yaml` | DGD with in-process KVCR using process-local host memory |
| `agg-memory-service.yaml` | DGD with a `kvcr-services` sidecar running the KVCR memory service, whose pool and Guard survive an engine restart |
| `deploy.sh` | Renders and applies the selected variant and its content-addressed script ConfigMap |
| `services.sh` | Sidecar wrapper that starts the KVCR memory service and, on worker 0, the state agent |
