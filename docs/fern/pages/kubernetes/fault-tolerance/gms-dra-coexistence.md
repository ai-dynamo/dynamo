---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: GMS with Device Plugin and DRA
subtitle: Verify GPU Memory Service shadow engine failover on clusters where the NVIDIA device plugin and the DRA driver allocate the same GPUs.
---

> [!WARNING]
> On clusters where the NVIDIA device plugin also serves `nvidia.com/gpu`
> workloads, the DRA scheduler cannot see device-plugin allocations and can
> place the GMS claim on a GPU that is already fully occupied. The engine then
> fails at startup with a `Free memory on device ... is less than desired GPU
> memory utilization` error, and recreating the pod allocates the same device.

Most existing clusters run the device plugin, so this situation is easy to hit
when trying GMS out. Since Shadow Engine Failover is still experimental, a
temporary workaround is enough to verify the feature: reserve the occupied GPUs
with a blocker ResourceClaim before deploying the GMS workload, so the
generated claims land on free devices. Apply
[gms-dra-blocker.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/gms-dra-blocker.yaml)
with the CEL expression and the request `count` adjusted to the occupied device
UUIDs on your node (list them with `nvidia-smi -L`, or map DRA device names to
UUIDs from the ResourceSlices as shown in the manifest comments), then
deploy the GMS example as usual and confirm the allocation with
`kubectl get resourceclaims`. The blocker pod holds the devices in the DRA
ledger without starting any CUDA process, so the occupied GPUs keep serving
their existing workloads. Delete the blocker Pod and ResourceClaim when the
verification is done. The ResourceClaim spec is immutable, so when the set of
occupied GPUs changes, delete both objects and apply the manifest again with
the updated UUIDs.
