<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Pre-Deployment Check Script

This directory contains a pre-deployment check script that verifies your Kubernetes cluster meets the requirements for deploying Dynamo.

- For NCCL tests, please refer to the [NCCL tests](https://docs.nebius.com/kubernetes/gpu/nccl-test#run-tests) for more details.

For the latest pre-deployment check instructions, see the [main branch version of this README](https://github.com/ai-dynamo/dynamo/blob/main/deploy/pre-deployment/README.md).

## Usage

Run the pre-deployment check before deploying Dynamo:

```bash
./pre-deployment-check.sh
```

## What it checks

The script performs few checks and provides a detailed summary:

### 1. kubectl Connectivity
- Verifies that `kubectl` is installed and kubectl can connect to your Kubernetes cluster

### 2. Default StorageClass
- Verifies that a default StorageClass is configured in your cluster
- If no default StorageClass is found:
  - Lists all available StorageClasses in the cluster with full details
  - Provides a sample command to set a StorageClass as default
  - References the official Kubernetes documentation for detailed guidance

### 3. Cluster GPU Resources
- Checks for GPU-enabled nodes in the cluster using label `nvidia.com/gpu.present=true`

## Sample Output

### Complete Script Output Example:
```
========================================
  Dynamo Pre-Deployment Check Script
========================================

--- Checking kubectl connectivity ---
✅ kubectl is available and cluster is accessible

--- Checking for default StorageClass ---
❌ No default StorageClass found

Dynamo requires a default StorageClass for persistent volume provisioning.
Please configure a default StorageClass before proceeding with deployment.

Available StorageClasses in your cluster:
NAME                                 PROVISIONER                     RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
my-default-storage-class (default)   compute.csi.mock                Delete          WaitForFirstConsumer   true                   65d
fast-ssd-storage                     kubernetes.io/gce-pd            Delete          Immediate              true                   30d

To set a StorageClass as default, use the following command:
kubectl patch storageclass <storage-class-name> -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

Example with your first available StorageClass:
kubectl patch storageclass my-default-storage-class -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

For more information on managing default StorageClasses, visit:
https://kubernetes.io/docs/tasks/administer-cluster/change-default-storage-class/

--- Checking cluster gpu resources ---
✅ Found 17 gpu node(s) in the cluster
Node information:

--- Pre-Deployment Check Summary ---
✅ kubectl Connectivity: PASSED
❌ Default StorageClass: FAILED
✅ Cluster Resources: PASSED

Summary: 2 passed, 1 failed
❌ 1 pre-deployment check(s) failed.
Please address the issues above before proceeding with deployment.
```

### When all checks pass:
```
========================================
  Dynamo Pre-Deployment Check Script
========================================


--- Checking kubectl connectivity ---
✅ kubectl is available and cluster is accessible

--- Checking for default StorageClass ---
✅ Default StorageClass found
  - NAME                               PROVISIONER      RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
my-default-storage-class (default)   compute.csi.mock   Delete          WaitForFirstConsumer   true                   65d

--- Checking cluster gpu resources ---
✅ Found 17 gpu node(s) in the cluster
Node information:


--- Pre-Deployment Check Summary ---
✅ kubectl Connectivity: PASSED
✅ Default StorageClass: PASSED
✅ Cluster Resources: PASSED

Summary: 3 passed, 0 failed
🎉 All pre-deployment checks passed!
Your cluster is ready for Dynamo deployment.
```

## Check Status Summary

The script provides a comprehensive summary showing the status of each check:

| Check Name | Description | Pass/Fail Status |
|------------|-------------|------------------|
| **kubectl Connectivity** | Verifies kubectl installation and cluster access | ✅ PASSED / ❌ FAILED |
| **Default StorageClass** | Checks for default StorageClass annotation | ✅ PASSED / ❌ FAILED |
| **Cluster Resources** | Validates GPU nodes availability | ✅ PASSED / ❌ FAILED |

## Setting a Default StorageClass

If you need to set a default StorageClass, use the following command:

```bash
kubectl patch storageclass <storage-class-name> -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

Replace `<storage-class-name>` with the name of your desired StorageClass.

## Troubleshooting

### Multiple Default StorageClasses
If you have multiple StorageClasses marked as default, the script will warn you:
```
⚠️  Warning: Multiple default StorageClasses detected
   This may cause unpredictable behavior. Consider having only one default StorageClass.
```

To remove the default annotation from a StorageClass:
```bash
kubectl patch storageclass <storage-class-name> -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'
```

### No GPU Nodes Found
If no GPU nodes are found, ensure your cluster has nodes with the `nvidia.com/gpu.present=true` label.

### No StorageClasses Available
If no StorageClasses are available in your cluster, you'll need to:
1. Install a storage provisioner (e.g., for cloud providers, local storage, etc.)
2. Create appropriate StorageClass resources
3. Mark one as default

## Reference

For more information on managing default StorageClasses, visit:
[Kubernetes Documentation - Change the default StorageClass](https://kubernetes.io/docs/tasks/administer-cluster/change-default-storage-class/)