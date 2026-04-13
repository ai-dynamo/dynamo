<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Create an Amazon EFS File System for Amazon EKS

This guide walks through creating an Amazon EFS file system and connecting it to your EKS cluster. The EFS CSI Driver was already installed as an addon via `eksctl.yaml` during cluster creation. Now we need to create the actual file system and make it available to Kubernetes workloads.

This filesystem will be used by Dynamo to store shared model weights and compilation cache across nodes.

## Prerequisites

- EKS cluster created following the [README](README.md)
- Environment variables set:

```bash
export AWS_REGION="us-east-1"
export CLUSTER_NAME="ai-dynamo"
export DYNAMO_NAMESPACE="dynamo-system"
```

## Retrieve VPC and Subnet Information

Get the VPC ID associated with your EKS cluster:

```bash
export VPC_ID=$(aws eks describe-cluster \
  --name "$CLUSTER_NAME" \
  --region "$AWS_REGION" \
  --query "cluster.resourcesVpcConfig.vpcId" \
  --output text)
```

Get the CIDR range for the VPC (used for the security group rule):

```bash
export VPC_CIDR=$(aws ec2 describe-vpcs \
  --vpc-ids "$VPC_ID" \
  --region "$AWS_REGION" \
  --query "Vpcs[0].CidrBlock" \
  --output text)
```

## Create a Security Group for EFS

Create a security group that allows NFS traffic (port 2049) from within the VPC:

```bash
export EFS_SG_ID=$(aws ec2 create-security-group \
  --group-name "${CLUSTER_NAME}-efs-sg" \
  --description "Security group for EFS access from EKS" \
  --vpc-id "$VPC_ID" \
  --region "$AWS_REGION" \
  --query "GroupId" \
  --output text)
```

Add an inbound rule to allow NFS traffic from the VPC CIDR:

```bash
aws ec2 authorize-security-group-ingress \
  --group-id "$EFS_SG_ID" \
  --protocol tcp \
  --port 2049 \
  --cidr "$VPC_CIDR" \
  --region "$AWS_REGION"
```

## Create the EFS File System

```bash
export EFS_FS_ID=$(aws efs create-file-system \
  --performance-mode generalPurpose \
  --throughput-mode elastic \
  --encrypted \
  --region "$AWS_REGION" \
  --tags Key=Name,Value=dynamo-efs \
  --query "FileSystemId" \
  --output text)
```

Wait for the file system to become available:

```bash
aws efs describe-file-systems \
  --file-system-id "$EFS_FS_ID" \
  --region "$AWS_REGION" \
  --query "FileSystems[0].LifeCycleState" \
  --output text
```

You should see `available` before proceeding.

## Create Mount Targets

Mount targets allow your EKS nodes to access the EFS file system. You need one mount target per subnet where your nodes run.

Get the subnet IDs used by your EKS cluster:

```bash
export SUBNET_IDS=$(aws eks describe-cluster \
  --name "$CLUSTER_NAME" \
  --region "$AWS_REGION" \
  --query "cluster.resourcesVpcConfig.subnetIds[]" \
  --output text)

echo "Subnet IDs: $SUBNET_IDS"
```

Create a mount target in each subnet:

```bash
for SUBNET_ID in $(echo "$SUBNET_IDS" | tr '\t' '\n'); do
  echo "Creating mount target in subnet: $SUBNET_ID"
  aws efs create-mount-target \
    --file-system-id "$EFS_FS_ID" \
    --subnet-id "$SUBNET_ID" \
    --security-groups "$EFS_SG_ID" \
    --region "$AWS_REGION" 2>/dev/null || echo "  Mount target already exists or subnet is in a duplicate AZ (this is OK)"
done
```

> **Note:** EFS allows only one mount target per Availability Zone. If multiple subnets are in the same AZ, the command will fail for the duplicates, which is expected and safe to ignore.

Verify mount targets are available:

```bash
aws efs describe-mount-targets \
  --file-system-id "$EFS_FS_ID" \
  --region "$AWS_REGION" \
  --query "MountTargets[*].{SubnetId:SubnetId,AZ:AvailabilityZoneName,State:LifeCycleState}" \
  --output table
```

Wait until all mount targets show `available` in the State column before proceeding.

## Create Kubernetes StorageClass

Create a StorageClass that uses the EFS CSI driver with dynamic provisioning:

```bash
kubectl apply -f - << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc-dynamic
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: "${EFS_FS_ID}"
  directoryPerms: "777"
  uid: "1000"
  gid: "1000"
EOF
```

## Create a PersistentVolumeClaim

We create three separate PVCs because different Dynamo recipe examples reference each one individually:
* `model-cache` stores downloaded model weights (e.g. from HuggingFace).
* `compilation-cache` stores vLLM/TRT-LLM compilation artifacts.
* `perf-cache` stores benchmark traces and performance results.

```bash
# Create the namespace we will use for Dynamo if not already exists
kubectl create namespace ${DYNAMO_NAMESPACE}

# Create PVCs
kubectl apply -f - << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: ${DYNAMO_NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
  storageClassName: "efs-sc-dynamic"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: compilation-cache
  namespace: ${DYNAMO_NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
  storageClassName: "efs-sc-dynamic"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perf-cache
  namespace: ${DYNAMO_NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
  storageClassName: "efs-sc-dynamic"
EOF
```

> **Note:** EFS is elastic, the `storage` value in the PVC is required by Kubernetes but does not limit the actual storage. EFS will grow and shrink automatically.

## Verify

Confirm the PVC is bound:

```bash
kubectl get pvc -n ${DYNAMO_NAMESPACE}
```

You should see output similar to:

```
NAME                STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS     VOLUMEATTRIBUTESCLASS   AGE
compilation-cache   Bound    pvc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   5Gi        RWX            efs-sc-dynamic   <unset>                 41s
model-cache         Bound    pvc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   5Gi        RWX            efs-sc-dynamic   <unset>                 42s
perf-cache          Bound    pvc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   5Gi        RWX            efs-sc-dynamic   <unset>                 41s
```

## Cleanup

To delete the EFS resources when no longer needed:

```bash
# Delete the Kubernetes resources
kubectl delete pvc model-cache compilation-cache perf-cache -n ${DYNAMO_NAMESPACE}
kubectl delete storageclass efs-sc-dynamic

# Delete mount targets
for MT_ID in $(aws efs describe-mount-targets --file-system-id $EFS_FS_ID --region $AWS_REGION --query "MountTargets[*].MountTargetId" --output text); do
  aws efs delete-mount-target --mount-target-id $MT_ID --region $AWS_REGION
done

# Delete the EFS file system
aws efs delete-file-system --file-system-id $EFS_FS_ID --region $AWS_REGION

# Delete the security group
aws ec2 delete-security-group --group-id $EFS_SG_ID --region $AWS_REGION
```
