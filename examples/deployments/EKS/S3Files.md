<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Create an Amazon S3 Files file system for Amazon EKS

This guide walks through creating an Amazon S3 Files file system and connecting it to your EKS cluster. The EFS CSI Driver was already installed as an addon via `eksctl.yaml` during cluster creation. Now we need to create the actual file system and make it available to Kubernetes workloads.

This filesystem will be used by Dynamo to store shared model weights and compilation cache across nodes.

## Prerequisites

- EKS cluster created following the [README](README.md)
- AWS CLI version (>2.34.26) that supports Amazon S3 Files
- Environment variables set:

```bash
export AWS_REGION="us-east-1"
export CLUSTER_NAME="ai-dynamo"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --output json | jq -r '.Account')
export S3_BUCKET_NAME="${CLUSTER_NAME}-s3files-${AWS_ACCOUNT_ID}-${AWS_REGION}"
export DYNAMO_NAMESPACE="dynamo-system"
```

## Create S3 bucket

Create an S3 bucket with versioning enabled - required for S3 Files.

```bash
aws s3 mb --region "$AWS_REGION" s3://${S3_BUCKET_NAME}
aws s3api put-bucket-versioning --bucket "$S3_BUCKET_NAME" --versioning-configuration Status=Enabled
```

## Create S3 Files file system IAM Role

Create the [S3 Files IAM Role](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-files-prereq-policies.html#s3-files-prereq-iam), this allows accessing the S3 Bucket from the S3 Files file system and attaching the file system to AWS compute resources.

```bash
aws iam create-role --role-name "${CLUSTER_NAME}-S3FilesRole" --assume-role-policy-document "$(envsubst < templates/s3files-trustpolicy.json)"
aws iam create-policy --policy-name "${CLUSTER_NAME}-S3FilesPolicy" --policy-document "$(envsubst < templates/s3files-policy.json)"
aws iam attach-role-policy --role-name "${CLUSTER_NAME}-S3FilesRole" --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${CLUSTER_NAME}-S3FilesPolicy"
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

## Create a Security Group for S3 Files

Create a security group that allows NFS traffic (port 2049) from within the VPC:

```bash
export S3FILES_SG_ID=$(aws ec2 create-security-group \
  --group-name "${CLUSTER_NAME}-s3files-sg" \
  --description "Security group for S3 Files access from EKS" \
  --vpc-id "$VPC_ID" \
  --region "$AWS_REGION" \
  --query "GroupId" \
  --output text)
```

Add an inbound rule to allow NFS traffic from the VPC CIDR:

```bash
aws ec2 authorize-security-group-ingress \
  --group-id "$S3FILES_SG_ID" \
  --protocol tcp \
  --port 2049 \
  --cidr "$VPC_CIDR" \
  --region "$AWS_REGION"
```

## Create the S3 Files file system

```bash
export S3FILES_FS_ID=$(aws s3files create-file-system \
  --region "$AWS_REGION" \
  --bucket "arn:aws:s3:::$S3_BUCKET_NAME" \
  --role-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CLUSTER_NAME}-S3FilesRole" \
  --tags key=Name,value="fs-${S3_BUCKET_NAME}" \
  --query "fileSystemId" \
  --output text)
```

Wait for the file system to become available:

```bash
aws s3files get-file-system \
  --region "$AWS_REGION" \
  --file-system-id "$S3FILES_FS_ID" \
  --query "status" \
  --output text
```

You should see `available` before proceeding.

## Create Mount Targets

Mount targets allow your EKS nodes to access the S3 Files file system. You need one mount target per subnet where your nodes run.

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
  aws s3files create-mount-target \
    --file-system-id "$S3FILES_FS_ID" \
    --subnet-id "$SUBNET_ID" \
    --security-groups "$S3FILES_SG_ID" \
    --region "$AWS_REGION" 2>/dev/null || echo "  Mount target already exists or subnet is in a duplicate AZ (this is OK)"
done
```

> **Note:** S3 Files file system allows only one mount target per Availability Zone. If multiple subnets are in the same AZ, the command will fail for the duplicates, which is expected and safe to ignore.

Verify mount targets are available:

```bash
aws s3files list-mount-targets \
  --file-system-id "$S3FILES_FS_ID" \
  --region "$AWS_REGION" \
  --query "mountTargets[*].{subnetId:subnetId,AZ:availabilityZoneId,State:status}" \
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
  name: s3files-sc-dynamic
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: s3files-ap
  fileSystemId: "${S3FILES_FS_ID}"
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
  storageClassName: "s3files-sc-dynamic"
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
  storageClassName: "s3files-sc-dynamic"
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
  storageClassName: "s3files-sc-dynamic"
EOF
```

> **Note:**  The `storage` value in the PVC is required by Kubernetes but does not limit the actual storage.

## Verify

Confirm the PVC is bound:

```bash
kubectl get pvc -n ${DYNAMO_NAMESPACE}
```

You should see output similar to:

```
NAME                STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS         VOLUMEATTRIBUTESCLASS   AGE
compilation-cache   Bound    pvc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   5Gi        RWX            s3files-sc-dynamic   <unset>                 41s
model-cache         Bound    pvc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   5Gi        RWX            s3files-sc-dynamic   <unset>                 42s
perf-cache          Bound    pvc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   5Gi        RWX            s3files-sc-dynamic   <unset>                 41s
```

## Cleanup

To delete the S3 Files file system resources when no longer needed:

```bash
# Delete the Kubernetes resources
kubectl delete pvc model-cache compilation-cache perf-cache -n ${DYNAMO_NAMESPACE}
kubectl delete storageclass s3files-sc-dynamic

# Delete mount targets
for MT_ID in $(aws s3files list-mount-targets --file-system-id $S3FILES_FS_ID --region $AWS_REGION --query "mountTargets[*].mountTargetId" --output text); do
  aws s3files delete-mount-target --mount-target-id $MT_ID --region $AWS_REGION
done

# Delete the S3 Files file system
aws s3files delete-file-system --file-system-id $S3FILES_FS_ID --region $AWS_REGION

# Delete the security group
aws ec2 delete-security-group --group-id $S3FILES_SG_ID --region $AWS_REGION

# Delete IAM roles
aws iam detach-role-policy --role-name "${CLUSTER_NAME}-S3FilesRole" --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${CLUSTER_NAME}-S3FilesPolicy"
aws iam delete-role --role-name "${CLUSTER_NAME}-S3FilesRole"

# Delete IAM policies
aws iam delete-policy --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${CLUSTER_NAME}-S3FilesPolicy"
```

### Delete the S3 Bucket

For faster deletion process, you can delete the S3 Bucket from the AWS Management Console.

```bash
# Delete S3 bucket and all if it's content (THIS WILL DELETE ALL THE BUCKET CONTENTS)
# Delete all objects versions
aws s3api list-object-versions --region "$AWS_REGION" --bucket "$S3_BUCKET_NAME" | \
jq -r '.Versions[] | [.Key, .VersionId] | @tsv' | \
while read -r key version; do
    aws s3api delete-object --region "$AWS_REGION" --bucket "$S3_BUCKET_NAME" --key "$key" --version-id "$version"
done
# Clean objects delete markers
aws s3api list-object-versions --region "$AWS_REGION" --bucket "$S3_BUCKET_NAME" --query 'DeleteMarkers[?IsLatest==`true`].[Key,VersionId]' --output text | while read key versionId; do
  aws s3api delete-object --region "$AWS_REGION" --bucket "$S3_BUCKET_NAME" --key "$key" --version-id "$versionId"
done
# Delete bucket
aws s3 rb s3://${S3_BUCKET_NAME} --force
```
