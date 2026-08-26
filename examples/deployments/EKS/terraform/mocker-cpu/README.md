<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CPU Mocker on Amazon EKS with Terraform

This template creates an Amazon EKS cluster and deploys an aggregated NVIDIA Dynamo
`DynamoGraphDeployment` (DGD) backed by CPU-only Mocker workers. The EKS cluster subnets used for
control-plane network interfaces span two Availability Zones (AZs), while the only data-plane node
group, the frontend, and every Mocker pod are constrained to one selected AZ.

This is a development and simulation environment, not a highly available production architecture.
It creates billable AWS resources, including an EKS control plane, a NAT gateway, and EC2 instances.

## Placement Model

Amazon EKS requires cluster subnets in at least two AZs. The template satisfies that requirement with
dedicated cluster subnets, then enforces single-AZ workload placement at two levels:

- The `mocker-cpu` managed node group receives exactly one private subnet, in the workload AZ.
- Both DGD components select `dynamo.nvidia.com/purpose=mocker-cpu` and
  `topology.kubernetes.io/zone=<workload-az>`.

The Mocker pods use preferred host anti-affinity, so Kubernetes spreads them across the CPU nodes when
capacity permits without moving them out of the selected AZ. A single NAT gateway is also placed in
that AZ to avoid cross-AZ egress on image and tokenizer downloads.

## Prerequisites

- Terraform 1.10 or later
- AWS CLI v2 authenticated to the target account
- `kubectl`
- AWS permissions to create VPC, EKS, EC2, IAM, and related resources

The template pulls released Dynamo artifacts from NGC and downloads tokenizer metadata for the public
`Qwen/Qwen3-0.6B` model. It does not require GPUs or a Hugging Face token.

## Deploy

From this directory, create the local variables file:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and replace `203.0.113.10/32` with the public IPv4 address of the machine that
runs Terraform and `kubectl`. Keep the `/32` suffix; the template rejects an unrestricted
`0.0.0.0/0`. You can find that address with:

```bash
curl -s https://checkip.amazonaws.com
```

Keep the `/32` suffix. Set `workload_availability_zone` when you need a specific AZ; when it is `null`,
Terraform selects the first EKS-eligible AZ returned for the Region.

Initialize, review, and apply the configuration:

```bash
terraform init
terraform validate
terraform plan -out=mocker.tfplan
terraform apply mocker.tfplan
```

Creation usually takes several minutes. Terraform installs the Dynamo platform first, waits for its
operator and CRDs, and then installs the local Mocker chart.

## Verify

Load the Terraform outputs and configure `kubectl`:

```bash
AWS_REGION="$(terraform output -raw aws_region)"
CLUSTER_NAME="$(terraform output -raw cluster_name)"
DYNAMO_NAMESPACE="$(terraform output -raw dynamo_namespace)"
DGD_NAME="$(terraform output -raw dgd_name)"
MODEL_NAME="$(terraform output -raw model_name)"
WORKLOAD_AZ="$(terraform output -raw workload_availability_zone)"

aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"
```

Wait for the DGD, and then inspect every pod's placement:

```bash
kubectl wait -n "$DYNAMO_NAMESPACE" \
  --for=condition=Ready "dgd/$DGD_NAME" \
  --timeout=600s

kubectl get nodes \
  -L topology.kubernetes.io/zone,dynamo.nvidia.com/purpose

kubectl get pods -n "$DYNAMO_NAMESPACE" \
  -l "nvidia.com/dynamo-graph-deployment-name=$DGD_NAME" \
  -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,REQUESTED_AZ:.spec.nodeSelector.topology\\.kubernetes\\.io/zone
```

Every node and every `REQUESTED_AZ` value should match `$WORKLOAD_AZ`.

Forward the frontend service:

```bash
kubectl port-forward -n "$DYNAMO_NAMESPACE" "svc/$DGD_NAME-frontend" 8000:8000
```

In another terminal, send an OpenAI-compatible request:

```bash
MODEL_NAME="$(terraform output -raw model_name)"

curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<EOF
  {
    "model": "${MODEL_NAME}",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }
EOF
```

A response confirms that the frontend discovered a Mocker worker and routed the request through the
DGD. A healthy frontend alone does not prove worker discovery, so keep the request as part of the
verification.

## Is single-AZ placement the Dynamo default?

No. A DGD describes components and pod templates; without scheduling constraints, Kubernetes can
place replicas on any eligible node. Dynamo can operate a CPU Mocker deployment across AZs when pod
networking is routable. A balanced multi-AZ design can improve availability, but it also introduces
inter-AZ latency, data transfer charges, and dependencies.

For real disaggregated GPU serving, treat an AZ as a KV-transfer domain. AWS Elastic Fabric Adapter
(EFA) device and OS-bypass traffic cannot span AZs; ordinary IP traffic through the Elastic Network
Adapter (ENA) remains routable. Dynamo's experimental topology-aware KV-transfer policy can keep
prefill-to-decode routing within a zone, but it does not schedule pods or create per-zone capacity.
Multi-AZ designs should define balanced prefill and decode pools in each AZ and pin those components.

See:

- [Amazon EKS VPC and subnet requirements](https://docs.aws.amazon.com/eks/latest/userguide/network-reqs.html)
- [EFA limitations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Dynamo topology-aware KV transfer](../../../../../docs/fern/pages/developer-guide/knowledge-base/kubernetes/multinode/topology-aware-kv-transfer.md)
- [Dynamo Mocker simulation](../../../../../docs/fern/pages/kubernetes/operations/simulation-with-dynosim/mocker-live-simulation.mdx)

## Main Inputs

| Input | Default | Purpose |
|---|---|---|
| `aws_region` | `us-west-2` | AWS Region for all resources |
| `workload_availability_zone` | First eligible AZ | AZ used by all worker nodes and DGD pods |
| `cluster_endpoint_public_access_cidrs` | Required | Networks allowed to reach the public EKS API |
| `vpc_cidr` | `10.42.0.0/16` | `/16` through `/20` VPC CIDR; derived subnets are `/20` through `/24` |
| `kubernetes_version` | `1.35` | EKS Kubernetes minor version |
| `node_instance_types` | `m7i.xlarge`, `m6i.xlarge`, `m5.xlarge` | Managed node group capacity choices |
| `node_group_scaling.desired_size` | `2` | Initial number of CPU nodes |
| `dynamo_version` | `1.4.0` | Dynamo platform chart and planner image tag |
| `mocker_replicas` | `3` | Mocker pods, with one simulated worker per pod |
| `model_name` | `Qwen/Qwen3-0.6B` | Tokenizer and API model identifier |

## Destroy

Stop any running port-forward, then remove the deployment and its AWS infrastructure:

```bash
terraform destroy
```

Terraform destroys the Mocker release before the operator and cluster because the Helm releases have
explicit dependencies.
