<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reserved GPU compute

This app adds GPU capacity without updating the CPU Mocker stacks. Deploy each
layer independently: asset storage, stable VPC/EKS foundation (new regions only),
and reserved GPU compute. Application charts belong in separate workload stacks.
Externally owned On-Demand Capacity Reservations are referenced, never created,
modified, extended or cancelled by this app.

The compute stack creates an EKS managed node group with a custom launch template
that targets one reservation ID and one instance type in its exact AZ. There is no
unreserved fallback. Fixed minimum/desired/maximum counts prevent automatic scale
down. GPU nodes, launch templates and their role have retain policies; stack deletion
protection is enabled. These guards preserve capacity across application redeploys.
Deliberate compute updates, node failure and replacement, or reservation expiry still
require operational planning. Reservation expiry ends capacity assurance; it does
not automatically stop ordinary On-Demand instances or their billing.

Node groups cannot move between EKS clusters. Preserve the existing cluster name,
VPC and subnet. Review changesets before changing these settings. Updating a launch
template or AMI deliberately replaces nodes; MINIMAL updates terminate one old node
before its replacement so a full reservation needs no surge slot. Retention preserves
old resources on replacement and can leave billable resources requiring manual cleanup.
Never update the EKS-owned Auto Scaling Group directly.

## Configure

Use Node.js 20 or newer and install the locked dependencies with `pnpm install --frozen-lockfile`.
AWS CLI credentials must remain valid throughout deployment. This entry point uses
current operator credentials, with an S3-only asset stack; it creates no persistent
CDK deployment or CloudFormation execution roles. Existing CPU stack commands are unchanged.

Create `config/gpu-capacity.local.json` (ignored by Git), following this shape with
operator-supplied values. All identifiers below are synthetic. `releaseVersion` must
be a pinned EKS 1.35 NVIDIA release, resolved from the regional SSM parameter
`/aws/service/eks/optimized-ami/1.35/amazon-linux-2023/x86_64/nvidia/recommended/release_version`.
Set `hourlyRate` to the current regional Linux On-Demand price; the preflight checks it.

```json
{
  "regions": [{
    "region": "us-west-2",
    "availabilityZone": "us-west-2a",
    "reservationId": "cr-00000000000000000",
    "instanceType": "g6.4xlarge",
    "count": 2,
    "hourlyRate": 1.3232,
    "releaseVersion": "1.35.9-20261023",
    "clusterName": "example-gpu-cluster",
    "subnetId": "subnet-00000000000000000"
  }]
}
```

For a new region, omit `subnetId` initially and add `foundation` with a nonoverlapping
VPC CIDR and exactly two supported AZs, including the reservation AZ:

```json
"foundation": {
  "cidr": "10.44.0.0/16",
  "availabilityZones": ["us-west-2a", "us-west-2b"]
}
```

The fleet permits two to four nodes per region, at most four regions/16 nodes, and
at most $7/hour/node. Initial supported types are g5/g6/g6e at 2xlarge or 4xlarge.
The cap covers this configuration; check the account for other GPU fleets separately.

## Deploy

Run the read-only preflight before provisioning. It checks live reservation ownership,
state, type, AZ and free slots, AWS prices, and existing cluster/subnet compatibility.
For a new VPC also check that its CIDR does not overlap networks you plan to peer.

```bash
python3 scripts/gpu_preflight.py config/gpu-capacity.local.json
pnpm build
pnpm test
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION=us-west-2
pnpm cdk deploy --app 'node --require ts-node/register bin/gpu-compute.ts' \
  -c gpuConfigPath=config/gpu-capacity.local.json -c gpuRegion=us-west-2 -c gpuMode=assets
```

For a new region, deploy the foundation next using the same command with
`gpuMode=foundation`. It creates a private EKS API, two private subnets with NAT
egress, and a pinned NVIDIA device plugin through a VPC-local kubectl provider.
Copy its `GpuSubnetId` output into the local configuration. Do not create a foundation
for an existing cluster. Foundation stacks contain no worker nodes or application charts.

Then deploy `gpuMode=compute`. Repeat the sequence for each configured region.
For an existing cluster install the NVIDIA plugin through its existing private
admin path; EKS GPU AMIs include drivers but not this Kubernetes plugin:

```bash
helm template nvidia-device-plugin nvidia-device-plugin \
  --repo https://nvidia.github.io/k8s-device-plugin --version 0.17.1 \
  --namespace kube-system --set-string 'nodeSelector.dynamo\.nvidia\.com/purpose=gpu' \
  --set-json 'affinity=null' | kubectl apply -f -
```

The explicit selector restricts the plugin to these GPU nodes without depending on
a separate feature-discovery installation. Nodes use encrypted gp3 root disks,
IMDSv2, no SSH key, and SSM diagnostics access. They are initially untainted so new
clusters can schedule CoreDNS and system pods on this fixed capacity.

## Verify and operate

Confirm each managed node group is ACTIVE, both nodes are Ready with
`nvidia.com/gpu: 1` allocatable, and EC2 instances report the expected
`CapacityReservationId`. Run `nvidia-smi` through SSM, then a small GPU pod before
placing inference workloads. Keep full operator evidence outside the repository.

Check reservation expiry dates separately. Reservations charge for unused slots;
instances occupying them are not charged twice. Additional EKS control planes, NAT,
storage and network traffic have separate costs. Cleanup is deliberate: deletion
protection and retention prevent a normal stack destroy from releasing the fleet.
No command here cancels or extends a reservation.
