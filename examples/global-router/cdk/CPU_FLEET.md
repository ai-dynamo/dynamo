<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isolated CPU development fleets

`bin/cpu-fleet.ts` creates independently named CPU EKS deployments for parallel
Global Router development. Each fleet owns its VPCs, clusters, admin roles,
asset buckets, and private regional mesh. Global Frontend/Discovery workloads
are deployed separately from this infrastructure entry point.

The regional foundation includes the existing Dynamo operator, a CPU Mocker DGD,
its local Frontend and Relay, and optional fresh Karmada control plane. Mockers
simulate a model; they do not load GPU model weights.

## Private configuration

Keep account identifiers, AMIs, images and deployment metadata in an ignored
`config/cpu-fleet.local.json`. Its top-level fields are:

- `prefix`: a lowercase DNS label, up to 16 characters, such as `example`.
- `accountId`: the target AWS account ID.
- `deployments`: member IDs mapped to complete `DeploymentConfig` objects from
  `lib/config.ts`. Use `ohio`, `west`, `tokyo` and `saopaulo` for a four-region fleet.
- `mcs`: after foundations exist, the `members` and `links` schema described in
  [MCS on existing clusters](MCS_FLEET.md), populated exclusively from this fleet.

Every cluster name and bastion IAM role name must start with the prefix. Use
unique private VPC CIDRs and prefixed DGD, runtime and capacity pool names.
Configure CPU instance types, with two nodes per member for the small lab setup.
Select a pinned compatible Mocker image and Relay image as in the CPU example.

This entry point makes the admin hosts **SSM-only**: no launch key, SSH ingress or
bastion Elastic IP. The legacy `bastionConfig.keyPairName` field is still required
by the shared configuration type but is unused here; an `unused-ssm-only` value
is sufficient. Set explicit regional `bastionConfig.amiId` values for AL2023 with
SSM Agent. Each new admin role receives the SSM managed-instance policy and
administration access to its own cluster.

For egress without Elastic IP allocations, set `vpcConfig.natInstanceType` to a
small CPU instance type and `natInstanceAmiId` to a matching regional AL2023 AMI.
The NAT instance admits forwarded traffic only from its own VPC. Without these
options, the shared VPC construct retains its managed NAT gateway behavior.
NAT instances are a development option with one-instance availability limits.

Configure `multiCluster` per member with only the other new VPC CIDRs. A fresh
Ohio hub may set `karmadaControlPlane`; this entry point installs it directly.
Never point a second fleet at an existing fleet's hub or reinitialize that hub.

## Deploy in independent phases

Use Node.js 20 or newer and the locked package dependencies. Inspect every
synthesized stack name before deployment. No existing foundation should appear
in a new fleet's deployment plan.

```bash
pnpm exec cdk synth --app 'node --require ts-node/register bin/cpu-fleet.ts' \
  -c fleetConfigPath=config/cpu-fleet.local.json -c member=ohio -c mode=assets
```

Deploy `assets` and then `foundation` for each configured member. Asset buckets
are independently named `<prefix>-global-router-assets-<account>-<region>`;
this does not bootstrap or modify an existing deployment's asset stack.
Foundation stack names are `<clusterName>-cdk` and enable termination protection.

After the new foundations complete, collect their VPC IDs, route tables, cluster
security groups and kubectl provider tokens into the local `mcs` configuration.
Deploy `peerings` for the members owning links, fill the returned IDs, then deploy
`routes` for every member. A four-region mesh needs six links. The last member
owns no links and does not need an empty peering stack. The networking stack names
include the fleet prefix.

Administration can use each member's own SSM host. For bounded registration from
the new Ohio host, `mode=admin` creates an independent access-entry stack granting
that host's role administration in a selected new member. Ohio already grants its
own host access. Remove these temporary admin stacks immediately after registration
and validation, including on failure. Disable termination protection only for the
specific temporary admin stack being removed, not for the foundations or mesh.
Follow the token lifetime, agent namespace and certificate-rotation instructions
in [the MCS runbook](MCS_FLEET.md).

## Validate actual runtime state

The operator versions worker runtime namespaces. Read
`DynamoGraphDeployment.status.components.decode.runtimeNamespace`; the worker
namespace includes a generation suffix and differs from the Frontend namespace.
Set `dgds[].workerRuntimeNamespace` to that observed value and update the new
Relay release. Its chart checksum rolls the Relay when the configuration changes.
Refresh the binding after worker-spec rollouts that change the generation.

Do not use a Relay's open TCP ports as proof of discovery. Check its pool catalogue
and serving-readiness stream for the expected model and worker namespace.
The operator owns `<dgdName>-frontend`; use that Service without adopting it into
a second application stack. Propagate ServiceExports and dependent imports, not
the regional Services themselves.

For Global Router validation, require only this fleet's four pools and all of its
router replicas, exercise both streaming and unary inference, and verify the
selected destinations. Keep any original fleet's instance IDs, stack update times
and serving pod identities as a separate preservation baseline.
