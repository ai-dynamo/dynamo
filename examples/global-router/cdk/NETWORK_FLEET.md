<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Networking for existing regional clusters

The `bin/network-fleet.ts` entry point connects two through four existing regional
clusters. It imports each cluster's private kubectl provider, security group and
route tables without owning worker nodes, GPU reservations or cluster foundations.
Discovery and session affinity use DynamoDB through the separately deployed global
services. This networking layer supplies the private paths between advertised
router, frontend and PoolRelay endpoints.

## Operator configuration

Create `config/network-fleet.local.json`, which Git ignores. The top-level
object has `members` and `links`. Keep actual deployment identifiers out of
checked-in examples. Run with the same account and Node.js prerequisites as
[Reserved GPU compute](GPU_COMPUTE.md); its S3 asset stacks must exist in each region.
This entry point uses the current operator session without creating deployment roles.

Each member supplies:

| Field | Meaning |
|---|---|
| `id` | Stable alphanumeric configuration key, such as `ohio` |
| `region`, `clusterName` | Existing EKS cluster location and name |
| `vpcId`, `cidr` | Existing VPC and its nonoverlapping private CIDR |
| `routeTableIds` | Ordered list of private/public subnet route tables needing peer access |
| `clusterSecurityGroupId` | Existing EKS cluster security group used by the nodes |
| `kubectlServiceToken` | Existing CDK kubectl framework Lambda ARN for that cluster |

Preserve member keys, route-table ordering and link ownership across redeployments.
Inspect existing rules before deployment: this app must not duplicate security group
rules or routes already owned elsewhere. VPC peering is not transitive: a four-member
fleet requires six direct links. Specify every unordered member pair exactly once.

For an existing connection managed by another stack:

```json
{ "left": "ohio", "right": "west", "peeringId": "pcx-00000000000000000", "external": true }
```

An external link leaves both its connection and all its routes with their existing
owner. Verify its current private pod routes; this app does not repair external links.
For a new connection, omit `external` and initially omit `peeringId`:

```json
{ "left": "ohio", "right": "tokyo" }
```

The left member owns the new peering. After creating it, copy its output ID into
that link's `peeringId`. Keep it non-external: this stack still owns the connection.

## Deploy the three independent phases

Install locked dependencies, build and run the CDK tests. Select the member owning
each new link and deploy `networkMode=peerings`. Copy the outputs into local settings,
then deploy `networkMode=routes` for each member. Literal peering IDs avoid cross-region
CloudFormation export dependencies.

```bash
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
pnpm exec cdk deploy --app 'node --require ts-node/register bin/network-fleet.ts' \
  -c networkConfigPath=config/network-fleet.local.json -c networkMember=ohio -c networkMode=peerings
```

Run the same command with `networkMode=routes` after filling the IDs. Review each
changeset: routes must add private peer CIDRs through the intended peering, with
no default-route, gateway or existing destination replacement. Existing Ohio/West
network stacks are independent and are not deployed by this app.

Deploy `networkMode=member` for each member. It excludes peer CIDRs from CNI
source NAT and adds peer-only TCP and ICMP Path MTU Discovery rules. CNI DaemonSet
pods roll; worker instances remain in place. It installs no discovery controllers
or Kubernetes service import/export resources.

All new stacks enable deletion protection. Retained network and Kubernetes resources
require deliberate cleanup; a stack destroy does not undo this fleet configuration.

## Validate the data plane

Check that every member can reach the private advertised endpoints of the other
members. A four-region fleet has twelve directed remote paths. Confirm security
groups, subnet routes and source NAT exclusions together; a DynamoDB registry
entry alone does not prove that its endpoint is reachable.

EKS Service CIDRs may overlap. Do not route a remote cluster's ClusterIP or Service
CIDR. Advertise reachable pod addresses or private load balancer endpoints, and
use nonoverlapping VPC CIDRs for direct regional paths. Run inference through the
global entry points and verify the selected pools separately from networking.
