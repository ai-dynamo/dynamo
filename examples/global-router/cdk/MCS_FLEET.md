<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MCS on existing regional clusters

The `bin/mcs-fleet.ts` entry point extends the Karmada/MCS support from
[PR #16963](https://github.com/ai-dynamo/dynamo/pull/16963) to two through four
existing regional clusters. It imports each cluster's private kubectl provider,
security group and route tables. It never owns worker nodes, GPU reservations,
cluster foundations or the running Karmada control plane.

Use the existing Ohio hub when extending a running fleet. Do not run `karmadactl
init`, `deinit` or the chart migration steps over an existing installation.
The original CPU entry point still supports a fresh CDK-managed hub, but this
entry point does not install or adopt its chart. A hostPath-backed existing etcd
remains dependent on its current node; preserving it does not provide database HA.

## Operator configuration

Create `config/mcs-fleet.local.json`, which Git and Docker ignore. The top-level
object has `members` and `links`. Keep actual deployment identifiers out of
checked-in examples. Run with the same account and Node.js prerequisites as
[Reserved GPU compute](GPU_COMPUTE.md); its S3 asset stacks must exist in each region.
This entry point uses the current operator session without creating deployment roles.

Each member supplies:

| Field | Meaning |
|---|---|
| `id` | Stable alphanumeric configuration key, such as `ohio` |
| `siteId` | Karmada member name, matching the Router's site identity |
| `region`, `clusterName` | Existing EKS cluster location and name |
| `vpcId`, `cidr` | Existing VPC and its nonoverlapping private CIDR |
| `routeTableIds` | Ordered list of private/public subnet route tables needing peer access |
| `clusterSecurityGroupId` | Existing EKS cluster security group used by the nodes |
| `kubectlServiceToken` | Existing CDK kubectl framework Lambda ARN for that cluster |
| `hubApiService` | Hub only: `{ "namespace": "karmada-system", "name": "karmada-apiserver" }` |

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
each new link and deploy `mcsMode=peerings`. Copy the outputs into local settings,
then deploy `mcsMode=routes` for each member. Literal peering IDs avoid cross-region
CloudFormation export dependencies.

```bash
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
pnpm exec cdk deploy --app 'node --require ts-node/register bin/mcs-fleet.ts' \
  -c mcsConfigPath=config/mcs-fleet.local.json -c mcsMember=ohio -c mcsMode=peerings
```

Run the same command with `mcsMode=routes` after filling the IDs. Review each
changeset: routes must add private peer CIDRs through the intended peering, with
no default-route, gateway or existing destination replacement. Existing Ohio/West
network stacks are independent and are not deployed by this app.

Deploy `mcsMode=member` for each member. It installs retained MCS CRDs, excludes
peer CIDRs from CNI source NAT, and adds peer-only TCP and ICMP Path MTU Discovery
rules. CNI DaemonSet pods roll; worker instances remain in place. The hub's existing
API Service source ranges expand to the configured VPCs while remaining internal.
No live Relay propagation policies or registrations are changed.

All new stacks enable deletion protection. Retained network and Kubernetes resources
require deliberate cleanup; a stack destroy does not undo this fleet configuration.

## Register new members

Use `karmadactl` 1.19.0 and the existing hub's working internal API endpoint. A saved
kubeconfig may still point to an expired localhost port-forward: check `/readyz`
using a private temporary copy pointed at the internal NLB first. Label hub-only
coordination and independently managed test namespaces with
`namespace.karmada.io/skip-auto-propagation=true` before creating them on the
Karmada API, so its namespace controller does not try to adopt existing member namespaces.
Keep credentials
on the administration host and do not print bootstrap tokens or commit kubeconfigs.

Before registering with `--namespace=karmada-agent`, create that namespace in the
Karmada API as well as in the member cluster. The agent elects its leader against
the Karmada API; a missing lease namespace leaves an otherwise running agent
`Ready=Unknown` with `ClusterStatusNeverUpdated`. For example:

```bash
kubectl --kubeconfig karmada-apiserver.config apply -f - <<'YAML'
apiVersion: v1
kind: Namespace
metadata:
  name: karmada-agent
  labels:
    namespace.karmada.io/skip-auto-propagation: "true"
YAML
```

Create a short-lived registration token on the hub, then run the printed register
command with the new member's kubeconfig, `--cluster-name=<siteId>`, region/provider
metadata and `--namespace=karmada-agent`. Use CA-hash verification and certificate
rotation. Delete the bootstrap token after registration. If temporary operator
access is needed, bound registration with a timeout and revoke that access in a
finally path immediately afterward, including on failure. Worker roles do not need
cluster-admin access.

## Validate discovery and the data plane

Use an isolated namespace and policies selecting the four explicit members. Do not
edit production Relay policies to run a test. Create each test Service directly on its source member. Propagate its ServiceExport
from Karmada to the explicit source members, without propagating the Service itself.
Karmada 1.19 initializes its export informer after an applied ServiceExport Work;
installing CRDs and creating only local exports is insufficient on a fresh agent.
If adopting an already-created test export, scope `conflictResolution: Overwrite`
to that ServiceExport policy only.
Create the matching ServiceImport and PropagationPolicy on the hub with dependency
propagation enabled. Check:

- All four Karmada members are Pull and Ready.
- The hub collects each exported endpoint and every importing member receives all
  four pod addresses in its derived Service's EndpointSlices.
- Each member can reach the other three pod addresses: twelve directed remote paths.
- Each member can use its local `derived-<service>.<namespace>.svc.cluster.local`
  name to reach the imported backends.
- GPU node IDs and existing hub/member workloads remain unchanged and healthy.

This setup does not install a `clusterset.local` DNS integration. The EKS service
CIDRs may overlap because derived Services use cluster-local IP allocation; do not
route a remote cluster's Service IP or Service CIDR. Direct cross-region paths use
the nonoverlapping VPC pod addresses.

Remove test clients, imports/policies and server exports after recording results.
Leave the member agents, CRDs, private network and existing hub installed. Actual
Global Router discovery integration and model-serving benchmarks are separate tests.
