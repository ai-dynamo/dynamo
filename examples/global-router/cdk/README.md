<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CPU Mocker on Amazon EKS with AWS CDK

Run the commands below from Dynamo’s `examples/global-router/cdk` directory.

This TypeScript AWS CDK v2 template creates an Amazon EKS cluster and deploys one or more
aggregated NVIDIA Dynamo `DynamoGraphDeployment` (DGD) resources backed by CPU-only Mocker
workers. Deployment settings are ordinary typed objects in `lib/const.ts`; there is no JSON
configuration loader.

Each entry in `dgdPools` creates one infrastructure capacity pool: a managed node group in one
selected private subnet and Availability Zone (AZ). Each entry in `dgds` creates one Mocker Helm
release and lists one or more eligible pools. A pool can serve multiple DGDs, and a DGD can use
multiple pools. The EKS control-plane network interfaces still span two AZs, as required by EKS.

The Kubernetes API is private-only. Administration runs through an EC2 bastion with an Elastic IP
address and SSH. This is a development and simulation environment, not a highly available
production architecture. It creates billable AWS resources, including an EKS control plane, a NAT
gateway, EC2 instances, two Elastic IP addresses (one for the NAT gateway and one for the bastion),
one internal Network Load Balancer per DGD, and CDK bootstrap storage.

## Placement and Access Model

Amazon EKS requires cluster subnets in at least two AZs. The VPC construct creates dedicated
cluster subnets in both configured AZs, plus numbered pool subnet slots through the highest slot in
use. CDK creates each slot in both VPC AZs, but the stack passes only the subnet in the pool's
selected AZ to that pool's managed node group. The matching subnet in the other AZ does not receive
nodes from that pool.

The stack configures each infrastructure pool as follows:

- The managed node group receives exactly one pool-specific private subnet.
- The node group is labeled `dynamo.nvidia.com/dgd-pool=<pool-name>` and
  `dynamo.nvidia.com/purpose=mocker-cpu`.

Every component in a DGD selects `dynamo.nvidia.com/purpose=mocker-cpu` and uses required node
affinity with `dynamo.nvidia.com/dgd-pool In (<poolNames>)`. With one referenced pool, every DGD pod
is eligible for only that pool's subnet and AZ. With multiple referenced pools, each pod can run in
any of their subnets and AZs.

Pool references define eligibility, not assignment or balance. Kubernetes schedules each pod
independently and can place all replicas in one referenced pool when capacity permits. The Mocker
workers use preferred host anti-affinity, which spreads them across hostnames when possible but does
not guarantee even distribution across pools or AZs. Multiple DGDs can share a pool and compete for
its capacity. System pods and workloads without matching constraints can also use the nodes. Strict
isolation requires a separate system node group plus pool taints and matching DGD tolerations.

The stack places its only NAT gateway in `VpcConfig.availabilityZones[0]`. A pool in the second AZ
therefore uses cross-AZ NAT egress, which has cost and availability tradeoffs. CDK's Helm provider
runs in the two cluster subnets and also uses the NAT gateway for NGC downloads; it is deployment
infrastructure, not a Mocker instance.

The bastion is an Amazon Linux 2023 instance in a public subnet with an Elastic IP address. Its
security group exposes only TCP 22. Amazon EC2 attaches the key pair named by `keyPairName` when the
instance launches. The stack does not modify the Amazon Linux SSH configuration or run guest
bootstrap commands. The instance role is granted cluster-admin access through an EKS access entry.
The stack adds TCP 443 ingress from the bastion security group to the private EKS API security
groups. The bastion and cluster must share a Region and VPC in this template.

The deployed bastion configurations pin their current regional AMI IDs. Leave these pins unchanged
for application rollouts: a deploy-time lookup of the latest Amazon Linux image can replace the
bastion during an otherwise unrelated stack update. Upgrade a bastion AMI deliberately, accounting
for its new SSH host key and the loss of local tools/files on its root volume. Use a CloudFormation
change set to check replacements before executing a deployment; a template-only CDK diff does not
show changes in values resolved from SSM at deployment time.

The two private EKS cluster subnets are tagged `kubernetes.io/role/internal-elb=1` so an internal
Network Load Balancer can discover placement subnets. The stack installs AWS Load Balancer Controller
v2.14.1 (Helm chart 1.14.1) into `kube-system` in every configured cluster. CDK creates a dedicated
per-cluster IRSA role and service account using its bundled v2.14.1 controller IAM policy. Each DGD
release also creates a private internal NLB Service targeting only that DGD's Frontend pods. The
operator-owned Frontend ClusterIP remains available locally through the bastion's `kubectl
port-forward`. User-facing public regional ingress is a separate path and is not the target for the
Global Router.

## Global Router private Frontend path

The agreed first POC places the Global Router in Ohio. Ohio and West use non-overlapping VPC ranges
`10.42.0.0/16` and `10.43.0.0/16`. Deploy the regional stacks separately with
`deployment=ohio` and `deployment=west`. The `-c deployment=both` mode then creates only a single
inter-Region VPC peering connection and private routes in both directions. The peering connection
and routes do not create public ingress. Public regional ingress addresses are client entry points;
the Global Router uses a separate private endpoint for each DGD Frontend.

The Mocker Helm chart emits a `DynamoGraphDeployment`; the Dynamo 1.4.0 operator creates
`<dgd>-frontend` as a ClusterIP Service on port 8000, with named container port `http`. The pinned
operator source defines the [Frontend container port](https://github.com/ai-dynamo/dynamo/blob/v1.4.0/deploy/operator/internal/dynamo/component_frontend.go#L34-L40)
and [generated service port and selector](https://github.com/ai-dynamo/dynamo/blob/v1.4.0/deploy/operator/internal/dynamo/graph.go#L904-L960). DGD `spec.services.Frontend.annotations` can
add Service annotations, but the operator's generated Service leaves `spec.type` unset (Kubernetes
defaults it to ClusterIP), and the DGD API has no Service type override. Adding load balancer
annotations to that generated Service alone therefore does not make it externally reachable.
See Dynamo's [1.4.0 component Service generator](https://github.com/ai-dynamo/dynamo/blob/v1.4.0/deploy/operator/internal/dynamo/graph.go#L904-L966).

This stack uses standard EKS managed node groups and installs the controller through CDK's native
`AlbController` construct. CDK orders its Helm release after the cluster OIDC provider, the IRSA
service account and role, and the managed node groups. Each DGD Helm release explicitly depends on
the controller. Its chart creates a second, private Service named `<dgd-name>-private`; this Service
uses the DGD's Frontend-only chart label plus a unique DGD label, TCP port 8000, and an internal NLB
with IP targets. It restricts load balancer source ranges to the configured
`privateFrontendSourceCidrs`. Both regional deployment configurations currently allow the Ohio VPC
CIDR (`10.42.0.0/16`), where the first Global Router will run. If a West regional Router is
added later, Ohio Frontend source ranges must also allow the West VPC CIDR (`10.43.0.0/16`).

After deployment, set `DYNAMO_NAMESPACE` to the stack's `DynamoNamespace` output (currently
`dynamo`), then get each NLB's private DNS name with:

```sh
DYNAMO_NAMESPACE=dynamo
kubectl get service -n "$DYNAMO_NAMESPACE" <dgd-name>-private \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

The endpoint to put in that DGD's later Global Router JSON configuration is
`http://<private-nlb-dns>:8000`. The router calls the selected pool's local Frontend directly. With
the `both` mode networking stacks deployed and the peering active, Ohio can reach the West endpoint
over private routes. AWS's [EKS NLB guide](https://docs.aws.amazon.com/eks/latest/userguide/network-load-balancing.html)
documents the internal NLB and IP-target requirements; the [AWS Load Balancer Controller v2.14.1
Service annotation reference](https://kubernetes-sigs.github.io/aws-load-balancer-controller/v2.14/guide/service/annotations/)
documents these Service settings.

## Optional private KV DC Relay workload

Each named deployment may set `kvDcRelayImage` in `lib/const.ts` to an explicit image tag or
sha256 digest. When set, CDK installs one Relay Deployment and internal NLB Service per DGD in the
same Kubernetes namespace (`dynamo`). The Relay uses the operator-created DGD discovery service
account `<dgd-name>-k8s-service-discovery` and is scheduled only on the pools assigned to that
DGD. Its own runtime namespace stays at `dynamo-<dgd-name>` for Frontend stats. It watches the
worker runtime namespace for KV catalog, readiness, and events. The Relay image is not built or
published by this CDK project. The current POC configuration pins the published Relay image digest
in each Region.

The Service listens on TCP 50051 for the WAN gRPC endpoint and TCP 50052 for stats gRPC. Both use
internal NLBs and `loadBalancerSourceRanges` from `privateRelaySourceCidrs`; the Ohio and West
configurations allow both regional VPC ranges (`10.42.0.0/16` and `10.43.0.0/16`) so a Router in
either VPC can connect. The AWS Load Balancer Controller manages the NLB security groups and target
rules. The controller must be installed and healthy, and the regional VPC peering routes from the
`deployment=both` networking stacks must be active for cross-region access. Do not add a manual
load-balancer security group annotation without also taking responsibility for target-node ingress.

When the Relay image is enabled, `KvDcRelayEndpointCommands` reports how to read each Service's
`status.loadBalancer.ingress[0].hostname`. After deployment, use that internal DNS name in the
matching Router pool config: `relay_grpc_url` as `http://<private-relay-nlb>:50051` and
`stats_grpc_url` as `http://<private-relay-nlb>:50052`. This post-deploy wiring is manual for the
POC. Relay readiness verifies its listeners; Router readiness still depends on receiving fresh
Relay data. No AWS resources are created for Relay while `kvDcRelayImage` is unset.

## Optional Ohio Global Router workload

The Ohio stack can install the packaged Rust Global Router alongside the CPU Mocker DGDs. It is
opt-in and needs an explicit image URI/tag plus a local JSON config file; the CDK has no implicit
image or `latest` default. For example, after publishing an image and preparing the config:

```bash
pnpm exec cdk deploy -c deployment=ohio \
  -c routerImage="197002356433.dkr.ecr.us-east-2.amazonaws.com/global-router-poc:<tag>" \
  -c routerConfigPath=/path/to/router-ohio.json
```

For two-region routing, deploy `deployment=both` first to add peering, then install the Router
in Ohio using the checked-in `config/router-ohio-poc.json`:

```bash
pnpm exec cdk deploy -c deployment=ohio \
  -c routerImage="197002356433.dkr.ecr.us-east-2.amazonaws.com/global-router-poc@sha256:2d7e76ba22051bbfb7d4f3cb17626f878d8fdcb9186a2214b880df04de4d7dc2" \
  -c routerConfigPath=config/router-ohio-poc.json
```

That config records the private NLB names assigned on 2026-09-29. Refresh an endpoint if its
Service is replaced. Global View derives each stable pool ID from the configured site ID,
Kubernetes namespace, and DGD name. The Mocker pools now set `stats_only_aggregated: false` so
the Router consumes the Relay KV catalog, serving readiness, and CKF stream. Verify those streams
before deploying the Router configuration; an empty catalog does not make a pool routable.

The worker runtime namespaces in `lib/const.ts` and `config/router-ohio-poc.json` are temporary
pins to the active operator worker generations. The Frontend stays in the unsuffixed namespace,
while each decode worker advertises KV events in a namespace with a worker hash suffix. Read the
current values from each cluster before deployment:

```bash
kubectl -n dynamo get dgd mocker-ohio-1 -o jsonpath='{.status.components.decode.runtimeNamespace}'
kubectl -n dynamo get dgd mocker-west-1 -o jsonpath='{.status.components.decode.runtimeNamespace}'
```

Run each command against its corresponding cluster. When a worker rollout changes a suffix,
update the matching DGD `workerRuntimeNamespace` in `lib/const.ts` and the Router pool's
`runtime_namespace`, then redeploy the affected Relay and Router. Their pod templates include
ConfigMap checksums so changed configuration restarts the processes. Discovery from DGD status
should replace these static pins after the POC. Keep the Router's optional
`kv_aware_token_id_completions` flag disabled while validating the catalog and CKF stream. After
both pools report fresh overlap, enable it with a Router image that supports the flag. The first
serving experiment uses `/v1/completions` with `prompt` as an array of token IDs; chat and text
prompts continue using load-based selection. Each deployed DGD explicitly sets its local Frontend
`frontendRouterMode` to `kv`, so the selected pool can route to the worker holding the prefix.
The Global Router logs pool decisions at info level with `RUST_LOG=warn,dynamo_llm::global_view=info`;
these records include the selected pool and matched-prefix token count.

The supplied JSON must set `listen` to `0.0.0.0:8080`; CDK mounts it at
`/etc/global-router/config.json` from a ConfigMap. In each pool object, set `namespace` to the
Kubernetes namespace `dynamo`, and `runtime_namespace` to the current **worker** runtime namespace
from DGD status. Populate each pool's `private_frontend_base_url` with the internal Frontend NLB URL
obtained from its Kubernetes Service after the regional DGD stacks are deployed. The Router config
also requires `relay_grpc_url` and `stats_grpc_url` for every pool. When `kvDcRelayImage` is
configured, the optional private Relay workload above supplies both endpoints; use its internal
NLB hostname after deployment. Otherwise, supply both URLs from a separate Relay deployment. Until its Relay feeds produce
fresh pool state, `/readyz` returns 503 and the pod remains unready. Helm does not wait for Router
readiness while installing this optional workload.

The Router is exposed only by a ClusterIP Service. Connect to the Ohio bastion using
`ConnectToBastion`, set `DYNAMO_NAMESPACE` to the stack's `DynamoNamespace` output (currently
`dynamo`), then run the conditional `PortForwardGlobalRouter` command on the bastion. Its generated
command uses the configured namespace. The equivalent command is:

```bash
DYNAMO_NAMESPACE=dynamo
kubectl port-forward -n "$DYNAMO_NAMESPACE" svc/global-router 8080:8080
```

In another shell on the bastion, test `http://127.0.0.1:8080`, for example:

```bash
curl -i http://127.0.0.1:8080/readyz
curl -i http://127.0.0.1:8080/pools
```

`/readyz` may return 503 until Relay feeds produce fresh pool state. This creates no public or
VPC-wide listener: the Router stays behind ClusterIP and access uses the existing bastion SSH
connection and private EKS API. The Global View diagnostics feature is enabled in the Router image
build. The optional `-L 8080:127.0.0.1:8080` SSH tunnel can be used to test from a local workstation
instead, with curl directed to local `127.0.0.1:8080`.

> [!WARNING]
> TCP 22 is reachable from `0.0.0.0/0`. Possession of the launch private key gates login, but this
> still exposes the SSH daemon to the public internet. Rotate keys promptly and destroy the stack
> when the environment is not in use.

## Prerequisites

- Node.js 20 or later
- pnpm 11.7.0
- AWS CLI v2 authenticated to the target account
- An SSH client
- An existing EC2 key pair matching `keyPairName` in the configured Region, with its private key
  available locally
- AWS permissions to bootstrap CDK and create VPC, EKS, EC2, IAM, Elastic Load Balancing, Lambda, S3, and related resources

Install an EKS-compatible `kubectl` client on the bastion after connecting. Local `kubectl` and a
public administrator CIDR are not required.

The template pulls released Dynamo artifacts from NGC and downloads tokenizer metadata for each
configured public model. It does not require GPUs or a Hugging Face token.

## Configure

Edit `lib/const.ts` and the bastion key-pair name in `lib/keypairs.ts`. Regional configurations are
collected in `DEPLOYMENTS` and selected by name with CDK context. Ohio remains the default:

```bash
pnpm exec cdk synth
pnpm exec cdk synth -c deployment=ohio
pnpm exec cdk synth -c deployment=west
pnpm exec cdk synth -c deployment=both
```

An unknown deployment name fails with the available names. Use `-c deployment=west` for
us-west-2. After deploying both regional VPC/EKS stacks separately, `deployment=both` synthesizes
only two network stacks: one owns the peering connection and Ohio private routes, and the other owns
West private routes. `POC_PEERING_OHIO` and `POC_PEERING_WEST` in `lib/const.ts` pin the VPC and
private route table IDs from the deployed stacks; refresh them if a regional VPC is rebuilt.
Deploy these two stacks with `pnpm exec cdk deploy --all -c deployment=both` after bootstrapping
both Regions. Each private subnet route table gets a native CloudFormation route to the opposite VPC CIDR. Public
subnet route tables are not changed. The regional load balancer controller applies security group
ingress from the NLB Service source ranges: West allows the Ohio VPC CIDR, so no broad worker
ingress rule is added. The private NLB DNS hostname resolves to its private node addresses; VPC
peering DNS-resolution options for EC2 public hostnames are not needed for that NLB name. Verify DNS
resolution and connectivity from Ohio after peering becomes active.

The West VPC uses `10.43.0.0/16` to keep its network range distinct from Ohio. West is configured to
use the EC2 key pair `mocker-west-admin`, imported on 2026-09-29 from the public half of the supplied
Ohio bastion key. The same private key can access both bastions. CDK bootstrap completed in West on
2026-09-29; a new account still needs its own regional bootstrap.

The regional `global-router-poc` ECR repositories contain the POC images built from Dynamo
`a47ad40894`. CDK does not build or push images. Each entry in `DEPLOYMENTS` pins its
region-local `mockerImage` and `kvDcRelayImage` by digest. The planner digest requires
`mockerRuntimeVersionOverride: "1.6.0"` on both DGD components because the operator cannot infer
a runtime version from a digest-only image reference. If omitted, the chart keeps using
`nvcr.io/nvidia/ai-dynamo/dynamo-planner` at the configured Dynamo version. The objects separate
settings by responsibility:

- `MOCKER_VPC_US_EAST_2` defines the Region, two standard AZs, and VPC CIDR.
- `US_EAST_2_BASTION_KEY_PAIR` in `lib/keypairs.ts` defines the SSH key pair attached at launch.
- `BASTION_CONFIG_OHIO` defines the public admin instance, IAM role, and key pair.
- `MOCKER_US_EAST_2_CONFIG` defines the EKS cluster.
- `MOCKER_CPU_NODE_GROUP` defines reusable CPU instance types and scaling.
- `DYNAMO_PLATFORM_1_4_0` defines the shared Dynamo version and namespace.
- `MOCKER_OHIO_1_POOL` combines a pool name, subnet slot, AZ, and node-group configuration.
- `MOCKER_OHIO_1_DGD` defines a DGD, its eligible pools, and its Mocker workload.
- `US_EAST2_DEPLOYMENT`, `US_WEST2_DEPLOYMENT`, and `DEPLOYMENTS` define the named regional deployments.

## Build and publish a Dynamo planner POC image

For a Dynamo source revision that should supply the planner image in this POC, build from that
checked-out revision and publish the same commit-derived tag to both regional repositories. This uses
Dynamo's developer-convenience target; Dynamo documents its official build and release pipeline as
the supported path for production images. The two `global-router-poc` ECR repositories are
`197002356433.dkr.ecr.us-east-2.amazonaws.com/global-router-poc` and
`197002356433.dkr.ecr.us-west-2.amazonaws.com/global-router-poc`. Confirm you are authorized for
account `197002356433` before running the push steps.

From a shell with Docker, AWS CLI v2, and GNU Make installed, set `DYNAMO_DIR` to the local Dynamo
checkout at the desired revision. These commands build once for the current Linux/amd64 platform,
then tag and push the identical image to each region:

```bash
export DYNAMO_DIR=/path/to/dynamo
test -z "$(git -C "$DYNAMO_DIR" status --porcelain)"  # Build a clean source revision
export IMAGE_TAG="gv-$(git -C "$DYNAMO_DIR" rev-parse HEAD)"
export EAST_REGISTRY=197002356433.dkr.ecr.us-east-2.amazonaws.com
export WEST_REGISTRY=197002356433.dkr.ecr.us-west-2.amazonaws.com
export EAST_REPO="$EAST_REGISTRY/global-router-poc"
export WEST_REPO="$WEST_REGISTRY/global-router-poc"

aws ecr get-login-password --region us-east-2 \
  | docker login --username AWS --password-stdin 197002356433.dkr.ecr.us-east-2.amazonaws.com
aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin 197002356433.dkr.ecr.us-west-2.amazonaws.com

make -C "$DYNAMO_DIR/deploy/operator" docker-build-planner \
  REGISTRY="$EAST_REGISTRY" TAG="$IMAGE_TAG"
docker tag "$EAST_REGISTRY/dynamo-planner:$IMAGE_TAG" "$EAST_REPO:$IMAGE_TAG"
docker tag "$EAST_REGISTRY/dynamo-planner:$IMAGE_TAG" "$WEST_REPO:$IMAGE_TAG"
docker push "$EAST_REPO:$IMAGE_TAG"
docker push "$WEST_REPO:$IMAGE_TAG"
```

The tag contains the full checked-out Dynamo commit SHA; retain the image digest with the
deployment record. ECR tag immutability is not configured by this CDK project, so avoid reusing tags. After publishing, set `mockerImage` in each regional configuration in
`lib/const.ts` to its matching URI and tag, for example
`197002356433.dkr.ecr.us-east-2.amazonaws.com/global-router-poc:<tag>` for Ohio and
`197002356433.dkr.ecr.us-west-2.amazonaws.com/global-router-poc:<tag>` for West. The CDK chart
uses this image for the Mocker DGD's frontend and worker.
This planner build does not create the standalone Rust KV DC Relay or Global Router images.
Build those from their dedicated Dockerfiles in the Dynamo source checkout and publish them with
explicit revision tags. Do not use the planner image for either component.

To add infrastructure capacity, append a typed object to `dgdPools`:

```typescript
{
  name: "mocker-cpu-ohio-2",
  subnetSlot: 2,
  availabilityZone: "us-east-2a",
  nodeGroupConfig: ANOTHER_NODE_GROUP,
}
```

To add a DGD, append a separate typed object to `dgds` and reference one or more configured pool
names:

```typescript
{
  dgdName: "mocker-ohio-2",
  poolNames: ["mocker-cpu-ohio-1", "mocker-cpu-ohio-2"],
  mockerReplicas: 4,
  modelName: "Qwen/Qwen3-32B",
  mockerSpeedupRatio: 1,
}
```

Adding a pool does not create a DGD, and adding a DGD does not create capacity. Pool names must be
unique lowercase DNS labels no longer than 56 characters. DGD names must be unique lowercase DNS
labels no longer than 53 characters. Every `poolNames` entry must be unique within its DGD and must
name a configured pool. The DGD name `dynamo-platform` is reserved for the shared platform Helm
release. All DGDs use the one version and namespace in `dynamoPlatformConfig`.

Pool and bastion instance types must be x86-64. The stack validates this because its EKS node groups
and bastion use Amazon Linux 2023 x86-64 images.

`subnetSlot` must be a unique integer from 1 through 6. The explicit slot gives the pool a stable
CIDR position: adding or reordering another pool does not renumber existing pool subnets. If slots
are skipped, the VPC reserves the lower slot subnet groups through the highest configured slot.
Treat a deployed pool's slot as persistent. Changing its slot deliberately moves the managed node
group to another subnet and is a disruptive capacity change.

The pool name drives node-group names, labels, tags, and output mappings. The DGD name drives its
Helm release and Kubernetes resource name. Each pool AZ must be one of the VPC's two configured AZs.

With the current `/16` VPC and `/20` subnet layout, the template supports six pool subnet slots. Each
slot consumes two subnet ranges because CDK creates it in both VPC AZs. Change the VPC/subnet design
before exceeding that limit.

Confirm that both configured AZs are standard AZs available to your account:

```bash
aws ec2 describe-availability-zones \
  --region us-east-2 \
  --filters Name=state,Values=available Name=zone-type,Values=availability-zone \
  --query 'AvailabilityZones[].[ZoneName,ZoneId]' \
  --output table
```

The configured key pair must already exist in the selected Region. EC2 installs that key when it
launches the instance. If you delete and recreate the key pair under the same name, destroy and
redeploy this development stack so the bastion receives the new public key.

## Install and Test

Install the pinned pnpm version directly with npm, install the locked dependencies, and run the
local checks:

```bash
npm install --global pnpm@11.7.0
pnpm --version
pnpm install --frozen-lockfile
pnpm run build
pnpm test
```

The tests synthesize configurations with multiple infrastructure pools and DGDs. They verify
distinct private, NAT-routed single-subnet node groups, both VPC AZ choices, DGD affinity to one or
more referenced pools, stable subnet CIDRs when pools are added or reordered, independent pool and
DGD settings, the private-only API endpoint, public SSH bastion, launch-key selection, absence of
bastion bootstrap commands and creation signals, EKS access entry, encrypted gp3 volumes, and Helm
deployment order.

## Deploy

Bootstrap CDK once per account and Region. East is already bootstrapped; bootstrap West before its
first deployment. The West key pair must also exist before deployment. These commands select the
West configuration and Region consistently:

```bash
DEPLOYMENT=west
AWS_REGION=us-west-2
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

pnpm exec cdk bootstrap "aws://${AWS_ACCOUNT_ID}/${AWS_REGION}"
pnpm exec cdk synth -c "deployment=${DEPLOYMENT}"
pnpm exec cdk diff -c "deployment=${DEPLOYMENT}"
pnpm exec cdk deploy -c "deployment=${DEPLOYMENT}"
```

Creation usually takes several minutes. The bastion sends no guest bootstrap signal; configure its
tools manually after the deployment completes. The stack creates every managed node group first,
waits for the shared Dynamo platform chart and its operator to become ready, and then installs one
local Mocker chart per DGD. Each DGD chart depends on the node groups named in its `poolNames`. The
reverse dependency order keeps the operator and referenced nodes available while Helm removes each
DGD during stack deletion.

The synthesized stack exposes separate aggregate outputs for infrastructure and workloads:

- Pool-ordered outputs are `DgdPoolNames`, `DgdPoolPlacements` (`pool=AZ`), and
  `DgdPoolSubnetIds` (`pool=subnet-id`).
- DGD-ordered outputs are `DgdNames`, `DgdPoolMappings` (`dgd=pool1+pool2`), `ModelNames`, and
  `PortForwardFrontends`.
- Bastion outputs are `BastionInstanceId`, `BastionPublicIp`, `BastionKeyPairName`, and
  `ConnectToBastion`. When the optional Ohio Global Router is enabled, `PortForwardGlobalRouter` is
  also emitted; run it on the bastion, then curl the forwarded port from another bastion shell.

Each group uses bytewise name order. The DGD name, model, mapping, and port-forward entries remain
index-aligned. Subnet allocation comes from `subnetSlot`, so it does not depend on declaration or
name order.

## Verify

Resolve the bastion's Elastic IP, select the private key for the configured key-pair name, and
connect with an SSH tunnel for the frontend port:

```bash
AWS_REGION=us-west-2
CLUSTER_NAME=sachal-mocker-west-cluster
STACK_NAME="${CLUSTER_NAME}-cdk"
BASTION_IP="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='BastionPublicIp'].OutputValue | [0]" \
  --output text)"
BASTION_KEY_PAIR_NAME="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='BastionKeyPairName'].OutputValue | [0]" \
  --output text)"
DGD_NAMES="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='DgdNames'].OutputValue | [0]" \
  --output text)"
DGD_POOL_NAMES="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='DgdPoolNames'].OutputValue | [0]" \
  --output text)"
DGD_POOL_PLACEMENTS="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='DgdPoolPlacements'].OutputValue | [0]" \
  --output text)"
DGD_POOL_SUBNET_IDS="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='DgdPoolSubnetIds'].OutputValue | [0]" \
  --output text)"
DGD_POOL_MAPPINGS="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='DgdPoolMappings'].OutputValue | [0]" \
  --output text)"
PORT_FORWARD_FRONTENDS="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='PortForwardFrontends'].OutputValue | [0]" \
  --output text)"
KEY_FILE="/path/to/private-key.pem"

printf '%s\n' \
  "Launch key pair: $BASTION_KEY_PAIR_NAME" \
  "$DGD_NAMES" \
  "$DGD_POOL_NAMES" \
  "$DGD_POOL_PLACEMENTS" \
  "$DGD_POOL_SUBNET_IDS" \
  "$DGD_POOL_MAPPINGS" \
  "$PORT_FORWARD_FRONTENDS"
chmod 400 "$KEY_FILE"
ssh -i "$KEY_FILE" -L 8000:127.0.0.1:8000 "ec2-user@$BASTION_IP"
```

`DgdPoolNames`, `DgdPoolPlacements`, and `DgdPoolSubnetIds` describe the infrastructure pools in the
same order. `DgdNames` and `DgdPoolMappings` describe which pools each DGD can use. On the bastion,
set the cluster values from `lib/const.ts`, install `kubectl`, wait for all DGDs, and inspect the node
labels and pod constraints. The commands below use the West configuration:

```bash
export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION="$AWS_REGION"
export CLUSTER_NAME=sachal-mocker-west-cluster
export DYNAMO_NAMESPACE=dynamo

KUBECTL_VERSION=1.35.6
KUBECTL_RELEASE_DATE=2026-07-05
KUBECTL_BASE_URL="https://s3.us-west-2.amazonaws.com/amazon-eks"
KUBECTL_URL="${KUBECTL_BASE_URL}/${KUBECTL_VERSION}/${KUBECTL_RELEASE_DATE}/bin/linux/amd64/kubectl"

curl --fail --location --remote-name "$KUBECTL_URL"
curl --fail --location --remote-name "${KUBECTL_URL}.sha256"
sha256sum --check kubectl.sha256
sudo install --owner root --group root --mode 0755 kubectl /usr/local/bin/kubectl
rm kubectl kubectl.sha256

aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"

kubectl wait -n "$DYNAMO_NAMESPACE" \
  --for=condition=Ready dgd --all \
  --timeout=600s

mapfile -t DGD_NAME_LIST < <(
  kubectl get dgd -n "$DYNAMO_NAMESPACE" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | LC_ALL=C sort
)

kubectl get nodes \
  -o 'custom-columns=NODE:.metadata.name,POOL:.metadata.labels.dynamo\.nvidia\.com/dgd-pool,AZ:.metadata.labels.topology\.kubernetes\.io/zone'

for dgd_name in "${DGD_NAME_LIST[@]}"; do
  kubectl get pods -n "$DYNAMO_NAMESPACE" \
    -l "nvidia.com/dynamo-graph-deployment-name=$dgd_name" \
    -o 'custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,PURPOSE:.spec.nodeSelector.dynamo\.nvidia\.com/purpose,ELIGIBLE_POOLS:.spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values'
done
```

For each pod, find its scheduled node in the node table. That node's `POOL` must be one of the DGD's
entries in `DgdPoolMappings`, and its `AZ` must match that pool's entry in `DgdPoolPlacements`.
`ELIGIBLE_POOLS` is the allowed set from the required `In` affinity; it does not identify which pool
Kubernetes selected or promise balanced placement across that set.
This lookup assumes the configured Dynamo namespace is dedicated to this stack; if it contains
other DGDs, select only the names printed in the `DgdNames` stack output.

In the same bastion session, choose the first DGD and forward its frontend service:

```bash
DGD_NAME="${DGD_NAME_LIST[0]}"

echo "Forwarding $DGD_NAME"
kubectl port-forward -n "$DYNAMO_NAMESPACE" "svc/$DGD_NAME-frontend" 8000:8000
```

Keep that command running. The SSH `-L` option carries the bastion's loopback port 8000 to your
machine. In a second local terminal, read the correspondingly ordered model output and send an
OpenAI-compatible request:

```bash
AWS_REGION=us-west-2
CLUSTER_NAME=sachal-mocker-west-cluster
STACK_NAME="${CLUSTER_NAME}-cdk"
MODEL_NAMES="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ModelNames'].OutputValue | [0]" \
  --output text)"
MODEL_NAME="${MODEL_NAMES%%,*}"

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
DGD. Repeat the port-forward and select the model at the same index to verify another DGD. A
healthy frontend alone does not prove worker discovery, so keep the request as part of the
verification.

## Is Single-AZ Placement the Dynamo Default?

No. A DGD describes components and pod templates; without scheduling constraints, Kubernetes can
place replicas on any eligible node. In this stack, each infrastructure pool is always one managed
node group in one subnet and AZ. A DGD that references one pool is therefore restricted to that
pool's subnet and AZ. A DGD that references multiple pools may schedule replicas across their
subnets and AZs, but the required `In` affinity only defines eligibility. It does not balance
replicas or require cross-pool spread. Add explicit topology-spread or other hard scheduling rules
when the workload requires that guarantee.

Dynamo can operate a CPU Mocker deployment across AZs when pod networking is routable. For real
disaggregated GPU serving, treat an AZ as a KV-transfer domain: AWS Elastic Fabric Adapter (EFA)
device and OS-bypass traffic cannot span AZs, although ordinary Elastic Network Adapter (ENA) IP
traffic can. Dynamo's experimental topology-aware KV-transfer policy constrains routing, not pod
scheduling or per-zone capacity. A multi-AZ design should create balanced prefill and decode pools
in each AZ and pin those components.

See:

- [Amazon EKS cluster endpoint access](https://docs.aws.amazon.com/eks/latest/userguide/cluster-endpoint.html)
- [Amazon EC2 key pairs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
- [Amazon EKS VPC and subnet requirements](https://docs.aws.amazon.com/eks/latest/userguide/network-reqs.html)
- [EFA limitations](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Dynamo topology-aware KV transfer](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/kubernetes/multinode/topology-aware-kv-transfer.md)
- [Dynamo Mocker simulation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/operations/simulation-with-dynosim/mocker-live-simulation.mdx)

## Main Configuration

| Object and key | Current value | Purpose |
|---|---|---|
| `MOCKER_VPC_US_EAST_2.region` | `us-east-2` | AWS Region for this stack |
| `MOCKER_VPC_US_EAST_2.availabilityZones` | `us-east-2a`, `us-east-2b` | AZs used for cluster and pool subnet groups |
| `MOCKER_VPC_US_EAST_2.cidrBlock` | `10.42.0.0/16` | VPC CIDR; derived subnets are `/20` |
| `US_EAST_2_BASTION_KEY_PAIR` | `sachalm-mac` | EC2 key pair attached to the bastion at launch |
| `BASTION_CONFIG_OHIO.instanceType` | `t3.small` | Low-cost SSH and `kubectl` admin host |
| `MOCKER_CPU_NODE_GROUP.instanceTypes` | `m5.xlarge` | Default x86-64 managed node-group capacity choice |
| `MOCKER_CPU_NODE_GROUP.scaling.desiredSize` | `2` | Initial number of CPU nodes in the default pool |
| `MOCKER_OHIO_1_POOL.name` | `mocker-cpu-ohio-1` | Pool identity used for labels and DGD references |
| `MOCKER_OHIO_1_POOL.subnetSlot` | `1` | Stable workload-subnet CIDR slot for the pool |
| `MOCKER_OHIO_1_POOL.availabilityZone` | `us-east-2a` | Subnet and AZ selected for the pool |
| `MOCKER_OHIO_1_POOL.nodeGroupConfig` | `MOCKER_CPU_NODE_GROUP` | EC2 instance types and scaling for the pool |
| `DYNAMO_PLATFORM_1_4_0.dynamoVersion` | `1.4.0` | Shared Dynamo platform version |
| `DYNAMO_PLATFORM_1_4_0.dynamoNamespace` | `dynamo` | Shared Kubernetes namespace |
| `MOCKER_OHIO_1_DGD.dgdName` | `mocker-ohio-1` | Helm release and Kubernetes resource name |
| `MOCKER_OHIO_1_DGD.poolNames` | `mocker-cpu-ohio-1` | Pools eligible to host this DGD |
| `MOCKER_OHIO_1_DGD.mockerReplicas` | `2` | Mocker pods, with one simulated worker per pod |
| `MOCKER_OHIO_1_DGD.modelName` | `Qwen/Qwen3-32B` | Tokenizer and API model identifier |
| `MOCKER_OHIO_1_DGD.mockerSpeedupRatio` | `1` | Simulated worker speed multiplier |
| `US_EAST2_DEPLOYMENT.dgdPools` | `MOCKER_OHIO_1_POOL` | Infrastructure pools created by this deployment |
| `US_EAST2_DEPLOYMENT.dgds` | `MOCKER_OHIO_1_DGD` | DGD workloads created by this deployment |

## Destroy

Stop the port-forward and SSH session, then remove the deployment and its AWS infrastructure:

```bash
pnpm exec cdk destroy -c deployment=west
```

The shared CDK bootstrap stack is not removed. Keep it for other CDK deployments, or remove it
separately only after confirming that no other stack uses its assets and roles.
