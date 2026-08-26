<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CPU Mocker on Amazon EKS with AWS CDK

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
and CDK bootstrap storage.

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
- AWS permissions to bootstrap CDK and create VPC, EKS, EC2, IAM, Lambda, S3, and related resources

Install an EKS-compatible `kubectl` client on the bastion after connecting. Local `kubectl` and a
public administrator CIDR are not required.

The template pulls released Dynamo artifacts from NGC and downloads tokenizer metadata for each
configured public model. It does not require GPUs or a Hugging Face token.

## Configure

Edit `lib/const.ts` and the bastion key-pair name in `lib/keypairs.ts`. The objects separate settings
by responsibility:

- `MOCKER_VPC_US_EAST_2` defines the Region, two standard AZs, and VPC CIDR.
- `US_EAST_2_BASTION_KEY_PAIR` in `lib/keypairs.ts` defines the SSH key pair attached at launch.
- `BASTION_CONFIG_OHIO` defines the public admin instance, IAM role, and key pair.
- `MOCKER_US_EAST_2_CONFIG` defines the EKS cluster.
- `MOCKER_CPU_NODE_GROUP` defines reusable CPU instance types and scaling.
- `DYNAMO_PLATFORM_1_4_0` defines the shared Dynamo version and namespace.
- `MOCKER_OHIO_1_POOL` combines a pool name, subnet slot, AZ, and node-group configuration.
- `MOCKER_OHIO_1_DGD` defines a DGD, its eligible pools, and its Mocker workload.
- `US_EAST2_DEPLOYMENT` selects the platform, infrastructure pools, and DGDs created by the CDK app.

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

Read the selected Region from the TypeScript constants, bootstrap the account once, then review and
deploy the stack:

```bash
AWS_REGION="$(node --require ts-node/register -e \
  'process.stdout.write(require("./lib/const").US_EAST2_DEPLOYMENT.clusterConfig.region)')"
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

pnpm exec cdk bootstrap "aws://${AWS_ACCOUNT_ID}/${AWS_REGION}"
pnpm exec cdk synth
pnpm exec cdk diff
pnpm exec cdk deploy
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
  `ConnectToBastion`.

Each group uses bytewise name order. The DGD name, model, mapping, and port-forward entries remain
index-aligned. Subnet allocation comes from `subnetSlot`, so it does not depend on declaration or
name order.

## Verify

Resolve the bastion's Elastic IP, select the private key for the configured key-pair name, and
connect with an SSH tunnel for the frontend port:

```bash
AWS_REGION="$(node --require ts-node/register -e \
  'process.stdout.write(require("./lib/const").US_EAST2_DEPLOYMENT.clusterConfig.region)')"
CLUSTER_NAME="$(node --require ts-node/register -e \
  'process.stdout.write(require("./lib/const").US_EAST2_DEPLOYMENT.clusterConfig.clusterName)')"
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
labels and pod constraints. The commands below use the default configuration:

```bash
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION="$AWS_REGION"
export CLUSTER_NAME=sachal-mocker-cluster
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
AWS_REGION="$(node --require ts-node/register -e \
  'process.stdout.write(require("./lib/const").US_EAST2_DEPLOYMENT.clusterConfig.region)')"
CLUSTER_NAME="$(node --require ts-node/register -e \
  'process.stdout.write(require("./lib/const").US_EAST2_DEPLOYMENT.clusterConfig.clusterName)')"
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
- [Single-DGD Terraform template](../../terraform/mocker-cpu/)

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
pnpm exec cdk destroy
```

The shared CDK bootstrap stack is not removed. Keep it for other CDK deployments, or remove it
separately only after confirming that no other stack uses its assets and roles.
