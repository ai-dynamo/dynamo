<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Real GPU Qwen pools

This example runs real Dynamo vLLM workers for
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
It uses ordinary Deployments, Services and an isolated etcd instance, reusing the
isolated-pool support in `rr-backend`. It provisions no EC2 instances, node groups,
operators, CRDs, or capacity reservations.

The model revision and Linux AMD64
[Dynamo vLLM runtime](https://docs.nvidia.com/dynamo/v1.4.0/knowledge-base/modular-components/backends/v-llm/overview)
image digest are pinned in `manifests.py`. The model is public and requires no gated
model token. Workers use FP16 weights, a 4,096-token context, eight concurrent
sequences, eager execution, prefix caching, and a GPU memory fraction of 0.75.

Each pool has two workers requesting one NVIDIA GPU each, plus one Dynamo
Frontend, one real Relay, and single-member etcd. Workers select existing nodes
labeled `dynamo.nvidia.com/purpose=gpu`. The NVIDIA device plugin must already
advertise one free GPU on each of two nodes. Backend support pods use those nodes
as well, sharing the cached runtime image. Discovery uses etcd, requests use TCP,
and events use ZMQ. The Frontend uses round-robin worker selection.

Model files are cached under the dedicated host directory
`/var/lib/dynamo/model-cache/qwen3-4b-instruct-2507`. A bounded init container sets
ownership of that directory for runtime UID 1000; it does not recursively change
other host paths. The cache survives worker pod replacement. Worker rolling
updates use zero surge and at most one unavailable worker, so replacing a worker
does not require a third GPU. Removing the workload leaves the node cache intact.

## Render and deploy

Use a separate namespace from Mocker validation so this serving surface lists
only the real Qwen model. Supply immutable frontend and Relay image digests.
Use the pinned vLLM runtime for the Frontend as well to match worker protocols.

```bash
python3 examples/global-router/qwen-gpu/manifests.py \
  --namespace global-router-qwen \
  --site-id example-region \
  --frontend-image "$DYNAMO_VLLM_IMAGE" \
  --relay-image "$RELAY_IMAGE" > /tmp/qwen-backend.json
kubectl apply -f /tmp/qwen-backend.json
```

For private EKS access through an existing CDK kubectl provider, the renderer's
output can be applied with `rr-backend/cluster.py`. That helper retains the shared
namespace on stack removal. Etcd uses ephemeral storage for this POC; model caches
are separate from its lifecycle.

Deploy the Global Ingress/GDS resources with `rr-control-plane/manifests.py`.
Configure each pool's source cluster, namespace, DGD/service stem `qwen-gpu`, model
`Qwen/Qwen3-4B-Instruct-2507`, and location in the GDS routing file. Set the desired
Global Ingress and GDS Deployment replica counts before applying. Export/import
`global-router`, `qwen-gpu-frontend`, and `qwen-gpu-pool-relay` with
`rr-backend/mcs.py`. Each GDS reads imported EndpointSlices from its local cluster.

Validate container imports and the frontend CLI locally before deployment, then
stage the first GPU pool and verify generated answers before expanding regions.
A CPU-only host cannot initialize the vLLM worker's CUDA device configuration.

## Validate real inference

`smoke.py` checks that the endpoint lists only the configured real model, verifies
arithmetic and factual answers, consumes a complete SSE stream, and exercises text
completions. It can run from an allowed client or as a short-lived Python pod with
no Kubernetes token or API permissions:

```bash
TARGET_URL=http://qwen-gpu-frontend:8000 \
  python3 examples/global-router/qwen-gpu/smoke.py
```

For a pod, the script also writes its result to `/dev/termination-log`. Inspect
worker readiness and `nvidia-smi` independently: CUDA engine processes and resident
GPU memory distinguish this deployment from an advertised model name alone.
The smoke test is functional validation, not a throughput benchmark.

## Regional ALB

`alb.py` renders a separate CloudFormation stack containing one internet-facing
ALB, a listener on HTTP port 80, a target group, a NodePort Service selecting the
Global Ingress replicas, and narrowly scoped security-group rules. ALB ingress
accepts only the approved bastion's public IPv4 `/32`; broad ranges and private
source ranges are rejected. The ALB can send only TCP traffic to the configured
NodePort within the existing VPC. Node security groups admit that port from the
ALB security group. There is no public NodePort ingress rule.

The ALB targets the existing reserved GPU node instance IDs through NodePort
30800. Kubernetes forwards to healthy Global Ingress pods, so pod replacement and
replica changes require no target-group edits. Node identities remain managed by
the independent compute stacks. Refresh the operator-local target IDs if nodes
are intentionally replaced. This example does not install a load-balancer
controller or grant a new AWS IAM role.

Supply an operator-local JSON file with `bastionCidr`, `vpcId`, `vpcCidr`,
`publicSubnetIds` (at least two availability zones), `gpuInstanceIds`,
`gpuSecurityGroups`, `clusterName`, `kubectlProviderServiceToken`, and optionally
`namespace` and `nodePort`. Keep real identifiers outside Git and image contexts.
The namespace and Global Ingress deployment must already exist. The configured
NodePort must be free in the cluster.

```bash
python3 examples/global-router/qwen-gpu/alb.py \
  --config /path/to/alb.local.json > /tmp/qwen-alb.json
aws cloudformation create-stack \
  --region "$AWS_REGION" --stack-name example-qwen-alb \
  --template-body file:///tmp/qwen-alb.json
```

Wait for stack completion and healthy targets, then use its `Url` output from
the approved bastion. Requests may route to any ready regional GPU pool. These
HTTP endpoints are restricted test endpoints; no TLS certificate or vanity DNS
name is provisioned by this example.

```bash
curl "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"What is 7 times 8?"}],"max_tokens":64,"temperature":0}'
```

Run `smoke.py` against every regional `Url`. Check security-group rules and
confirm that requests from an unapproved source cannot reach the listener. Remove
short-lived validation pods and their temporary RBAC after collecting evidence;
keep the serving and ALB stacks deployed for interactive use.
