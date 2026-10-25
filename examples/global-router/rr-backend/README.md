<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Isolated CPU Mocker pools

These renderers create a real Dynamo Frontend, two CPU Mocker workers, a KV DC
Relay, and a private single-member etcd instance in an isolated namespace. They
use ordinary Deployments and ClusterIP Services, without installing a Dynamo
operator, changing CRDs, requesting GPUs, or provisioning nodes. Each region has
its own discovery store. Runtime requests use TCP and events use ZMQ.

Supply immutable images for a planner containing `dynamo.frontend` and
`dynamo.mocker`, and the KV DC Relay POC executable that accepts the generated
JSON configuration. Fresh planner containers download the public model's
configuration and tokenizer; Mocker does not load model weights. Validate the
chosen image pair locally before deploying it. The intended smoke includes
`/v1/models`, unary chat/completions, streaming completions through `[DONE]`, and
the Relay's pool catalog and serving-readiness streams.

```bash
python3 manifests.py \
  --site-id example-site \
  --planner-image "$PLANNER_IMAGE" \
  --relay-image "$RELAY_IMAGE" > /tmp/backend.json
```

Optional `--node-selector KEY=VALUE` arguments select existing nodes. The default
model is `Qwen/Qwen3-32B`; the Frontend uses round-robin worker routing. Per region,
the five pods request 1.45 CPU cores and 2.75 GiB of memory in total. The generated
Services are `rr-mocker-frontend` (`http:8000`), `rr-mocker-pool-relay`
(`wan-grpc:50051`, `stats-grpc:50052`), and `rr-mocker-etcd` (`client:2379`). The
Relay Service selects actual Relay pods. No Relay state-push receiver is
configured on the Frontend.

The etcd data directory uses `emptyDir`: this is an ephemeral validation pool,
not a durable production deployment. Frontend readiness requires the configured
model to appear in `/v1/models`; worker probes use the runtime's health endpoint.

## Existing private cluster access

`cluster.py` applies manifests through an existing CDK kubectl provider. Its
operator-local fleet JSON has a `regions` array containing `region`,
`clusterName`, `kubectlProviderServiceToken`, and `kubectlProviderHandlerArn`.
Keep that file and deployment evidence outside the repository and container
build context. It creates only a CloudFormation stack wrapping the supplied
Kubernetes objects. The shared Namespace resource is retained on deletion so
removing a backend stack cannot delete separately deployed ingress/discovery.

```bash
python3 cluster.py --fleet "$FLEET_CONFIG" --region "$AWS_REGION" \
  --evidence /tmp/rr-evidence deploy \
  --stack example-rr-backend --manifest /tmp/backend.json
python3 cluster.py --fleet "$FLEET_CONFIG" --region "$AWS_REGION" \
  --evidence /tmp/rr-evidence read --kind pods
```

Stack submission does not imply pod readiness. Inspect stack status and the
returned Pod conditions before starting traffic.

## Karmada registration

`mcs.py` renders hub resources for the two backend Services and the separately
deployed `global-router` Service (`http:8000`). Apply its output to the Karmada
API after the member namespaces exist. Supply every intended member name:

```bash
python3 mcs.py --member example-east --member example-west > /tmp/rr-mcs.json
kubectl --kubeconfig "$KARMADA_KUBECONFIG" apply -f /tmp/rr-mcs.json
```

The hub Namespace disables automatic namespace propagation at creation. Export
policies propagate only the named ServiceExports, with overwrite scoped to those
isolated objects. Successfully applied export Works are necessary to initialize
Karmada v1.19's source informers. Import policies propagate dependencies so each
member receives derived Services and EndpointSlices. Regional Services remain
locally owned. Wait for all expected ready Frontend and Relay EndpointSlices
before testing ingress; catalog convergence and ingress polling are asynchronous.
