<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill workers discover decode workers

A prefill worker keeps track of the running decode workers, reads an address
that each decode worker publishes in discovery, and runs an operation against
it. Here the operation is connecting to a small TCP server on the decode
worker. The prefill worker never asks a decode worker for its address.

- `decode.py` starts a TCP server, a stand-in for whatever the prefill side
  needs to reach (for example, a KV receiver). It publishes the server's
  address in the runtime config of its model card, with
  `ModelRuntimeConfig.set_engine_specific("side_channel", ...)` and
  `register_model`. A real decode worker already registers its model, so only
  the `set_engine_specific` call is new.
- `prefill.py` serves `prefill.generate` (mock output) and calls
  `watch_runtime_configs()` on the decode endpoint. It returns a view of every
  ready decode worker's runtime config, keyed by instance ID, that discovery
  keeps up to date: `snapshot()` reads it, and `changed()` waits for a decode
  worker to join, leave, or publish new values. The prefill worker connects to
  each new decode worker and drops the ones that leave.

`Endpoint.watch_runtime_configs()` is new, so a released or nightly
`ai-dynamo-runtime` may not have it yet. Build the Python bindings from this
source tree first, as described in
[Building from Source](../../../../../docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md).

## Run locally

On one machine, the file discovery backend needs no etcd or NATS. Start each
command in its own terminal, from this directory:

```bash
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/pd-discovery DYN_LOG=warn

python3 decode.py
python3 decode.py
python3 prefill.py
python3 prefill.py
```

Without `DYN_DISCOVERY_BACKEND`, the workers use etcd at `ETCD_ENDPOINTS`.

Each prefill worker connects to every decode worker, and prints:

```text
prefill 5a0c58c0314098a0: decode 124b86db51ca2566 at 192.168.1.242:40807 -> 'ack from decode 124b86db51ca2566'
```

Start another decode worker and every prefill worker connects to it right
away. Stop one and every prefill worker prints:

```text
prefill 5a0c58c0314098a0: decode 7391da11afde812d at 192.168.1.242:38553 left
```

## Run on Kubernetes

`k8s.yaml` runs two decode and two prefill workers with Kubernetes discovery
(`DYN_DISCOVERY_BACKEND=kubernetes`) and without the Dynamo operator. It
creates by hand what the operator creates for a DynamoGraphDeployment: RBAC
for `DynamoWorkerMetadata` resources and EndpointSlices, a Service with the
discovery labels, and the pod name, namespace, and UID in each pod's
environment. The `Dockerfile` builds an image with the Dynamo runtime and the
two scripts.

From this directory, on minikube:

```bash
# Build the runtime wheel from this source tree into this directory.
(cd ../.. && maturin build --release -o examples/pd_discovery)
docker build --build-arg DYNAMO_RUNTIME=$(ls ai_dynamo_runtime-*.whl) -t pd-discovery:latest .
minikube image load pd-discovery:latest

# Skip the CRD if the Dynamo platform is installed.
kubectl apply -f ../../../../../deploy/operator/config/crd/bases/nvidia.com_dynamoworkermetadatas.yaml
kubectl apply -f k8s.yaml

kubectl -n pd-discovery logs -f -l role=prefill --prefix
```

On another cluster, push the image to a registry that the cluster can pull
from and set `image` in `k8s.yaml`. To watch the prefill workers react, scale
the decode workers or delete a decode pod:

```bash
kubectl -n pd-discovery scale deploy/decode --replicas=3
kubectl -n pd-discovery delete pod <decode-pod>
```

Remove everything with `kubectl delete -f k8s.yaml`.

## Notes

- Reading `snapshot()` makes no network calls. The prefill worker wakes when
  `changed()` returns, and at least every `--interval` seconds to retry failed
  connections.
- A worker that shuts down gracefully leaves the view right away. A worker that
  is killed, which includes Ctrl-C in this example, leaves when its discovery
  lease expires: 10 seconds with the file backend, 30 seconds with etcd
  (`ETCD_LEASE_TTL`). With Kubernetes discovery, a worker leaves when its pod
  stops being ready, for example when the pod starts terminating.
- Each decode worker publishes a new `incarnation` value when it starts. With
  Kubernetes discovery, the instance ID comes from the pod name, so a container
  that restarts in its pod keeps its ID; the new incarnation tells the prefill
  worker to treat it as a new worker.
- The model card checksum that the frontend compares across the workers of a
  model does not include the runtime config, so per-worker values such as this
  address do not split the workers into separate sets.
- `--host` sets the address that the decode worker publishes. The default, the
  machine's hostname resolved to an IP, works on one machine and is the pod IP
  on Kubernetes. Across machines, pass an IP that the prefill workers can
  reach.
