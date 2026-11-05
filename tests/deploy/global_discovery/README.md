<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multi-cluster Global Discovery deploy test

`test_global_discovery.py` runs the most basic multi-cluster case of the Global
Discovery Service on two Kubernetes clusters. In CI they are two vClusters on
one host cluster.

```text
hub cluster                          member cluster
Global Discovery Service  <--------  DynamoPoolRelay (operator-managed)
  --pool-discovery registrations       POST /pools on every heartbeat
  --registration-storage memory      DGD: Frontend + mocker worker
```

The relay reaches the service through the hub Service's ClusterIP. vCluster
syncs every pod and Service to the host cluster, so an address from one
vCluster is routable from a pod of the other. That is the routable network the
Kubernetes discovery modes assume.

The test checks that:

1. `GET /pools` on the hub lists the member's pool under its key (the relay's
   cluster ID and the DGD's namespace and name), with the model the relay read
   from the DGD, the relay's region and the frontend URL it registered, and
   calls the list complete.
2. Heartbeats keep the pool listed for three leases.
3. Once the relay is gone, the pool leaves `GET /pools` within one lease.

The operator does not set the relay's Global Discovery flags yet, so the test
passes them as `DYN_RELAY_*` variables in the `DynamoPoolRelay` pod template.

## In CI

`post-merge-ci.yml` runs it after every merge to `main`:

- `global-discovery-service-build` builds `container/Dockerfile.global-discovery-service`.
- `deploy-test-global-discovery` creates a host namespace for the test and the
  hub and member vClusters in it, each with the operator, and runs the test.
- `deploy-cleanup-global-discovery` deletes both vClusters and the namespace.

The build and the test are non-blocking while the multi-cluster pieces land.
They have `continue-on-error`, `deploy-status-check` does not wait for them,
and the Slack notice leaves out jobs whose names contain `Global Discovery`.

## Run it yourself

You need two clusters whose pods can reach each other's ClusterIPs, the Dynamo
operator in the member, and `kubectl`. Two vClusters on one host cluster work:

```bash
NS=gd-e2e
vcluster create gd-hub --namespace "$NS" --connect=false
vcluster create gd-member --namespace "$NS" --connect=false
kubectl port-forward -n "$NS" svc/gd-hub 8443:443 &
kubectl port-forward -n "$NS" svc/gd-member 8444:443 &
vcluster connect gd-hub --namespace "$NS" --server=https://127.0.0.1:8443 --print > hub.kubeconfig
vcluster connect gd-member --namespace "$NS" --server=https://127.0.0.1:8444 --print > member.kubeconfig
```

Install the Dynamo platform Helm chart in the member, as
`.github/actions/setup-dynamo-operator` does. An `hf-token-secret` with an
`HF_TOKEN` key in the member is optional; the mocker downloads the model's
configuration and tokenizer.

Build and push the service image, then run the test with a `dynamo-frontend`
image from the same revision as the operator:

```bash
docker build -f container/Dockerfile.global-discovery-service -t <registry>/global-discovery-service:<tag> .
docker push <registry>/global-discovery-service:<tag>

python3 -m pytest tests/deploy/global_discovery/test_global_discovery.py -s \
  --hub-kubeconfig=hub.kubeconfig \
  --member-kubeconfig=member.kubeconfig \
  --frontend-image=<dynamo-frontend image> \
  --global-discovery-image=<registry>/global-discovery-service:<tag>
```

`--namespace` (default `default`) is the namespace the test uses in both
clusters. On failure the test saves both clusters' state and logs to its output
directory.

## Not covered yet

| Case | Waits on |
|------|----------|
| `--registration-storage dynamodb`: two service replicas share one table (DynamoDB Local in the hub), and both list the pool | DynamoDB registration storage in the service |
| Two memory-storage instances share registrations | Registration sharing between service instances |
| `--pool-discovery mcs`: the operator exports the pool's Services, an MCS implementation imports them into the hub, and the pool is listed with its relay identities | The operator's ServiceExports for pools, and an MCS implementation that runs in a vCluster |
| `--pool-discovery karmada` | A Karmada control plane in the hub |
| A rolling update of the worker keeps the pool listed under its model | The PoolRelay watching both worker revisions during a rollout |
| Global Router replicas in the hub (`--router`), the relay's state stream, and a request through the listed frontend | The Global Router's deployment |

In the MCS modes, the relay's `clusterId` must be the member's name in the
ClusterSet. Cilium's MCS cannot run in a vCluster, since it is part of the
host's network plugin.
