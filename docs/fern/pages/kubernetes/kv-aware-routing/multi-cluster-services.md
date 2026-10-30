---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy Multi-Cluster Services
subtitle: Share Services between Kubernetes clusters for the Global Discovery Service without Karmada
---

**Experimental.** NVIDIA Dynamo includes an implementation of the Kubernetes Multi-Cluster Services
(MCS) API ([KEP-1645](https://github.com/kubernetes/enhancements/tree/master/keps/sig-multicluster/1645-multi-cluster-services-api),
`multicluster.x-k8s.io`) in
[`deploy/global-discovery-mcs`](https://github.com/ai-dynamo/dynamo/tree/main/deploy/global-discovery-mcs).
It has two parts:

- `mcs-registry`, a registry server that holds the clusters of a clusterset, the Services each
  cluster exports and their endpoints, and merges the exports of each Service into one
  ServiceImport.
- `mcs-controller`, which runs in every member cluster. It sends the cluster's ServiceExports and
  their endpoints to the registry, and creates the ServiceImports, derived Services and
  EndpointSlices of every exported Service in its own cluster.

The Global Discovery Service, run with `--pool-discovery mcs`, reads those EndpointSlices to find
the Global Router pools and replicas of every cluster.

For every flag, condition and metric, see
[Multi-Cluster Services Configuration](../../reference/components/global-discovery-mcs-configuration.md).
For the data model and the consistency rules, see
[Multi-Cluster Services Registry](../../developer-guide/knowledge-base/modular-components/router/multi-cluster-services-registry.md).

## Choose an Implementation

The Global Discovery Service reads imports from this implementation (`--pool-discovery mcs`) or
from Karmada (`--pool-discovery karmada`). Both read the same ServiceExport in the cluster that runs
the Service, and both need pod addresses that are routable between the clusters.

| | Dynamo MCS (`mcs`) | Karmada (`karmada`) |
|---|---|---|
| Control plane | `mcs-registry` in one cluster, or in several regions over one DynamoDB Global Table | The Karmada control plane |
| Where imports appear | Every member cluster that runs `mcs-controller`, in the namespace of the exported Service | The Karmada control plane, and member clusters that have a ServiceImport for the Service |
| Longest exported Service name | 63 characters | 55 characters (Karmada names the imported Service `derived-<name>`) |
| Imported EndpointSlices | Labelled `multicluster.kubernetes.io/service-name` and `multicluster.kubernetes.io/source-cluster` | Named `imported-<cluster>-<slice>` |

Use the Dynamo implementation when all of the following hold:

- **One owner and a trusted network.** The registry does not authenticate callers: any client
  that reaches its ports can register a cluster and write exports for it. Run it on a private
  network that only the member clusters reach, and use TLS on its listeners where that network
  is shared.
- **A pod network across clusters.** Imported EndpointSlices hold the pod addresses of the source
  cluster. Consumers, the Global Discovery Service and the Global Router included, dial those
  addresses directly, for example over peered VPCs with non-overlapping pod CIDRs. Neither
  implementation checks that the addresses are reachable.
- **No single control-plane cluster, or several regions.** The registry runs in each region over
  one DynamoDB Global Table, and each controller fails over between registry URLs.

Use Karmada when the clusters are already Karmada members and you manage them through its control
plane.

## Prerequisites

- Two or more Kubernetes clusters, and kubectl and Helm access to each of them.
- A clusterset name shared by every cluster and a cluster id unique to each cluster, both DNS
  labels, for example clusterset `prod` and cluster ids `use1-a` and `usw2-a`. The cluster id is
  the `multicluster.kubernetes.io/source-cluster` label of every EndpointSlice imported from that
  cluster.
- Pod addresses routable between the clusters, and network rules that allow each cluster's pod
  CIDR to reach the exported ports.
- A network path from every member cluster to the registry's gRPC port, `9090`.
- For the DynamoDB store: an AWS account, permission to create a DynamoDB table, and an IAM role
  for the registry. The module's
  [IAM document](https://github.com/ai-dynamo/dynamo/blob/main/deploy/global-discovery-mcs/docs/iam.md)
  lists the policies.

Build the `global-discovery-mcs` image, which holds `mcs-registry`, `mcs-controller`, `mcs-admin` and the
CRDs, and push it to a registry your clusters can pull from:

```bash
cd deploy/global-discovery-mcs
make docker-build docker-push IMG=my-registry.example.com/global-discovery-mcs:dev
```

The commands below install the Helm chart
[`deploy/helm/charts/global-discovery-mcs`](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/global-discovery-mcs)
from a checkout of the repository. They set only the values each step needs; the
[chart README](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/global-discovery-mcs/README.md)
lists every value, including the Service, TLS and ServiceAccount settings.

## Install the Registry

Run one registry for the clusterset, in any cluster that every member cluster can reach: a cluster
of its own, or one of the members. Choose a store:

| Store | Replicas | Use |
|---|---|---|
| `memory` | Exactly one | Evaluation. A restart loses every record. The controllers register again on their next heartbeat (every 10 s with the default 30 s lease) and send their exports again; in between, every cluster deletes the imports the registry no longer has. |
| `dynamodb` | Any number, in one or more regions | Production. Records survive restarts, and a DynamoDB Global Table replicates them between regions. |

The registry and every controller must agree on the transport. Choose one path before you install
either of them:

| Path | Registry | Controllers | Use |
|---|---|---|---|
| TLS | `registry.tls.secretName` names a Secret that holds the registry certificate | An `https://` URL, and `controller.registryTlsCa.configMapName` when a private CA signed the certificate | Production |
| Plaintext | `registry.tls.secretName` empty (the default) | An `http://` URL and `controller.registryAllowInsecureHttp=true` | Evaluation on a private network |

The chart refuses an `http://` controller URL without `controller.registryAllowInsecureHttp=true`.
It cannot check a registry in another cluster, so an `https://` URL in front of a plaintext
registry, or the reverse, shows up as a controller that cannot register its cluster.

The two choices are independent: the commands below pair the memory store with the plaintext path
and the DynamoDB store with the TLS path. To run the memory store over TLS, add
`registry.tls.secretName` to the first command and follow the TLS steps. To run the DynamoDB store
without TLS, leave it out of the second command and follow the plaintext steps.

### Evaluation: Memory Store without TLS

```bash
helm install mcs-registry deploy/helm/charts/global-discovery-mcs \
  --namespace mcs-system --create-namespace \
  --set registry.enabled=true \
  --set controller.enabled=false \
  --set image.repository=my-registry.example.com/global-discovery-mcs \
  --set image.tag=dev
```

The registry serves plaintext (h2c) on the gRPC and REST ports. Install the controllers with an
`http://` URL and `controller.registryAllowInsecureHttp=true`, as the next section shows.

### Production: DynamoDB Store with TLS

Create the table once with `mcs-admin` (`/mcs-admin` in the image, or `bin/mcs-admin` after
`make build`), using the credentials of an administrator role:

```bash
mcs-admin create-table --table=global-discovery-mcs --region=us-east-1
```

Create the registry's certificate as a `kubernetes.io/tls` Secret in the registry's namespace, for
example with cert-manager or from a certificate that your CA issued. The certificate must list, as
a subject alternative name (SAN), the host of every URL in `controller.registryUrls`: the
controller checks the URL's host against the certificate. This page uses
`mcs-registry.example.internal`. When the registry's cluster also runs a controller from the same
release, that controller dials the registry Service and the certificate also needs the name
`mcs-registry.mcs-system.svc`.

```bash
kubectl create namespace mcs-system
kubectl -n mcs-system create secret tls mcs-registry-tls \
  --cert=registry.crt --key=registry.key
```

Install the registry with the store, the table name and the Secret:

```bash
helm install mcs-registry deploy/helm/charts/global-discovery-mcs \
  --namespace mcs-system \
  --set registry.enabled=true \
  --set controller.enabled=false \
  --set registry.store=dynamodb \
  --set registry.dynamodb.table=global-discovery-mcs \
  --set registry.tls.secretName=mcs-registry-tls \
  --set image.repository=my-registry.example.com/global-discovery-mcs \
  --set image.tag=dev
```

With `registry.tls.secretName`, the registry serves TLS on the gRPC and REST ports
(`--tls-cert-file` and `--tls-key-file`). The admin port stays plaintext. The registry reads the key
pair at startup, so restart it after you rotate the certificate. The controllers need the CA
certificate that signed it, as the next section shows.

The registry takes its credentials from the default AWS chain, and its region from `--region` or
from `AWS_REGION`, which IAM roles for service accounts (IRSA) and EKS Pod Identity set. Bind its
ServiceAccount to the registry role.

To run the registry in several regions, add a Global Table replica of the table in each region,
as the module's
[storage document](https://github.com/ai-dynamo/dynamo/blob/main/deploy/global-discovery-mcs/docs/storage.md#global-table-replicas)
describes, and install the registry in each region with the same table name. Each registry reads
and writes the replica of its own region. A write usually reaches the other regions within a few
seconds, with no guaranteed bound; see
[Guarantees and Typical Timing](../../developer-guide/knowledge-base/modular-components/router/multi-cluster-services-registry.md#guarantees-and-typical-timing).
Give every regional hostname that a controller dials a SAN in the certificate of the registries
behind it.

### Expose the Registry

Expose the gRPC port, `9090`, to the member clusters, for example through an internal load
balancer. The REST port, `8080`, serves the same API as JSON and is useful for inspection; the
admin port, `9091`, serves `/metrics`, `/healthz` and `/readyz`.

To terminate TLS at a load balancer instead of in the registry, leave `registry.tls.secretName`
empty, give the load balancer a certificate for the host in `controller.registryUrls`, and have it
forward HTTP/2 (gRPC) to port `9090`, for example an NLB TLS listener. The controllers then follow
the TLS path: an `https://` URL, `controller.registryTlsCa` when a private CA signed the load
balancer's certificate, and `controller.registryAllowInsecureHttp` left `false`. The hop from the
load balancer to the registry is plaintext, so keep it on the private network. In the registry's
own cluster, set `controller.registryUrls` to the same `https://` URL, because the default URL of
a release without `registry.tls.secretName` is plaintext.

### Check the Registry

Check that the registry is ready. Forward its admin port:

```bash
kubectl -n mcs-system port-forward deployment/mcs-registry 9091:9091
```

In another terminal:

```bash
curl -s localhost:9091/readyz
```

## Install the Controller in Every Member Cluster

Install the controller in each member cluster with that cluster's id and region. The clusterset
name is the same everywhere. Use the sequence that matches the registry's transport.

### Production: TLS

Give the controller the CA certificate that signed the registry certificate, as a ConfigMap in the
controller's namespace, and set the registry URL to an `https://` URL whose host is one of the
certificate's SANs:

```bash
kubectl create namespace mcs-system
kubectl -n mcs-system create configmap mcs-registry-ca --from-file=ca.crt=ca.crt
helm install mcs-controller deploy/helm/charts/global-discovery-mcs \
  --namespace mcs-system \
  --set registry.enabled=false \
  --set controller.enabled=true \
  --set 'controller.registryUrls={https://mcs-registry.example.internal:9090}' \
  --set controller.registryTlsCa.configMapName=mcs-registry-ca \
  --set controller.clustersetId=prod \
  --set controller.clusterId=use1-a \
  --set controller.region=us-east-1 \
  --set image.repository=my-registry.example.com/global-discovery-mcs \
  --set image.tag=dev
```

When a CA in the system trust roots of the image signed the certificate, skip the ConfigMap and
`controller.registryTlsCa.configMapName`. To read the bundle from a Secret instead, set
`controller.registryTlsCa.secretName`; the bundle's key is `ca.crt` unless you set
`controller.registryTlsCa.key`.

### Evaluation: plaintext

Set an `http://` URL and opt in to plaintext explicitly:

```bash
helm install mcs-controller deploy/helm/charts/global-discovery-mcs \
  --namespace mcs-system --create-namespace \
  --set registry.enabled=false \
  --set controller.enabled=true \
  --set 'controller.registryUrls={http://mcs-registry.example.internal:9090}' \
  --set controller.registryAllowInsecureHttp=true \
  --set controller.clustersetId=prod \
  --set controller.clusterId=use1-a \
  --set controller.region=us-east-1 \
  --set image.repository=my-registry.example.com/global-discovery-mcs \
  --set image.tag=dev
```

### In the Registry's Cluster

In the cluster that runs the registry, install one release with both `registry.enabled=true` and
`controller.enabled=true` instead. Without `controller.registryUrls`, its controller dials the
registry Service of the release, `https://mcs-registry.<namespace>.svc:9090` with
`registry.tls.secretName` and `http://mcs-registry.<namespace>.svc:9090` without it. On the TLS
path, also set `controller.registryTlsCa.configMapName` and give the certificate the name
`mcs-registry.<namespace>.svc`. On the plaintext path, set
`controller.registryAllowInsecureHttp=true`, because the chart refuses the plaintext URL otherwise.

The values map to `mcs-controller` flags:

- `controller.registryUrls` (`--registry-url`): the registry's gRPC URLs, in failover order. Put
  a global DNS name first and per-region names after it. A URL without a port uses `443` for
  `https` and `80` for `http`, so name the port when the registry serves on `9090`. Every URL
  uses the same scheme.
- `controller.registryTlsCa` (`--registry-tls-ca-file`): the PEM bundle of the CAs that sign the
  registry certificates. When unset, the controller verifies the registry against the system roots.
- `controller.registryAllowInsecureHttp` (`--registry-allow-insecure-http`): permits `http://`
  URLs. It defaults to `false`; the chart and the controller both refuse an `http://` URL without
  it.
- `controller.clustersetId` (`--clusterset-id`) and `controller.clusterId` (`--cluster-id`): when
  unset, the controller reads them from the KEP-2149 ClusterProperty objects `clusterset.k8s.io`
  and `cluster.clusterset.k8s.io`.
- `controller.region` (`--region`): when unset, the controller uses the most common
  `topology.kubernetes.io/region` node label.

The controller uses leader election: only the leader writes to the registry and applies imports,
and a second replica takes over when the leader stops. The leader registers the cluster with a
30 s lease and renews it every 10 s. If the renewals stop, the registry removes the cluster's
endpoints from every import once the lease expires.

Check that every cluster registered. In the registry's cluster, forward its REST port:

```bash
kubectl -n mcs-system port-forward deployment/mcs-registry 8080:8080
```

In another terminal, list the clusters of the clusterset. Every cluster id appears with
`"state":"LEASE_STATE_ACTIVE"` in its lease:

```bash
curl -s localhost:8080/v1/clustersets/prod/clusters
```

When the registry serves TLS, its REST port does too. Verify the certificate against the CA and
resolve the certificate's host to the forwarded port:

```bash
curl -s --cacert ca.crt --resolve mcs-registry.example.internal:8080:127.0.0.1 \
  https://mcs-registry.example.internal:8080/v1/clustersets/prod/clusters
```

## Export Services

An exported Service appears as a ServiceImport, with a derived Service and EndpointSlices, in every
member cluster where its namespace exists. Export a Service in one of three ways.

Label the Service. The controller creates a ServiceExport of the same name, owned by the Service,
and deletes it when the label is removed:

```bash
kubectl -n dynamo label service my-dgd-pool-relay dynamo.nvidia.com/mcs-export=true
```

Or create the ServiceExport yourself, in the namespace and with the name of the Service:

```yaml
apiVersion: multicluster.x-k8s.io/v1beta1
kind: ServiceExport
metadata:
  name: my-dgd-pool-relay
  namespace: dynamo
```

Or let the Dynamo operator create the ServiceExports. Set `mcsImplementation` to `mcs` in the
operator configuration (`dynamo-operator.dynamo.multiCluster.mcsImplementation` in the platform
chart), and annotate each DynamoGraphDeployment (DGD) that serves as a Global Router pool:

```bash
kubectl -n dynamo annotate dynamographdeployment my-dgd nvidia.com/global-router-pool=true
```

The operator enables its `multiClusterExport` feature gate when the cluster serves the ServiceExport
API, and then creates ServiceExports for `<DGD name>-pool-relay` and `<DGD name>-frontend` of every
annotated DGD. Removing the annotation deletes them.

For the Global Discovery Service, export in every cluster:

- the PoolRelay Service of each pool, `<DGD name>-pool-relay`;
- the Global Router Service, in every cluster that runs Global Router replicas.

Keep the port names and numbers of an exported Service the same in every cluster. When exports of
one Service disagree, the oldest export wins and every participating cluster's ServiceExport reports
`Conflict=True`.

A ServiceImport lands in the namespace of the exported Service. In a cluster without that namespace,
the controller holds the import until the namespace is created; `--create-namespaces` creates it
instead.

## Verify the Imports

In the exporting cluster, check the ServiceExport's conditions:

```bash
kubectl -n dynamo get serviceexport my-dgd-pool-relay \
  -o jsonpath='{range .status.conditions[*]}{.type}={.status} {.reason}{"\n"}{end}'
```

A working export reports:

```text
Valid=True Valid
Ready=True Ready
Conflict=False NoConflicts
```

In any member cluster, list the ServiceImports, then the derived Services and the imported
EndpointSlices with their source clusters:

```bash
kubectl -n dynamo get serviceimports
kubectl -n dynamo get services,endpointslices -l multicluster.kubernetes.io/service-name \
  -L multicluster.kubernetes.io/service-name,multicluster.kubernetes.io/source-cluster
```

Each exported Service has one derived Service named `derived-<hash>` and one set of EndpointSlices
per exporting cluster, the local cluster included. The ServiceExport and ServiceImport objects do
not carry the `multicluster.kubernetes.io/service-name` label, so list them without the selector.

## Run the Global Discovery Service with MCS

The Global Discovery Service does not call the registry. Run it in any member cluster where
`mcs-controller` imports the clusterset; it watches the imported EndpointSlices with its pod's
credentials and needs `get`, `list` and `watch` on `endpointslices` in every namespace.

Start from the example manifest
[`container/global-discovery-service.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/container/global-discovery-service.yaml),
which holds the RBAC, Deployment and Service, and set `--pool-discovery=mcs`:

```yaml
args:
  - --listen=0.0.0.0:8080
  - --pool-discovery=mcs
  - --router-namespace=global-discovery
  - --router-service=global-router
  - --router-port-name=http
```

- `--router-namespace` and `--router-service` name the exported Global Router Service, and
  `--router-port-name` its TCP port that PoolRelays connect to. All three are required with
  `--pool-discovery mcs`.
- `--pool-service-suffix` (default `-pool-relay`) names each pool's PoolRelay Service,
  `<DGD name><suffix>`.

Check the lists it serves. Forward its port:

```bash
kubectl -n global-discovery port-forward service/global-discovery-service 8080:8080
```

In another terminal:

```bash
curl -s localhost:8080/pools
curl -s localhost:8080/routers
```

`/pools` lists one pool per source cluster, namespace and DGD whose PoolRelay Service has a Ready
endpoint; `key.cluster_id` is the cluster id of the exporting cluster. `/routers` lists every Ready
Global Router replica as `<cluster id>/<pod name>` with its URL.

The module's end-to-end test, `make -C deploy/global-discovery-mcs e2e`, runs this setup on three local kind clusters.

## Troubleshooting

| Symptom | Check |
|---|---|
| ServiceExport `Valid=False`, reason `NoService` | Create the Service with the ServiceExport's name in the same namespace. |
| ServiceExport `Valid=False`, reason `InvalidServiceType` | ExternalName Services cannot be exported. |
| ServiceExport `Ready=Unknown`, reason `Pending`, for more than a few seconds | The controller has not registered the cluster. Check the controller logs, the registry URL and port, that the URL scheme matches the registry's TLS setting, that the certificate names the URL's host and the controller trusts its CA, and network access to the registry. |
| ServiceExport `Ready=False`, reason `Failed` | The registry rejected the export, for example above 64 ports, labels or 32 annotations. The `ExportFailed` Event names the reason: `kubectl -n dynamo events --for serviceexport/my-dgd-pool-relay`. |
| ServiceExport `Ready=False`, reason `ClusterLeaseExpired` | The cluster's heartbeats stopped reaching the registry. Check the controller leader and its logs. |
| ServiceExport `Conflict=True` | The exports of the Service differ between clusters; the reason names the field (`PortConflict`, `TypeConflict` and so on). Align the Services; the oldest export wins in the meantime. |
| No ServiceImport in a cluster | The namespace is missing (the controller holds the import), `--watch-namespaces` excludes it, or a ServiceImport of another MCS implementation exists with that name. The controller skips ServiceImports it did not create unless `--adopt-unmanaged-imports` is set. |
| A cluster's endpoints are missing from an import | Its lease expired or it did not export the Service. List the clusters and imports through the registry's REST port, or run `mcs-admin describe` against the DynamoDB store. |
| The Global Discovery Service lists no pools | Check that the imported EndpointSlices exist in its cluster, that the PoolRelay Service names end with `--pool-service-suffix`, that the endpoints are Ready, and that its ServiceAccount can list EndpointSlices in every namespace. |
| The Global Discovery Service lists no routers | Check `--router-namespace`, `--router-service` and that `--router-port-name` names a TCP port of the exported router Service. |

To read the controller logs:

```bash
kubectl -n mcs-system logs deployment/mcs-controller
```

To inspect what the registry holds, list the imports through its REST port (forwarded as above, with the TLS options when the registry serves TLS):

```bash
curl -s 'localhost:8080/v1/clustersets/prod/imports?includeEndpoints=true'
```

With the DynamoDB store, `mcs-admin` reads the table directly, with the registry role's
permissions. `describe` lists the clusters with their lease states, the exports and the endpoint
sets:

```bash
mcs-admin --store=dynamodb --region=us-east-1 describe --clusterset=prod
```

To remove a cluster that will not return, together with its exports, run
`mcs-admin --store=dynamodb --region=us-east-1 delete-cluster --clusterset=prod --cluster=use1-a --yes`.

## CRDs and Other MCS Implementations

The chart installs three CRDs when `crds.install=true`: ServiceExport and ServiceImport, copied from
[`sigs.k8s.io/mcs-api`](https://github.com/kubernetes-sigs/mcs-api) v0.5.2, and the KEP-2149
ClusterProperty (`about.k8s.io/v1alpha1`). The mcs-api v0.5.2 CRDs serve both
`multicluster.x-k8s.io/v1alpha1` and `v1beta1`, and store `v1beta1`. The controller reads and
writes `v1beta1`; clients that write `v1alpha1`, such as the Dynamo operator and Karmada, keep
working against the same CRDs.

Set `crds.install=false` when another installation already owns the CRDs in a cluster, for example
Karmada or another release of this chart. The cluster must still serve
`multicluster.x-k8s.io/v1beta1`; replace a CRD that serves only `v1alpha1` with the mcs-api v0.5.2
version first.

Within one cluster, run one MCS implementation per namespace. The controller acts on every
ServiceExport in the namespaces it watches: it adds its finalizer, writes the export to the
registry and sets the export's conditions. Limit it with `--watch-namespaces` where another
implementation exports Services from the same cluster.
