<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# global-discovery-mcs

Multi-Cluster Services registry and controller for the Dynamo Global Router

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 0.1.0](https://img.shields.io/badge/AppVersion-0.1.0-informational?style=flat-square)

This chart deploys the Kubernetes Multi-Cluster Services API (KEP-1645,
`multicluster.x-k8s.io/v1beta1`) of the Dynamo Global Router, built from
[`deploy/global-discovery-mcs`](../../../global-discovery-mcs):

- `mcs-registry` (`registry.enabled`, off by default): the registry server.
  It keeps the clusters of every clusterset with their ServiceExports and
  endpoints in memory or in one DynamoDB table, merges them into
  ServiceImports and serves the `dynamo.mcs.v1` API over gRPC and REST.
- `mcs-controller` (`controller.enabled`, on by default): runs in every
  member cluster. It registers the cluster with the registry, exports the
  cluster's ServiceExports and materialises the clusterset's ServiceImports,
  derived Services and EndpointSlices, which the Global Discovery Service
  reads.
- The ServiceExport, ServiceImport and ClusterProperty CRDs (`crds.install`).

The module's [README](../../../global-discovery-mcs/README.md) lists every
flag; [docs/mcs-controller.md](../../../global-discovery-mcs/docs/mcs-controller.md)
describes the controller, and [docs/iam.md](../../../global-discovery-mcs/docs/iam.md)
the registry's AWS permissions.

## Prerequisites

- Kubernetes 1.30 or later.
- Helm 3.8 or later.
- The `global-discovery-mcs` image (`image.repository`), built with
  `make docker-build` in `deploy/global-discovery-mcs`.
- For `registry.store=dynamodb`: the DynamoDB table, created once with
  `mcs-admin create-table`, and an IAM role for the registry pods.
- Pod IPs routable between the member clusters for ClusterSetIP imports
  (see "Reachability prerequisite" in docs/mcs-controller.md).

## Installing

The chart has no default registry URL: install it with the registry, with
`controller.registryUrls`, or both. The resources are named after their
component, not the release: the Deployments, Services and ServiceAccounts
are `mcs-registry` and `mcs-controller` (`fullnameOverride` changes the
`mcs` prefix), so install one release per namespace. One release may enable
both components.

A member cluster whose controller talks to registries elsewhere:

```bash
helm install mcs deploy/helm/charts/global-discovery-mcs \
  --namespace mcs-system --create-namespace \
  --set 'controller.registryUrls={https://mcs.us-east-1.example.internal:9090,https://mcs.us-west-2.example.internal:9090}' \
  --set controller.registryTlsCa.configMapName=mcs-registry-ca \
  --set controller.clustersetId=prod \
  --set controller.clusterId=use1-a \
  --set controller.region=us-east-1
```

A registry cluster serving the DynamoDB store over TLS behind an internal
NLB, with IRSA:

```bash
helm install mcs deploy/helm/charts/global-discovery-mcs \
  --namespace mcs-system --create-namespace \
  --set controller.enabled=false \
  --set registry.enabled=true \
  --set registry.store=dynamodb \
  --set registry.dynamodb.table=global-discovery-mcs \
  --set registry.replicas=2 \
  --set registry.podDisruptionBudget.enabled=true \
  --set registry.tls.secretName=mcs-registry-tls \
  --set registry.service.type=LoadBalancer \
  --set registry.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"=external \
  --set registry.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-scheme"=internal \
  --set registry.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-nlb-target-type"=ip \
  --set registry.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=arn:aws:iam::123456789012:role/mcs-registry
```

The same as a values file:

```yaml
controller:
  enabled: false
registry:
  enabled: true
  store: dynamodb
  dynamodb:
    table: global-discovery-mcs
  replicas: 2
  podDisruptionBudget:
    enabled: true
  tls:
    secretName: mcs-registry-tls
  service:
    type: LoadBalancer
    annotations:
      service.beta.kubernetes.io/aws-load-balancer-type: external
      service.beta.kubernetes.io/aws-load-balancer-scheme: internal
      service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: ip
  serviceAccount:
    annotations:
      eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/mcs-registry
```

One cluster with both components and the memory store, for development
(for example on kind):

```bash
helm install mcs deploy/helm/charts/global-discovery-mcs \
  --namespace mcs-system --create-namespace \
  --set image.repository=global-discovery-mcs --set image.tag=dev \
  --set registry.enabled=true \
  --set controller.registryAllowInsecureHttp=true \
  --set controller.clustersetId=dev \
  --set controller.clusterId=kind \
  --set controller.region=local
kubectl -n mcs-system port-forward deployment/mcs-registry 8080 9091
kubectl -n mcs-system logs deployment/mcs-controller
```

With `registry.enabled` and no `controller.registryUrls`, the controller
uses the registry Service of the release,
`http://mcs-registry.<namespace>.svc:9090`, or `https://` with
`registry.tls.secretName`. A plaintext URL needs
`controller.registryAllowInsecureHttp=true` (`--registry-allow-insecure-http`);
the chart fails otherwise, as the controller would.

Install the CRDs from one release per cluster: a second release with
`crds.install=true` fails on Helm's ownership check. Set `crds.install=false`
in the others.

## CRDs

`crds.install` (default `true`) renders three CRDs from `files/crds`, a copy
of `deploy/global-discovery-mcs/config/crd/bases`:

- `serviceexports.multicluster.x-k8s.io` and
  `serviceimports.multicluster.x-k8s.io`, unchanged from `sigs.k8s.io/mcs-api`
  v0.5.2. They serve `v1alpha1` and `v1beta1` and store `v1beta1`; the
  controller uses `v1beta1`.
- `clusterproperties.about.k8s.io` (KEP-2149), with the
  `api-approved.kubernetes.io` annotation the protected `about.k8s.io` group
  requires.

The CRDs are templates rather than files in `crds/`, so `crds.install`
switches them off and `helm upgrade` updates them. With `crds.keep` (default
`true`) they carry `helm.sh/resource-policy: keep`: `helm uninstall`, or an
upgrade to `crds.install=false`, leaves them and every ServiceExport and
ServiceImport in place.

Set `crds.install=false` when the cluster already serves these CRDs, for
example with Karmada or another MCS implementation, or when another release
owns them. Another implementation's CRDs must serve
`multicluster.x-k8s.io/v1beta1`; the controller skips ServiceImports it did
not create unless `controller.adoptUnmanagedImports` is set. Without the
ClusterProperty CRD the controller reads no ClusterProperty, so set
`controller.clustersetId` and `controller.clusterId`.

## Registry

- **Store.** `registry.store=memory` keeps everything in the memory of one
  replica and loses it on restart: it requires `registry.replicas=1`, and
  both the values schema and the templates refuse more. Run several
  replicas with `registry.store=dynamodb`, which reads and writes the
  replica of the DynamoDB Global Table (`registry.dynamodb.table`) in
  `registry.region` (default `$AWS_REGION`).
- **AWS credentials.** The registry uses the AWS SDK default credential
  chain, so DynamoDB credentials come from the pod identity: annotate the
  ServiceAccount with `eks.amazonaws.com/role-arn` for IRSA
  (`registry.serviceAccount.annotations`), or create an EKS Pod Identity
  association for it. The role's policy and trust policy are in
  [docs/iam.md](../../../global-discovery-mcs/docs/iam.md). The chart creates
  no AWS credentials Secret; for dynamodb-local, set dummy
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_REGION` in
  `registry.extraEnv` and `registry.dynamodb.endpoint`.
- **TLS.** `registry.tls.secretName` names a `kubernetes.io/tls` Secret (for
  example from cert-manager) that enables TLS on the gRPC and REST
  listeners. The admin listener stays plaintext. The registry reads the key
  pair at startup, so restart it after a certificate rotation.
- **Exposure.** The Service `mcs-registry` exposes gRPC (9090), REST (8080)
  and admin (9091: `/metrics`, `/healthz`, `/readyz`, `/debug/pprof`); the
  ports are `registry.ports`. The registry has no authentication: expose it
  to other clusters only on private networks, for example through an
  internal NLB (`registry.service.type=LoadBalancer` with the AWS Load
  Balancer Controller's annotations in `registry.service.annotations`, as
  above), and keep `registry.watch.bookmarkInterval` below the load
  balancer's idle timeout. The admin port is on the same Service, so the
  load balancer exposes it too.
- **Shutdown.** On SIGTERM the registry reports not ready for
  `registry.shutdownDrain`, then stops within `registry.shutdownTimeout`;
  keep `registry.terminationGracePeriodSeconds` above their sum.
- **Flags.** Every `mcs-registry` flag also reads `MCS_<FLAG_NAME>` from the
  environment, but a flag on the command line wins and the chart passes its
  values as flags: change the values or `registry.extraArgs`, not
  `registry.extraEnv`.

## Controller

- **Registry.** `controller.registryUrls` lists the registries in failover
  order (one `--registry-url` each), for example
  `--set 'controller.registryUrls={https://mcs.example.internal:9090}'`.
  `controller.registryTlsCa` mounts a CA bundle from a ConfigMap or Secret
  for `--registry-tls-ca-file`; without it the system roots verify the
  registry. `controller.registryAllowInsecureHttp` passes
  `--registry-allow-insecure-http` for `http://` URLs.
- **Identity.** `controller.clustersetId` and `controller.clusterId` are DNS
  labels; empty ids come from the ClusterProperty objects
  (`controller.clusterIdentitySource=auto`). The cluster id is the
  `source-cluster` other clusters see; run exactly one controller per
  cluster id. `controller.region` sets the region of the registration;
  empty uses the nodes' `topology.kubernetes.io/region` label.
- **Replicas and leader election.** Two replicas by default, with leader
  election on: only the leader writes to the registry and applies imports,
  holding the Lease `mcs-controller.dynamo.nvidia.com` in the release
  namespace. `POD_NAMESPACE` comes from the downward API and is the default
  of `--operator-namespace`. More than one replica requires leader election.
- **RBAC.** The ClusterRole is rendered from `files/controller-role.yaml`,
  which `make manifests` in `deploy/global-discovery-mcs` generates from the
  controller's `+kubebuilder:rbac` markers. A Role and RoleBinding in the
  release namespace grant the leader election Lease and its Events.
- **Metrics.** `/metrics` on port 8080 behind a ClusterIP Service;
  `controller.metrics.serviceMonitor.enabled` adds a prometheus-operator
  ServiceMonitor (`registry.metrics.serviceMonitor.enabled` does the same
  for the registry's admin port).

## Development

```bash
make -C deploy/helm/charts/global-discovery-mcs lint       # helm lint and renders; fails when files/crds is stale
make -C deploy/helm/charts/global-discovery-mcs test       # helm-unittest (installs the plugin)
make -C deploy/helm/charts/global-discovery-mcs docs       # regenerate this README with helm-docs
make -C deploy/helm/charts/global-discovery-mcs sync-crds  # copy the module's CRDs into files/crds
```

- `files/controller-role.yaml` is generated: run `make manifests` in
  `deploy/global-discovery-mcs`, then copy its rules into
  `tests/rbac_test.yaml`.
- `files/crds` is a copy: change `deploy/global-discovery-mcs/config/crd/bases`
  (`make sync-crds` there), then run `make sync-crds` here.
  `make check` in the module fails while the copies differ.
- Edit `README.md.gotmpl` and the `# --` comments in `values.yaml`, then run
  `make docs`; do not edit `README.md`.

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| nameOverride | string | `""` | Override the chart name in the `app.kubernetes.io/name` label. |
| fullnameOverride | string | `""` | Prefix of the resource names: the components are `<prefix>-registry` and `<prefix>-controller`. Empty uses `mcs`, so the Deployments are `mcs-registry` and `mcs-controller`. |
| image.repository | string | `"nvcr.io/nvidia/ai-dynamo/global-discovery-mcs"` | Image holding `/mcs-registry` (the entrypoint), `/mcs-controller` and `/mcs-admin`, built from deploy/global-discovery-mcs/Dockerfile. |
| image.tag | string | `""` | Image tag. Empty uses the chart appVersion. |
| image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| imagePullSecrets | list | `[]` | Image pull secrets of the registry and controller pods, for example `[{name: ngc-secret}]`. |
| podSecurityContext | object | `{"runAsGroup":65532,"runAsNonRoot":true,"runAsUser":65532,"seccompProfile":{"type":"RuntimeDefault"}}` | Pod security context of both components. The image runs as the distroless nonroot user 65532. |
| securityContext | object | `{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]},"readOnlyRootFilesystem":true}` | Container security context of both components. |
| rbac.create | bool | `true` | Create the controller ClusterRole (rendered from files/controller-role.yaml), its ClusterRoleBinding and the leader election Role and RoleBinding. |
| crds.install | bool | `true` | Render the ServiceExport and ServiceImport CRDs (sigs.k8s.io/mcs-api v0.5.2) and the KEP-2149 ClusterProperty CRD. Set false when Karmada or another MCS implementation already serves them. |
| crds.keep | bool | `true` | Annotate the CRDs with `helm.sh/resource-policy: keep`, so that `helm uninstall` and `crds.install=false` leave them, and every ServiceExport and ServiceImport, in place. |
| registry.enabled | bool | `false` | Deploy mcs-registry. Usually enabled in the registry clusters only. |
| registry.replicas | int | `1` | Registry replicas. `store: memory` keeps the state in the replica's memory and requires exactly one replica (the chart refuses more); scale out with `store: dynamodb`. |
| registry.store | string | `"memory"` | Store backend: `memory` or `dynamodb` (`--store`). |
| registry.dynamodb.table | string | `"global-discovery-mcs"` | DynamoDB table (`--dynamodb-table`); create it with `mcs-admin create-table`. |
| registry.dynamodb.endpoint | string | `""` | DynamoDB endpoint override, for dynamodb-local (`--dynamodb-endpoint`). Empty uses the AWS endpoint of the region. |
| registry.region | string | `""` | Region reported by GetServerInfo and used by the DynamoDB store (`--region`). Empty uses `$AWS_REGION`, which IRSA and EKS Pod Identity inject. |
| registry.refreshInterval | string | `"2s"` | Full reload of each clusterset from the store (`--refresh-interval`). |
| registry.leaseEvalInterval | string | `"1s"` | Lease evaluation without a store read (`--lease-eval-interval`). |
| registry.clustersetDiscoveryInterval | string | `"30s"` | Listing of the store's clustersets (`--clusterset-discovery-interval`). |
| registry.leaseSkew | string | `"2s"` | Clock skew allowance before a lease expires, 0s to 1m (`--lease-skew`). |
| registry.retention | string | `"24h"` | How long the records of an expired cluster are kept, at least 5m (`--retention`). |
| registry.compactionInterval | string | `"10m"` | Interval of the retention cleanup (`--compaction-interval`). |
| registry.watch.bookmarkInterval | string | `"5s"` | Interval of watch bookmarks, 1s to 5m (`--bookmark-interval`). Keep it below the idle timeout of any load balancer in front of the registry. |
| registry.watch.history | int | `10000` | Events kept per clusterset for resume (`--watch-history`). |
| registry.watch.queueSize | int | `1024` | Per-watcher queue; an overflowing watcher is re-snapshotted (`--watch-queue-size`). |
| registry.watch.maxWatchers | int | `5000` | Concurrent watch streams per replica (`--max-watchers`). |
| registry.maxMessageBytes | int | `4194304` | gRPC message size limit in bytes, at least 1 MiB (`--max-message-bytes`). |
| registry.advertiseEndpoints | list | `[]` | Endpoints returned by GetServerInfo as client fallbacks (`--advertise-endpoints`), each `{region, grpcUrl, httpUrl, primary}`; at most one primary. |
| registry.tls.secretName | string | `""` | Name of an existing `kubernetes.io/tls` Secret (keys `tls.crt`, `tls.key`) that enables TLS on the gRPC and REST listeners (`--tls-cert-file`, `--tls-key-file`). Empty serves plaintext (h2c). The registry reads the key pair at startup; restart it after a rotation. |
| registry.shutdownDrain | string | `"5s"` | Time reported not ready before the replica stops serving (`--shutdown-drain`). |
| registry.shutdownTimeout | string | `"25s"` | Graceful stop of the gRPC and REST servers (`--shutdown-timeout`). |
| registry.terminationGracePeriodSeconds | int | `45` | Pod termination grace period in seconds. Keep it above `shutdownDrain` plus `shutdownTimeout`. |
| registry.logLevel | string | `"info"` | Log level: `debug`, `info`, `warn` or `error` (`--log-level`). |
| registry.otel.enabled | bool | `false` | OTLP tracing (`--otel-enabled`), configured by `OTEL_*` variables in `registry.extraEnv`. |
| registry.ports.grpc | int | `9090` | gRPC port (`--grpc-listen-address`), on the container and the Service. |
| registry.ports.http | int | `8080` | REST gateway port (`--http-listen-address`), on the container and the Service. |
| registry.ports.admin | int | `9091` | Admin port (`--admin-listen-address`): `/metrics`, `/healthz`, `/readyz` and `/debug/pprof`, on the container and the Service. |
| registry.service.type | string | `"ClusterIP"` | Service type. The registry has no authentication: expose it only on private networks (for example an internal NLB). |
| registry.service.annotations | object | `{}` | Service annotations, for example the AWS Load Balancer Controller's. |
| registry.serviceAccount.create | bool | `true` | Create the registry ServiceAccount. |
| registry.serviceAccount.name | string | `""` | ServiceAccount name. Empty uses `<fullname>-registry`. |
| registry.serviceAccount.annotations | object | `{}` | ServiceAccount annotations, for example `eks.amazonaws.com/role-arn` for IRSA (see deploy/global-discovery-mcs/docs/iam.md). |
| registry.automountServiceAccountToken | bool | `false` | Mount the Kubernetes API token into the registry pods. The registry does not call the Kubernetes API; IRSA and EKS Pod Identity mount their own token. |
| registry.podDisruptionBudget.enabled | bool | `false` | Create a PodDisruptionBudget for the registry. |
| registry.podDisruptionBudget.minAvailable | int | `1` | Minimum available registry pods, a number or a percentage. Ignored when `maxUnavailable` is set. |
| registry.podDisruptionBudget.maxUnavailable | int | `nil` | Maximum unavailable registry pods, a number or a percentage; set it instead of `minAvailable`. |
| registry.metrics.serviceMonitor.enabled | bool | `false` | Create a prometheus-operator ServiceMonitor that scrapes `/metrics` on the admin port. |
| registry.metrics.serviceMonitor.interval | string | `"30s"` | Scrape interval. |
| registry.metrics.serviceMonitor.labels | object | `{}` | Extra ServiceMonitor labels, for example the `release` label a Prometheus selects. |
| registry.readinessProbe.periodSeconds | int | `5` | Readiness probe period against `/readyz` on the admin port (store ping and a fresh view). |
| registry.readinessProbe.timeoutSeconds | int | `3` | Readiness probe timeout. |
| registry.readinessProbe.failureThreshold | int | `3` | Readiness probe failure threshold. |
| registry.livenessProbe.periodSeconds | int | `10` | Liveness probe period against `/healthz` on the admin port. |
| registry.livenessProbe.timeoutSeconds | int | `3` | Liveness probe timeout. |
| registry.livenessProbe.failureThreshold | int | `3` | Liveness probe failure threshold. |
| registry.extraArgs | list | `[]` | Extra registry arguments, appended after the chart's flags. |
| registry.extraEnv | list | `[]` | Extra registry environment variables (EnvVar objects), for example `OTEL_*` or `AWS_*`. The chart passes its values as flags, which win over the `MCS_*` fallbacks. |
| registry.extraVolumes | list | `[]` | Extra registry pod volumes. |
| registry.extraVolumeMounts | list | `[]` | Extra registry container volume mounts. |
| registry.resources | object | `{"limits":{"memory":"1Gi"},"requests":{"cpu":"100m","memory":"128Mi"}}` | Registry container resources. |
| registry.nodeSelector | object | `{}` | Registry pod node selector. |
| registry.tolerations | list | `[]` | Registry pod tolerations. |
| registry.affinity | object | `{}` | Registry pod affinity. |
| registry.topologySpreadConstraints | list | `[]` | Registry pod topology spread constraints. |
| registry.priorityClassName | string | `""` | Registry pod priority class. |
| registry.podAnnotations | object | `{}` | Registry pod annotations. |
| registry.podLabels | object | `{}` | Registry pod labels. |
| controller.enabled | bool | `true` | Deploy mcs-controller. One installation runs in every member cluster. |
| controller.replicas | int | `2` | Controller replicas. Only the leader writes to the registry and applies imports; more than one replica requires `leaderElect`. |
| controller.registryUrls | list | `[]` | Registry URLs in failover order, all `https://` or all `http://` (one `--registry-url` each), for example `--set 'controller.registryUrls={https://mcs.example.internal:9090}'`. Empty uses the registry Service of this release when `registry.enabled`. |
| controller.registryAllowInsecureHttp | bool | `false` | Permit `http://` registry URLs (`--registry-allow-insecure-http`). |
| controller.registryTlsCa.configMapName | string | `""` | ConfigMap holding the PEM bundle of the CAs that sign the registry certificates, mounted for `--registry-tls-ca-file`. Empty with `secretName` empty uses the system roots. |
| controller.registryTlsCa.secretName | string | `""` | Secret holding the CA bundle, instead of `configMapName`. |
| controller.registryTlsCa.key | string | `"ca.crt"` | Key of the CA bundle in the ConfigMap or Secret. |
| controller.clustersetId | string | `""` | Clusterset name, a DNS label (`--clusterset-id`). Empty reads the ClusterProperty `clusterset.k8s.io`. |
| controller.clusterId | string | `""` | Cluster id, a DNS label unique in the clusterset (`--cluster-id`). Empty reads the ClusterProperty `cluster.clusterset.k8s.io`. |
| controller.clusterIdentitySource | string | `"auto"` | Where the ids come from: `auto` (flags win, else ClusterProperty), `flags` or `clusterproperty` (`--cluster-identity-source`). |
| controller.region | string | `""` | Region of this cluster (`--region`). Empty uses the most common `topology.kubernetes.io/region` node label. |
| controller.clusterLabels | object | `{}` | Labels of the cluster registration (`--cluster-labels`). |
| controller.leaseTtl | string | `"30s"` | Cluster lease TTL, 10s to 5m; heartbeats run every TTL/3 (`--lease-ttl`). |
| controller.watchNamespaces | list | `[]` | Namespaces exported from and imported into (`--watch-namespaces`). Empty watches every namespace. |
| controller.createNamespaces | bool | `false` | Create missing import namespaces (`--create-namespaces`). |
| controller.adoptUnmanagedImports | bool | `false` | Take over ServiceImports of other implementations (`--adopt-unmanaged-imports`). |
| controller.writeClusterProperties | bool | `false` | Write the ClusterProperty objects of the resolved identity (`--write-cluster-properties`). |
| controller.deregisterOnShutdown | bool | `false` | Deregister the cluster on stop, deleting its exports in the registry (`--deregister-on-shutdown`). Leave it off for restarts and upgrades. |
| controller.endpointsPerSlice | int | `100` | Endpoints per imported EndpointSlice, 1 to 1000 (`--endpoints-per-slice`). |
| controller.maxConcurrentReconciles | int | `4` | Workers per controller (`--max-concurrent-reconciles`). |
| controller.leaderElect | bool | `true` | Leader election (`--leader-elect`) with a Lease in the release namespace (`POD_NAMESPACE`); required for more than one replica. |
| controller.otel.enabled | bool | `false` | OTLP tracing (`--otel-enabled`), configured by `OTEL_*` variables in `controller.extraEnv`. |
| controller.log.level | string | `"info"` | Log level: `debug`, `info`, `error`, `panic` or an integer verbosity (`--zap-log-level`). Empty keeps the controller-runtime default. |
| controller.log.encoder | string | `"json"` | Log encoder: `json` or `console` (`--zap-encoder`). Empty keeps the controller-runtime default. |
| controller.log.stacktraceLevel | string | `"error"` | Level at and above which stack traces are logged: `info`, `error` or `panic` (`--zap-stacktrace-level`). Empty keeps the controller-runtime default. |
| controller.log.timeEncoding | string | `"rfc3339"` | Time encoding: `epoch`, `millis`, `nano`, `iso8601`, `rfc3339` or `rfc3339nano` (`--zap-time-encoding`). Empty keeps the controller-runtime default. |
| controller.log.development | bool | `false` | Development logging defaults (`--zap-devel`). |
| controller.healthProbePort | int | `8081` | Health probe port serving `/healthz` and `/readyz` (`--health-probe-bind-address`). |
| controller.metrics.enabled | bool | `true` | Serve `/metrics` (`--metrics-bind-address`) and create the metrics Service. False passes `--metrics-bind-address=0`. |
| controller.metrics.port | int | `8080` | Metrics port, on the container and the metrics Service. |
| controller.metrics.serviceMonitor.enabled | bool | `false` | Create a prometheus-operator ServiceMonitor for the metrics Service. |
| controller.metrics.serviceMonitor.interval | string | `"30s"` | Scrape interval. |
| controller.metrics.serviceMonitor.labels | object | `{}` | Extra ServiceMonitor labels, for example the `release` label a Prometheus selects. |
| controller.terminationGracePeriodSeconds | int | `30` | Pod termination grace period in seconds. |
| controller.serviceAccount.create | bool | `true` | Create the controller ServiceAccount. |
| controller.serviceAccount.name | string | `""` | ServiceAccount name. Empty uses `<fullname>-controller`. |
| controller.serviceAccount.annotations | object | `{}` | ServiceAccount annotations. |
| controller.readinessProbe.periodSeconds | int | `10` | Readiness probe period against `/readyz` (registry connected and, on the leader, the cluster registered). |
| controller.readinessProbe.timeoutSeconds | int | `3` | Readiness probe timeout. |
| controller.readinessProbe.failureThreshold | int | `3` | Readiness probe failure threshold. |
| controller.livenessProbe.periodSeconds | int | `20` | Liveness probe period against `/healthz`. |
| controller.livenessProbe.timeoutSeconds | int | `3` | Liveness probe timeout. |
| controller.livenessProbe.failureThreshold | int | `3` | Liveness probe failure threshold. |
| controller.extraArgs | list | `[]` | Extra controller arguments, appended after the chart's flags. |
| controller.extraEnv | list | `[]` | Extra controller environment variables (EnvVar objects), for example `OTEL_*`. |
| controller.extraVolumes | list | `[]` | Extra controller pod volumes. |
| controller.extraVolumeMounts | list | `[]` | Extra controller container volume mounts. |
| controller.resources | object | `{"limits":{"memory":"512Mi"},"requests":{"cpu":"50m","memory":"128Mi"}}` | Controller container resources. |
| controller.nodeSelector | object | `{}` | Controller pod node selector. |
| controller.tolerations | list | `[]` | Controller pod tolerations. |
| controller.affinity | object | `{}` | Controller pod affinity. |
| controller.topologySpreadConstraints | list | `[]` | Controller pod topology spread constraints. |
| controller.priorityClassName | string | `""` | Controller pod priority class. |
| controller.podAnnotations | object | `{}` | Controller pod annotations. |
| controller.podLabels | object | `{}` | Controller pod labels. |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)
