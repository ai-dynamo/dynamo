<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DynamoPoolRelay operator prototype

`nvidia.com/v1alpha1 DynamoPoolRelay` deploys an active-active Pool Relay independently of
inference DGD lifecycles, with two replicas by default. The operator derives its source scope and connection
from selected DGDs. The CRD and `spec.clusterId` follow the Pool Relay vocabulary
in [DEP #15742](https://github.com/ai-dynamo/dynamo/issues/15742).
This is an experimental single-version API. It has no conversion webhook or
compatibility alias for the unpublished `DynamoRelay` prototype.
Existing local prototype CRs must move their DGD selection into `spec.sources.dgds`
before installing the updated CRD and operator.

## Install and configure

Use a Relay image supporting file sources, connection-revision guards,
`GET /engine/state` acknowledgements, and `logicalNamespace` source identities.
Build the Relay image with the separate `fix/relay-logical-source-namespaces`
runtime change to include `logicalNamespace` support. This operator branch does
not include Relay runtime changes. Older images that reject unknown source fields
cannot consume files containing this field during worker namespace rollouts. Files without the field
retain their existing behavior and canonical revision.

The operator maps `spec.clusterId` to the runtime's existing `--dc-id` flag and
launches `dynamo.kv_dc_relay`, probing `dynamo.kvrelay.v1.KvEventRelay`. These
legacy runtime names remain supported as aliases under DEP #15742. Runtime
module, CLI and gRPC renames belong to the separate runtime changes.
The operator always registers the DynamoPoolRelay controller. Install the generated
CRD and operator RBAC before starting it. A cluster-wide Helm installation with
`dynamo-operator.upgradeCRD=true` applies the CRD from the operator image. When CRD
upgrades are managed separately, including namespace-restricted installations,
apply the version-matched CRDs before upgrading the operator. Create a
DynamoPoolRelay resource to deploy a Relay.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoPoolRelay
metadata:
  name: dc-a
  namespace: inference
spec:
  clusterId: dc-a
  replicas: 2
  podTemplate:
    spec:
      containers:
        - name: main
          image: registry.example.com/dynamo:relay
  sources:
    dgds:
      labelSelector:
        matchLabels:
          relay-group: production
      refs:
        - name: special-model
```

`spec.sources.dgds` groups `labelSelector` and `refs`. Their union selects each
DGD once, only in the Relay's Kubernetes namespace. Label selection matches DGD
`metadata.labels`. An explicit `labelSelector: {}` selects all DGDs in that
namespace; an omitted selector selects nothing. Supply a selector or nonempty refs.
Explicit references must contain unique, valid names; a missing referenced DGD is unresolved.
Deleting DGDs are withdrawn. Recreating a DGD with the same namespace and
connection preserves the source identity.

The operator derives worker discovery namespaces, event/request transports,
NATS settings, and applicable transport TLS settings. It follows the existing
DGD env precedence and operator defaults, including discovery annotations on
DGD metadata, graph spec, and component Pod templates. Multinode components
with complete `roles[].podTemplate` sources are resolved per role; every selected
worker role must use a compatible transport. Do not put `eventPlane`, `nats`, or
`dynamoNamespaces` in the Relay CR. Users of the earlier local prototype must
migrate those fields to DGD selection and worker connection configuration.

## Credentials and connection compatibility

Configure NATS on the selected DGD's worker using the normal runtime envs. For
example, these entries belong in its main-container `env`:

```yaml
- name: DYN_EVENT_PLANE
  value: nats
- name: NATS_SERVER
  valueFrom:
    secretKeyRef:
      name: worker-nats
      key: url
- name: NATS_AUTH_CREDENTIALS_FILE
  value: /etc/nats/client.creds
```

For the credentials-file example, the worker must also declare a Secret volume
and corresponding mount exposing `/etc/nats/client.creds`. The operator inherits
only required connection env references and file keys, with read-only mounts.
It supports literal env values, `secretKeyRef`, `configMapKeyRef`, and supported
Secret/ConfigMap file mounts, including fixed `subPath`. Secret values are read
to resolve and validate configuration; references are retained in the generated
Pod. Unrelated worker envs and mounted keys are not copied.

One Relay process requires one compatible connection across every selected
worker, prefill, and decode component. Conflicts are reported instead of silently
omitting a source. Brokered ZMQ must follow a selected worker's declared broker
configuration; another broker's presence in the namespace proves nothing.

Transport settings supplied through worker `envFrom`, shell-derived values,
image-internal configuration, `subPathExpr`,
unsupported volume sources, and global Dynamo namespaces are outside this
phase. Kubernetes pod discovery is required. Source files are limited to 4096
sources and 1 MiB. DGDs whose normalized Dynamo namespaces collide are rejected.

Changes to referenced credentials, TLS data, or connection configuration update
the Deployment with `RollingUpdate` (`maxUnavailable: 0`, `maxSurge: 1`). The operator computes a keyed content revision
using an owned Secret named `<relay-resource-name>-revision-key`. Only relevant
keys affect the revision; metadata and unrelated data changes do not. The key
Secret is private to the operator and is not mounted into Relay. Removing it
causes a new key and connection revision on reconciliation.

## Live sources and status

The owned ConfigMap projects `sources.json` without a `subPath` mount. The file
contains `version: 1`, `connectionRevision`, and `sources: [{"namespace": "..."}]`.
Entries for a hashed worker runtime namespace also include `logicalNamespace`
with the stable DGD namespace;
it contains no DGD identity, credentials, or stored revision. Both operator and
Relay compute SHA256 from the canonical document with sources sorted by namespace.
The launcher uses `--sources-file` and `DYN_RELAY_CONNECTION_REVISION` supplied by the
operator. File mode cannot be combined with discovery filters. Standalone file
mode may omit connectionRevision when the environment guard is also absent.

Relay discovers workers and resolves KV domains through the physical `namespace`.
The optional `logicalNamespace` changes only the public identity in the catalog,
serving readiness, and endpoint status. When the DGD's active worker runtime
namespace changes, the operator updates the physical source while preserving
that public identity. Relay withdraws the old pool and starts a new pool lifetime;
it does not carry KV state or readiness across the switch. An alias-only change
also starts a new public lifetime.

Multiple physical sources may share a logical namespace for different components
or endpoints. If they claim the same logical namespace/component/endpoint, Relay
withholds all conflicting endpoints and reports `sources.logicalEndpointConflicts`
in `GET /engine/state` until the conflict is resolved.

Compatible source additions/removals update the file without changing the Pod
template. Kubernetes projection is asynchronous; Relay polls the projected file
and reports the applied revision. Unchanged sources keep their watches. An empty
selection means zero sources and preserves an existing connection. Invalid files
are rejected. A file for another connection revision cannot be applied to the
old process during a rollout.

If an addition is unresolved or incompatible, the operator retains the last
valid connection and still withdraws sources no longer selected. Inspect both
configuration and scope conditions to distinguish partial withdrawal from full
application of the desired selection.

- `SelectionResolved`: explicit references exist and selection is resolved.
- `ConfigurationResolved`: selected workers have one supported connection.
- `SourcesApplied`: every requested current replica acknowledged the generated scope;
  no superseded non-terminating Pods remain.
- `Ready`: all requested replicas are updated and available, the rollout is complete,
  and their gRPC listener probes succeed. It does not imply KV recovery or model readiness.
- `desiredSourcesRevision` and `appliedSourcesRevision`: generated and acknowledged
  scope identities. The applied revision is empty until all replicas acknowledge it.
  The operator verifies each Pod UID, current ReplicaSet template, and Deployment ownership.
- `replicas`, `updatedReplicas`, `availableReplicas`, and `sourcesAppliedReplicas`:
  observed Deployment counts and the number of current replicas acknowledging the desired scope.

The `GET /engine/state` acknowledgement endpoint is Pod-local HTTP port 8081 and is not published
by the Service. Operator-to-Pod connectivity to that port is required. By default the Service exposes cluster-internal gRPC on 5561. Configure
`spec.service.ports` to publish a sidecar instead; the port named `grpc` determines
`status.endpoint`. The operator does not create a WAN ingress or TLS proxy. System and recovery response ports are not exposed through the Service.

`clusterId` is immutable. `spec.replicas` defaults to two and must be at least one.
Every replica actively observes the complete source set and maintains independent
KV state and publication incarnations. There is no leader election or shared KV
state. A single replica provides no redundancy. Use native Pod scheduling settings
to spread replicas across nodes or zones when those failure domains matter.

Keep a consumer's catalog and subscriptions on the same replica through a persistent
gRPC connection or an explicit Pod endpoint. A ClusterIP Service balances connections;
separate connections may reach different replicas. After a replica or incarnation
change, discard its previous state and bootstrap the new catalog and subscriptions.
Do not combine deltas, sequence numbers, or load observations from independent producers.
Rolling updates preserve available listeners, but do not migrate open streams or
prove KV recovery readiness on a replacement replica.

The operator owns its infrastructure, never adopts foreign resources, and never
owns, edits, or deletes inference DGDs. Image belongs to the container named `main`; imagePullSecrets belongs to
`podTemplate.spec`. Both remain explicit;
use immutable image tags or digests. Omitting resources requests 1 CPU/1Gi and
limits memory to 4Gi. Namespace restriction and exclusion filters still apply.

## Local validation

Regenerate DeepCopy with controller-gen and CRD/RBAC with `make manifests`.
With `KUBEBUILDER_ASSETS` set to local envtest assets, controller tests run a real
isolated API server and manager, including admission, defaulting, dependency
watches, Secret creation/rotation, and label-only withdrawal without rollout.
Without that variable the envtest case explicitly skips.

```bash
go test ./internal/controller/poolrelay -count=1
go test ./api/v1alpha1 -count=1
go build -o /tmp/relay-operator-manager ./cmd
```

Runtime scope and native-launcher tests live under
`lib/llm/src/kv_dc_relay/sources_watch.rs` and
`components/src/dynamo/kv_dc_relay/tests/test_native_sources_reload.py`.
The native test requires matching Python bindings. These checks do not alone
prove end-to-end KV recovery or Kubernetes volume projection; validate those
with the compatible Relay image before enabling the controller in a cluster.

## Pod template and sidecars

`spec.podTemplate` is required. Exactly one container must be named `main`,
regardless of its position. Other containers are sidecars. Set main resources,
env, probes, and securityContext on that container; native scheduling, volumes,
imagePullSecrets and initContainers belong to `podTemplate.spec`. Labels and
annotations belong to `podTemplate.metadata`. Top-level image/resources/
imagePullSecrets are removed without compatibility support.

The operator protects its identity labels, service account, scope and connection
volumes, derived transport envs, and launcher arguments. Main command and envFrom
are unsupported. Main args allow `--bind` and `--expected-unique-blocks` only;
the listener port remains 5561. Sidecars retain their own command, args and env.
Removing managed overrides removes them from the Deployment. Admission-owned
fields are preserved through server-side apply.

For Envoy mTLS, bind main to `127.0.0.1:5561`, configure Envoy to forward there,
and publish its named port through `spec.service.ports`. A `grpc` Service port
is required; port 8081 cannot be published. Declare both main startupProbe and
readinessProbe as exec probes with this command:

```yaml
exec:
  command: [python3, -m, dynamo.kv_dc_relay.healthcheck]
timeoutSeconds: 5
```

This command reads the ready field from `/engine/state` on the system HTTP port with a bounded timeout;
it does not require worker/KV recovery readiness. It requires a Relay image built
with this implementation. Sidecar readiness, when configured, also gates Pod
availability. Declare sidecar Secret/ConfigMap mounts normally. Projected data
updates through kubelet and Envoy can reload through SDS; arbitrary sidecar env
or credential changes do not automatically trigger a rollout.


Worker `envFrom` references containing only backend settings (for example HF_TOKEN)
are accepted. Secret and ConfigMap keys are checked with the declared prefix;
contents are never inherited into Relay. Transport keys must use explicit env
entries. Referenced sources are watched: adding a transport key blocks the desired
configuration, and removing it unblocks reconciliation. Missing required sources
remain unresolved; missing optional sources are allowed and still watched.

## Source modes and coordinated migration

Without `--sources-file`, Relay discovers sources using its existing filters.
With it, Relay reads and reloads only the namespaces in that JSON file. Missing
or invalid initial files fail startup; invalid updates preserve the last applied
configuration. Recreating a DGD with the same namespace and connection does not
change the source file or restart the Relay.

`GET /engine/state` works in both modes on the existing system HTTP server. It returns
`mode`, optional `podUID` and `connectionRevision`, listener `ready`, and `sources`
with `desiredRevision`, `appliedRevision`, `count`, and bounded `lastError`.
Discovery mode has no file revisions. Reload failures do not alone make a
working listener unready. Runtime health and the gRPC catalog are unchanged;
HTTP `/scope`, `/sources`, and `/catalog` are not available.

The unpublished `DynamoRelay` prototype is not migrated automatically to
`DynamoPoolRelay`. Its `dcId` field becomes `clusterId`; the runtime aliases
described above are retained.
Build and publish the matching images first; deployment is a separate operation.
Before a later rollout, save the existing CRD, ConfigMap, Deployment, and image
digests. Stop the old controller and Relay, install the new CRD status schema,
and switch to the matching new images and sources file. Preserve Service
identity, sidecars, and worker deployments. Never leave an old process consuming
the new file format. Rollback restores the saved images, schema, file, and args
together. Image publication alone does not authorize this migration.

The state callback uses the existing engine route registry; no runtime HTTP extensions are required. Update Relay and operator together when adopting this contract.
