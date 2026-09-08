<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Beta cluster Kustomization starter

Use this starter to create the private, reusable cluster-binding layer for
portable Dynamo recipes. A cluster owner copies it once, replaces every
placeholder with values approved for that cluster, and keeps the filled copy
outside the upstream recipe contribution. A recipe developer can then point the
filled Kustomization at different portable recipes that satisfy the same cluster
contract.

This starter targets only `nvidia.com/v1beta1` `DynamoGraphDeployment` (DGD)
objects with canonical aggregate or disaggregated component names. It does not
convert alpha recipes to beta. Select and adapt a portable base from the
[representative recipe catalog](../README.md), and follow the
[recipe contribution guide](../../CONTRIBUTING.md) when shipping that base.

> [!WARNING]
> The checked-in starter is deliberately incomplete. Values such as
> `your-worker-node-pool`, `your-model-cache-claim`,
> and `your-worker-startup-failure-threshold` prevent it from accidentally
> binding to a real cluster. You must never apply an unfilled copy, and must
> not render one beyond placeholder preflight.

Recipe templates and contributed `deploy.yaml` files must contain concrete,
portable values and no unresolved placeholders. The deliberate-placeholder
exception applies only to this copy-and-fill cluster scaffold.

## Understand the three API versions

The three version strings in this directory describe different objects:

| File or object | API and kind | Meaning |
| --- | --- | --- |
| Each scaffold building block | `kustomize.config.k8s.io/v1alpha1`, `kind: Component` | Kustomize's Component configuration API. This remains `v1alpha1`. |
| The root `kustomization.yaml` | `kustomize.config.k8s.io/v1beta1`, `kind: Kustomization` | Kustomize's ordinary composition API. |
| The patched recipe resource | `nvidia.com/v1beta1`, `kind: DynamoGraphDeployment` | The Dynamo API targeted by every patch in this starter. |

Changing the Component configuration API to `v1beta1` does not make a patch
target beta. The patch target's `group`, `version`, and `kind` select the DGD.

The starter uses guarded JSON 6902 patches. It does not use strategic merge and
does not select `recipes/kustomize/components/dynamo-openapi`. The OpenAPI
Component remains part of the separate legacy strategic-merge matrix workflow.

## Install the pinned renderer

Use the standalone Kustomize v5.8.1 binary for filling, validation, rendering,
and review. Patch layering and validation are qualified against that exact
version.

```bash
kustomize version
```

Confirm that the command reports `v5.8.1` before continuing. Do not assume that
`kubectl kustomize` or `kubectl apply -k` uses the same version: `kubectl`
embeds its own Kustomize release. `kubectl apply -k` is a deployment operation,
not an automatic substitute for the qualified standalone validation renderer.

The validator requires Python 3.9 or later and PyYAML.

## Discover the cluster bindings

Choose a namespace first and use the same namespace for the DGD, PVC, and
application Secrets.

```bash
export NAMESPACE=your-namespace
kubectl get namespace "$NAMESPACE"
kubectl get nodes -o wide --show-labels
kubectl describe nodes
kubectl get storageclass
kubectl get pvc -n "$NAMESPACE"
kubectl get secret -n "$NAMESPACE"
kubectl get runtimeclass
kubectl get priorityclass
kubectl api-resources | rg 'ComputeDomain|ResourceClaim|DeviceClass'
```

Complete every applicable row before filling the patches:

| Binding | What to discover or provide |
| --- | --- |
| Namespace | An existing namespace authorized for the workload. Namespace selection stays outside the portable recipe. |
| Model cache | A pre-existing, bound, pre-populated PVC in the target namespace. The portable examples refer to it as `shared-model-cache`. Record the physical claim name and storage class. Select `cache-binding` only when the physical claim has a different name. This starter changes references only; it does not provision storage or download a model. If the private cluster configuration also provisions that external PVC, replace its deliberate `your-storage-class-name` placeholder there. |
| Registry access | A namespace-local image-pull Secret and its exact name. Confirm that it can pull every private runtime image used by the selected recipe. |
| Node policy | Approved frontend and worker label keys and values, node taints and matching tolerations, and allocatable CPU, memory, GPUs, and extended resources. |
| Scheduler policy | The approved `schedulerName`. Scheduler profiles are cluster configuration and may not be discoverable as Kubernetes API objects; obtain the name from the cluster owner. |
| Runtime and priority | Available `RuntimeClass` and `PriorityClass` names, or confirmation that the cluster does not require them. |
| Probe policy | Determine whether the operator's two-hour worker startup allowance is sufficient. Select the optional `probes` Component only for a cluster-wide startup-budget adjustment. |
| Network interfaces | Provider-approved socket, RDMA, or EFA interface names. Node labels do not expose Linux interface names; use the provider's runbook or an approved diagnostic Pod instead of guessing. |
| Network resources | Extended-resource keys and counts from node capacity, plus any required Pod annotations, device names, endpoints, or host bindings. |
| Placement | The approved topology label or clique key and the Kubernetes-version requirements of the selected affinity fields. |
| ComputeDomain and DRA | Installed controllers and drivers, available device classes, and the cluster-side realization of the recipe's logical claims. The portable recipe retains the logical ComputeDomain and claim chain. |

Provider networking and physical Dynamic Resource Allocation (DRA) details are
not portable. When a reviewed beta provider Component exists, select it for the
mechanism and keep only site values in the private cluster copy. Otherwise,
fill the local network Component according to the provider's documentation.

## Copy and fill the starter

1. Copy this directory into the private cluster configuration.
2. Replace `your-base-recipe.yaml` in the root `kustomization.yaml` with the
   path to one portable beta recipe. The resource may be a multi-document YAML
   file, but it must contain exactly one beta DGD. Review every referenced path;
   the validation and render commands below allow an explicitly referenced base
   to live outside the copied scaffold directory. Keep `resources:` to that one
   base file; provision PVCs, Secrets, namespaces, and other cluster objects in
   their own cluster workflow rather than adding them to this validated case.
3. Retain the aggregate, disaggregated, or both topology sets that this cluster
   configuration will support. Delete unselected topology and concern
   directories from the private copy.
4. Replace every placeholder in every retained YAML file with discovered
   cluster values. Delete unselected hook snippets. The copied tree must not
   retain dormant, unfilled YAML that the preflight scan would report.
5. Add only the guarded case-local patches required by that recipe.
6. Run the placeholder scan, validator, render inspection, server-side dry run,
   and apply commands in this README.

Use this preflight from the copied scaffold root:

```bash
if grep -R -n -E --include='*.yaml' \
  'your-[[:alnum:].~/_-]+' kustomization.yaml components patches
then
  echo 'ERROR: unresolved recipe placeholders remain' >&2
  exit 1
else
  placeholder_scan_status=$?
  if [ "$placeholder_scan_status" -ne 1 ]; then
    exit "$placeholder_scan_status"
  fi
fi
```

The preflight fails when it finds a placeholder, succeeds only when `grep`
returns status 1 for no matches, and propagates any execution error greater
than 1. Also inspect the diff of the private copy; the scan cannot decide
whether a syntactically real value belongs to the target cluster.

## Select Components in the required order

Each Component owns one concern:

| Component | Purpose |
| --- | --- |
| `cache-binding` | Replaces each canonical worker's `shared-model-cache` PVC claim reference with the pre-existing physical claim. It leaves the logical volume and mount name unchanged. |
| `registry-credentials` | Adds `imagePullSecrets` to the canonical Pod templates. It remains independent of cache binding because a cacheless recipe may still use private images. |
| `probes` | Optionally replaces the operator's worker startup allowance with one cluster-wide policy. It is not selected by default. |
| `scheduling` | Adds node selection, tolerations, affinity, scheduler, runtime class, and priority to the canonical components. |
| `network-interface` | The checked-in generic scaffold illustrates cluster-owned `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` environment variables and the site RDMA extended-resource request. Use a qualified provider Component instead when one owns the required mechanism. |
| `placement` | Optionally extends the worker affinity with a topology or clique constraint. It depends on `scheduling` having already created `affinity`. |

Physical networking environment variables remain forbidden in portable bases.
Qualified provider `network-interface` operations are the carrier for
`UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME`; the checked-in
generic scaffold illustrates only the names that its operations actually
carry. A site that needs `NCCL_IB_HCA` must qualify a provider operation and
extend the networking allowlist and positive and negative tests before use.
Support also requires a documented design decision. Do not remove the
forbidden-name validation.

The RDMA placeholder appears inside a JSON Pointer. Convert the discovered
Kubernetes resource key to RFC 6901 form in both the request and limit paths:
escape `~` as `~0` and `/` as `~1`. For example,
`rdma.example.com/device` becomes `rdma.example.com~1device`. Keep the decoded
key and quantity identical in `requests` and `limits`.

The root example selects the standard concerns to make the binding surface
visible; the optional `probes` override remains unselected.
Deselect an entire Component when the cluster does not require that concern;
remove its reference and delete its unfilled directory from the private copy
when no other case uses it. For example, omit `registry-credentials` for images
that need no pull Secret, omit `network-interface` when the workload needs no
site interface or extended network resource, and omit `placement` when no
topology constraint is needed.
Within `scheduling`, remove a complete `add` operation for an unused policy
field instead of inventing a value. Keep its `affinity` operation whenever
`placement` is selected. A cluster that relies entirely on default scheduling
may deselect both `scheduling` and `placement`.
Within `network-interface`, remove the complete environment-variable or
extended-resource `add` operations that the provider does not require. Keep
each extended-resource request and limit pair together with the same key and
quantity.

List Components in this exact order:

1. `cache-binding`
2. `registry-credentials`
3. `probes`, when required
4. `scheduling`
5. `network-interface`
6. `placement`, when required

The order is part of the contract. In particular, never place `placement`
before `scheduling`. The validator reports order violations as
`component-order`. Placement requires earlier same-topology scheduling; a
missing scheduling Component or missing scheduling-owned affinity parent is a
`component-dependency` failure. Keep scheduling's affinity `add` operation
whenever placement is selected.

Aggregate placement requires Kubernetes 1.33+ with
`MatchLabelKeysInPodAffinity` enabled. A cluster that disables this feature
must omit the aggregate placement Component or use a separately qualified
alternative.

### Root aggregate example

The root case for an aggregate recipe has canonical `Frontend` and `Worker`
components:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
sortOptions:
  order: fifo
resources:
  - your-base-recipe.yaml
components:
  - components/cache-binding/agg
  - components/registry-credentials/agg
  - components/scheduling/agg
  - components/network-interface/agg
  - components/placement/agg
```

Remove `components/placement/agg` when the cluster does not require the optional
placement policy.

### Disaggregated selection

For a canonical `Frontend`, `PrefillWorker`, and `DecodeWorker` recipe, replace
each `/agg` suffix with `/disagg`. Keep the same Component order:

```yaml
components:
  - components/cache-binding/disagg
  - components/registry-credentials/disagg
  - components/scheduling/disagg
  - components/network-interface/disagg
  - components/placement/disagg
```

Beta Frontends retrieve model metadata from workers and do not mount the model
cache. The cache-binding Components therefore target backend workers only.

The cache-binding guards deliberately do not depend on a worker's cache volume
or mount name. They guard the worker position and the first volume's
`persistentVolumeClaim.claimName: shared-model-cache`, then replace only the
claim name. This allows a copied recipe to rename its logical volume and mount
without weakening the physical-claim precondition.

### Cacheless and optional components

- A cacheless recipe removes the complete cache bundle from its portable base
  and deselects `cache-binding`. It may still select registry credentials.
- The beta examples run offline against a pre-populated cache. Keep the cache
  bundle, and either retain the canonical `shared-model-cache` physical claim
  or select `cache-binding` when the cluster uses a different claim name.
- Optional `planner` or `epp` components appear after the canonical components.
  The topology Components intentionally target only the canonical positions.
  Add guarded case-local image-pull, scheduling, and networking patches for an
  optional component when it needs the same cluster policy.

### Optional worker startup budget

By default, omit `probes` and rely on the probes supplied by the Dynamo
operator. Select `components/probes/<topology>` only when a cluster-wide
condition, such as consistently slow cache storage, requires a different
worker startup allowance. Place it after `registry-credentials` and before
`scheduling`.

Replace each failure-threshold placeholder with an unquoted positive integer.
The shipped probe checks `/live` on the named `system` port every 10 seconds,
so its startup allowance is `10 × failureThreshold` seconds. The aggregate
Component uses `your-worker-startup-failure-threshold`; the disaggregated
Component uses `your-prefill-startup-failure-threshold` and
`your-decode-startup-failure-threshold` for the two worker roles.

The Component adds a complete `startupProbe`. The operator does not merge an
explicit probe field by field with its default: the complete field replaces the
corresponding default on the leader/main worker container.
For multinode follower probes, backend/operator-specific behavior strips or
replaces them. A base that already owns `startupProbe` must fail the validator's
derived-absence check. Choose either a documented recipe-owned probe or this
cluster-owned override; do not select both.

## Override networking hooks

The files under `patches/` are examples for case-local overrides. They are not
reusable Components. Select only the file whose tested old value matches the
portable base, replace its one obvious placeholder value, and list it under the
root Kustomization's `patches:` field after `components:`.

The replacement remains a YAML string. Replace only the text inside the
existing single quotes; for example:

```yaml
value: &kv-transfer-config '{"key":"value"}'
```

Keeping the outer quotes prevents YAML from converting the JSON text into a
mapping before it reaches the environment variable.

All three files replace the hook on both canonical worker roles:

| File | Select when |
| --- | --- |
| `patches/vllm-kv-transfer-config.yaml` | The vLLM disaggregated base's first worker environment entry is `KV_TRANSFER_CONFIG` with the common beta default `{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda"}`. |
| `patches/vllm-compute-domain-kv-transfer-config.yaml` | The vLLM ComputeDomain base uses the minimal default `{"kv_connector":"NixlConnector","kv_role":"kv_both"}`. |
| `patches/sglang-nixl-backend.yaml` | The SGLang disaggregated base's first worker environment entry is `SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX`. |

For example:

```yaml
patches:
  - target:
      group: nvidia.com
      version: v1beta1
      kind: DynamoGraphDeployment
    path: patches/vllm-kv-transfer-config.yaml
```

The tests on component identity, container name, hook name, and old value are
intentional. Do not weaken them to make one snippet match a different base.

TensorRT-LLM transfer selection has no worker environment hook in the catalog.
Its `cache_transceiver_config` remains part of the paired prefill and decode
engine ConfigMaps. Change and qualify that coordinated runtime bundle in the
portable recipe; do not use a vLLM or SGLang hook patch.

## Validate, render, and apply

Run the repository-level validator from the Dynamo repository root against the
copied scaffold. The first argument is the pre-render portable base; the second
is the copied scaffold's root Kustomization:

```bash
python3 scripts/validate-recipe-kustomization.py \
  <cluster-kustomization>/your-base-recipe.yaml \
  <cluster-kustomization>/kustomization.yaml
```

The validator uses `kustomize` from `PATH` by default. To select an explicitly
qualified binary instead, use the override:

```bash
python3 scripts/validate-recipe-kustomization.py \
  <cluster-kustomization>/your-base-recipe.yaml \
  <cluster-kustomization>/kustomization.yaml \
  --kustomize-bin /explicit/path/to/kustomize
```

The validator checks that the build targets exactly one beta DGD, verifies the
canonical component positions, rejects base-owned cluster fields and duplicate
environment names, replays the selected Component and case patch operations in
order, and compares that replay with the Kustomize render. Its supported patch
target is deliberately limited to exact `group`, `version`, and `kind` fields;
name, namespace, label, annotation, and other selectors are rejected instead
of approximating Kustomize's selector semantics.

Inspect and server-side dry-run the pinned render before applying it:

```bash
kustomize build --load-restrictor LoadRestrictionsNone .
kustomize build --load-restrictor LoadRestrictionsNone . | \
  kubectl apply --dry-run=server -f - -n "$NAMESPACE"
kustomize build --load-restrictor LoadRestrictionsNone . | \
  kubectl apply -f - -n "$NAMESPACE"
```

This pipeline guarantees that the manifest sent to `kubectl` came from the
standalone v5.8.1 renderer. `LoadRestrictionsNone` is required when the selected
base or a shared Component is outside the copied directory, so use only paths
reviewed as part of the private cluster configuration. `kubectl apply -k .` is
a deployment operation and not an automatic substitute for the qualified
validation renderer. Validate with standalone Kustomize v5.8.1, then apply the
validated standalone render through the pipeline above.

Rendering and admission do not prove scheduling, readiness, correct responses,
or performance. Exercise the deployment on the intended hardware and network
before treating the portable recipe as qualified.

## Recognize guard and duplicate failures

### Renamed component

Renaming canonical `Worker` to another value causes a JSON Patch `test` to fail
during `kustomize build`. The error identifies the guarded JSON Pointer instead
of appending a new component.

```text
Error: accumulating components: accumulateDirectory: "recursed accumulation of path '/private/tmp/recipe-kustomization/components/cache-binding/agg': testing value /spec/components/1/name failed: test failed"
```

Do not remove the test. Restore the canonical name in a new or refreshed base,
or create a deliberately scoped case patch for a legacy, nonconforming base and
review it as a separate compatibility path.

### Duplicate environment append

Kustomize can successfully append a second environment entry with the same
name. In the v5.8.1 verification fixture, a base value of `ens2f0` followed by
the Component value `eth0` rendered this contiguous excerpt:

```yaml
          - name: NCCL_SOCKET_IFNAME
            value: ens2f0
          - name: NCCL_SOCKET_IFNAME
            value: eth0
          - name: GLOO_SOCKET_IFNAME
            value: eth0
```

Kubernetes does not make this a safe override contract.
`scripts/validate-recipe-kustomization.py` rejects the duplicate at the layer
that appends it. Override a Component-added value with a guarded `test` plus
`replace` in the case patch instead of appending the same name again.

## Keep the upstream boundary clean

Commit portable recipe bases, this generic starter, and genuinely reusable
site-neutral mechanisms upstream. Do not commit a filled cluster copy,
rendered output containing site values, namespace identities, credentials,
physical claim names, node-pool labels, interface names, endpoints, or device
assignments. The private Kustomization is qualification input for a recipe, not
part of the portable recipe contribution.
