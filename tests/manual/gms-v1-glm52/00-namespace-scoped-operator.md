<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 00 — Namespace-scoped operator for `schwinns` + cluster-wide webhook patch

Two gates must be ON for a GMS-enabled `DynamoCheckpoint` to be admitted:

| Gate | Const | Source | Live cluster-wide operator |
|---|---|---|---|
| `checkpoint` | `features.Checkpoint` | `features/gates.go:52,190` ← `config.Checkpoint.Enabled` | **OFF** — live ConfigMap `dynamo-platform-dynamo-operator-config` has no `checkpoint:` section (only the stale `last-applied-configuration` annotation does) |
| `gmsSnapshot` | `features.GMSSnapshot` | `features/gates.go:40,154,183` ← `DYN_OPERATOR_ALLOW_GMS_SNAPSHOT=1` | **ON** |

With `Checkpoint` OFF, `webhook/validation/dynamocheckpoint.go:64-70` rejects every
`DynamoCheckpoint` with `spec: Forbidden: checkpoint functionality is disabled in
the operator configuration`. The prior successful runs in `schwinns` (see
`05-prior-attempts.md`) date from ~14-24 days ago, when the gate was on; it has
since drifted off.

Plan: install **our own namespace-restricted operator** in `schwinns` with both
gates ON, **and** patch the cluster-wide webhooks to stop intercepting `schwinns`.

## Step 0 — Clean up the failed leftover release

`schwinns` has a FAILED release `dynamo-platform-gmscr` and its orphaned webhook
objects. They will fight ours (same namespace, `failurePolicy: Fail`, pointing at
a Service that no longer has endpoints → every apply times out).

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
NS=schwinns

helm list --kube-context "$CTX" -n "$NS" -a
helm uninstall dynamo-platform-gmscr --kube-context "$CTX" -n "$NS" || true

# Helm does not always reap these; delete explicitly.
kubectl --context "$CTX" delete validatingwebhookconfiguration \
  dynamo-platform-gmscr-dynamo-operator-validating-schwinns --ignore-not-found
kubectl --context "$CTX" delete mutatingwebhookconfiguration \
  dynamo-platform-gmscr-dynamo-operator-mutating-schwinns --ignore-not-found

# Nothing else should be left selecting our namespace.
kubectl --context "$CTX" get validatingwebhookconfiguration,mutatingwebhookconfiguration \
  -l nvidia.com/dynamo-operator-namespace=schwinns
```

That naming (`<fullname>-{validating,mutating}-<namespace>`) is exactly what the
chart renders in namespaced mode
(`deploy/helm/charts/platform/components/operator/templates/webhook-configuration.yaml:130-134`).

## Step 1 — Patch the cluster-wide webhooks to exclude `schwinns` (REQUIRED)

### Why this is unavoidable

| Fact | Evidence |
|---|---|
| Namespaced install scopes **its own** webhooks to the target ns | `webhook-configuration.yaml:163-167` renders `namespaceSelector.matchLabels."kubernetes.io/metadata.name": schwinns` (repeated for all 5 validating + 4 mutating entries) |
| Cluster-wide install renders **no** `namespaceSelector` | Same block: the `{{- else if .Values.namespaceRestriction.enabled }}` branch does not fire in cluster-wide mode → matches every namespace |
| You cannot fix this via Helm values | `_validation.tpl:100-102` hard-`fail`s on any non-empty `webhook.namespaceSelector` |
| `failurePolicy` defaults to `Fail` | `components/operator/values.yaml:216` |

So the API server calls **both** webhook servers for our objects, and the
cluster-wide one has `Checkpoint` OFF.

### Does the ownership Lease save us?

Partly — but **not for admission in the way we need**. The mechanism is real:

- Our operator creates/renews Lease `dynamo-operator-namespace-scope` in `schwinns`
  (`namespace_scope/lease_manager.go:39`, `cmd/main.go:335-358`).
- The cluster-wide operator watches those Leases (`cmd/main.go:383-404`,
  `lease_watcher.go:231-235`) and logs
  `Excluding namespace from cluster-wide operator processing`
  (`lease_watcher.go:217`) — the same line already visible for `mohammed-snap`.
- Its validators *are* wrapped in `LeaseAwareValidator`, which returns
  `nil, nil` for excluded namespaces before the gate check
  (`webhook/common.go:114-118,145-163`).

> [!WARNING]
> Do **not** rely on this alone. It is a race and a single point of failure:
> the Lease has a 30 s duration / 10 s renewal (`values.yaml:41-46`), and
> `lease_watcher.go:240-253` expires it aggressively. Any operator restart,
> eviction, or >30 s stall re-arms the cluster-wide validator mid-experiment and
> your next apply fails with the gate error. The Lease also does not exist until
> **after** our operator is Running, so the very first apply can race it.
>
> Patch the webhooks. The Lease then acts as a second layer of defence.

### The patch

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster

# Confirm the live object names first.
kubectl --context "$CTX" get validatingwebhookconfiguration,mutatingwebhookconfiguration \
  | grep dynamo-platform

VWC=dynamo-platform-dynamo-operator-validating
MWC=dynamo-platform-dynamo-operator-mutating

patch_wh() {  # $1 = resource kind, $2 = name
  kubectl --context "$CTX" get "$1" "$2" -o json \
  | jq '.webhooks |= map(.namespaceSelector = {
          matchExpressions: [
            { key: "kubernetes.io/metadata.name",
              operator: "NotIn",
              values: ["schwinns"] }
          ]
        })' \
  | kubectl --context "$CTX" apply -f -
}

patch_wh validatingwebhookconfiguration "$VWC"
patch_wh mutatingwebhookconfiguration   "$MWC"
```

Verify it took:

```bash
kubectl --context "$CTX" get validatingwebhookconfiguration "$VWC" \
  -o jsonpath='{.webhooks[?(@.name=="vdynamocheckpoint.kb.io")].namespaceSelector}{"\n"}'
# -> {"matchExpressions":[{"key":"kubernetes.io/metadata.name","operator":"NotIn","values":["schwinns"]}]}
```

### Racing Flux (approved)

FluxCD Kustomization `dynamo-platform` reconciles every **2 m** and reverts this.
Run a repatch loop in a spare terminal for the whole experiment:

```bash
# Re-applies every 20s; cheap and idempotent. Ctrl-C when done.
while true; do
  for pair in "validatingwebhookconfiguration $VWC" "mutatingwebhookconfiguration $MWC"; do
    set -- $pair
    current=$(kubectl --context "$CTX" get "$1" "$2" \
      -o jsonpath='{.webhooks[0].namespaceSelector.matchExpressions[0].values[0]}' 2>/dev/null)
    if [ "$current" != "schwinns" ]; then
      echo "$(date -Is) repatching $2"
      patch_wh "$1" "$2"
    fi
  done
  sleep 20
done
```

Alternative to the loop — suspend Flux for the duration (cleaner, needs perms):

```bash
flux --context "$CTX" suspend kustomization dynamo-platform
# ... run the experiment ...
flux --context "$CTX" resume kustomization dynamo-platform
```

<!-- UNVERIFIED: exact Flux Kustomization namespace (assumed flux-system) and
     whether you hold suspend permission. Check: flux get kustomizations -A -->

## Step 2 — Install the namespace-scoped operator

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
NS=schwinns
REL=schwinns-gmsv1
CHART=deploy/helm/charts/platform      # from the repo root of branch rebase-test

helm dependency build "$CHART"

helm upgrade --install "$REL" "$CHART" \
  --kube-context "$CTX" --namespace "$NS" \
  \
  `# --- namespace-restricted mode --------------------------------------` \
  --set dynamo-operator.namespaceRestriction.enabled=true \
  --set dynamo-operator.namespaceRestriction.targetNamespace="$NS" \
  --set dynamo-operator.upgradeCRD=false \
  \
  `# --- no duplicate cluster infra --------------------------------------` \
  --set global.etcd.install=false \
  --set global.nats.install=false \
  --set global.grove.install=false \
  --set 'global.kai-scheduler.install=false' \
  \
  `# --- THE GATES --------------------------------------------------------` \
  --set dynamo-operator.checkpoint.enabled=true \
  --set dynamo-operator.featureGates.gmsSnapshot=true \
  \
  `# --- snapshot artifact storage: REUSE the existing bound PVC ----------` \
  --set dynamo-operator.checkpoint.cleanupImage=busybox:1.36 \
  --set dynamo-operator.checkpoint.storage.type=pvc \
  --set dynamo-operator.checkpoint.storage.pvc.pvcName=snapshot-pvc \
  --set dynamo-operator.checkpoint.storage.pvc.basePath=/checkpoints \
  --set dynamo-operator.checkpoint.storage.pvc.create=false \
  \
  `# --- operator image ---------------------------------------------------` \
  --set dynamo-operator.controllerManager.manager.image.repository=dynamoci.azurecr.io/ai-dynamo/dynamo \
  --set dynamo-operator.controllerManager.manager.image.tag=4860c4604c7b3af0df950bc1568c887fb2a70e4a-operator \
  \
  --set 'imagePullSecrets[0].name=acr-token-secret' \
  --set dynamo-operator.gpuDiscovery.enabled=false
```

### Why each flag

| Flag | Verified reason |
|---|---|
| `namespaceRestriction.enabled=true` + `targetNamespace` | Renders namespaced RBAC, ns-scoped webhooks, `namespace.restricted` in config (`operator-config.yaml:42-56`), starts the Lease (`cmd/main.go:275-277,335`) |
| `upgradeCRD=false` | **Mandatory.** `_validation.tpl:97-99` hard-fails the combination; also drops the `crd-apply` initContainer (`deployment.yaml:67`) so we never mutate cluster-scoped CRDs |
| `global.*.install=false` | Chart defaults, pinned explicitly so no duplicate NATS/etcd/Grove/KAI lands in `schwinns` |
| `checkpoint.enabled=true` | Emits the `checkpoint:` block (`operator-config.yaml:166-202`) → `features.Checkpoint` ON |
| `featureGates.gmsSnapshot=true` | Emits `DYN_OPERATOR_ALLOW_GMS_SNAPSHOT=1` (`deployment.yaml:94-96`). Setting that env var directly via `env[]` is **hard-failed** by `deployment.yaml:90-92` |
| `checkpoint.storage.pvc.create=false` + `pvcName=snapshot-pvc` | Reuses the already-bound 2 Ti `vast` RWX PVC. `checkpoint/storage.go:EnsureStoragePVC` errors if the PVC is missing and `create` is false — it exists, so this is correct and avoids a second 1 Ti claim |
| `storage.type=pvc` | Only `pvc` is implemented, in **both** the operator (`checkpoint/storage.go:43-47`) and the Go agent (`deploy/snapshot/internal/types/config.go` `Validate()`) |
| `gpuDiscovery.enabled=false` | `gpu-discovery-preflight.yaml:24-70` template-fails without cluster RBAC read; only affects DGDR auto-profiling, unused here |

> [!WARNING]
> **Operator version skew is a real, already-observed hazard.** The chart README
> (`deploy/helm/charts/platform/README.md:136-138`) says the cluster-wide operator
> should be the **same or newer** than the namespaced one. Here it is *older*
> (`operator-07-17-26-main-1f74ef8` vs `4860c460`), and CRDs are cluster-scoped
> and owned by the cluster-wide operator.
>
> **Proof this bites:** in the saved reference,
> `spec.gpuMemoryService.extraClientContainers: ["gms-saver"]` is present in
> `last-applied-configuration` but **absent from the live object** — the installed
> CRD schema pruned it. The repo's own CRD at 1.4.0 *does* define it
> (`config/crd/bases/nvidia.com_dynamocheckpoints.yaml:90`).
>
> Consequence for us: `gms-saver` may not be auto-wired as a GMS client.
> `20-dynamocheckpoint.yaml` therefore hand-writes the full client contract
> (claim + mount + `GMS_SOCKET_DIR`) on `gms-saver` so it works **either way**.
>
> **Check before you trust the field:**
> ```bash
> kubectl --context $CTX get crd dynamocheckpoints.nvidia.com -o yaml \
>   | grep -c extraClientContainers      # 0 => pruned
> ```
> **Safer fallback:** use the cluster-wide operator's own image
> `dynamoci.azurecr.io/ai-dynamo/dynamo:operator-07-17-26-main-1f74ef8`, which is
> guaranteed schema-compatible with the installed CRDs. PR #12011 changes no
> operator Go code, so nothing is lost by doing so.

## Step 3 — Verify

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster; NS=schwinns

kubectl --context "$CTX" -n "$NS" get pod -l control-plane=controller-manager

# Gate 1: checkpoint block present in OUR config.
kubectl --context "$CTX" -n "$NS" get cm -o name | grep operator-config
kubectl --context "$CTX" -n "$NS" get cm schwinns-gmsv1-dynamo-operator-config \
  -o jsonpath='{.data.config\.yaml}' | grep -A10 '^checkpoint:'

# Gate 2: GMS env var.
kubectl --context "$CTX" -n "$NS" get deploy -l control-plane=controller-manager \
  -o jsonpath='{.items[0].spec.template.spec.containers[0].env}' | tr ',' '\n' | grep GMS

# Lease: must exist, RenewTime advancing every ~10s.
kubectl --context "$CTX" -n "$NS" get lease dynamo-operator-namespace-scope -o yaml

# Cluster-wide operator should now log the exclusion.
kubectl --context "$CTX" -n dynamo-system logs \
  deploy/dynamo-platform-dynamo-operator-controller-manager --tail=200 \
  | grep -i "Excluding namespace"
```

<!-- UNVERIFIED: rendered ConfigMap name assumes fullname == "<release>-dynamo-operator".
     Confirm with: kubectl -n schwinns get cm | grep gmsv1 -->

## Step 4 — Verify the EXISTING snapshot-agent (do NOT reinstall)

`schwinns` already runs a healthy snapshot-agent DaemonSet (41 d old, 1/1 Ready),
already pinned to s2877, image
`dynamoci.azurecr.io/ai-dynamo/dynamo:329aba5d0c91c4520c5ce8a1707310fa68fccc69-snapshot-agent`,
storage `snapshot-pvc`, `accessMode: podMount`. Installing a second one would
contend for the same node, PVC, and CRIU/runtime sockets.

```bash
kubectl --context "$CTX" -n "$NS" get ds -l app.kubernetes.io/component=snapshot-agent
kubectl --context "$CTX" -n "$NS" get pod -l app.kubernetes.io/component=snapshot-agent -o wide
kubectl --context "$CTX" -n "$NS" logs -l app.kubernetes.io/component=snapshot-agent --tail=50
```

Expect `DESIRED=CURRENT=READY=1` on `cluster-0967a26d-pool-14bee067-prctr-s2877`.

> [!NOTE]
> `accessMode: podMount` means the **workload** pod mounts `snapshot-pvc` and the
> agent reaches it via `/host/proc/<pid>/root` (`snapshot/values.yaml:39-48`).
> That is why `20-dynamocheckpoint.yaml` mounts `checkpoint-storage` at
> `/checkpoints` on `main` **and** on `gms-saver` — matching the reference.

## Step 5 — Gate smoke test (before spending GPU time)

```bash
kubectl --context "$CTX" -n "$NS" apply --dry-run=server -f - <<'EOF'
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: gate-probe
spec:
  identity: {model: probe, backendFramework: vllm}
  gpuMemoryService: {enabled: true, mode: intraPod}
  job:
    podTemplateSpec:
      spec:
        containers: [{name: main, image: "busybox:1.36"}]
EOF
```

| Response | Meaning |
|---|---|
| `spec: Forbidden: checkpoint functionality is disabled...` | The **cluster-wide** operator admitted it → webhook patch reverted or never applied. Re-patch (Step 1). |
| `spec.gpuMemoryService: Forbidden: GMS + Snapshot is temporarily disabled...` | `featureGates.gmsSnapshot` did not take on our operator. |
| `...volumes: Required value: must contain the GMS shared volume "gms-intrapod-control"` | 🎉 **Both gates ON, our operator is admitting.** This is `validateDynamoCheckpointJobConfig` (`dynamocheckpoint.go:134-158`). Proceed. |
| Hangs / `context deadline exceeded` calling a webhook | A stale webhook (Step 0) still points at a dead Service. |

## Teardown

```bash
helm uninstall schwinns-gmsv1 --kube-context "$CTX" -n "$NS"
kubectl --context "$CTX" -n "$NS" get lease dynamo-operator-namespace-scope  # should be gone
# Stop the repatch loop and let Flux restore the cluster-wide webhooks (<=2m),
# or force it:  flux --context "$CTX" reconcile kustomization dynamo-platform
```
