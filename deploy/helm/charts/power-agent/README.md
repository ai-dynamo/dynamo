<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Power Agent Helm Chart

Privileged per-node DaemonSet that enforces per-GPU NVML power caps on
Dynamo worker pods. Watches pods for the
`dynamo.nvidia.com/gpu-power-limit` annotation, parses
`/proc/{pid}/cgroup` to map GPU processes to pod UIDs, and calls
`nvmlDeviceSetPowerManagementLimit()`.

Supersedes the raw `deploy/power_agent/{daemonset,rbac,dev-pod}.yaml`
manifests removed in the same commit. Design rationale lives in
[`docs/design-docs/power-agent-helm-chart-plan.md`](../../../../docs/design-docs/power-agent-helm-chart-plan.md).

## Prerequisites

- Kubernetes cluster with GPU nodes labeled `nvidia.com/gpu.present: "true"`.
- NVIDIA Container Toolkit installed on each GPU node with the `nvidia`
  runtime class registered.
- A privileged DaemonSet with `hostPID: true` is acceptable on the
  target cluster.

The chart uses the canonical NVIDIA monitoring-agent pattern:
`privileged: true` + `runtimeClassName: nvidia` +
`NVIDIA_VISIBLE_DEVICES=all`. This injects `libnvidia-ml.so` and exposes
every GPU on the node **without consuming an `nvidia.com/gpu` resource
claim** (the same pattern DCGM Exporter uses). Because the container is
privileged, the
`accept-nvidia-visible-devices-envvar-when-unprivileged=false` runtime
config does not apply.

## Production install

The chart requires an explicit image tag — `:latest` is rejected at
template time. Pin a release tag or `sha256:digest`:

```bash
helm install power-agent ./deploy/helm/charts/power-agent \
  --namespace dynamo-system \
  --create-namespace \
  --set image.tag=v1.0.0 \
  --set agent.safeDefaultWatts=500
```

Per-SKU `safeDefaultWatts` (≈70% of TDP):

| SKU         | Recommended `safeDefaultWatts` |
|-------------|-------------------------------:|
| H200 SXM    | `500` |
| H100 SXM    | `490` |
| A100 SXM 80GB | `280` |
| B200 SXM    | (consult SKU TDP) |

Verify:

```bash
kubectl rollout status daemonset/power-agent -n dynamo-system
kubectl logs -n dynamo-system -l app.kubernetes.io/name=power-agent --tail=50
kubectl get pods -n dynamo-system -l app.kubernetes.io/name=power-agent -o wide
```

## Dev install (iterating on `power_agent.py`)

Dev mode renders a single Pod pinned to one GPU node (instead of a
DaemonSet) that mounts `power_agent.py` from an externally-created
ConfigMap. This shortens the edit-deploy-test loop: edit the script
locally, update the ConfigMap, restart the Pod.

> **Step 1 (REQUIRED before `helm install`).** Create the script ConfigMap.
>
> ```bash
> kubectl create configmap dynamo-power-agent-script \
>   --from-file=power_agent.py=components/power_agent/power_agent.py \
>   -n $NAMESPACE
> ```
>
> Without this ConfigMap the dev Pod stays `Pending` with a
> `MountVolume.SetUp failed for volume "script"` event. The chart does
> not auto-create the ConfigMap because the iteration loop is
> `kubectl create cm --from-file=... --dry-run=client -o yaml | kubectl apply -f -`
> (faster than `helm upgrade --set-file`).

Step 2: install in dev mode.

```bash
helm install power-agent ./deploy/helm/charts/power-agent \
  --namespace $NAMESPACE \
  --set image.tag=v1.0.0 \
  --set daemonset.enabled=false \
  --set dev.enabled=true \
  --set dev.nodeName=<gpu-node-name>
```

`daemonset.enabled` and `dev.enabled` are mutually exclusive — the
chart fails to render at `helm template` / `helm install` time if both
are true.

**RBAC scope in dev mode.** Dev mode automatically uses namespace-scoped
RBAC (Role + RoleBinding in the release namespace) instead of cluster-wide
RBAC, because the Pod is pinned to one node and runs the agent with
`--namespace=$POD_NAMESPACE`. This is principle-of-least-privilege by
default. If you genuinely need to test cross-namespace multi-pod conflict
resolution while iterating, override with:

```bash
--set dev.namespaceRestrictedOverride=true
```

Update flow (after editing `power_agent.py`):

```bash
kubectl create configmap dynamo-power-agent-script \
  --from-file=power_agent.py=components/power_agent/power_agent.py \
  -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

kubectl delete pod power-agent-dev -n $NAMESPACE
# Helm re-creates the Pod with the fresh ConfigMap content on next reconcile.
```

## Important values

| Key | Meaning | Default |
|-----|---------|---------|
| `image.repository` | Power Agent image registry path | `nvcr.io/nvidia/dynamo/power-agent` |
| `image.tag` | **Required** — pin to a release tag or `sha256:digest` (no default) | `""` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `imagePullSecrets` | Image pull secrets list | `[]` |
| `agent.safeDefaultWatts` | Per-SKU fail-closed cap when annotation parsing fails or multi-pod conflicts can't agree (≈70% of TDP) | `500` |
| `agent.prometheusPort` | Port exposing `/metrics` | `9100` |
| `runtimeClassName` | NVIDIA runtime class (injects `libnvidia-ml.so`) | `nvidia` |
| `env` | Container env (`NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`) | `{NVIDIA_VISIBLE_DEVICES: "all", NVIDIA_DRIVER_CAPABILITIES: "compute,utility"}` |
| `nodeSelector` | Target only GPU nodes | `{nvidia.com/gpu.present: "true"}` |
| `tolerations` | Tolerate the standard GPU taint | `[{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` |
| `resources` | Per-container CPU/mem requests + limits | small — see `values.yaml` |
| `state.hostPath` | Directory for `managed_gpus.json` UUID-gated cold-start state | `/var/lib/dynamo-power-agent` |
| `serviceAccount.create` | Create a ServiceAccount | `true` |
| `serviceAccount.name` | Override SA name (default: chart fullname) | `""` |
| `rbac.create` | Create RBAC resources | `true` |
| `rbac.namespaceRestricted` | `true` → Role+RoleBinding; `false` → ClusterRole+ClusterRoleBinding. Dev mode forces `true` regardless. | `false` |
| `daemonset.enabled` | Render production DaemonSet | `true` |
| `daemonset.updateStrategy.rollingUpdate.maxUnavailable` | Conservative `1` by default — see "Rollout strategy on large fleets" below | `1` |
| `daemonset.podLabels` / `podAnnotations` / `affinity` | Standard DS overrides | `{}` |
| `dev.enabled` | Render dev Pod instead of DaemonSet (mutually exclusive) | `false` |
| `dev.nodeName` | Required when `dev.enabled=true`; GPU node to pin to | `""` |
| `dev.scriptConfigMap` | Externally created ConfigMap holding `power_agent.py` | `dynamo-power-agent-script` |
| `dev.image.repository` / `dev.image.tag` | Dev container image (uses `vllm-runtime` which ships pynvml) | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` |
| `dev.namespaceRestrictedOverride` | Set `true` to allow cluster-wide RBAC in dev mode (rare; for cross-namespace multi-pod testing only) | `false` |

See [`values.yaml`](./values.yaml) for the full configuration surface and
per-key rationale comments.

## Rollout strategy on large fleets

The default `daemonset.updateStrategy.rollingUpdate.maxUnavailable: 1`
is deliberately conservative: power-cap enforcement is safety-critical,
so only one node's worth of GPUs ever loses enforcement during a
rollout. On large fleets (hundreds of GPU nodes) where a 1-at-a-time
rollout would take too long, override:

```bash
--set daemonset.updateStrategy.rollingUpdate.maxUnavailable=10%
```

The trade-off is more concurrent enforcement gaps during the rollout
window; safe when the planner's reconcile interval (15 s) is much
shorter than the rollout duration.

## Uninstall

```bash
helm uninstall power-agent -n $NAMESPACE
```

The chart does **not** clean up `/var/lib/dynamo-power-agent/managed_gpus.json`
on host nodes. That hostPath state persists until either:

- Manual cleanup: `rm -f /var/lib/dynamo-power-agent/managed_gpus.json`
  on each affected node, **or**
- A subsequent agent install reads it during cold-start orphan recovery
  (UUID-gated; safe to leave in place across installs).

If you want a clean slate before reinstalling on a different image tag
or with different `safeDefaultWatts`, clear the file first. Otherwise
the persistent state is harmless and helps the agent restore default
TGP on GPUs it previously managed but no longer sees.

## Troubleshooting

**`MountVolume.SetUp failed for volume "script"`** in dev mode: the
script ConfigMap doesn't exist. Run the `kubectl create configmap`
recipe from the Dev install section above.

**`Error: image.tag is required (pin to a release tag or sha256:digest; :latest is not supported)`** at install time: pass `--set image.tag=<pinned>`. The chart deliberately rejects unset / `:latest` tags to keep deployments reproducible.

**`Error: daemonset.enabled and dev.enabled are mutually exclusive`** at install time: pick one mode. The chart cannot render both a DaemonSet and a single-Pod dev harness from one release.

**No NVML caps being applied at runtime**: verify the workload pods
have the `dynamo.nvidia.com/gpu-power-limit` annotation. The planner
emits it when `enable_power_awareness=true` in its config; see
[`examples/deployments/powerplanner/disagg-power-aware.yaml`](../../../../examples/deployments/powerplanner/disagg-power-aware.yaml).
