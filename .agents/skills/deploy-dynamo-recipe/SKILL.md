---
name: deploy-dynamo-recipe
description: Deploys one assigned DynamoGraphDeployment and proves it with an OpenAI-compatible smoke test. Use when user-interviewer has captured the user-provided baseline DGD or hypothesis-challenger has approved a later DGD.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - kubernetes
    - recipes
    - deployment
---

# Deploy Dynamo Recipe

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Deploy exactly one assigned Dynamo Kubernetes DGD and return a small smoke-test artifact. This skill does not search
the recipe catalog, choose or substitute a DGD, tune knobs, benchmark performance, or create new recipes.

Input ownership:

- First Optimization Iteration: `user-interviewer` provides the canonical user-provided DGD path and SHA256.
- Subsequent Optimization Iterations: `hypothesis-challenger` provides the candidate `deploy.yaml` or DGD.
- The synthesized `user_workload.yaml` supplies the Kubernetes context, namespace, optional storage class, and
  baseline DGD path-and-hash record.

## Inputs

Required:

- assigned DGD manifest path and SHA256
- handoff provenance: `user-interviewer` for iteration 0 or `hypothesis-challenger` for iteration > 0
- exact `<EXP_ROOT>/user_workload.yaml` path and SHA256
- target namespace and `kubectl` context from `user_workload.yaml`
- experiment root created by `user-interviewer`
- zero-based optimization iteration
- previous deployment root for iteration > 0

Optional:

- storage class, only when model-cache PVCs need one
- smoke prompt; default to `Simply output the phrase: NVIDIA Dynamo`

Secrets:

- Never ask the user to paste token values into the agent conversation.
- Treat Kubernetes secrets referenced by the selected manifests as pre-existing cluster prerequisites.
- Check referenced secrets only by name. If one is missing, record a blocker; do not ask for its value or create it.
- Current recipes commonly expect `hf-token-secret` with key `HF_TOKEN` for gated Hugging Face model access.

## Workflow

Recompute the supplied `user_workload.yaml` SHA256 before using its Kubernetes and workload context. Recompute the
assigned DGD SHA256 and require it to match the handoff before creating run-scoped copies. At iteration 0, also
require the assigned path and SHA256 to equal `deployment.dgd_path` and `deployment.dgd_sha256` in
`user_workload.yaml`.

### 1. Create The Deployment Directory

Create exactly one directory for the assigned candidate:

```text
<EXP_ROOT>/artifacts/deploy-iter-<NNN>/
```

Create `applied_manifests/` beneath it. Copy the assigned DGD and every explicitly handed-off support manifest used by
the deployment into that directory with stable names such as `deploy.yaml`, `model-cache.yaml`,
`model-download.yaml`, and `model-validate.yaml`. Which support manifests exist is recipe-dependent: some recipes ship
precision-suffixed download manifests (`recipes/deepseek-v4/*` ship `model-download-fp8.yaml` and
`model-download-nvfp4.yaml`), some fold the download into the model-cache manifest, and few ship a validation job at
all. Copy what the handoff actually contains. Never modify the handed-off source files.

Update these run-scoped copies in place when a compatibility fix is required, then reapply them. Record every change
and reason in `deployment_ledger.json`; do not retain numbered intermediate copies. After a successful smoke test,
`applied_manifests/` must contain exactly one final file per manifest type used, and those files must be the exact set
that produced the successful deployment. If the deployment is blocked, retain only the latest attempted copies and mark
the ledger blocked. Create `logs/` only when a targeted failure log must be retained.

### 2. Validate The Assigned DGD

Run read-only checks first:

```bash
set -euo pipefail
kubectl --context "${KUBE_CONTEXT}" get namespace "${NAMESPACE}"
# CRD presence gate: a Forbidden here is tolerated because the server dry-run below
# re-checks it authoritatively; a confirmed absence stops before any mutation.
crds="$(kubectl --context "${KUBE_CONTEXT}" get crd 2>&1 || true)"
case "${crds}" in
  *Forbidden*) echo "WARN: cluster-scope CRD list forbidden for this identity; deferring to server dry-run" ;;
  *dynamographdeployment*) : ;;
  *) echo "Dynamo CRDs missing"; exit 1 ;;
esac
# Advisory reads: storage classes and node inventory inform sizing but a namespace-scoped
# identity may lack cluster-scope list rights. Record a Forbidden as a run limitation; do not fail.
kubectl --context "${KUBE_CONTEXT}" get storageclass || echo "WARN: storageclass list forbidden; record as limitation"
kubectl --context "${KUBE_CONTEXT}" get nodes -o wide || echo "WARN: node list forbidden; record as limitation"
```

Every kubectl call in this skill pins `--context "${KUBE_CONTEXT}"` (the contract's `kube_context`); never rely on
the ambient current-context.

Validate the selected path without mutating the cluster:

```bash
kubectl --context "${KUBE_CONTEXT}" apply --dry-run=server -n "${NAMESPACE}" \
  -f <assigned-dgd-yaml>
```

Review the assigned DGD and any support manifests explicitly included in the handoff. Check:

- DGD name and frontend service name
- model-cache PVCs and storage class needs
- model download or validation jobs
- secrets referenced by `secretKeyRef`, `envFromSecret`, or `imagePullSecrets`
- GPU requests, node selectors, tolerations, and GPU SKU expectations

Stop before mutation if required namespace, CRDs, PVC prerequisites, secret names, storage class, images, or GPU
capacity are missing. Also verify before mutation that the assigned manifest changes no knob listed in the
contract's `resources.pinned` and that total concurrent GPU holdings stay within `resources.gpu_ceiling`.

When checking GPU capacity, count every pod that is bound to a node (`spec.nodeName` set) and not in a terminal phase
(`Succeeded`/`Failed`) as holding its full GPU request. Do not filter on `phase == Running`: pods in
`ImagePullBackOff`, `ContainerCreating`, or init hold their reservations. Exclude nodes whose taints the
assigned manifest does not already tolerate, and never add new tolerations for other tenants' reservation taints.
Evaluate fit by expanding the DGD into its full multiset of pod demands (every component, every replica) and placing
them against per-node free blocks while decrementing remaining capacity — two pods cannot count the same free GPUs.
Honor each pod's node selectors, required affinity/anti-affinity, and tolerations during placement. If the DGD cannot
be faithfully expanded into pod demands, report capacity as unknown, not sufficient. When `resources.gpu_ceiling` is
set in the workload contract, also verify the run's total concurrent GPU holdings stay within it.

If a manifest must change only to work with the target cluster, such as resolving a storage class placeholder or adding
a required node-taint toleration, update only the copy under `applied_manifests/`. Preserve the handed-off source
and record the exact change and reason in `deployment_ledger.json`. Do not change performance knobs.

### 3. Retire The Previous Iteration

For iteration > 0, read the previous deployment ledger and delete only its DGD by exact name, namespace, and context.
Wait for the DGD and its operator-owned workloads to terminate before applying the new candidate. Record the deletion
in the new deployment ledger.

```bash
set -euo pipefail
kubectl --context "${PREVIOUS_KUBE_CONTEXT}" delete dynamographdeployment "${PREVIOUS_DGD}" \
  -n "${PREVIOUS_NAMESPACE}" --wait=true --timeout=10m
kubectl --context "${PREVIOUS_KUBE_CONTEXT}" wait --for=delete pod \
  -l nvidia.com/dynamo-graph-deployment-name="${PREVIOUS_DGD}" \
  -n "${PREVIOUS_NAMESPACE}" --timeout=10m
```

Do not delete or modify the previous deployment directory or its successful YAML. Create new run-scoped copies in the
new iteration directory. Preserve shared PVCs, model-cache jobs, namespaces, and secrets.

### 4. Apply Support Manifests

Follow user-provided deployment instructions when they give a specific sequence. Otherwise:

If the effective cluster context differs from what `<EXP_ROOT>/manifest.yaml` records, update the manifest's
cluster-context entry before mutating anything.

Read each support manifest's `kind` and `metadata.name`; never infer a Kubernetes resource name from its filename. Set
`DOWNLOAD_MANIFEST` and `DOWNLOAD_JOB` from the handed-off download manifest, and `VALIDATE_MANIFEST` and
`VALIDATE_JOB` from the validation manifest. Both are optional: skip the corresponding block when the recipe ships no
such manifest, and never assume the literal filename `model-download.yaml`.

```bash
set -euo pipefail
kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/model-cache.yaml" -n "${NAMESPACE}"
kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/${DOWNLOAD_MANIFEST}" -n "${NAMESPACE}"
job_state=""
for _ in $(seq 1 200); do  # 200 x 30s = 100 min bound
  # NOTE: match by substring - a successful Job on Kubernetes 1.31+ carries BOTH
  # SuccessCriteriaMet and Complete conditions, so the jsonpath returns them space-separated.
  job_state="$(kubectl --context "${KUBE_CONTEXT}" get "job/${DOWNLOAD_JOB}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.conditions[?(@.status=="True")].type}')"
  case "${job_state}" in *Failed*) echo "download job failed"; exit 1;; *Complete*) break;; esac
  sleep 30
done
case "${job_state}" in *Complete*) : ;; *) echo "download job timed out"; exit 1;; esac
```

If a validation job exists, run it after download and before the DGD:

```bash
set -euo pipefail
kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/${VALIDATE_MANIFEST}" -n "${NAMESPACE}"
job_state=""
for _ in $(seq 1 120); do  # 120 x 30s = 60 min bound
  job_state="$(kubectl --context "${KUBE_CONTEXT}" get "job/${VALIDATE_JOB}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.conditions[?(@.status=="True")].type}')"
  case "${job_state}" in *Failed*) echo "validate job failed"; exit 1;; *Complete*) break;; esac
  sleep 30
done
case "${job_state}" in *Complete*) : ;; *) echo "validate job timed out"; exit 1;; esac
```

### 5. Apply The Assigned DGD

Apply only the run-scoped copy of the assigned manifest:

```bash
set -euo pipefail
kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/deploy.yaml" -n "${NAMESPACE}"
kubectl --context "${KUBE_CONTEXT}" get dynamographdeployment -n "${NAMESPACE}"
kubectl --context "${KUBE_CONTEXT}" get pods -n "${NAMESPACE}" -o wide
kubectl --context "${KUBE_CONTEXT}" get svc -n "${NAMESPACE}"
```

Do not apply sibling variants. Do not change engine arguments, GPU counts, replica topology, routing, or other
performance settings unless the parent supplied them as part of the assigned candidate. Kubernetes-only compatibility
patches are allowed when required and recorded.

### 6. Wait For Readiness

Healthy signals:

- model-cache PVC is `Bound`
- required model download/validation jobs are `Complete`
- DGD exists without unresolved reconciliation errors
- every component and replica declared by the selected DGD is `Running` and ready
- frontend service exists

On failure, inspect the DGD status, events, and logs for the affected component before making a minimal run-scoped
compatibility patch. Record the readiness state, diagnosis, relevant error excerpt, and patch in `deployment_ledger.json`.
Do not generate broad Kubernetes snapshots, endpoint-response copies, successful pod logs, or other evidence files.
Persist additional logs under `logs/` only when failure output is needed beyond the ledger excerpt. If no
diagnosis-backed patch remains, stop or hand off to troubleshooting; do not loop blindly.

### 7. Smoke Test

Identify the OpenAI endpoint first: standard recipes expose a frontend Service; gateway-integrated (gaie)
variants have NO frontend Service — the frontend runs as a sidecar in each worker pod, so port-forward a worker
pod's port 8000 instead. Direct-routing sidecars additionally require the worker instance id the gateway
would inject: pass `-H "x-dynamo-worker-instance-id: <decimal instance id>"` on completion requests (the id
appears in the sidecar's registration logs; convert from hex). A 400 naming "Direct routing mode" means this
header is missing, not that the deployment is broken.

Run the port-forward and the smoke test in ONE shell session (the trap, `PF_PID`, and the captured bodies do not
survive across separate command invocations). Capture HTTP status and response body separately; do not treat JSON
parsing alone as success:

```bash
set -euo pipefail
SERVED_MODEL="<served-model-name>"
SMOKE_DIR="${DEPLOY_ROOT}/smoke"
mkdir -p "${SMOKE_DIR}"

kubectl --context "${KUBE_CONTEXT}" port-forward svc/<frontend-service> 8000:8000 -n "${NAMESPACE}" &
PF_PID=$!
trap 'kill "${PF_PID}" 2>/dev/null' EXIT

ready=0
for _ in $(seq 1 30); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models || true)"
  [ "${code}" -ge 100 ] && { ready=1; break; }  # any HTTP response = port forwards; smoke gates judge health
  sleep 2
done
[ "${ready}" = "1" ] || { echo "port-forward never became reachable"; exit 1; }

# Worker registration lags pod readiness (the frontend lists a model only after the
# worker's generate endpoint registers with discovery); wait bounded, don't fail on the first poll.
listed=0
for _ in $(seq 1 30); do  # 5 min bound
  models_code="$(curl -sS -o "${SMOKE_DIR}/models_body.json" -w '%{http_code}' http://127.0.0.1:8000/v1/models || true)"
  if [ "${models_code}" -ge 200 ] && [ "${models_code}" -lt 300 ] && \
     jq -e --arg model "${SERVED_MODEL}" 'any(.data[]?; .id == $model)' "${SMOKE_DIR}/models_body.json" >/dev/null; then
    listed=1; break
  fi
  sleep 10
done
[ "${listed}" = "1" ] || { echo "served model never listed (last code ${models_code})"; exit 1; }

api_request="$(jq -nc --arg model "${SERVED_MODEL}" '{
  model: $model,
  messages: [{role: "user", content: "Simply output the phrase: NVIDIA Dynamo"}],
  max_tokens: 100,
  temperature: 0
}')"
api_code="$(curl -sS -o "${SMOKE_DIR}/api_body.json" -w '%{http_code}' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "${api_request}")"
[ "${api_code}" -ge 200 ] && [ "${api_code}" -lt 300 ] || { echo "chat endpoint ${api_code}"; exit 1; }
jq -e '.object == "chat.completion" and (.choices | type == "array" and length > 0) and (.error | not)' \
  "${SMOKE_DIR}/api_body.json" >/dev/null || { echo "chat response failed structural check"; exit 1; }

echo "smoke_success=1 models_code=${models_code} api_code=${api_code}"
```

The script exits non-zero on ANY failed gate, so `success: 1` in `smoke_test_artifact.json` may be written only
when it printed `smoke_success=1`. The response bodies live under `${DEPLOY_ROOT}/smoke/`, never a shared /tmp
path, so a stale body from a previous run can never satisfy the checks.

Set `success` to `1` only when both captured HTTP codes are 2xx AND both structural checks pass; record both codes in `smoke_test_artifact.json`.

After a successful smoke test, record durable config-engagement evidence in `deployment_ledger.json` per
`agent-docs/rules/verification/config-engagement.md`: the Kubernetes pod-spec fields or startup-log lines proving
the candidate's changed knob is live (applied YAML plus a passing smoke request are not sufficient by themselves). Preserve the full chat
response before validation and write it unchanged to `api_response`; on failure, preserve the full API error body.

## Required Output

Write `${DEPLOY_ROOT}/smoke_test_artifact.json`:

```json
{
  "api_request": {},
  "api_response": {},
  "success": 0
}
```

- `api_request`: full OpenAI-compatible request body sent to the endpoint.
- `api_response`: full parsed response body, or error body if the smoke test fails.
- `success`: `1` when the smoke test passes, otherwise `0`.

Also write `deployment_ledger.json`, including the DGD name, Kubernetes context and namespace, assigned source DGD
path and SHA256, final applied-manifest paths, compatibility patches and their reasons, readiness state, concise
diagnostics, blockers, and cleanup commands.

## Out Of Scope

- catalog search, DGD selection, or DGD substitution
- benchmark execution or AIPerf result parsing
- optimization hypotheses or challenger reviews
- authoring new recipes from scratch
- cluster setup
- reading or storing secret values

## References

- `../../../agent-docs/guides/deployment/kubernetes-recipe-workflow.md`
- `../../../agent-docs/rules/execution/run-artifacts.md`
