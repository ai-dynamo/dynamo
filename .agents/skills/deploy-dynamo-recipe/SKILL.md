---
name: deploy-dynamo-recipe
description: Deploy one already-selected Dynamo Kubernetes recipe or DGD manifest and prove it with an OpenAI-compatible smoke test. Use when recipe-explorer or hypothesis-challenger has already selected the deploy.yaml/DGD.
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

## Purpose

Deploy exactly one selected Dynamo Kubernetes manifest and return a small smoke-test artifact. This skill does not
search the recipe catalog, choose between variants, tune knobs, benchmark performance, or create new recipes.

Input ownership:

- First Optimization Iteration: `recipe-explorer` provides the selected `deploy.yaml` or DGD.
- Subsequent Optimization Iterations: `hypothesis-challenger` provides the candidate `deploy.yaml` or DGD.
- The user-provided `target_workload.yaml` supplies the Kubernetes context, namespace, and optional storage class.

## Inputs

Required:

- selected `deploy.yaml`, DGD manifest, or recipe variant directory
- user-provided `target_workload.yaml`
- target namespace and `kubectl` context from `target_workload.yaml`
- experiment root created by `recipe-explorer`
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

### 1. Create The Deployment Directory

Create exactly one directory for the assigned candidate:

```text
<EXP_ROOT>/artifacts/deploy-iter-<NNN>/
```

Create `applied_manifests/` beneath it. Copy the assigned DGD and every support manifest used by the deployment into
that directory with stable names such as `deploy.yaml`, `model-cache.yaml`, `model-download.yaml`, and
`model-validate.yaml`. Never modify the tracked recipe source.

Update these run-scoped copies in place when a compatibility fix is required, then reapply them. Record every change
and reason in `deployment_ledger.json`; do not retain numbered intermediate copies. After a successful smoke test,
`applied_manifests/` must contain exactly one final file per manifest type used, and those files must be the exact set
that produced the successful deployment. If the deployment is blocked, retain only the latest attempted copies and mark
the ledger blocked. Create `logs/` only when a targeted failure log must be retained.

### 2. Validate The Selected Manifest

Run read-only checks first:

```bash
kubectl config current-context
kubectl get namespace "${NAMESPACE}"
kubectl get crd | grep -i dynamo
kubectl get storageclass
kubectl get nodes -o wide
```

Validate the selected path without mutating the cluster:

```bash
kubectl --context "${KUBE_CONTEXT}" apply --dry-run=client -n "${NAMESPACE}" \
  -f <deploy-yaml-or-recipe-dir>
```

Review the selected manifest and any sibling support manifests it requires. Check:

- DGD name and frontend service name
- model-cache PVCs and storage class needs
- model download or validation jobs
- secrets referenced by `secretKeyRef`, `envFromSecret`, or `imagePullSecrets`
- GPU requests, node selectors, tolerations, and GPU SKU expectations

Stop before mutation if required namespace, CRDs, PVC prerequisites, secret names, storage class, images, or GPU
capacity are missing.

If a manifest must change only to work with the target cluster, such as resolving a storage class placeholder or adding
a required node-taint toleration, update only the copy under `applied_manifests/`. Preserve the tracked recipe source
and record the exact change and reason in `deployment_ledger.json`. Do not change performance knobs.

### 3. Retire The Previous Iteration

For iteration > 0, read the previous deployment ledger and delete only its DGD by exact name, namespace, and context.
Wait for the DGD and its operator-owned workloads to terminate before applying the new candidate. Record the deletion
in the new deployment ledger.

```bash
kubectl --context "${PREVIOUS_KUBE_CONTEXT}" delete dynamographdeployment "${PREVIOUS_DGD}" \
  -n "${PREVIOUS_NAMESPACE}" --wait=true --timeout=10m
```

Do not delete or modify the previous deployment directory or its successful YAML. Create new run-scoped copies in the
new iteration directory. Preserve shared PVCs, model-cache jobs, namespaces, and secrets.

### 4. Apply Support Manifests

Follow the selected recipe README when it gives a specific sequence. Otherwise:

Read each support manifest's `kind` and `metadata.name`; never infer a Kubernetes resource name from its filename. Set
`DOWNLOAD_JOB` and `VALIDATE_JOB` from the corresponding Job manifests.

```bash
kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/model-cache.yaml" -n "${NAMESPACE}"
kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/model-download.yaml" -n "${NAMESPACE}"
kubectl wait --for=condition=Complete "job/${DOWNLOAD_JOB}" -n "${NAMESPACE}" --timeout=6000s
```

If a validation job exists, run it after download and before the DGD:

```bash
kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/model-validate.yaml" -n "${NAMESPACE}"
kubectl wait --for=condition=Complete "job/${VALIDATE_JOB}" -n "${NAMESPACE}" --timeout=3600s
```

### 5. Apply The Selected DGD

Apply only the run-scoped copy of the assigned manifest:

```bash
kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/deploy.yaml" -n "${NAMESPACE}"
kubectl get dynamographdeployment -n "${NAMESPACE}"
kubectl get pods -n "${NAMESPACE}" -o wide
kubectl get svc -n "${NAMESPACE}"
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
Persist one targeted file under `logs/` only when failure output is needed beyond the ledger excerpt. If no
diagnosis-backed patch remains, stop or hand off to troubleshooting; do not loop blindly.

### 7. Smoke Test

Port-forward the frontend service:

```bash
kubectl port-forward svc/<frontend-service> 8000:8000 -n "${NAMESPACE}"
```

Capture HTTP status and response body separately. Do not treat JSON parsing alone as success. Check `/v1/models`, then
send one chat completion:

```bash
SERVED_MODEL="<served-model-name>"

models_response="$(curl -sS --fail-with-body http://127.0.0.1:8000/v1/models)"
jq -e --arg model "${SERVED_MODEL}" 'any(.data[]?; .id == $model)' <<<"${models_response}"

api_request="$(jq -nc --arg model "${SERVED_MODEL}" '{
  model: $model,
  messages: [{role: "user", content: "Simply output the phrase: NVIDIA Dynamo"}],
  max_tokens: 100,
  temperature: 0
}')"
api_response="$(curl -sS --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "${api_request}")"
jq -e '.object == "chat.completion" and (.choices | type == "array" and length > 0) and (.error | not)' \
  <<<"${api_response}"
```

Set `success` to `1` only when both endpoints return 2xx and pass the structural checks above. Preserve the full chat
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

Also write `deployment_ledger.json`, including the DGD name, Kubernetes context and namespace, source recipe path,
final applied-manifest paths, compatibility patches and their reasons, readiness state, concise diagnostics, blockers,
and cleanup commands.

## Out Of Scope

- recipe discovery or variant selection
- benchmark execution or AIPerf result parsing
- optimization hypotheses or challenger reviews
- authoring new recipes from scratch
- cluster setup
- reading or storing secret values

## References

- `../../../docs/kubernetes/kubernetes-recipe-workflow.md`
- `../../../docs/reference/run-artifacts.md`
