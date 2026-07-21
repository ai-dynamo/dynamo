# Kubernetes Recipe Deployment Workflow

Use this reference after another agent has already selected a Dynamo `deploy.yaml` or DGD manifest.

## Artifact Setup

Create `<EXP_ROOT>/artifacts/deploy-iter-<NNN>/applied_manifests/`. Copy every manifest used into it with a
stable filename and apply only those run-scoped copies. Update a copy in place when a compatibility fix is required and
record the change in `deployment_ledger.json`; do not retain numbered intermediate copies. After a successful smoke
test, keep exactly one final file per manifest type used. Create `logs/` only for targeted failure output that must be
retained beyond the deployment ledger.

## Read-Only Preflight

```bash
kubectl config current-context
kubectl get namespace "${NAMESPACE}"
kubectl get crd | grep -i dynamo
kubectl get storageclass
kubectl get nodes -o wide
```

Check secrets only by name. Never print, decode, or persist secret values.

## Common Apply Sequence

For iteration > 0, read the previous deployment ledger. Using the recorded Kubernetes context and namespace, delete
only the previous DGD by its exact name, then wait for its operator-owned workloads to exit. Preserve the previous
deployment directory and all shared PVCs, model-cache jobs, namespaces, and secrets.

Apply model cache resources when the recipe requires them.

```bash
kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/model-cache.yaml" -n "${NAMESPACE}"
kubectl get pvc -n "${NAMESPACE}"
```

Run model download and validation jobs when present. Read each Job name from its manifest's `metadata.name`; never infer
the resource name from the filename.

```bash
kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/model-download.yaml" -n "${NAMESPACE}"
kubectl wait --for=condition=Complete "job/${DOWNLOAD_JOB}" -n "${NAMESPACE}" --timeout=6000s

kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/model-validate.yaml" -n "${NAMESPACE}"
kubectl wait --for=condition=Complete "job/${VALIDATE_JOB}" -n "${NAMESPACE}" --timeout=3600s
```

Apply the selected DGD from its run-scoped copy:

```bash
kubectl apply -f "${DEPLOY_ROOT}/applied_manifests/deploy.yaml" -n "${NAMESPACE}"
kubectl get dynamographdeployment -n "${NAMESPACE}"
kubectl get pods -n "${NAMESPACE}" -o wide
kubectl get svc -n "${NAMESPACE}"
```

## Readiness Signals

- PVCs are `Bound`
- model download and validation jobs are `Complete`
- DGD reports no unresolved reconciliation errors
- every component and replica declared by the selected DGD is `Running` and ready
- frontend service exists
- no unresolved scheduling, mount, image pull, or crash-loop events remain

On failure, inspect the DGD status, events, and logs for the affected component before making a minimal run-scoped
compatibility patch.

## Smoke Test

Find the frontend service:

```bash
kubectl get svc -n "${NAMESPACE}" | grep frontend
```

Port-forward:

```bash
kubectl port-forward svc/<frontend-service> 8000:8000 -n "${NAMESPACE}"
```

Capture HTTP status and response body separately; JSON parsing alone does not prove success. Verify the served model and
one chat completion:

```bash
SERVED_MODEL="<served-model-name>"
models_response="$(curl -sS --fail-with-body http://127.0.0.1:8000/v1/models)"
jq -e --arg model "${SERVED_MODEL}" 'any(.data[]?; .id == $model)' <<<"${models_response}"
api_request="$(jq -nc --arg model "${SERVED_MODEL}" '{
  model: $model,
  messages: [{role: "user", content: "Simply output the phrase: NVIDIA Dynamo"}],
  max_tokens: 16,
  temperature: 0
}')"
api_response="$(curl -sS --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "${api_request}")"
jq -e '.object == "chat.completion" and (.choices | type == "array" and length > 0) and (.error | not)' \
  <<<"${api_response}"
```

The smoke test succeeds only when both endpoints return 2xx and pass these structural checks. Preserve the full chat
response as `api_response`, including the full API error body on failure.

## Common Blockers

- missing namespace, CRD, storage class, PVC, or referenced secret
- model access not accepted upstream
- image pull failure
- requested GPU count or SKU unavailable
- node selectors or tolerations do not match the cluster
- model download or validation job failed
- frontend service missing
- `/v1/models` or `/v1/chat/completions` returns an error body

Record the diagnosis, relevant error excerpt, and any compatibility patch in `deployment_ledger.json`. Do not dump
namespace state, endpoint-response copies, successful pod logs, or unrelated command output. Save one targeted file
under `logs/` only when failure output is needed beyond the ledger excerpt.
