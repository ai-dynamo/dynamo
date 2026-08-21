<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kubernetes Recipe Deployment Workflow

Use this reference after another agent has already selected a Dynamo `deploy.yaml` or DGD manifest.

## Artifact Setup

Create `<EXP_ROOT>/artifacts/deploy-iter-<NNN>/applied_manifests/`. This deployment iteration directory,
`<EXP_ROOT>/artifacts/deploy-iter-<NNN>/`, is `DEPLOY_ROOT` in the commands below (see
`agent-docs/rules/execution/run-artifacts.md` for the full definitions); export it before applying anything. Copy every manifest used into it with a
stable filename and apply only those run-scoped copies. Update a copy in place when a compatibility fix is required and
record the change in `deployment_ledger.json`; do not retain numbered intermediate copies. After a successful smoke
test, keep exactly one final file per manifest type used. Create `logs/` only for targeted failure output that must be
retained beyond the deployment ledger.

## Read-Only Preflight

```bash
kubectl --context "${KUBE_CONTEXT}" get namespace "${NAMESPACE}"
# CRD presence gate: a Forbidden here is tolerated because deploy-dynamo-recipe's server
# dry-run re-checks it authoritatively; a confirmed absence stops before any mutation.
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

Check secrets only by name. Never print, decode, or persist secret values.

## Common Apply Sequence

For iteration > 0, read the previous deployment ledger. Using the recorded Kubernetes context and namespace, delete
only the previous DGD by its exact name, then wait for its operator-owned workloads to exit. Preserve the previous
deployment directory and all shared PVCs, model-cache jobs, namespaces, and secrets.

Apply model cache resources when the recipe requires them.

```bash
kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/model-cache.yaml" -n "${NAMESPACE}"
kubectl --context "${KUBE_CONTEXT}" get pvc -n "${NAMESPACE}"
```

Run model download and validation jobs when present. Read each Job name from its manifest's `metadata.name`; never infer
the resource name from the filename. Both the existence and the filename of these manifests are recipe-dependent — some
recipes ship precision-suffixed download manifests, and some fold the download into the model-cache manifest — so take
the filename from the handoff and skip the block entirely when the recipe ships no such job.

```bash
kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/${DOWNLOAD_MANIFEST}" -n "${NAMESPACE}"
# Poll the job true condition with a bounded loop (Complete -> proceed, Failed -> exit 1); a Failed job must
# fail fast, not burn the timeout. Use the scripted poll block from deploy-dynamo-recipe SKILL.md.

kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/${VALIDATE_MANIFEST}" -n "${NAMESPACE}"
# Same bounded Complete/Failed poll as the download job (60 min bound).
```

Apply the selected DGD from its run-scoped copy:

```bash
kubectl --context "${KUBE_CONTEXT}" apply -f "${DEPLOY_ROOT}/applied_manifests/deploy.yaml" -n "${NAMESPACE}"
kubectl --context "${KUBE_CONTEXT}" get dynamographdeployment -n "${NAMESPACE}"
kubectl --context "${KUBE_CONTEXT}" get pods -n "${NAMESPACE}" -o wide
kubectl --context "${KUBE_CONTEXT}" get svc -n "${NAMESPACE}"
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
kubectl --context "${KUBE_CONTEXT}" get svc -n "${NAMESPACE}" | grep frontend
```

Port-forward and smoke-test using the single gated script in `deploy-dynamo-recipe`'s SKILL.md ("Run the
port-forward and the smoke test in ONE shell session"): it backgrounds the port-forward with a readiness poll and
trap teardown, captures HTTP status codes separately from response bodies, stores the bodies under
`${DEPLOY_ROOT}/smoke/`, and exits non-zero on any failed gate. Do not reconstruct the smoke test from memory;
that script is the reference.

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
