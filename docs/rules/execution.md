# Execution Rules

Concise execution discipline for Dynamo agents.

## Fail Fast

- Before mutating a cluster, verify the active `kubectl` context, namespace, required CRDs, storage class, secrets, and
  selected manifests.
- Prefer dry-run validation when possible:

```bash
kubectl apply --dry-run=server -f <manifest> -n <namespace>
```

- Do not apply a recipe while required placeholders, PVCs, secrets, image pull requirements, GPU resources, or model
  access are unresolved.

## Long-Running Work

- Model download jobs, model validation jobs, DGD reconciliation, pod startup, and AIPerf jobs can take many minutes.
- Use bounded waits, but do not wait blindly. During long waits, check job status, pod readiness, recent events,
  relevant logs, and DGD status.
- Avoid high frequency agent-side polling. Use coarse checks unless actively debugging startup readiness.

## Failure Handling

- On timeout or failure, inspect evidence such as the following before retrying:
  - job logs for failed jobs
  - pod logs for crashed containers
  - `kubectl describe pod` and events for scheduling, mount, image pull, or GPU allocation failures
  - DGD status and events for deployment failures
  - endpoint response body for smoke-test failures
- Retry only after a concrete fix.
- Stop after repeated identical failures and report the blocker with evidence.
