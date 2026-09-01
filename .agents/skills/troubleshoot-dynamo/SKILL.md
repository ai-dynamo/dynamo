---
name: troubleshoot-dynamo
description: Diagnose failed or unhealthy Dynamo deployments. Use when pods, model-cache jobs, PVCs, workers, frontend/router health, endpoints, or benchmark jobs fail; use deploy-dynamo-recipe or dynamo-router-starter before this for normal bring-up.
license: Apache-2.0
metadata:
  author: Dan Gil <dagil@nvidia.com>
  tags:
    - dynamo
    - kubernetes
    - troubleshooting
    - day-2
---

# Troubleshoot Dynamo

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Purpose

Turn a Dynamo failure into a clear problem class, strongest signal, and next
action. Start with read-only evidence, avoid secrets, and fix one layer at a
time.

## Prerequisites

- Python 3.10+ on the operator machine.
- `kubectl` configured with read access to the target namespace.
- Permission to read pods, pod logs (`pods/log`), events, jobs, PVCs, services, and
  `DynamoGraphDeployment` resources in the target namespace (NOT secrets).
- Permission to read cluster-scoped nodes and storage classes. Without it, the
  bundle is still useful but is reported as incomplete.
- Network reachability to the cluster API server.

## Instructions

### 1. Collect A Read-Only Bundle

Run:

```bash
python3 scripts/collect_dynamo_debug_bundle.py \
  --namespace "${NAMESPACE}"
```

If the user names a deployment, include it:

```bash
python3 scripts/collect_dynamo_debug_bundle.py \
  --namespace "${NAMESPACE}" \
  --deployment-name <deployment-name>
```

`--deployment-name` limits pod listings, descriptions, and logs to pods with
the deployment's `nvidia.com/dynamo-graph-deployment-name` label. Namespace
summaries such as Services, Jobs, and PVCs remain namespace-wide. Add
`--selector <label-selector>` only when a narrower pod subset is useful; it is
combined with the deployment label.

Do not request Kubernetes Secret resources. Treat the bundle as sensitive:
descriptions and logs receive best-effort redaction, but inspect the output
before sharing it.

Pass a new or empty directory to `--outdir` or `--output-dir`. The collector
rejects nonempty directories so stale artifacts from another scope cannot be
mistaken for current evidence.

### 2. Classify The Failure

Use `references/failure-decision-tree.md` and classify into one primary bucket:

- cluster/platform
- namespace/secret
- model cache/PVC/download
- image pull/runtime image
- GPU scheduling/resources
- operator/DynamoGraphDeployment reconciliation
- frontend/router
- worker/backend
- endpoint/API
- benchmark/perf job

### 3. Debug Top Down

Check in this order:

1. namespace, storage class, GPU nodes, and model-access Secret requirements
2. PVC and model-download job
3. `DynamoGraphDeployment` status and events
4. pod status, `describe pod`, and container logs
5. frontend service and port-forward
6. `/v1/models`
7. `/v1/chat/completions`
8. benchmark job only after endpoint smoke test passes

### 4. Fix One Layer At A Time

Prefer the smallest reversible change:

- create a missing namespace or required model-access Secret
- patch `storageClassName`
- patch image tag or image pull secret
- reduce GPU request only if the recipe can still be valid
- switch KV router to approximate mode only if workers do not publish events
- restart failed jobs after fixing the underlying config

After each fix, rerun the relevant readiness check before moving deeper.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/collect_dynamo_debug_bundle.py` | Collect a read-only debug bundle (pods, events, jobs, PVCs, CR status) | `--namespace`, `--deployment-name`, `--outdir` (`--output-dir` alias) |

Invoke via the agentskills.io `run_script()` protocol:

```python
run_script("scripts/collect_dynamo_debug_bundle.py", args=["--namespace", "dynamo-demo"])
```

## Examples

Collect everything in a namespace for triage:

```bash
python3 scripts/collect_dynamo_debug_bundle.py --namespace dynamo-demo
```

Scope to a single failing deployment:

```bash
python3 scripts/collect_dynamo_debug_bundle.py \
  --namespace dynamo-demo \
  --deployment-name qwen-vllm-disagg
```

Equivalent through the agent protocol:

```python
run_script("scripts/collect_dynamo_debug_bundle.py", args=["--namespace", "dynamo-demo", "--deployment-name", "qwen-vllm-disagg"])
```

## Output Contract

Return:

- problem class
- evidence checked
- strongest signal
- likely cause
- exact next command or patch
- what was ruled out
- whether it is safe to continue deployment or benchmarking

## Limitations

- Read-only. Never mutates the cluster; remediation commands are returned, not executed.
- Does not request Kubernetes Secret resources. Pod descriptions and logs
  receive best-effort redaction, but custom credential formats can evade it;
  inspect the bundle before sharing. Some authentication failures may still
  need user-side inspection.
- `--deployment-name` scopes pod details and logs, but namespace summaries
  remain namespace-wide.
- User-provided output directories must be empty; the collector never deletes
  or silently mixes existing artifacts.
- Any failed top-level read, pod discovery, pod description, or current-log
  read makes the script return nonzero and records its name in `summary.json`;
  the successfully collected evidence is still preserved. Missing
  `--previous` logs are expected for containers that have not restarted and
  are reported separately without making the bundle incomplete.
- Does not validate disagg transport — use `dynamo-interconnect-check` for that.

## Troubleshooting

| Symptom | Likely cause | Next step |
|---|---|---|
| `kubectl` returns Forbidden on events/pods | Service account lacks read RBAC | Ask operator for read-only role binding on the namespace |
| Bundle reports `complete: false` | One or more required collection commands failed | Inspect `failed_commands` in `summary.json` and the matching result files |
| Bundle missing `DynamoGraphDeployment` status | Operator not installed or different namespace | Verify `dynamo-platform` operator is installed and watching the namespace |
| Model-download job in `Pending` | PVC unbound or configured model-access Secret missing | Fix PVC binding or create the Secret referenced by the model-download job, then rerun the job |
| Worker pods `CrashLoopBackOff` | Image/runtime mismatch or GPU not available | Inspect container logs; check `nvidia.com/gpu` allocatable on nodes |


## References

- Read `references/failure-decision-tree.md` for bucket-specific checks.
- Use `scripts/collect_dynamo_debug_bundle.py` for read-only bundle collection.
