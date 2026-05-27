---
name: dynamo-recipe-runner
description: Select, validate, patch, and deploy existing NVIDIA Dynamo Kubernetes recipes. Use for model/backend/GPU/deployment-mode recipe bring-up; use router-starter for router-only mode work and troubleshoot for broken deployments.
tags:
  - dynamo
  - kubernetes
  - recipes
  - bring-up
---

# Dynamo Recipe Runner

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Goal

Get from user intent to a working Dynamo recipe endpoint with minimal back and
forth. Do not create new guide content. Operate on the existing `recipes/`
tree, patch the smallest necessary set of manifests, deploy when the user has
cluster access, and prove success with an OpenAI-compatible smoke request.

## Required Inputs

Collect or infer these before changing manifests:

- recipe target: model, framework (`vllm`, `sglang`, `trtllm`, `tokenspeed`), deployment mode, and GPU type/count
- Kubernetes context and namespace
- Hugging Face secret name, usually `hf-token-secret`
- storage class for model cache PVCs
- runtime image tag if the recipe uses a placeholder or stale test image
- whether to run commands or only produce exact commands

If a required value is missing and cannot be inferred from the selected recipe,
ask for only that value.

## Workflow

### 1. Preflight

Run read-only checks first:

```bash
git status --short
python3 scripts/recipe_tool.py list --format table
kubectl config current-context
kubectl get storageclass
kubectl get nodes -o wide
kubectl get namespace "${NAMESPACE}"
kubectl get secret hf-token-secret -n "${NAMESPACE}"
```

If `kubectl` is unavailable or the cluster is unreachable, continue by
selecting and validating the recipe, then return exact commands instead of
pretending the deployment ran.

### 2. Select The Recipe

Use the recipe matrix from `recipes/README.md` and the scanner:

```bash
python3 scripts/recipe_tool.py list \
  --query qwen --framework vllm --mode disagg --format table
```

Prefer an exact existing recipe. Do not invent new manifests unless the user
explicitly asks to author a new recipe.

### 3. Inspect And Validate

Read the selected recipe README, model-cache manifests, `deploy.yaml`, and
`perf.yaml` if present. Then run:

```bash
python3 scripts/recipe_tool.py validate \
  recipes/<model>/<framework>/<mode>
```

Resolve reported blockers before applying manifests: storage class, model cache
PVC, image tag, HF token secret, GPU count, frontend service name, and router
mode.

### 4. Patch Minimal Values

Patch only recipe-specific values needed for this run. Do not reformat whole
YAML files. Common patches:

- `storageClassName`
- image repository/tag
- model path or model cache mount path
- GPU resource requests/limits
- frontend `DYN_ROUTER_MODE`
- namespace only when a manifest hardcodes it

Never write Hugging Face tokens into files or logs. Use Kubernetes secrets.

### 5. Deploy

Follow the selected recipe README when it differs from the default sequence.
The default sequence is:

```bash
kubectl apply -f recipes/<model>/model-cache/ -n "${NAMESPACE}"
kubectl wait --for=condition=Complete job/model-download -n "${NAMESPACE}" --timeout=6000s
kubectl apply -f recipes/<model>/<framework>/<mode>/deploy.yaml -n "${NAMESPACE}"
kubectl get dynamographdeployment -n "${NAMESPACE}"
kubectl get pods -n "${NAMESPACE}" -o wide
```

Wait for the frontend and workers to be ready before testing.

### 6. Smoke Test

Port-forward the frontend service, then verify `/v1/models` and one chat
completion:

```bash
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n "${NAMESPACE}"
curl http://127.0.0.1:8000/v1/models
```

If `dynamo-router-starter` is also installed, prefer its `scripts/check_router_health.py`
for the full OpenAI-compatible smoke test. If this fails, switch to
`dynamo-troubleshoot`.

## Output Contract

Return:

- selected recipe path and why it was selected
- exact values patched
- commands run or commands to run
- endpoint and smoke-test result
- unresolved blockers, if any
- next troubleshooting step when deployment does not become healthy

## References

- Read `references/k8s-recipe-workflow.md` for command templates and readiness checks.
- Use `scripts/recipe_tool.py` for recipe discovery and lightweight validation.
