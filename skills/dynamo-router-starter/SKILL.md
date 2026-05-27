---
name: dynamo-router-starter
description: Start or patch Dynamo router modes and run router endpoint smoke checks. Use for round-robin, KV-aware, least-loaded, or device-aware routing setup; use recipe-runner for recipe deployment and troubleshoot for failure diagnosis.
tags:
  - dynamo
  - router
  - smoke-test
  - bring-up
license: Apache-2.0
---

# Dynamo Router Starter

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Goal

Make Dynamo routing feel easy by getting a baseline router mode running, enabling
KV-aware routing when appropriate, and proving the endpoint works. Keep the user
focused on exact commands and success signals, not router internals.

## Required Inputs

Collect or infer:

- local Python/CLI or Kubernetes recipe path
- desired mode: `round-robin`, `kv`, `least-loaded`, `device-aware-weighted`, `direct`, or `random`
- frontend port or Kubernetes frontend service
- whether workers publish KV events; if not, use approximate KV mode
- model name for smoke requests, if `/v1/models` cannot discover it

## Workflow

### 1. Establish A Baseline

For local bring-up with already registered workers:

```bash
python3 -m dynamo.frontend --router-mode round-robin --http-port 8000
```

For Kubernetes, inspect the selected recipe `deploy.yaml` and locate the
frontend service. If the recipe is not already deployed, use
`dynamo-recipe-runner` first.

### 2. Enable KV Routing

For local frontend:

```bash
python3 -m dynamo.frontend --router-mode kv --http-port 8000
```

For Kubernetes, patch only the frontend service env:

```yaml
envs:
  - name: DYN_ROUTER_MODE
    value: kv
```

If backend workers are not publishing KV cache events, set approximate mode
instead of leaving the router waiting for events:

```yaml
envs:
  - name: DYN_ROUTER_USE_KV_EVENTS
    value: "false"
```

### 3. Smoke Test

After port-forwarding the frontend service or starting local frontend, run:

```bash
python3 scripts/check_router_health.py \
  --base-url http://127.0.0.1:8000
```

This must verify `/v1/models` and, when a model is discoverable, one
`/v1/chat/completions` request.

### 4. Compare Modes Carefully

When comparing round-robin vs KV routing:

- use the same model, workers, prompt set, concurrency, and sampling settings
- send repeated-prefix prompts if demonstrating KV reuse
- label the result as a smoke comparison unless enough benchmark samples were collected
- do not claim throughput improvement from a single chat request

If the endpoint is unhealthy or workers are missing, switch to
`dynamo-troubleshoot`.

## Output Contract

Return:

- mode selected and why
- local command or Kubernetes env patch
- frontend service or URL
- smoke-test result
- any limitation, such as approximate KV mode or missing worker KV events
- next command to run for a fuller comparison

## References

- Read `references/router-modes.md` for the compact mode/env map.
- Use `scripts/check_router_health.py` for endpoint smoke tests.
