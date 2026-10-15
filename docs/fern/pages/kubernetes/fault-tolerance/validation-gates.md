---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Validation Gates
subtitle: Run your own checks against a new worker before it takes client traffic, so a broken worker stalls a rollout instead of serving.
---

A worker can pass its probes and still be broken in ways only a real request shows, such as a bad
tool-call parser or the wrong weight revision. Validation gates run your checks against a new worker
before it becomes routable. Until every gate passes, its pod isn't Ready and, while its
`before_publish` gates run, it gets no requests at all; liveness is unaffected, so it isn't restarted.
Nothing changes when gates aren't set.

## Configure gates

```yaml
env:
- name: DYN_WORKER_VALIDATION_GATES
  value: |
    {
      "completion": {"exec": ["python3", "-m", "dynamo.validation.completion"], "timeout_secs": 600},
      "smoke": {"url": "http://127.0.0.1:9100/validate/smoke", "timeout_secs": 900}
    }
```

Each gate runs once and must reach a verdict within `timeout_secs`. Gates run in parallel within their
[phase](#phases), starting as soon as the worker's engine is serving.

| Gate | Runs | Verdict |
|---|---|---|
| `exec` | The command in the worker container | Exit status 0 passes; otherwise the end of stderr is the reason. |
| `url` | A `POST` to a check runner on loopback, usually a sidecar | `200` passes; any other status fails, with the end of the body as the reason. A runner that isn't reachable yet is retried until the deadline. |

A misconfigured gate stops the worker at startup. A failed gate keeps the worker held for the life of
the process; delete the pod to retry. A restarted worker container runs every gate again.

## What a check sees

Every check gets `DYN_VALIDATION_GATE` (its name), `DYN_VALIDATION_MODEL`, and `OPENAI_BASE_URL`: an
OpenAI-compatible endpoint on `127.0.0.1` that serves the worker's model through the same
preprocessor, tool-call and reasoning parsers as the frontend. `url` gates receive them as a JSON
request body. Workers without a chat or completions surface, such as prefill workers, get no
`OPENAI_BASE_URL`.

## Phases

Each gate's optional `"phase"` sets when it runs relative to the worker's model card being written to
discovery:

- `before_publish` (default): the card isn't written while the gate runs, so no frontend routes to the
  worker under any router. The worker is quiet, so measurements are comparable across pods.
- `after_publish`: runs once every `before_publish` gate has passed and the card has been written. Use
  it for a check that needs the frontend to see the worker, such as warming up on traffic the
  deployment routes to it. The deployment must keep client traffic off the worker by other means.

```json
{
  "completion": {"exec": ["python3", "-m", "dynamo.validation.completion"], "timeout_secs": 300},
  "soak": {"exec": ["python3", "/checks/soak.py"], "timeout_secs": 600},
  "warm": {"exec": ["python3", "/checks/warm.py"], "timeout_secs": 600, "phase": "after_publish"}
}
```

A failed `before_publish` gate means the card is never written. `/health` reports 503 until every
gate in both phases passes. Under etcd discovery the frontend routes on cards, not pod readiness, so
use `after_publish` only when the deployment keeps traffic off the worker by other means.

## Run checks in a sidecar

When the worker image can't carry your checks, run them in a sidecar with its own image:

```bash
python3 -m dynamo.validation.runner --port 9100 --check smoke='python3 /checks/smoke.py'
```

The runner answers `POST /validate/<name>` by running that check with the gate's environment. See
[`agg_validation_gates.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_validation_gates.yaml).

## Roll out with gates

Set `nvidia.com/deployment-rolling-update-max-unavailable: "0"` on the worker so an old pod leaves
only when a new one is Ready. A failing gate then stalls the rollout with the old pods serving.
Nothing rolls back on its own (Kubernetes may report the rollout as stalled after
`progressDeadlineSeconds`): fix the check or roll back.

Watch a held worker on its system port: `/health` lists each gate's state and failure reason, and
`/live` stays 200.

## Scope and known bypasses

Gates cover aggregated, prefill and decode workers that register model cards, under etcd or
pod-mode Kubernetes discovery. Not covered: multinode deployments, multimodal encode workers (they
are called worker to worker), the virtual planner (it counts endpoint instances, not cards), and a
`worker_id` pinned through the standalone router.
