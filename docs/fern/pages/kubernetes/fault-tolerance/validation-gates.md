---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Validation Gates
subtitle: Run your own checks against a new worker before it takes client traffic, so a broken worker stalls a rollout instead of serving.
---

A worker can pass its probes and still serve broken responses: engine flags that change how it handles
reasoning or tool calls, a kernel choice that regresses a feature, a card that configures the frontend's
chat template or parsers wrongly, or a fault on one node. Validation gates call your checks against a
new worker before it becomes routable. Until every check passes, its pod isn't Ready; liveness is
unaffected, so it isn't restarted. Nothing changes when the gates aren't set.

## Configure gates

```yaml
env:
- name: DYN_PUBLISH_GATE
  value: http://127.0.0.1:9100/checks/completion,http://127.0.0.1:9100/checks/smoke
- name: DYN_READY_GATE
  value: http://127.0.0.1:9100/checks/warmup
```

Each variable is a comma-separated list of check URLs on loopback; every one must pass. Unset means no
gate at that release point.

| Gate | Holds | Checks are called |
|---|---|---|
| `DYN_PUBLISH_GATE` | The model card's discovery write, and `/health` | Once the worker's engine is serving |
| `DYN_READY_GATE` | `/health` only | Once every publish check has passed and the card is written |

The worker `POST`s to each URL once, in parallel within a gate. A `200` passes; any other status fails,
and the start of the response body is the reason. A runner that can't be reached yet, or drops the
connection before answering, is retried with backoff, so a check may run more than once for one worker.
The worker puts no deadline on a check: the check runner bounds its own checks, and one that never
answers keeps the worker held.

A misconfigured gate stops the worker at startup. A failed check keeps the worker held for the life of
the process; delete the pod to retry. A restarted worker container calls every check again, and each
call must run the check again rather than return an earlier verdict.

## What each gate means

| | etcd discovery | Kubernetes discovery |
|---|---|---|
| Publish checks running | No card, so no frontend routes to the worker. | Not Ready, so not routable. |
| Ready checks running | Card written, so **routable**. Not Ready, so the rollout doesn't advance. | Not Ready, so not routable. |
| Every check passed | Routable and Ready. | Routable and Ready. |

The publish gate keeps client traffic off the worker under either backend and any router; most
deployments only need it. Use the ready gate for checks that need the frontend to see the worker, such
as warming up on traffic the deployment routes to it. Under etcd the deployment must keep client
traffic off the worker by other means while those checks run.

## What a check sees

Each request has a JSON body:

```json
{"gate": "publish", "base_url": "http://127.0.0.1:41813/v1", "models": ["Qwen/Qwen3-0.6B"]}
```

`base_url` is an OpenAI-compatible endpoint on `127.0.0.1` that serves the worker's model through the
same preprocessor, chat template, tool-call and reasoning parsers as the frontend, so a check sees what
a client would get from this worker. It's `null` for workers without a chat or completions surface,
such as prefill workers. The endpoint stops once the worker is released.

## Run checks

The reference runner turns ordinary one-shot commands into check URLs:

```bash
python3 -m dynamo.validation.runner --port 9100 --timeout 600 \
    --check completion='python3 -m dynamo.validation.completion' \
    --check smoke='python3 /checks/smoke.py'
```

It answers `POST /checks/<name>` by running that command with `OPENAI_BASE_URL`,
`DYN_VALIDATION_MODEL` and `DYN_VALIDATION_GATE` set from the request. Exit status 0 is a `200`;
otherwise it answers `500` with the end of stderr. A check that runs past `--timeout` fails. Whether a
problem fails the check or only logs a warning is up to the check.

Run it as a sidecar with its own image, as in
[`agg_validation_gates.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_validation_gates.yaml),
or, where the platform doesn't allow sidecars, start it next to the worker in the worker container's
command. Any server on loopback that follows the same contract works.

## Roll out with gates

Set `nvidia.com/deployment-rolling-update-max-unavailable: "0"` on the worker so an old pod leaves
only when a new one is Ready. A failing check then stalls the rollout with the old pods serving.
Nothing rolls back on its own (Kubernetes may report the rollout as stalled after
`progressDeadlineSeconds`): fix the check or roll back.

Watch a held worker on its system port: `/health` lists each check's state and failure reason under
`validation.publish` and `validation.ready`, and `/live` stays 200.

## Scope and known bypasses

Gates cover aggregated, prefill and decode workers that register model cards, under etcd or
pod-mode Kubernetes discovery, each checking itself alone; checks that need several pods, such as a
request through prefill and decode, are out of scope. Not covered: multinode deployments, multimodal
encode workers (they are called worker to worker), the virtual planner (it counts endpoint instances,
not cards), and a `worker_id` pinned through the standalone router.
