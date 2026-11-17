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
unaffected, so it isn't restarted. Nothing changes when the gate isn't set.

## Configure the gate

```yaml
env:
- name: DYN_PUBLISH_GATE
  value: http://127.0.0.1:9100/checks/completion,http://127.0.0.1:9100/checks/smoke
```

`DYN_PUBLISH_GATE` is a comma-separated list of check URLs on loopback; every one must pass. Unset means
no gate. While the checks run, the worker holds its model card's discovery write and reports not ready
on `/health`; once every check passes, it writes the card and reports ready.

The worker `POST`s to each URL once, in parallel. A `200` passes; any other status fails,
and the start of the response body is the reason. A runner that can't be reached yet, or drops the
connection before answering, is retried with backoff, so a check may run more than once for one worker.
The worker puts no deadline on a check: the check runner bounds its own checks, and one that never
answers keeps the worker held.

A misconfigured gate stops the worker at startup. A failed check keeps the worker held for the life of
the process; delete the pod to retry. A restarted worker container calls every check again, and each
call must run the check again rather than return an earlier verdict.

## What the gate means

| | etcd discovery | Kubernetes discovery |
|---|---|---|
| Checks running | No card, so no frontend routes to the worker. Not Ready. | No card and not Ready, so not routable. |
| Every check passed | Card written: routable and Ready. | Card written: Ready and routable. |

The gate keeps client traffic off the worker under either backend and any router. Under etcd
discovery it's the only way to: frontends route to any worker with a card, whatever its pod's
readiness. Under Kubernetes discovery a pod that isn't Ready is already unroutable, so the gate's job
there is to make "Ready" mean "checked".

Checks that need frontends to see the worker before it takes traffic, such as a request through each
frontend or warming its KV cache, aren't covered here; see
[DEP #17083](https://github.com/ai-dynamo/dynamo/issues/17083).

## What a check sees

Each request has a JSON body with the one thing a check runner can't be configured with ahead of time:

```json
{"base_url": "http://127.0.0.1:41813/v1"}
```

`base_url` is an OpenAI-compatible endpoint on `127.0.0.1`, on a port chosen when the worker starts. It
serves the worker's model through the same preprocessor, chat template, tool-call and reasoning
parsers as the frontend, so a check sees what a client would get from this worker; its `/models` lists
the model. It's `null` for workers without a chat or completions surface, such as prefill workers. The
endpoint stops once the worker is released. Anything else a check needs, such as which gate it serves,
comes from the deployment spec: configure it on the check runner.

## Run checks

The reference runner turns ordinary one-shot commands into check URLs:

```bash
python3 -m dynamo.common.validation.runner --port 9100 --timeout 600 \
    --check completion='python3 -m dynamo.common.validation.completion' \
    --check smoke='python3 /checks/smoke.py'
```

It answers `POST /checks/<name>` by running that command with `OPENAI_BASE_URL` set from the request
and the rest of its own environment. The `completion` check requests `DYN_VALIDATION_MODEL` if it's
set, and otherwise the first model on `/models`. Exit status 0 is a `200`;
otherwise it answers `500` with the end of stderr. A check that runs past `--timeout` fails. Whether a
problem fails the check or only logs a warning is up to the check.

Run it as a sidecar with its own image, as in
[`agg_validation_gates.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_validation_gates.yaml).
Where the platform doesn't allow sidecars, start it next to the worker in the worker container's
command; the runtime calls it the same way. Any server on loopback that follows the same contract
works.

The same setup works under Kubernetes discovery. There, a validation sidecar with an ordinary
`readinessProbe` is the recommended pattern, since Pod readiness already decides routing; support for
it in the reference runner is coming in a follow-up.

## Roll out with gates

Set `nvidia.com/deployment-rolling-update-max-unavailable: "0"` on the worker so an old pod leaves
only when a new one is Ready. A failing check then stalls the rollout with the old pods serving.
Nothing rolls back on its own (Kubernetes may report the rollout as stalled after
`progressDeadlineSeconds`): fix the check or roll back.

Watch a held worker on its system port: `/health` lists each check's state and failure reason under
`validation.publish`, and `/live` stays 200.

## Scope and known bypasses

Gates cover aggregated, prefill and decode workers that register model cards, under etcd or
pod-mode Kubernetes discovery, each checking itself alone; checks that need several pods, such as a
request through prefill and decode, are out of scope. Not covered: multinode deployments, multimodal
encode workers (they are called worker to worker), the virtual planner (it counts endpoint instances,
not cards), and a `worker_id` pinned through the standalone router.
