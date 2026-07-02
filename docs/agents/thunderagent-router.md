---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: ThunderAgent Program Scheduler
subtitle: Program-level scheduling with tool-boundary pause/resume on top of KV-aware routing
---

> **Experimental — not a released component.** Run it from a source checkout, not from a `pip install ai-dynamo`. The CLI flags, session headers, and lifecycle hooks are all unstable and will change.

`dynamo.thunderagent_router` is a standalone Dynamo router that schedules at the granularity of an agent run — the whole `LLM turn → tool call → next turn` loop — instead of individual requests. It wraps Dynamo's native KV router and adds a program-level scheduler with tool-boundary pause/resume on top of KV-aware routing, porting the scheduler from the [ThunderAgent](https://arxiv.org/abs/2602.13692) paper (Kang et al., 2026).

## The Problem

Agentic workloads (SWE-bench, browser-use, anything with a tool loop) make many short LLM calls separated by non-GPU work: `docker exec`, `pytest`, `curl`, waiting on a subagent. Between turns the agent's KV cache stays resident, holding blocks while doing nothing. A request-level router (vLLM's, SGLang's, Dynamo's stock `KvRouter`) sees each turn but not the agent behind it, which costs you two ways:

- **Cache-occupancy blowup.** With N agents at step K, the working set is `N × step_K_context`, most of it idle between turns. The engine evicts useful blocks under pressure or refuses admission, and every next turn pays a re-prefill tax.
- **No tool-boundary backpressure.** The router can't defer a hot session at a natural pause point — it can only cancel in-flight requests or queue them, both worse than waiting until the agent is between turns.

## The Scheduler

The algorithm groups requests by `program_id` (the header-derived `session_id`) and runs an outer scheduler that moves each program through `(REASONING | ACTING) × (ACTIVE | PAUSED)`. A program enters ACTING at a tool boundary. Under memory pressure the scheduler pauses ACTING programs — logically, with no decode preemption — so the engine is free to evict their KV. When utilization drops it resumes the smallest-token programs first, BFD-packing them back under threshold. The payoff is working-set accounting that counts programs rather than requests, plus pause/resume aimed at tool boundaries rather than arbitrary tokens.

This is an in-path Dynamo service that owns a `KvRouter` directly and registers as a model handler, so there is no extra proxy hop, and it reads real `prompt_tokens + completion_tokens` off each response rather than estimating token counts from raw bytes.

### Scheduler Tick

A single background task runs every `--scheduler-interval-seconds` (default `5.0`). Each tick takes a capacity snapshot and runs three phases in a fixed order:

```text
_apply_soft_demotes  →  _greedy_resume  →  _pause_until_safe
```

Resume runs **before** pause on purpose (upstream ThunderAgent ordering): a program paused this tick cannot resume until the next tick, which prevents a program from being paused and immediately resumed within one tick.

### Tool-Boundary Pause/Resume Semantics

- **Pause** is logical. The scheduler picks the smallest ACTING programs on an over-threshold worker first and pauses them; if no ACTING candidate exists it marks the smallest REASONING program for pause at its next tool boundary. There is no decode preemption — a paused program's in-flight turn is allowed to finish, and the program is held out of admission until a later tick resumes it.
- **Resume** is greedy and BFD-packed. When a worker has headroom (see the control loop below), the scheduler resumes the smallest-token paused programs first, fitting each back under threshold and accounting for `buffer_per_program`. Resumed requests get a transient priority boost so they re-enter ahead of fresh admissions, and a forced-resume cap (`--resume-timeout-seconds`) guarantees no program is starved indefinitely.

### Program Lifetime

A program is created on its first turn, keyed by `session_id`. Public session identity is carried in headers such as `x-dynamo-session-id`, `x-dynamo-parent-session-id`, and `x-dynamo-session-final`. Program bookkeeping must still be bounded by router policy, such as idle expiry or token-weight decay.

## Utilization-Driven Control Loop

Pause/resume is driven by per-worker utilization — the program working set as a fraction of the worker's retention budget. With SGLang HiCache enabled, Dynamo reads the worker's published GPU KV and host HiCache capacities and uses their sum. The host tier is included so native GPU-to-host spill can happen before this scheduler pauses programs. Mooncake is excluded: it is conditional content-addressed storage, not guaranteed per-program retention. The loop has three bands:

- At or above `pause-threshold`, the worker is over-subscribed; the tick pauses ACTING programs until utilization falls back to `pause-target`.
- In the `[soft-demote-threshold, pause-threshold)` band, programs are soft-demoted (a negative priority jump) but not paused — early backpressure before a hard pause is needed.
- Resume only fires once utilization has dropped at least `resume-hysteresis` below `pause-threshold`, so the loop does not oscillate between pause and resume on the threshold boundary.

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--pause-threshold` | `DYN_THUNDERAGENT_PAUSE_THRESHOLD` | `0.95` | Working-set fraction of the retention budget that fires a pause cycle. |
| `--soft-demote-threshold` | `DYN_THUNDERAGENT_SOFT_DEMOTE_THRESHOLD` | `0.80` | Soft-demote band start (negative priority jump in `[soft, pause)`). |
| `--pause-target` | `DYN_THUNDERAGENT_PAUSE_TARGET` | `0.80` | Setpoint that pause cycles drive utilization back down to. Must be `<= pause-threshold`. |
| `--resume-hysteresis` | `DYN_THUNDERAGENT_RESUME_HYSTERESIS` | `0.10` | Headroom below `pause-threshold` required before any resume. |
| `--resume-priority-boost` | `DYN_THUNDERAGENT_RESUME_PRIORITY_BOOST` | `1.0` | Priority seconds added to a request that just resumed. |
| `--resume-timeout-seconds` | `DYN_THUNDERAGENT_RESUME_TIMEOUT_SECONDS` | `1800.0` | Forced-resume cap. Mirrors ThunderAgent's `_wait_for_resume`. |
| `--scheduler-interval-seconds` | `DYN_THUNDERAGENT_SCHEDULER_INTERVAL_SECONDS` | `5.0` | Scheduler tick period. |
| `--soft-demote-priority-jump` | `DYN_THUNDERAGENT_SOFT_DEMOTE_PRIORITY_JUMP` | `-2.0` | Priority seconds applied to soft-demoted programs. |
| `--acting-token-weight` | `DYN_THUNDERAGENT_ACTING_TOKEN_WEIGHT` | `1.0` | Multiplier on `token_total` for ACTING programs in the **pause-side** working set. |
| `--acting-decay-tau-seconds` | `DYN_THUNDERAGENT_ACTING_DECAY_TAU_SECONDS` | `1.0` | Tau for exponential decay of ACTING tokens in the **resume-side** working set. |

> **Constraint:** `pause-target <= pause-threshold`. The service rejects configs that violate it (along with `0 <= resume-hysteresis <= pause-threshold` and `0 <= soft-demote-threshold <= pause-threshold`).

All `KvRouter` flags from `dynamo.router` (`--router-temperature`, `--use-kv-events`, `--router-track-output-blocks`, …) are also accepted and forwarded.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ dynamo.frontend  (HTTP + auth + tracing sink)               │
└────────────────────┬────────────────────────────────────────┘
                     │  chat completions, with session headers
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ dynamo.thunderagent_router  (this service)                  │
│  - ProgramTable: session_id → ProgramState                  │
│  - admission gate: before_request → was_paused?             │
│  - scheduler loop (every scheduler_interval_seconds):       │
│      _apply_soft_demotes → _greedy_resume → _pause_until_safe│
│  - sticky worker pin from program.assigned_worker_id        │
│  - after_request: real-token accounting                     │
└────────────────────┬────────────────────────────────────────┘
                     │  KvRouter.generate
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ KvRouter  (in-process; subscribes to KV events + FPM)       │
└────────────────────┬────────────────────────────────────────┘
                     │  per-worker dispatch
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ dynamo.vllm  (N workers; FPM publisher, KV events publisher)│
└─────────────────────────────────────────────────────────────┘
```

## Observability

The scheduler emits a per-tick INFO summary on each side of the control loop, so both pause and resume activity are visible at INFO without enabling DEBUG. Per-program detail stays at DEBUG.

**Pause side** — logged when a worker pauses or marks any program in a tick:

```text
scheduler.tick worker=<id> paused=<N> marked=<M> util=<X> -> <Y>
```

`paused` is the number of ACTING programs paused this tick, `marked` is the number of REASONING programs marked for pause at their next tool boundary, and `util=X -> Y` is the worker utilization before and after the pause cycle.

**Resume side** — logged when a worker resumes any program in a tick:

```text
scheduler.tick resumed=<N> still_paused=<M>
```

`resumed` is the number of programs resumed this tick and `still_paused` is the size of the paused table afterward. This line is symmetric to the pause-side summary; before it existed, pause was observable at INFO but resume was only visible at DEBUG, leaving a gap when reconstructing a control-loop cycle from INFO logs alone.

**Per-program detail (DEBUG):**

```text
Paused program <program_id> (tokens=<n>)
Resumed program <program_id> -> worker=<id> (tokens=<n>)
```

Enable these by lowering the log level for `dynamo.thunderagent_router`. They give the exact program identities behind each INFO summary count.

For per-request tracing (token counts, cache hits, worker placement), the router also integrates with [Agent Tracing](agent-tracing.md#enable-output): set `DYN_REQUEST_TRACE=1` on the frontend to land a `request_end` record per LLM call. Harness tool-event spans are separate: they require `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` plus a configured publisher.

## Reproducing with upstream Harbor and Pi

This is the maintained SWE-bench path: current upstream Harbor creates the task containers and runs Pi inside them; Dynamo, ThunderAgent, and the model stay on the host. No cache-specific engine flags are required for this smoke.

The current upstream Harbor Pi adapter does not install external Pi providers or preserve a session while it runs Pi with `--no-session`. The [`DynamoPi` adapter](https://github.com/ai-dynamo/agent-plugins/blob/main/pi-plugin/harbor_dynamo_pi.py) adds only those two missing pieces: it installs the mounted Dynamo provider and maps Harbor's already-assigned per-trial `agent.session_id` into `DYN_AGENT_SESSION_ID`. Thus every turn of one Harbor trial carries the same `x-dynamo-session-id`; identity remains a hint, not a cache command.

### 1. Check out the three source trees

```bash
export DYNAMO_DIR=$HOME/src/dynamo
export HARBOR_DIR=$HOME/src/harbor
export PLUGINS_DIR=$HOME/src/agent-plugins

git clone https://github.com/ai-dynamo/dynamo "$DYNAMO_DIR"
git clone https://github.com/harbor-framework/harbor "$HARBOR_DIR"
git clone https://github.com/ai-dynamo/agent-plugins "$PLUGINS_DIR"

cd "$DYNAMO_DIR/lib/bindings/python" && maturin develop --uv
cd "$DYNAMO_DIR" && uv pip install -e .
cd "$HARBOR_DIR" && uv sync

git -C "$DYNAMO_DIR" rev-parse HEAD
git -C "$HARBOR_DIR" rev-parse HEAD
git -C "$PLUGINS_DIR" rev-parse HEAD
```

The commands below were validated with Harbor `0.16.0`. Save the three printed revisions in the run artifact; do not substitute the old vendored ThunderAgent/Harbor launcher.

### 2. Start the Dynamo stack

Run this in one terminal and leave it running. File discovery, TCP request transport, and ZMQ events keep the stack self-contained: no etcd or NATS is required.

```bash
cd "$DYNAMO_DIR"
source .venv/bin/activate

export MODEL=zai-org/GLM-4.7-Flash
export RUN_DIR=$PWD/runs/ta-harbor-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/file-kv"

export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="$RUN_DIR/file-kv"
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_SINKS=jsonl_gz
export DYN_REQUEST_TRACE_OUTPUT_PATH="$RUN_DIR/request-trace"

python -m dynamo.frontend \
  --discovery-backend file \
  --request-plane tcp \
  --event-plane zmq \
  --router-mode round-robin \
  --router-reset-states \
  --http-host 0.0.0.0 \
  --http-port 8000 \
  >"$RUN_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!

DYN_SYSTEM_PORT=8081 python -m dynamo.sglang \
  --discovery-backend file \
  --request-plane tcp \
  --event-plane zmq \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp-size 2 \
  --trust-remote-code \
  --enable-metrics \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557","enable_kv_cache_events":true}' \
  --dyn-reasoning-parser glm45 \
  --dyn-tool-call-parser glm47 \
  >"$RUN_DIR/logs/sglang.log" 2>&1 &
SGLANG_PID=$!

python -m dynamo.thunderagent_router \
  --discovery-backend file \
  --request-plane tcp \
  --event-plane zmq \
  --endpoint dynamo.backend.generate \
  --model-name "$MODEL" \
  --model-path "$MODEL" \
  --router-block-size 16 \
  --router-reset-states \
  --dyn-reasoning-parser glm45 \
  --dyn-tool-call-parser glm47 \
  >"$RUN_DIR/logs/thunderagent.log" 2>&1 &
THUNDERAGENT_PID=$!

until curl -fsS http://127.0.0.1:8000/v1/models | grep -Fq "$MODEL"; do sleep 2; done
curl -fsS http://127.0.0.1:8000/v1/models
wait "$FRONTEND_PID" "$SGLANG_PID" "$THUNDERAGENT_PID"
```

The final `wait` keeps the stack alive. Stop it with `Ctrl-C`; the three PIDs are available for targeted cleanup if a process exits first.

### 3. Verify the container can reach the host frontend

In a second terminal, choose the host address reachable from Docker containers. Do this before starting a scored task so connection failures do not become benchmark failures.

```bash
export DYNAMO_HOST=$(ip route get 1.1.1.1 | awk '{print $7; exit}')
export DYNAMO_BASE_URL="http://${DYNAMO_HOST}:8000/v1"
curl -fsS "$DYNAMO_BASE_URL/models"
docker run --rm curlimages/curl:8.11.1 -fsS "$DYNAMO_BASE_URL/models"
```

### 4. Run an install-only Harbor smoke

This tests the current Harbor import path, read-only provider mount, Pi installation, and per-trial session plumbing without consuming a model rollout.

```bash
cd "$HARBOR_DIR"
source .venv/bin/activate
export PYTHONPATH="$PLUGINS_DIR/pi-plugin${PYTHONPATH:+:$PYTHONPATH}"
export PI_PLUGIN_DIR="$PLUGINS_DIR/pi-plugin"

harbor run \
  --dataset swebench-verified@1.0 \
  --include-task-name astropy__astropy-12907 \
  --agent harbor_dynamo_pi:DynamoPi \
  --model "dynamo/$MODEL" \
  --agent-kwarg version=0.72.1 \
  --agent-env "DYNAMO_BASE_URL=$DYNAMO_BASE_URL" \
  --agent-env DYNAMO_API_KEY=dummy \
  --agent-env DYN_REQUEST_TRACE=1 \
  --mounts "[{\"type\":\"bind\",\"source\":\"$PI_PLUGIN_DIR\",\"target\":\"/opt/pi-dynamo-provider\",\"read_only\":true}]" \
  --install-only \
  --n-concurrent 1 \
  --job-name ta-verified-install-smoke \
  --jobs-dir "$RUN_DIR/harbor" \
  --yes
```

### 5. Run one scored Verified task

Remove `--install-only`; verification is enabled by default. This is the first end-to-end correctness and trace check.

```bash
harbor run \
  --dataset swebench-verified@1.0 \
  --include-task-name astropy__astropy-12907 \
  --agent harbor_dynamo_pi:DynamoPi \
  --model "dynamo/$MODEL" \
  --agent-kwarg version=0.72.1 \
  --agent-env "DYNAMO_BASE_URL=$DYNAMO_BASE_URL" \
  --agent-env DYNAMO_API_KEY=dummy \
  --agent-env DYN_REQUEST_TRACE=1 \
  --mounts "[{\"type\":\"bind\",\"source\":\"$PI_PLUGIN_DIR\",\"target\":\"/opt/pi-dynamo-provider\",\"read_only\":true}]" \
  --n-concurrent 1 \
  --job-name ta-verified-one \
  --jobs-dir "$RUN_DIR/harbor" \
  --yes
```

Before scaling out, confirm both identities and scheduler behavior:

```bash
gzip -cd "$DYN_REQUEST_TRACE_OUTPUT_PATH".*.jsonl.gz \
  | jq -r '(.event // .).agent_context.session_id // empty' \
  | sort -u

rg 'scheduler\.tick|Paused program|Resumed program' "$RUN_DIR/logs/thunderagent.log"
find "$RUN_DIR/harbor/ta-verified-one" -name results.json -print
```

The trace command must report the Harbor trial session (normally `<trial-name>__agent`). An absence of `Paused program` or `Resumed program` is a valid low-pressure outcome, not a reason to infer scheduler activity.

### 6. Scale to a fixed Verified cohort, then Pro

For an exploratory load test, add `--n-tasks 32 --n-concurrent 32` to the Verified command. For a reported result, replace `--n-tasks` with repeated `--include-task-name <task-id>` arguments generated from a frozen task manifest; `--n-tasks` uses registry order and is not a stable paper cohort.

After the same path is clean, change only the dataset selector to start a small Pro smoke:

```bash
--dataset swebenchpro@1.0 --n-tasks 10 --n-concurrent 10
```

Pre-pull task images outside the measured interval, preserve `$RUN_DIR/harbor`, `$RUN_DIR/request-trace.*.jsonl.gz`, and the three server logs, then run the policy arms against fresh server stacks.

## References

- ThunderAgent paper: [arxiv.org/abs/2602.13692](https://arxiv.org/abs/2602.13692)
- Upstream ThunderAgent reference: [HaoKang-Timmy/ThunderAgent](https://github.com/HaoKang-Timmy/ThunderAgent)
- Pi Dynamo provider: [ai-dynamo/agent-plugins](https://github.com/ai-dynamo/agent-plugins/tree/main/pi-plugin)
- Dynamo KV router: [Router Guide](../components/router/router-guide.md)
- [Session IDs](session-ids.md), [Agent Tracing](agent-tracing.md), and [Agent Hints](agent-hints.md)
