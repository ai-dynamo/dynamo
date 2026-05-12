# `dynamo.thunderagent_router`

A standalone Dynamo routing service that adds **program-level scheduling**
with tool-boundary pause/resume on top of Dynamo's native KV-aware router.
Designed to capture ThunderAgent's outer-loop scheduling wins on agentic
workloads while replacing its weakest mechanisms with engine-true Dynamo
signals.

This is the v0 implementation: scope is items 1, 2, 3, 6 of the differentiator
list described in
[`~/memory/agent-orchestration-exploration/2026-05-04-thunderagent-router-v0-scoping.md`](../../../../../../memory/agent-orchestration-exploration/2026-05-04-thunderagent-router-v0-scoping.md).
Items 4, 5, 7 (workflow profile, subagent-aware lifecycle, fairness aging)
are deferred to follow-up PRs gated on ablation results.

---

## Attribution

This component is heavily inspired by **ThunderAgent**:

- Paper: <https://arxiv.org/abs/2602.13692>
- Public repository: <https://github.com/HaoKang-Timmy/ThunderAgent>
- See [`DESIGN.md`](DESIGN.md) for a walk-through of which mechanisms we port,
  where the upstream code drifts from the paper, and where this package
  deliberately deviates. Citation for the paper is at the bottom of this file.

The mechanism we adopt directly:

- **`program_id` as the schedulable unit.** A program is one agent run across
  many LLM turns and tool gaps. The scheduler operates on programs, not
  individual requests.
- **REASONING / ACTING state per program.** REASONING means a turn is on GPU;
  ACTING means the program is between turns (running a tool, waiting on a
  subagent, or otherwise off-GPU).
- **ACTIVE / PAUSED lifecycle.** A program in PAUSED is unregistered from
  every backend and waits in a global queue until the scheduler resumes it.
- **Tool-boundary admission gate.** ThunderAgent never interrupts an
  in-flight generation. It only blocks the *next* request for a paused
  program, and lets the current generation finish first when marking a
  REASONING program for pause.
- **Best-Fit-Decreasing resume.** When workers free up, paused programs are
  packed onto workers in priority order (continuation > new > acting) using
  BFD against per-worker remaining capacity.

We re-implement these primitives natively in Dynamo because the public
ThunderAgent codebase has several mechanisms that are weaker than what
Dynamo can already provide. The next section spells out exactly what we
kept, what we replaced, and why.

---

## What we kept vs. what we replaced

The table below summarises the deltas relative to the public ThunderAgent
repository at commit `aebad6421abe8e1fbf4e8fdca88f91346176cf29`.

| Concern                  | ThunderAgent (public repo)                               | `dynamo.thunderagent_router` v0                                                  |
| ------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Identity                 | `program_id` only (single-string)                        | Full `nvext.agent_context`: `workflow_type_id`, `workflow_id`, `program_id`, `parent_program_id` |
| Token accounting         | `chars / global_char_to_token_ratio` heuristic           | **Item 1.** Real `prompt_tokens` + `completion_tokens` from response usage; `len(token_ids)` for ISL |
| Capacity signal          | `/metrics` Prometheus polling per backend; `shared_tokens` heuristic that is dead code in the public repo | **Item 3.** `forward-pass-metrics` event-plane subscription; engine-true `sum_decode_kv_tokens` per worker |
| Per-worker capacity cap  | `max_total_num_tokens` from SGLang `/get_server_info`    | `runtime_config.max_num_batched_tokens` from each worker's published model deployment card |
| Acting-token decay       | `2^(-t)` with `t` in seconds, no fitted half-life        | Off in v0; revisit in follow-up using observed inter-turn gaps from agent traces |
| Resume placement         | BFD by remaining capacity; recompute prefix on the new worker | **Item 2.** `KvRouter.best_worker(last_prefix_token_ids)` -- target the warmest worker for the program's last-turn prefix |
| Pause severity           | Binary: pause or not                                     | **Item 6.** Soft (negative `priority_jump` in the next request) before hard pause; soft expires every scheduler tick |
| In-flight decode         | Never interrupted (we keep this)                         | Same: marked-for-pause REASONING programs only pause at next ACTING transition |
| Engine cache control     | None in public repo (`backend.unregister_program` is bookkeeping only) | Same in v0; demote/prefetch APIs are Phase 3 in the design doc and live in a separate workstream |
| Live KV migration        | Logical only -- pause + recompute on resume              | Same in v0; KV-aware resume *placement* mitigates this, but no actual KV transfer |
| Integration              | Standalone Python proxy intercepting OpenAI requests; strips `program_id` before forwarding | Standalone `dynamo_worker` that wraps Dynamo's native `KvRouter`; preserves `agent_context` end-to-end |

If a row is the same in both columns it means the concept is sound and
we are deliberately not changing it for v0.

### What we explicitly chose not to do in v0

These were considered and deferred so the v0 surface stays small enough to
ablate cleanly:

- **Item 4 — Workflow-profile-aware pause selection.** Pausing only programs
  whose predicted idle exceeds resume cost. Requires a `workflow_profile.json`
  artifact built offline from agent traces. See
  [`2026-04-14-oraculus-notes.md`](../../../../../../memory/agent-orchestration-exploration/2026-04-14-oraculus-notes.md).
- **Item 5 — Subagent-aware lifecycle.** Treat programs whose
  `parent_program_id` is set as subagents: never pause subagents (they ride
  the streaming-session path), and bias parent pause toward "child is slow".
  Encodes the finding in
  [`2026-04-30 session policy`](INDEX.md).
- **Item 7 — Step-count-aware fairness aging.** Bias resume priority by
  waited time so step=20 paused programs do not starve under step=2 floods.
- **Items 8, 9, 10, 11 from the design roadmap** (tier-aware cache status,
  external `warm_kv` / `prefetch_kv`, `demote_kv` / `pause_prefix`, full
  program scheduler + cache controller). These are separate workstreams
  tracked in
  [`2026-04-15-dynamo-program-scheduler-design.md`](../../../../../../memory/agent-orchestration-exploration/2026-04-15-dynamo-program-scheduler-design.md).

---

## Architecture

```
                         frontend (--router-mode kv | --router-mode round-robin | ...)
                                       │
                                       │  OpenAI request → preprocessor → PreprocessedRequest
                                       │  (nvext.agent_context survives end-to-end)
                                       ▼
              ┌─────────────────────────────────────────────┐
              │  python -m dynamo.thunderagent_router       │
              │                                             │
              │  ┌──────────────────────────────────────┐   │
              │  │ KvThunderAgentRouter                  │   │
              │  │  - ProgramTable (REASONING/ACTING,    │   │
              │  │    ACTIVE/PAUSED, soft-demote state)  │   │
              │  │  - before_request()  ← admission gate │   │
              │  │  - select_worker()   ← KV-aware resume│   │
              │  │  - after_request()   ← REASONING→ACTING│   │
              │  │  - _scheduler_tick() ← every 5 s      │   │
              │  └──────────────────────────────────────┘   │
              │              │             │                │
              │              │             │ snapshot()     │
              │              │             ▼                │
              │              │   ┌──────────────────┐       │
              │              │   │ FpmCapacityProvider│      │
              │              │   │  subscribes to     │      │
              │              │   │  forward-pass-     │      │
              │              │   │  metrics event plane│     │
              │              │   └──────────────────┘       │
              │              ▼                              │
              │      ┌──────────────────┐                   │
              │      │ KvRouter (PyO3)   │ ── best_worker(prefix)
              │      │ - KV indexer      │      kv-aware resume placement
              │      │ - scheduler queue │      (item 2)
              │      │ - WorkerLoadMon.  │
              │      └──────────────────┘                   │
              │              │                              │
              └──────────────┼──────────────────────────────┘
                             ▼
                       ┌─────────────┐    ┌─────────────┐
                       │ vLLM worker │    │ vLLM worker │ …
                       └─────────────┘    └─────────────┘
                       /metrics emitted via FPM ZMQ → event plane
```

The scheduler service registers
`{namespace}.thunderagent_router.generate` as a Dynamo endpoint. The
frontend's discovery picks it up the same way it picks up any other model
worker; no `--router-mode` change in the frontend is required.

The mapping between original ThunderAgent components and our equivalents:

| ThunderAgent concept                                | Our equivalent                                                |
| --------------------------------------------------- | ------------------------------------------------------------- |
| `MultiBackendRouter`                                | `KvThunderAgentRouter` (`router.py`)                          |
| `Program` dataclass + `ProgramStatus`/`ProgramState`| `program_state.py:Program` + `ProgramStatus`/`ProgramLifecycle`|
| `BackendState` per-backend metrics                  | `FpmCapacityProvider` snapshot per-worker (`capacity.py`)     |
| `update_program_before_request`                     | `before_request()` admission gate                             |
| `update_program_after_request`                      | `after_request()` REASONING→ACTING transition                 |
| `_scheduler_loop` / `_scheduled_check`              | `_scheduler_tick()` (every `--scheduler-interval-seconds`)    |
| `_pause_until_safe`                                 | `_pause_until_safe()` -- same semantics, different signal source |
| `_greedy_resume`                                    | `_greedy_resume()` BFD                                        |
| `_pause_program` / `_mark_program_for_pause`        | `_pause_acting()` / `program.marked_for_pause` flag           |
| `_resume_program`                                   | `_resume_program()`                                           |
| `_wait_for_resume`                                  | `asyncio.Event` per program with timeout-driven forced resume |
| `httpx.AsyncClient` proxy                           | `KvRouter.generate_from_request()` from PyO3 bindings         |

---

## Usage

### Launching

```bash
# 1. Start your Dynamo workers (vLLM example, with KV events on)
DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --block-size 64 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

DYN_SYSTEM_PORT=8082 CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --block-size 64 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# 2. Start the ThunderAgent router pointing at the worker endpoint
python -m dynamo.thunderagent_router \
    --endpoint dynamo.vllm.generate \
    --router-block-size 64 \
    --pause-threshold 0.95 \
    --soft-demote-threshold 0.80 \
    --resume-priority-boost 1.0 &

# 3. Start the frontend (any router mode -- the frontend just needs to find
#    a model handler, which our service registered)
python -m dynamo.frontend --router-mode round-robin &
```

### Sending requests

Clients use the standard OpenAI-compatible API. The only ThunderAgent-specific
contract is `nvext.agent_context.program_id`:

```python
import openai

client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
await client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "..."}],
    extra_body={
        "nvext": {
            "agent_context": {
                "workflow_type_id": "ms_agent",
                "workflow_id": "task-42",
                "program_id": "task-42:researcher",
                "parent_program_id": "task-42:root",  # optional
            }
        }
    },
)
```

Requests **without** `agent_context` are routed normally and bypass all
program-scheduling logic. This keeps the service backward-compatible for
mixed traffic.

### Configuration knobs

| Flag                                | Env var                                              | Default | Description                                                                          |
| ----------------------------------- | ---------------------------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `--endpoint`                        | `DYN_ROUTER_ENDPOINT`                                | --      | Worker endpoint to route to (e.g. `dynamo.vllm.generate`)                            |
| `--router-block-size`               | `DYN_ROUTER_BLOCK_SIZE`                              | 128     | KV cache block size                                                                  |
| `--pause-threshold`                 | `DYN_THUNDERAGENT_PAUSE_THRESHOLD`                   | 0.95    | Hard-pause when worker utilization >= this                                           |
| `--soft-demote-threshold`           | `DYN_THUNDERAGENT_SOFT_DEMOTE_THRESHOLD`             | 0.80    | Soft-demote (negative `priority_jump`) when utilization is between this and `pause-threshold` |
| `--soft-demote-priority-jump`       | `DYN_THUNDERAGENT_SOFT_DEMOTE_PRIORITY_JUMP`         | -2.0    | `priority_jump` (seconds) applied to soft-demoted programs                           |
| `--resume-priority-boost`           | `DYN_THUNDERAGENT_RESUME_PRIORITY_BOOST`             | 1.0     | `priority_jump` (seconds) added to a request that just resumed from hard pause       |
| `--resume-timeout-seconds`          | `DYN_THUNDERAGENT_RESUME_TIMEOUT_SECONDS`            | 1800.0  | Forced-resume after this many seconds; mirrors ThunderAgent's `_wait_for_resume`     |
| `--scheduler-interval-seconds`      | `DYN_THUNDERAGENT_SCHEDULER_INTERVAL_SECONDS`        | 5.0     | Scheduler tick period -- matches ThunderAgent's default                              |

All `KvRouter` flags from `dynamo.router` (`--router-temperature`,
`--use-kv-events`, `--router-track-output-blocks`, etc.) are also
accepted and forwarded.

---

## Lifecycle deep-dive

### When a request arrives

```
generate(request) called by Dynamo runtime
    │
    │ extract_program_id(request)
    ▼
  program_id?
    │
    │ no   ─────► forward to KvRouter.generate_from_request unchanged
    │
    │ yes
    ▼
  scheduler.before_request(program_id)
    │
    │ program is PAUSED?  ──── yes ─────► await waiting_event (with timeout)
    │                                       │
    │                                       │ on timeout: forced resume
    │                                       ▼
    │ no                            (continue)
    ▼
  PauseDecision { priority_jump, was_paused, assigned_worker_hint }
    │
    │ was_paused? ─── yes ──► scheduler.select_worker(program_id, token_ids, was_paused=True)
    │                          │   uses KvRouter.best_worker(last_prefix_token_ids)  ◄── ITEM 2
    │                          ▼
    │                       worker_id (soft pin) or None
    │
    │ fold priority_jump into routing.priority_jump  ◄── ITEM 6
    │
    ▼
  KvRouter.generate_from_request(preprocessed)
    │
    ▼
  on each chunk:
    - capture worker_id from disaggregated_params (first chunk)
    - update prompt_tokens / completion_tokens from completion_usage  ◄── ITEM 1
    - yield chunk to caller
    │
    ▼
  on stream end:
    scheduler.after_request(program_id, prompt_tokens, completion_tokens, last_prefix=token_ids)
        │
        │ marked_for_pause?  ── yes ─► transition to PAUSED, register in global queue
        ▼
       state: REASONING -> ACTING
```

### The scheduler tick (every `scheduler_interval_seconds`)

```
snapshot = capacity.snapshot()  # FPM event-plane data, item 3
    │
    ▼
apply_soft_demotes(snapshot)
    for each worker with soft_demote_threshold <= util < pause_threshold:
        for each ACTIVE program on that worker:
            program.soft_demoted_until = now + 1.5 * tick_interval
    │
    ▼
pause_until_safe(snapshot)
    for each worker with util >= pause_threshold:
        loop while effective_active > limit:
            smallest ACTING program on this worker  → hard pause
            else smallest REASONING program on this worker  → mark_for_pause
        (effective_active accounts for tokens we removed in this loop, since
         the FPM snapshot is stale until the next event)
    │
    ▼
greedy_resume(snapshot)
    candidates = paused_programs sorted by (priority_group, token_total)
        priority groups: REASONING (continuation) > NEW (step==1) > ACTING
    cumulative selection up to total remaining capacity
    BFD placement: largest first into highest-remaining worker
    on resume: signal asyncio.Event so any waiting before_request returns
```

---

## Differences from the v0 Rust prototype

A Rust `ProgramController` lives at
`lib/llm/src/kv_router/program_controller.rs` from earlier in this branch.
It implements the same state machine in-process beside `KvPushRouter`. The
Python service supersedes it for v0 because:

- The integration pattern (PR #8522, `dynamo.thompson_router`) is the
  precedent for stateful routing strategies. Mirroring it keeps the
  contribution model consistent and avoids a Rust trait extension.
- Faster iteration on the scheduler heuristics. v0 ablations matter more
  than in-process zero-hop latency for between-turn admission decisions
  measured against tool gaps in seconds.
- The Rust controller stays as the future "promote to Rust" path once the
  Python design is settled.

---

## Testing

```bash
PYTHONPATH=/path/to/dynamo/components/src:$PYTHONPATH \
    python -m pytest components/src/dynamo/thunderagent_router/tests/ -v
```

Coverage in v0:

- `tests/test_program_state.py` — pure state-machine validation: REASONING /
  ACTING transitions, real token accounting, prefix capture, release
  semantics, status counts.
- `tests/test_router.py` — scheduler logic with mocked `KvRouter` and
  capacity provider:
  - first turn admission is a no-op
  - real token totals end up on the program
  - KV-aware resume calls `kv_router.best_worker(last_prefix)`
  - non-paused turns skip KV-aware resume
  - hard pause + admission block + resume signal release
  - forced resume after timeout
  - soft demote applies the configured negative `priority_jump`
  - `pause_until_safe` pauses the smallest ACTING first

End-to-end / live validation is intentionally out of scope for unit tests;
see the bench harness section below.

---

## Benchmark harness (4-arm comparison)

The intended comparison rows (all on the same model, same workload):

| Arm | Stack                                               | Notes                                            |
| --- | --------------------------------------------------- | ------------------------------------------------ |
| A   | bare vLLM                                           | baseline                                         |
| B   | ThunderAgent + 2× vLLM                              | original reference; baseline for "outer-loop scheduling helps" |
| C   | Dynamo + 2× vLLM (no `thunderagent_router`)         | baseline for "Dynamo's KV-aware routing alone"   |
| D   | Dynamo + 2× vLLM + `dynamo.thunderagent_router`     | this PR                                          |

`examples/backends/vllm/launch/agg_router.sh` is the reference launcher for
arms C and D. Workload: concurrent ms-agent DeepResearch with
`DYN_AGENT_TRACE_*` enabled to capture lifecycle.

Replay-driven offline ablations reuse `python -m dynamo.replay` against
agent traces converted to Mooncake format; toggling individual v0 items
(1, 2, 3, 6) against a single trace lets us attribute wins.

---

## Citation

If you use this package for research, please cite the upstream ThunderAgent
paper:

```bibtex
@misc{kang2026thunderagentsimplefastprogramaware,
      title={ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System},
      author={Hao Kang and Ziyang Li and Xinyu Yang and Weili Xu and Yinfang Chen and Junxiong Wang and Beidi Chen and Tushar Krishna and Chenfeng Xu and Simran Arora},
      year={2026},
      eprint={2602.13692},
      archivePrefix={arXiv},
      primaryClass={cs.OS},
      url={https://arxiv.org/abs/2602.13692},
}
```

## References

- Original ThunderAgent paper: <https://arxiv.org/abs/2602.13692>
- Original ThunderAgent repository: <https://github.com/HaoKang-Timmy/ThunderAgent>
- PR #8522 -- `dynamo.thompson_router` standalone-router integration pattern this service mirrors.
- PR #7260 -- pluggable scheduling policy for the KV router queue (FCFS / WSPT). Our `priority_jump` boosts and demotes ride on top of this.
- Project memory:
  - [`2026-04-06-thunderagent-analysis.md`](../../../../../../memory/agent-orchestration-exploration/2026-04-06-thunderagent-analysis.md) -- public-repo gaps and how Dynamo signals address them.
  - [`2026-04-15-dynamo-program-scheduler-design.md`](../../../../../../memory/agent-orchestration-exploration/2026-04-15-dynamo-program-scheduler-design.md) -- phased design (this v0 covers Phase 0/1/2 admission; Phase 3+ cache APIs are separate).
  - [`2026-05-04-thunderagent-router-v0-scoping.md`](../../../../../../memory/agent-orchestration-exploration/2026-05-04-thunderagent-router-v0-scoping.md) -- v0 scoping decision (this PR's contract).
- Related Dynamo docs:
  - `docs/components/router/router-guide.md` -- KV router fundamentals.
  - `docs/components/frontend/nvext.md` -- the `nvext` extension surface.
  - `docs/features/agentic_workloads.md` -- `agent_context` schema.
