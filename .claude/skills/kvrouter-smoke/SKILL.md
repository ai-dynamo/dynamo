---
name: kvrouter-smoke
description: Smoke test for KV-aware routing — bring up dynamo.frontend (with embedded `--router-mode kv`) plus two dynamo.vllm workers (KVBM v2 connector, prefix caching ON). Issue R1 (cold prompt) and R2 (same prompt) via the frontend's OpenAI HTTP API; assert R2 lands on the same worker as R1 and that R2's response shows cached prompt tokens. NATS-path KV events only (no consolidator, no hub).
---

# Skill: kvrouter-smoke (Stage 1)

End-to-end smoke that proves the KV-aware router sends R2 (warm) to the
worker that holds R1's prefix-cache blocks.

## Stage scope

- **Stage 1 (this skill)**: NATS-path KV events. Workers publish G1
  prefix-cache events via `KvEventPublisher`; the embedded KV router in
  `dynamo.frontend` routes on G1 overlap. No `kvbm-consolidator`. No
  `kvbm-hub` / cross-instance P2P pull.

  `DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR=false` is *required* on each
  worker: `setup_kv_event_publisher` in `dynamo.vllm` will route
  `KvEventPublisher` to a consolidator output port whenever
  `consolidator_endpoints` is set, but the v2 connector path
  (`lib/bindings/kvbm/python/kvbm/v2/`) does not currently spawn a
  consolidator process — `lib/kvbm-consolidator/` is built but has no
  v2 consumer. Without this env var the publisher subscribes to a
  dead port and zero KV events flow, defeating kv-aware routing.

- **Stage 2 (future)**: spawn `lib/kvbm-consolidator` from the v2
  connector (e.g. via a new PyO3 binding around `ConsolidatorBuilder`).
  Pre-req: add `tracing::info!`-level per-batch logs at the
  consolidator ingress (vLLM ZMQ + kvbm-engine `EventsManager`) and
  egress so the smoke can assert both event streams arrive. The
  consolidator's current tracing is `warn!`/`error!` only — no
  positive-signal observability today.
- **Stage 3 (future)**: standalone consolidator binary + velo-based
  KVBM event stream.

## Topology

```
                +-----------------------------------+
   user --HTTP->| dynamo.frontend  (--router-mode kv)|
                |    embedded KvRouter, NATS-events  |
                +-----------------------------------+
                       |               |
            generate   |               | generate
                       v               v
              +---------------+ +---------------+
              | dynamo.vllm A | | dynamo.vllm B |
              | namespace=    | | namespace=    |
              |   dynamo      | |   dynamo      |
              | component=    | | component=    |
              |   backend     | |   backend     |
              | endpoint=     | | endpoint=     |
              |   generate    | |   generate    |
              | KVBM v2       | | KVBM v2       |
              | prefix-cache  | | prefix-cache  |
              | --kv-events   | | --kv-events   |
              |   tcp://*:5557| |   tcp://*:5567|
              +---------------+ +---------------+
                       \              /
                        \            /
                         v          v
              KV events  ZMQ -> KvEventPublisher -> NATS
                                                    |
                                                    v
                                          frontend's KvIndexer
```

Both workers register under the same `namespace.component.endpoint` so
the router treats them as two instances of the same logical service.
GMU is 0.15 per instance — same headroom as the P2P smoke for the
Spark's unified-memory GB10.

## Why no standalone `dynamo.router`?

The frontend's `--router-mode kv` *is* a KV router. Adding a separate
`dynamo.router` for an agg-only test creates a redundant indexer
without changing behavior. The standalone-router pattern is for split
prefill/decode pools (see `docs/components/router/`). Stage 2 may
reintroduce it if we need `get_overlap_scores` as a separate API
surface.

## Prereqs

- etcd reachable at `http://127.0.0.1:2379` (validated by the smoke).
- nats-server reachable at `nats://127.0.0.1:4222` (validated).
- venv with **both** `kvbm` (maturin develop on `lib/bindings/kvbm`)
  AND `ai-dynamo` + `ai-dynamo-runtime` (maturin develop on
  `lib/bindings/python`, then `uv pip install -e .` from repo root).
  Re-bump nccl after each maturin: see kvbm-maturin-dev skill.
- Default venv: `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox`
  (override via `KVBM_VENV`).

## Run

```bash
bash .claude/skills/kvrouter-smoke/kvrouter-smoke.sh
```

The script:

1. Validates etcd + NATS reachable.
2. Tears down stale vllm / dynamo / kvbm_hub processes.
3. Mints an experiment dir via `disagg-bringup/new-experiment.sh`.
4. Launches worker A on `--kv-events tcp://*:5557` (sequential — vLLM
   profiler races on unified memory).
5. Waits for worker A to register in etcd, then launches worker B on
   `--kv-events tcp://*:5567`.
6. Waits for both workers to be discoverable.
7. Launches `dynamo.frontend --router-mode kv --http-port 8080`.
8. Waits for the frontend's `/v1/models` to return the model.
9. **R1**: POST `/v1/completions` with a long prompt; record which
    worker handled it (by parsing the worker logs for the request id).
10. Sleep a few seconds to let G1 prefix-cache events flush over NATS
    into the router's indexer.
11. **R2**: POST the same prompt; record which worker handled it.
12. Hard assertions (exit non-zero on failure). Five signals from
    independent surfaces, structured per `tests/CLAUDE.md`:

    - **A. Co-location** (necessary): R1 and R2 routed to the same
      `worker_id`, parsed from the router's `Selected worker:`
      tracing event in `frontend.log`. Necessary but not sufficient
      — a router with zero KV signal can still deterministically
      pick the same worker via tiebreakers.

    - **B. KV-aware logic fired** (primary): R2's selected `logit` is
      strictly lower than R1's. The router emits one logit per
      worker; lower wins; overlap with the worker's cached prefix
      subtracts from `prefill_cost` (see
      `lib/kv-router/src/scheduling/selector.rs:107-168, 302-309`).
      If the logit didn't drop on R2, no overlap credit was applied
      and A is a false positive from a deterministic tiebreaker.

    - **C. Prefix cache actually served** (corroborating): R2's
      `usage.prompt_tokens_details.cached_tokens` > `prompt_tokens / 2`,
      from vLLM's OpenAI response. Independent of any router-internal
      log; rules out the case where routing worked but the chosen
      worker's prefix cache wasn't there to serve the request.

    - **D. Consolidator observed both ingress streams**: parses
      `kvbm_consolidator_audit event="ingress_zmq"` and
      `event="ingress_kvbm"` from worker logs (ANSI-stripped). Both
      counts must be > 0 across the two worker logs. Proves the
      v2 connector actually wires both vLLM ZMQ + kvbm-engine
      `EventsManager` into the consolidator's tracker.

    - **E. Consolidator emitted downstream**: parses
      `event="egress"` from worker logs. Count must be > 0. Proves
      the tracker → ZMQ publisher path is alive, not just the
      ingress side.

## Reuse

- `.claude/skills/disagg-bringup/new-experiment.sh` — experiment dir.
- Teardown pattern from `.claude/skills/disagg-teardown/` (inlined as
  step 2; we extend the pkill set to include `dynamo.vllm`,
  `dynamo.frontend`).
- Does **not** reuse `disagg-bringup/launch-*.sh` or
  `p2p-smoke/launch-instance.sh` — those use raw
  `vllm.entrypoints.openai.api_server` (no etcd registration) with
  KVBM disagg config + `--no-enable-prefix-caching`, both wrong for
  this smoke.

## Known limits

- Prompt must produce ≥ 1 full G1 block. Qwen3-0.6B's default
  `block_size=16` means ≥ 16 tokens — easy. The smoke uses ~320
  tokens for safety.
- Validation relies on log parsing for "which worker served the
  request". If structured tracing helpers land later
  (`tests/CLAUDE.md` guidance), swap the grep for the helper.
- The router takes a few seconds to ingest G1 store events from NATS
  before R2; the smoke sleeps before R2 to bound this.
