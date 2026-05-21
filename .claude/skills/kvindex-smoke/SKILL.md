---
name: kvindex-smoke
description: KV-index smoke — two plain aggregated KVBM instances publish G2 block events to a kvbm-hub running the KV indexer feature. Issue a request to A, see its blocks indexed and resolvable via the hub's /query API mapped to A; issue the same request to B, see the same block resolve to both A and B. No P/D specialization, no NATS, no consolidator.
---

# Skill: kvindex-smoke

End-to-end smoke for the hub-side KV indexer feature (`/v1/features/kv-index`).
Proves the connector→hub publish path and the hub's reverse-lookup query.

Cloned in structure from `disagg-bringup` / `disagg-smoke` but with a
**no-specialization** bringup: two ordinary aggregated vLLM instances
(`kv_role=kv_both`, no `disagg` block), each wired to publish its G2 block
create/remove events to the hub.

## Topology

```
        +-----------------------------+
        | kvbm_hub  --kv-index         |
        |  ZMQ SUB ingest (OS port)    |
        |  /v1/features/kv-index/*     |
        +-----------------------------+
            ^ PUB                ^ PUB
            | (ZMQ)              | (ZMQ)
   +-----------------+   +-----------------+
   | instance A :8000|   | instance B :8001|
   | vLLM + KVBM v2  |   | vLLM + KVBM v2  |
   | kv_role=kv_both |   | kv_role=kv_both |
   +-----------------+   +-----------------+
```

The hub URL reaches the connector via
`kv_connector_extra_config.leader.events.kv_index_hub_url` — injected into the
`--kv-transfer-config` JSON so it survives vLLM's EngineCore subprocess spawn
(an env var would be stripped). The connector probes
`GET {hub}/v1/features/kv-index/config`; when the feature is present and its
`block_size` matches the worker page size, it connects a ZMQ `PUB` to the
advertised endpoint and wires a publisher onto the block-registry
`EventsManager`.

## Prereqs

- **Each worktree has its own isolated `.sandbox` venv.** Never point
  `KVBM_VENV` at another worktree's `.sandbox` — `maturin develop` would
  overwrite that venv's `kvbm` `.so` with this branch's connector. Create this
  worktree's venv once:
    1. `kvbm-sandbox-venv` (`/dynamo:kvbm:sandbox-venv --fresh`) — builds
       `.sandbox` with torch+cu130 / vllm nightly / nccl≥2.29 / nixl. The
       deterministic path reuses a `requirements.release-pinned.txt` freeze.
       **ai-dynamo is NOT required** — this smoke drives raw
       `vllm.entrypoints.openai.api_server`, not `dynamo.frontend`.
    2. `kvbm-maturin-dev` (`/dynamo:kvbm:maturin-dev`) — builds **this
       worktree's** `kvbm` extension (the connector change lives in the `.so`),
       then re-bumps nccl.
  Override the venv path with `KVBM_VENV` only to point at *this* worktree's
  `.sandbox`.
- `kvbm_hub` builds from this worktree (the smoke's `start-hub.sh` rebuilds it
  incrementally, so it is never stale).
- A GPU big enough for two small instances (spark-gb10 profile: Qwen3-0.6B,
  GMU 0.15 each on GPU 0). No etcd / NATS / frontend required.

## Run

```bash
bash .claude/skills/kvindex-smoke/kvindex-smoke.sh
```

## Assertions (hard; exit non-zero on failure)

1. **Publisher wired** — each instance logs `kv-index publisher wired
   instance_id=<u128>`; the smoke extracts A's and B's ids.
2. **Indexed after A** — after a request to A, a block appears in a position
   bucket (`GET /hashes/by_position/X`) owned by A.
3. **Query maps to A** — `POST /query` with that block's hash returns a hit
   whose `instances` include A.
4. **Both after B** — after the same request to B, `POST /query` for the same
   hash returns both A and B.

## Files

| File | Purpose |
|---|---|
| `start-hub.sh` | Launch `kvbm_hub --kv-index` (no CD prefill dispatcher). |
| `launch-instance.sh` | One plain aggregated vLLM + KVBM v2 instance; injects `events.kv_index_hub_url`. |
| `kvindex-smoke.sh` | Orchestrator: bringup + the four assertions above. |

Reuses `disagg-bringup/{hardware-profiles,new-experiment}.sh` for sizing and
the experiment dir. Does **not** modify any `disagg-*` skill.
