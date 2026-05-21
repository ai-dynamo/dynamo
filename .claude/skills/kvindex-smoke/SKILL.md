---
name: kvindex-smoke
description: KV-index smoke — two plain aggregated KVBM instances publish G2 block events to a kvbm-hub running the KV indexer feature. Issue a request to A, see its blocks indexed and resolvable via the hub's /query API mapped to A; issue the same request to B, see the same block resolve to both A and B. No P/D specialization, no NATS, no consolidator.
---

# Skill: kvindex-smoke

End-to-end smoke for the hub-side KV indexer feature (`/v1/features/indexer`).
Proves the connector→hub publish path and the hub's reverse-lookup query.

Cloned in structure from `disagg-bringup` / `disagg-smoke` but with a
**no-specialization** bringup: two ordinary aggregated vLLM instances
(`kv_role=kv_both`, no `disagg` block), each wired to publish its G2 block
create/remove events to the hub.

## Topology

```
        +-----------------------------+
        | kvbm_hub --features indexer|
        |  ZMQ SUB ingest (OS port)    |
        |  /v1/config  (aggregate)     |
        |  /v1/features/indexer/*     |
        +-----------------------------+
            ^ PUB                ^ PUB
            | (ZMQ)              | (ZMQ)
   +-----------------+   +-----------------+
   | instance A :8000|   | instance B :8001|
   | vLLM + KVBM v2  |   | vLLM + KVBM v2  |
   | kv_role=kv_both |   | kv_role=kv_both |
   +-----------------+   +-----------------+
```

The connector reaches the hub via
`kv_connector_extra_config.leader.hub.{url, features:["indexer"]}` — injected
into the `--kv-transfer-config` JSON so it survives vLLM's EngineCore subprocess
spawn (an env var would be stripped). That blob is **rendered by `kvbmctl`** from
the live hub, so `block_size` / `max_model_len` / `block_layout` and the
`leader.hub` block all come from the hub aggregate; only free fields (tokio
workers, nixl backends, control metrics) are passed as `--kvbm` overrides.

At startup the connector pulls `GET {hub}/v1/config` (the aggregate), resolves
the `indexer` feature, registers `Feature::Indexer` (so the hub reclaims its
index entries on unregister), and — when block size matches the worker page
size — connects a ZMQ `PUB` to the advertised ingest endpoint and wires a
publisher onto the block-registry `EventsManager`. (The legacy
`GET /v1/features/indexer/config` endpoint still exists for back-compat but is
no longer the active path.)

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
- `kvbm_hub` **and `kvbmctl`** build from this worktree (the smoke's
  `start-hub.sh` rebuilds both incrementally, so neither is stale). `kvbmctl` is
  a default feature of `kvbm-hub`, so a plain `cargo build` produces it.
- A GPU big enough for two small instances (spark-gb10 profile: Qwen3-0.6B,
  GMU 0.15 each on GPU 0). No etcd / NATS / frontend required.

## Run

```bash
bash .claude/skills/kvindex-smoke/kvindex-smoke.sh
```

## Assertions (hard; exit non-zero on failure)

1. **Publisher wired** — each instance logs `indexer publisher wired
   instance_id=<u128>`; the smoke extracts A's and B's ids.
2. **Indexed after A** — after a request to A, a block appears in a position
   bucket (`GET /hashes/by_position/X`) owned by A.
3. **Query maps to A** — `POST /query` with that block's hash returns a hit
   whose `instances` include A.
4. **Both after B** — after the same request to B, `POST /query` for the same
   hash returns both A and B.

## Evaluating the index with `kvbmctl` (manual / debugging)

The orchestrator's assertions hit the hub's HTTP API directly (the `/query`
body needs the hash packed as 16 big-endian bytes — see `query_owners()`).
For manual inspection of a running smoke, `kvbmctl` exposes the same endpoints
as typed subcommands, and `query` does the `u128` → byte packing for you. With
the hub up on `:1337`:

```bash
export KVBMCTL_HUB=http://127.0.0.1:1337   # or pass --hub on each command
KVBMCTL=./target/debug/kvbmctl             # built by start-hub.sh

# Indexer config (assertion 0: sizing + ZMQ ingest endpoint)
$KVBMCTL get indexer config
# { "block_size": 16, "max_seq_len": 1024, "num_positions": 64,
#   "zmq_endpoint": "tcp://127.0.0.1:33273" }

# Assertion 2: which blocks are indexed at a position bucket, and who holds them
$KVBMCTL get indexer by-pos 0
# { "position": 0, "entries": [ { "hash_u128": "1665…3376",
#     "instances": ["<B-id>", "<A-id>"], … } ] }

# Assertions 3+4: resolve a hash (decimal u128 from by-pos) to its holders.
# kvbmctl packs the u128 into the wire byte-array — no manual encoding.
$KVBMCTL get indexer query 166542759488764189892533901512933376
# { "hit": { "hash_u128": "1665…3376", "instances": ["<B-id>", "<A-id>"], … } }
```

`--hub` (env `KVBMCTL_HUB`) may appear anywhere after the subcommand — it is a
global arg, so `kvbmctl get indexer by-pos 0 --hub $HUB` also works. These are
read-only; run them at any point while the smoke (or any hub with the
`indexer` feature) is up.

## Files

| File | Purpose |
|---|---|
| `start-hub.sh` | Thin wrapper over `kvbm-hub-bringup/start-hub.sh`: pins `KVBM_HUB_FEATURES=indexer` + derives `block_size`/`max_seq_len`/`g2` from the hardware profile. |
| `launch-instance.sh` | One plain aggregated vLLM + KVBM v2 instance; renders `--kv-transfer-config` (incl. `leader.hub`) via the shared `kvbm_hub_render_vllm` helper. |
| `kvindex-smoke.sh` | Orchestrator: bringup + the four assertions above. |

Reuses the **`kvbm-hub-bringup`** skill for the hub launcher (`start-hub.sh`)
and config rendering (`hub-lib.sh`'s `kvbm_hub_render_vllm`), and
`disagg-bringup/{hardware-profiles,new-experiment}.sh` for sizing and the
experiment dir. Does **not** modify any `disagg-*` skill.
