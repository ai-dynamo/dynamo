---
name: p2p-smoke
description: Hub-mediated P2P G2 block transfer smoke. Two same-model vLLMs + hub. R1 warms A's G2, hub /control/transfer/{open,pull,close} primitives copy blocks A→B via velo, R2 verifies B's G2 cache hit. Catches regressions in the leader transfer surface and the cross-instance pull plumbing.
user-invocable: true
disable-model-invocation: true
---

# Skill: KVBM P2P G2 transfer smoke

End-to-end exercise of the hub-orchestrated P2P block transfer path:

```
   ┌─────────────┐         ┌─────────┐         ┌─────────────┐
   │ instance_a  │◄────────│   hub   │────────►│ instance_b  │
   │  port 8000  │ velo    │  HTTP   │  velo   │  port 8002  │
   └─────────────┘         └─────────┘         └─────────────┘
                                ▲
                                │ /control/transfer/{open,pull,close}
                                │
                              client (smoke)
```

The smoke chains the three existing hub HTTP endpoints client-side:

1. `POST /v1/instances/{A}/control/transfer/open_session` — body has the ISL block hashes. Hub forwards to A's leader via velo; A opens a session over its matched G2 blocks. Returns a `TransferSessionCapability`.
2. `POST /v1/instances/{B}/control/transfer/pull_from_session` — body has the capability + `source_instance_id=A`. Hub forwards to B's leader; B opens a velo connection to A's session, pulls the blocks into B's G2 (long-poll). Returns the `pulled` list.
3. `POST /v1/instances/{A}/control/transfer/close_session` — cleanup.

A future iteration may collapse these into one hub orchestration endpoint (`/v1/features/p2p/copy_blocks`), at which point the smoke swaps in a single curl. The smoke stays the regression gate either way.

## What it asserts

- **R1 fired G1→G2 offload on A.** Greps A's audit log for `event="offload_register_complete"` with `dst=…G2`. This is the new audit event in `kvbm-engine/src/offload/pipeline.rs` (emitted on every batch register into the destination tier).
- **The hashes from A's audit are findable on A.** `open_session(find_mode=sync)` returns a `committed` list — must be non-empty for the same hashes A just registered.
- **B pulled the blocks.** `pull_from_session` returns a `pulled` list matching what was committed.
- **B's G2 now serves them.** R2 issues the same prompt to B; vLLM's `Cache Hit Rates` line surfaces a non-zero prefix cache match. The same `offload_register_complete` event should appear in B's audit log with B's hashes matching A's.

## Prerequisites

- A current Dynamo/KVBM environment with `kvbm` importable. In a dev
  container use `/opt/dynamo/venv`; in a local sandbox use
  `/dynamo:kvbm:sandbox-venv` + `/dynamo:kvbm:maturin-dev`.
- `target/debug/kvbm_hub` built: `cargo build --bin kvbm_hub`. The hub
  launcher rebuilds it incrementally unless `KVBM_HUB_SKIP_BUILD=1`.
- Hardware profile selected via `P2P_HARDWARE_PROFILE`. The default
  `h100-a100` profile uses two GPUs and the experiment model
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. The `spark-gb10` profile keeps
  the original one-GPU, low-memory Qwen path available.
- Profile defaults live in `../disagg-bringup/hardware-profiles.sh`; the smoke
  workflow sources that file and should not embed machine-specific sizing.
- Hub free at ports 1337/8337/1338, vLLMs free at 8000/8002.

## Run

```bash
bash .claude/skills/p2p-smoke/p2p-smoke.sh
```

Outputs an experiment directory under `$KVBM_EXPERIMENTS_DIR/<ts>-p2p-smoke/` containing:

| File | Contents |
|---|---|
| `hub.log` | kvbm_hub stdout/stderr |
| `instance_a.log` | A's vLLM + kvbm_audit stream |
| `instance_b.log` | B's vLLM + kvbm_audit stream |
| `trace.html` | three-lane visualization (A | Hub | B) via `p2p-trace.py` |

The script prints a validation report at the end and the trace.html path.

## Env vars

| Var | Default | What |
|---|---|---|
| `KVBM_REPO` | inferred from script location | Repo root for sibling skill scripts + hub binary |
| `P2P_HARDWARE_PROFILE` | `h100-a100` | Hardware sizing profile: `h100-a100`, `spark-gb10`, or `custom` |
| `P2P_MODEL` | profile-specific | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` for `h100-a100`; `Qwen/Qwen3-0.6B` for `spark-gb10` unless overridden |
| `P2P_GMU` | profile-specific | vLLM `--gpu-memory-utilization` for each instance |
| `P2P_CACHE_GB` | profile-specific | KVBM host cache size per instance |
| `P2P_A_CUDA_VISIBLE_DEVICES` | profile-specific | GPU for instance A |
| `P2P_B_CUDA_VISIBLE_DEVICES` | profile-specific | GPU for instance B |
| `P2P_MAX_MODEL_LEN` | profile-specific | vLLM `--max-model-len` |
| `P2P_BLOCKS` | `16` | Number of full G2 blocks the prompt should fill (advisory — prompt is hand-tuned) |
| `KVBM_BLOCK_LAYOUT` | `operational` | Threaded through `kv_connector_extra_config.default.block_layout`; both instances use the same value (P2P feature compat is enforced by the hub) |
| `KVBM_EXPERIMENT_LABEL` | `p2p-smoke` | Suffix on the experiment directory name |

Profile defaults:

| Profile | Model | GPUs | GMU | Cache | Max len |
|---|---|---:|---:|---:|---:|
| `h100-a100` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | A=0, B=1 | `0.70` | `16 GiB` | `2048` |
| `spark-gb10` | `Qwen/Qwen3-0.6B` | A=0, B=0 | `0.15` | `2 GiB` | `1024` |
| `custom` | caller-set | caller-set | caller-set | caller-set | caller-set |

## When to run this

- Before merging changes to:
  - `lib/kvbm-engine/src/leader/session/{leader,initiator,responder,server,staging}.rs` (session lifecycle on both sides).
  - `lib/kvbm-engine/src/leader/control/modules/transfer.rs` (HTTP→leader handlers).
  - `lib/kvbm-hub/src/features/control_plane/manager.rs` (hub→leader velo dispatch).
  - `lib/kvbm-engine/src/offload/pipeline.rs` (the dst-tier register site that emits `offload_register_complete`).
- As a tripwire after merging anything that touches velo's session/pull paths.

## What this skill does NOT cover (yet)

- **Multi-tier matching.** Always G2-only (`tiers.g3 = false`). A G3-staged source would need `tiers.g3 = true` and the holder's G3 pre-populated.
- **Asymmetric TP / PP.** Both instances run TP=1, identical layout. The hub's P2P feature has layout-compat gating that *would* reject mismatched groups, but this smoke doesn't exercise the rejection path.
- **Concurrent pulls.** Single open → single pull. The session API supports multiple pullers attaching to one session; not tested here.
- **Failure modes.** Session watchdog timeout, half-pull errors, source-instance teardown mid-pull — all separate smokes.

## Assets

| File | Purpose |
|---|---|
| `p2p-smoke.sh` | Orchestrator. Tears down, brings up hub + 2 vLLMs, drives the 3-call chain, runs R2, prints report. |
| `launch-instance.sh` | Parameterized single-instance launcher. `P2P_PORT` + `P2P_ROLE` chosen by the orchestrator. |
| `SKILL.md` | This file. |
| `../disagg-trace/p2p-trace.py` | HTML render of the kvbm_audit timeline across the three logs. Lanes are `Instance A | Hub | Instance B`. |

## See also

- `/dynamo:kvbm:disagg-smoke` (`two-request-smoke.sh`) — the conditional-disagg sibling that exercises the per-request CD flow. P2P differs in that there is no inference-side coordination; the hub primitives are driven externally by the smoke.
- `lib/kvbm-protocols/src/control/modules/transfer.rs` — request/response schemas for the three transfer endpoints. The wire shape changes here will likely break this smoke first.
