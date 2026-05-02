---
name: disagg-smoke
description: KVBM conditional-disagg two-request smoke (R1 cold-cache + R2 warm-decode). Modes — single-leader (default, uses KVBM_DISAGG_LEADER from env) or audit-equiv (runs the smoke twice — legacy then unified — and diffs the per-side kvbm_audit streams via the audit_diff binary).
---

# Skill: KVBM Disagg Smoke

End-to-end two-request smoke against a live `kvbm-hub` + Prefill + Decode topology. Drives `R1` (cold-cache, prefill computes 3 blocks) followed by `R2` (warm decode, prefill G2 reset → decode forwards 3 cached hashes, prefill pulls + forward-passes net-new). Owns its scripts; depends on `disagg-bringup` for launch helpers.

## Skill assets

| File | Purpose |
|---|---|
| `two-request-smoke.sh` | Drive R1 + R2 against a single leader. Mints an experiment dir, brings up hub + Prefill + Decode (via `disagg-bringup` scripts), runs the two prompts, prints a validation grep digest, renders `trace.html` via `disagg-trace`. Inherits `KVBM_DISAGG_LEADER` from env. |
| `audit-equiv.sh` | Run the smoke twice (`KVBM_DISAGG_LEADER=legacy` then `=unified`), then run `audit_diff` on per-side `kvbm_audit` streams to assert behavioral equivalence. |

## Modes

### Single-leader (default)

```bash
SKILL=/home/ryan/repos/dynamo/.claude/skills/disagg-smoke
bash "$SKILL/two-request-smoke.sh"                              # legacy (default)
KVBM_DISAGG_LEADER=unified bash "$SKILL/two-request-smoke.sh"   # unified
```

Output goes to `/tmp/kvbm-experiments/<ts>-two-request/`:
- `hub.log` / `prefill.log` / `decode.log`
- `trace.html` (rendered if `disagg-trace` is available)

### Audit-equivalence

```bash
bash "$SKILL/audit-equiv.sh"
```

Runs the smoke twice (legacy then unified). Per-side `audit_diff --normalize-request-ids --filter-prefixes session_factory_active_gauge` asserts the leader-emitted audit signatures match. Exit 0 on equivalence, 1 on divergence.

Prerequisite: `cargo build -p kvbm-connector --bin audit_diff` (the binary lives at `target/debug/audit_diff`).

## Prerequisites

- `/disagg-bringup` prerequisites (canonical venv, fresh bindings, kernels lib, hub binary).
- `audit_diff` binary built (only required for `audit-equiv` mode).

## Validation expectations (R1 + R2)

- **R1 prefill**: cold cache, ~0/N hit rates; G1→G2 register events fire after forward pass.
- **R1 decode**: full pull pipeline (`worker_pull_chunk_start → worker_session_pull_call → worker_session_pull_returned → worker_g2_to_g1_done`).
- **R2 decode** policy_decision: `matched_tokens >= 48` (3 blocks of warm cache from R1).
- **R2 prefill**: `cd_bound_ensure_started` with `num_sequence_hashes=3`, then `ensure_started_async_onboard` (NOT `ensure_started_zero_passthrough`).
- **R2 prefill** pull-from-decode events fire (`session_pull_request → session_pull_send → session_pull_rdma_*`).
- **R2 decode** pull-from-prefill events fire (the net-new blocks).
- All three logs: zero ERROR lines (excluding the standard UCX / kernel_config noise).

## See also

- `/disagg-bringup` — owns launch-prefill / launch-decode / start-hub / new-experiment scripts.
- `/disagg-teardown` — cleanup between runs.
- `/disagg-trace` — render `trace.html` from an experiment dir.
- `/disagg-hub-curl` — exercise hub HTTP endpoints during a run.
- `/kvbm-maturin-dev` — rebuild bindings.
